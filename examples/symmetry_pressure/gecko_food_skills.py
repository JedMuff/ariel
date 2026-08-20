"""
Symmetry-pressure investigation — food collection task with a state-based
skill-controller brain instead of a single monolithic network.

Outer loop:  (mu+lambda | mu,lambda | nsga2-efficiency | nsga2-symmetry) ES
             over bodies (tree / tree_symmetric / cppn genome, see --genome-type
             and genome_adapter.py) using the ariel.ec engine.
Inner loop:  Three separate CMA-ES runs per body — loco, turn-left, turn-right —
             each trained on their own single-skill task via
             shared.train_skill_for_body. At evaluation time the three trained
             networks are driven by a hand-coded, vision-gated SkillController
             (see run_skill_controller.py) that commits to whichever skill it
             selects for a minimum number of control steps before it is
             allowed to switch.
Task:        Collect food items (waypoints) in sequence. Proprioception + camera.
             Same fitness definition as gecko_food.py (see run_episode_food).

With --loco-only, the left/right turn training and the food-task evaluation are
skipped entirely: only the loco skill is trained and fitness is its own
(negative) forward-displacement score.

Independently of --loco-only, every individual is first gated on its own loco
fitness: if the loco skill isn't fast enough (fitness > LOCO_GATE_THRESH), the
turn skills and food-task evaluation are skipped for that individual too, and
its loco fitness is used as-is as the final fitness.

Data saved per individual during evolution (run_data.jsonl):
  gen, ind_id, parent_ids, fitness, loco_only, genome_type,
  loco_learning_curve, left_learning_curve, right_learning_curve,
  loco_eval_time_s, left_eval_time_s, right_eval_time_s, food_eval_time_s,
  trajectory, food_positions, skill_pct,
  control_cost, yz_symmetry, genome_hash, c_hinge, initial_height
"""

import argparse
import gc
import json
import math
import random
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import mujoco
import numpy as np
import torch
from rich.console import Console
from rich.traceback import install

from ariel.ec import EA, EAOperation, EASettings, Individual, Population
from ariel.simulation.controllers.utils.data_get import get_state_from_data as get_robot_state
from ariel.simulation.environments import SimpleFlatWorld

import genome_adapter
from shared import (
    LOCO_DURATION,
    RING_R_MAX,
    SETTLE_DURATION,
    CONTROL_STEP_FREQ,
    CTRL_ALPHA,
    GATE_HALF_HEIGHT,
    SPAWN_POSITION,
    TURN_DURATION,
    Network,
    SkillReward,
    action_control_cost,
    analyze_sections,
    fill_parameters,
    floor_id,
    genome_input_dim,
    isolate_green,
    nsga2_survivor_selection,
    rotor_geom_ids,
    sample_waypoints,
    train_skill_for_body,
)

install()
warnings.filterwarnings("ignore", message="TPA: apparent inconsistency",
                        category=UserWarning, module="cma")

console = Console()

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Symmetry-pressure: food collection with a state-based skill controller"
)
parser.add_argument("--budget",        type=int,   default=40)
parser.add_argument("--pop",           type=int,   default=20,  help="mu")
parser.add_argument("--lam",           type=int,   default=20,  help="lambda")
parser.add_argument("--brain-workers", type=int,   default=16,
                    help="Parallel workers for each skill's CMA-ES population "
                         "(matches CMA-ES popsize=16 by default)")
parser.add_argument("--reach-radius",  type=float, default=0.20)
parser.add_argument("--num-waypoints", type=int,   default=10)
parser.add_argument("--arena-radius",  type=float, default=3.0)
parser.add_argument("--max-modules",   type=int,   default=25)
parser.add_argument("--max-depth",     type=int,   default=25)
parser.add_argument("--genome-type",
                    choices=["tree", "tree_symmetric", "cppn"],
                    default="tree",
                    help="Body genome representation: plain TreeGenome, TreeGenome with "
                         "enforced y_zero bilateral symmetry, or a CPPN-NEAT genome")
parser.add_argument("--seed",          type=int,   default=42)
parser.add_argument("--strategy-type",
                    choices=["plus", "comma", "nsga2-efficiency", "nsga2-symmetry"],
                    default="plus")
parser.add_argument("--repeat-evals",  action="store_true",
                    help="Re-evaluate parents each generation (plus strategy only)")
parser.add_argument("--loco-only",     action="store_true",
                    help="Train only the loco skill via CMA-ES; skip left/right "
                         "turn training and the food-task evaluation")
parser.add_argument("--no-video",      action="store_true")
parser.add_argument("--time-limit",    type=float, default=None,
                    help="Wall-clock seconds; stop after current generation completes")
args = parser.parse_args()

BUDGET         = args.budget
MU             = args.pop
LAM            = args.lam
BRAIN_WORKERS  = max(1, args.brain_workers)
REACH_RADIUS   = max(0.05, args.reach_radius)
COLLECT_RADIUS = REACH_RADIUS
NUM_WAYPOINTS  = args.num_waypoints
ARENA_RADIUS   = max(1.0, args.arena_radius)
NUM_MODULES    = args.max_modules
MAX_DEPTH      = args.max_depth
GENOME_TYPE    = args.genome_type
BASE_SEED      = args.seed
STRATEGY       = args.strategy_type
REPEAT_EVALS   = args.repeat_evals
LOCO_ONLY      = args.loco_only
TIME_LIMIT     = args.time_limit

TOURNAMENT_SIZE = 4

SCRIPT_NAME = Path(__file__).stem
DATA = Path.cwd() / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True, parents=True)
CHECKPOINTS = DATA / "checkpoints"
CHECKPOINTS.mkdir(exist_ok=True, parents=True)
JSONL_PATH  = DATA / "run_data.jsonl"
TIMING_PATH = DATA / "timing.jsonl"

RNG = np.random.default_rng(BASE_SEED)

with (DATA / "run_config.json").open("w") as _fh:
    json.dump(vars(args), _fh, indent=2)

# ── Fixed skill-pipeline hyperparameters ──────────────────────────────────────
#
# CMA-ES hyperparameters, episode durations, and the hinge-glitch/height-penalty
# fitness terms shared with every other task script now live in shared.py
# (shared.train_skill_for_body) so all task scripts train skills identically.

LOCO_BUDGET          = 100     # CMA-ES generations
TURN_BUDGET          = 100     # CMA-ES generations (per direction)
COMMIT_STEPS         = 40
CENTRE_FWD_THRESH    = 0.4
LOCO_GATE_THRESH     = -1.0    # loco fitness must be <= this to train turn skills
FOOD_EVAL_DURATION   = 120.0   # 2 minutes

ADAPTER = genome_adapter.genome_adapter_from_cli(GENOME_TYPE, NUM_MODULES)


# ── Body-specific world builder (food task only; skill training lives in shared.py) ─


def _build_food_world_for_body(
    genome_dict: dict,
) -> tuple[mujoco.MjModel, mujoco.MjData, int, Optional[str]]:
    """Flat-world body + green food marker + body-mounted camera lookup."""
    spec = ADAPTER.to_spec(genome_dict)
    if spec is None:
        raise ValueError("Could not decode morphology")
    world = SimpleFlatWorld()
    try:
        world.spawn(spec, position=SPAWN_POSITION, correct_collision_with_floor=True)
    except Exception:
        world = SimpleFlatWorld()
        world.spawn(spec, position=SPAWN_POSITION, correct_collision_with_floor=False)

    marker = world.spec.worldbody.add_body(
        name="green_target", mocap=True, pos=[0.0, 0.0, GATE_HALF_HEIGHT]
    )
    marker.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[REACH_RADIUS, GATE_HALF_HEIGHT],
        rgba=[0, 1, 0, 0.7],
        contype=0, conaffinity=0,
    )

    model = world.spec.compile()
    data  = mujoco.MjData(model)
    target_mocap_id = model.body("green_target").mocapid[0]

    cam_name: Optional[str] = None
    for i in range(model.ncam):
        name = model.camera(i).name
        if ("camera" in name or "core" in name) and "overview" not in name:
            cam_name = name
            break

    return model, data, target_mocap_id, cam_name


# ── SkillController (matches run_skill_controller.py / tune_skill_pipeline.py) ─


class SkillController:
    """Stateful controller: commits to each skill for a minimum number of steps."""

    def __init__(
        self,
        rng: np.random.Generator,
        commit_steps: int = COMMIT_STEPS,
        centre_fwd_thresh: float = CENTRE_FWD_THRESH,
    ) -> None:
        self.last_turn_dir:     int   = 2 if rng.random() < 0.5 else 3
        self.commit_steps:      int   = commit_steps
        self.centre_fwd_thresh: float = centre_fwd_thresh
        self._current_skill:    int   = self.last_turn_dir
        self._steps_held:       int   = commit_steps

    def _decide(self, left: float, centre: float, right: float) -> int:
        any_visible = (left + centre + right) > 0.0
        if any_visible:
            if centre >= self.centre_fwd_thresh or (centre >= left and centre >= right):
                return 1
            elif left >= right:
                self.last_turn_dir = 2
                return 2
            else:
                self.last_turn_dir = 3
                return 3
        return self.last_turn_dir

    def select(self, left: float, centre: float, right: float) -> int:
        self._steps_held += 1
        if self._steps_held < self.commit_steps:
            return self._current_skill
        candidate = self._decide(left, centre, right)
        if candidate != self._current_skill:
            self._current_skill = candidate
            self._steps_held    = 0
        return self._current_skill


# ── Food-task evaluation (single episode, in-process) ────────────────────────


def _run_food_episode(
    genome_dict: dict,
    loco_weights: np.ndarray,
    left_weights: np.ndarray,
    right_weights: np.ndarray,
    waypoints: list[np.ndarray],
    seed: int,
) -> dict[str, Any]:
    model, data, target_mocap_id, cam_name = _build_food_world_for_body(genome_dict)
    input_dim  = len(get_robot_state(data))
    output_dim = model.nu

    def _load(w: np.ndarray) -> Network:
        net = Network(input_size=input_dim, output_size=output_dim)
        fill_parameters(net, w)
        return net

    loco_net  = _load(loco_weights)
    left_net  = _load(left_weights)
    right_net = _load(right_weights)
    renderer  = mujoco.Renderer(model, height=96, width=128)

    rotor_ids     = rotor_geom_ids(model)
    floor_geom_id = floor_id(model)

    rng = np.random.default_rng(seed)

    num_wps           = len(waypoints)
    current_wp_idx    = 0
    waypoints_reached = 0
    current_target    = waypoints[0]
    data.mocap_pos[target_mocap_id] = current_target
    min_dist = float("inf")

    while data.time < SETTLE_DURATION:
        mujoco.mj_step(model, data)
        dist = float(np.linalg.norm(np.array(data.qpos[:2]) - current_target[:2]))
        min_dist = min(min_dist, dist)
        if dist <= COLLECT_RADIUS:
            waypoints_reached += 1
            current_wp_idx    += 1
            if current_wp_idx < num_wps:
                current_target = waypoints[current_wp_idx]
                data.mocap_pos[target_mocap_id] = current_target
                min_dist = float("inf")

    core_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1_core")
    initial_height = float(data.xpos[core_id, 2])

    controller     = SkillController(rng, commit_steps=COMMIT_STEPS,
                                     centre_fwd_thresh=CENTRE_FWD_THRESH)
    step           = 0
    current_action = np.zeros(model.nu)
    trajectory: list[list[float]] = []
    ctrl_history: list[np.ndarray] = []
    skill_log: list[int] = []
    c_hinge = 0
    prev_rotor_contacts: set[int] = set()

    episode_end = SETTLE_DURATION + FOOD_EVAL_DURATION
    while data.time < episode_end and current_wp_idx < num_wps:
        if step % CONTROL_STEP_FREQ == 0:
            renderer.update_scene(data, camera=cam_name)
            img    = renderer.render()
            vision = analyze_sections(isolate_green(img))
            left_frac, centre_frac, right_frac = vision
            skill  = controller.select(left_frac, centre_frac, right_frac)
            skill_log.append(skill)

            state = get_robot_state(data).astype(np.float32)
            if skill == 1:
                raw_action = loco_net.forward(model, data, state)
            elif skill == 2:
                raw_action = left_net.forward(model, data, state)
            else:
                raw_action = right_net.forward(model, data, state)

            current_action = np.clip(
                current_action * (1.0 - CTRL_ALPHA) + raw_action * CTRL_ALPHA,
                -math.pi / 2, math.pi / 2,
            )
            trajectory.append([float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2])])

        data.ctrl[:] = current_action
        ctrl_history.append(current_action.copy())
        mujoco.mj_step(model, data)
        step += 1

        curr_rotor_contacts: set[int] = set()
        for k in range(data.ncon):
            c = data.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 == floor_geom_id and g2 in rotor_ids:
                curr_rotor_contacts.add(g2)
            elif g2 == floor_geom_id and g1 in rotor_ids:
                curr_rotor_contacts.add(g1)
        c_hinge += len(curr_rotor_contacts - prev_rotor_contacts)
        prev_rotor_contacts = curr_rotor_contacts

        dist = float(np.linalg.norm(np.array(data.qpos[:2]) - current_target[:2]))
        min_dist = min(min_dist, dist)
        if dist <= COLLECT_RADIUS:
            waypoints_reached += 1
            current_wp_idx    += 1
            if current_wp_idx < num_wps:
                current_target = waypoints[current_wp_idx]
                data.mocap_pos[target_mocap_id] = current_target
                min_dist = float("inf")

    renderer.close()

    final_dist = 0.0 if waypoints_reached >= num_wps else min_dist
    d_norm = float(np.clip((RING_R_MAX - final_dist) / RING_R_MAX, 0.0, 1.0))

    # Height penalty, hinge-contact penalty, and the hinge-glitch override are
    # deliberately not applied here: those concerns are handled during the
    # underlying loco/left/right skill training (see shared._skill_worker_eval)
    # that this episode's skills were trained with, not re-applied on top here.
    # Baseline shift: reaching the food eval at all (i.e. clearing the loco
    # gate) is worth -1, matching LOCO_GATE_THRESH so a gated-in individual
    # is never scored worse than one that was gated out.
    fitness = -1.0 - (waypoints_reached + d_norm)

    total_ctrl_steps = len(skill_log)
    skill_pct = {
        "loco":  100.0 * skill_log.count(1) / max(total_ctrl_steps, 1),
        "left":  100.0 * skill_log.count(2) / max(total_ctrl_steps, 1),
        "right": 100.0 * skill_log.count(3) / max(total_ctrl_steps, 1),
    }

    return {
        "fitness":        fitness,
        "trajectory":     trajectory,
        "control_cost":   action_control_cost(ctrl_history),
        "c_hinge":        c_hinge,
        "initial_height": initial_height,
        "skill_pct":      skill_pct,
    }


# ── Per-individual pipeline: loco -> left -> right -> food eval ──────────────


def _train_pipeline_for_body(
    genome_dict: dict,
    waypoints: list[np.ndarray],
    seed: int,
) -> dict[str, Any]:
    loco_w, loco_curve, loco_time = train_skill_for_body(
        SkillReward(kind="translate", angle_deg=0.0, duration=LOCO_DURATION),
        genome_dict, LOCO_BUDGET, BRAIN_WORKERS, seed, to_spec_fn=ADAPTER.to_spec,
    )
    loco_fitness = float(min(loco_curve)) if loco_curve else float("inf")

    if LOCO_ONLY or loco_fitness > LOCO_GATE_THRESH:
        return {
            "fitness":             loco_fitness,
            "loco_weights":        loco_w,
            "left_weights":        None,
            "right_weights":       None,
            "loco_learning_curve": loco_curve,
            "left_learning_curve": [],
            "right_learning_curve": [],
            "loco_eval_time_s":    loco_time,
            "left_eval_time_s":    0.0,
            "right_eval_time_s":   0.0,
            "food_eval_time_s":    0.0,
            "trajectory":          [],
            "control_cost":        0.0,
            "c_hinge":              0,
            "initial_height":       0.0,
            "skill_pct":            {},
        }

    left_w, left_curve, left_time = train_skill_for_body(
        SkillReward(kind="rotate", turn_sign=+1, duration=TURN_DURATION),
        genome_dict, TURN_BUDGET, BRAIN_WORKERS, seed + 1, to_spec_fn=ADAPTER.to_spec,
    )
    right_w, right_curve, right_time = train_skill_for_body(
        SkillReward(kind="rotate", turn_sign=-1, duration=TURN_DURATION),
        genome_dict, TURN_BUDGET, BRAIN_WORKERS, seed + 2, to_spec_fn=ADAPTER.to_spec,
    )

    t_food_start = time.perf_counter()
    food_result = _run_food_episode(
        genome_dict, loco_w, left_w, right_w, waypoints, seed + 3,
    )
    food_time = time.perf_counter() - t_food_start

    return {
        "fitness":              food_result["fitness"],
        "loco_weights":         loco_w,
        "left_weights":         left_w,
        "right_weights":        right_w,
        "loco_learning_curve":  loco_curve,
        "left_learning_curve":  left_curve,
        "right_learning_curve": right_curve,
        "loco_eval_time_s":     loco_time,
        "left_eval_time_s":     left_time,
        "right_eval_time_s":    right_time,
        "food_eval_time_s":     food_time,
        "trajectory":           food_result["trajectory"],
        "control_cost":         food_result["control_cost"],
        "c_hinge":              food_result["c_hinge"],
        "initial_height":       food_result["initial_height"],
        "skill_pct":            food_result["skill_pct"],
    }


# ── BodyBrainEvolution ────────────────────────────────────────────────────────


class BodyBrainEvolution:
    def __init__(self) -> None:
        self.config = EASettings(
            is_maximisation=False,
            num_steps=BUDGET,
            target_population_size=MU,
            output_folder=DATA,
            db_file_name=f"database_{int(time.time())}.db",
            db_handling="delete",
        )
        self.outer_gen: int = 0
        self.gen_waypoints: list[np.ndarray] = []
        self.selected_parents: list[Individual] = []
        self.best_seen_fitness:  float = float("inf")
        self.best_seen_genotype: Optional[dict] = None
        self.best_seen_weights:  Optional[dict[str, np.ndarray]] = None
        self.best_seen_waypoints: Optional[list[np.ndarray]] = None

    # -- operations -----------------------------------------------------------

    def parent_selection(self, population: Population) -> Population:
        contenders = list(population)
        k = min(TOURNAMENT_SIZE, len(contenders))
        winners: list[Individual] = []
        for _ in range(MU):
            bracket = RNG.choice(len(contenders), size=k, replace=False)
            winner = min(
                (contenders[i] for i in bracket),
                key=lambda ind: ind.fitness_ if ind.fitness_ is not None else float("inf"),
            )
            winners.append(winner)
        self.selected_parents = winners
        winner_ids = {id(w) for w in winners}
        for ind in population:
            ind.tags = {**ind.tags, "ps": id(ind) in winner_ids}
        return population

    def reproduction(self, population: Population) -> Population:
        parents = self.selected_parents or list(population)
        offspring = ADAPTER.make_offspring(parents, LAM, RNG, NUM_MODULES, MAX_DEPTH)

        if STRATEGY == "plus" and REPEAT_EVALS:
            for ind in parents:
                ind.requires_eval = True

        population.extend(offspring)
        return population

    def evaluate(self, population: Population) -> Population:
        gen_rng = np.random.default_rng(BASE_SEED + self.outer_gen)
        # First food is always 1.5 m directly ahead of spawn, so bodies must
        # master forward locomotion before turning is needed for later foods.
        # The core camera looks down world -Y at spawn (verified: core.py's
        # camera euler + the core module never being rotated by the genome),
        # so "ahead" for the vision-driven SkillController is -Y, not +X.
        self.gen_waypoints = [np.array([0.0, -1.5, GATE_HALF_HEIGHT])]
        self.gen_waypoints += sample_waypoints(gen_rng, n=NUM_WAYPOINTS - 1)
        food_positions = [[float(w[0]), float(w[1]), float(w[2])] for w in self.gen_waypoints]

        wp_str = "  ".join(f"({w[0]:.1f},{w[1]:.1f})" for w in self.gen_waypoints)
        console.rule(f"[bold magenta]Outer gen {self.outer_gen} — waypoints {wp_str}")

        to_eval = [
            ind for ind in population
            if ind.alive and ind.tags.get("valid", True) and ind.requires_eval
        ]

        t0 = time.time()
        eval_times: list[float] = []
        results: list[dict[str, Any]] = []
        for idx, ind in enumerate(to_eval):
            genome_dict = ind.genotype[ADAPTER.genotype_key]
            console.log(f"  [{idx + 1}/{len(to_eval)}] training body {ADAPTER.hash(genome_dict)[:8]}...")
            res = _train_pipeline_for_body(
                genome_dict=genome_dict,
                waypoints=self.gen_waypoints,
                seed=BASE_SEED + 1000 * self.outer_gen + idx,
            )
            results.append(res)
            eval_time = (res["loco_eval_time_s"] + res["left_eval_time_s"]
                         + res["right_eval_time_s"] + res["food_eval_time_s"])
            eval_times.append(eval_time)
        wall_elapsed = time.time() - t0

        for idx, (ind, res) in enumerate(zip(to_eval, results)):
            best_fit = float(res["fitness"])

            ind.fitness = best_fit if np.isfinite(best_fit) else float("inf")
            ind.tags = {
                **ind.tags,
                "loco_learning_curve":  res["loco_learning_curve"],
                "left_learning_curve":  res["left_learning_curve"],
                "right_learning_curve": res["right_learning_curve"],
                "loco_eval_time_s":     res["loco_eval_time_s"],
                "left_eval_time_s":     res["left_eval_time_s"],
                "right_eval_time_s":    res["right_eval_time_s"],
                "food_eval_time_s":     res["food_eval_time_s"],
                "trajectory":           res["trajectory"],
                "control_cost":         res["control_cost"],
                "yz_symmetry":          ADAPTER.symmetry_score(ind.genotype[ADAPTER.genotype_key]),
                "food_positions":       food_positions,
                "eval_time_s":          eval_times[idx],
                "c_hinge":              res["c_hinge"],
                "initial_height":       res["initial_height"],
                "skill_pct":            res["skill_pct"],
            }
            ind.requires_eval = False

            self._write_jsonl(ind, food_positions)

            weights = {
                "loco":  res["loco_weights"],
                "left":  res["left_weights"],
                "right": res["right_weights"],
            }
            self._save_checkpoint(
                tag=f"gen{self.outer_gen:03d}_body{idx:02d}",
                genotype=dict(ind.genotype[ADAPTER.genotype_key]),
                weights=weights,
                fitness=best_fit,
                waypoints=list(self.gen_waypoints),
            )

            if best_fit < self.best_seen_fitness and res["loco_weights"] is not None:
                self.best_seen_fitness   = best_fit
                self.best_seen_genotype  = dict(ind.genotype[ADAPTER.genotype_key])
                self.best_seen_weights   = {k: (v.copy() if v is not None else None)
                                             for k, v in weights.items()}
                self.best_seen_waypoints = list(self.gen_waypoints)

        finite = [r["fitness"] for r in results if np.isfinite(r["fitness"])]
        stats = (f"min={np.min(finite):.3f}  avg={np.mean(finite):.3f}"
                 if finite else "all-infinite")
        console.log(f"  {len(to_eval)} bodies in {wall_elapsed:.1f}s  {stats}")
        self._write_timing(wall_elapsed, eval_times)

        self.outer_gen += 1
        return population

    def survivor_selection(self, population: Population) -> Population:
        alive = [ind for ind in population if ind.alive]

        if STRATEGY == "plus":
            alive_sorted = sorted(alive, key=lambda i: i.fitness_ or float("inf"))
            survivors = set(id(i) for i in alive_sorted[:MU])
            for ind in population:
                if ind.alive and id(ind) not in survivors:
                    ind.alive = False

        elif STRATEGY == "comma":
            offspring_alive = [i for i in alive if not i.tags.get("ps", False)]
            offspring_sorted = sorted(offspring_alive, key=lambda i: i.fitness_ or float("inf"))
            survivors = set(id(i) for i in offspring_sorted[:MU])
            for ind in population:
                if ind.alive and id(ind) not in survivors:
                    ind.alive = False

        elif STRATEGY in ("nsga2-efficiency", "nsga2-symmetry"):
            valid   = [i for i in alive if np.isfinite(i.fitness_ or float("inf"))]
            invalid = [i for i in alive if not np.isfinite(i.fitness_ or float("inf"))]
            for ind in invalid:
                ind.alive = False

            if STRATEGY == "nsga2-efficiency":
                objectives = [
                    (i.fitness_ or float("inf"), i.tags.get("control_cost", float("inf")))
                    for i in valid
                ]
            else:
                objectives = [
                    (i.fitness_ or float("inf"), -i.tags.get("yz_symmetry", 0.0))
                    for i in valid
                ]

            chosen = nsga2_survivor_selection(valid, MU, objectives)
            survivors = set(id(i) for i in chosen)
            for ind in valid:
                if id(ind) not in survivors:
                    ind.alive = False

        survivors_alive = [i for i in population if i.alive]
        finite = [i.fitness_ for i in survivors_alive if i.fitness_ is not None and np.isfinite(i.fitness_)]
        if finite:
            console.log(f"[green]Survivors:[/green] avg={np.mean(finite):.3f}  min={np.min(finite):.3f}")
        return population

    # -- persistence ----------------------------------------------------------

    def _write_jsonl(self, ind: Individual, food_positions: list[list[float]]) -> None:
        record = {
            "gen":                   self.outer_gen,
            "ind_id":                ind.id,
            "parent_ids":            ind.genotype.get("parent_ids", []),
            "fitness":               ind.fitness_ if ind.fitness_ is not None else None,
            "loco_only":             LOCO_ONLY,
            "genome_type":           GENOME_TYPE,
            "loco_learning_curve":   ind.tags.get("loco_learning_curve", []),
            "left_learning_curve":   ind.tags.get("left_learning_curve", []),
            "right_learning_curve":  ind.tags.get("right_learning_curve", []),
            "loco_eval_time_s":      ind.tags.get("loco_eval_time_s", 0.0),
            "left_eval_time_s":      ind.tags.get("left_eval_time_s", 0.0),
            "right_eval_time_s":     ind.tags.get("right_eval_time_s", 0.0),
            "food_eval_time_s":      ind.tags.get("food_eval_time_s", 0.0),
            "trajectory":            ind.tags.get("trajectory", []),
            "food_positions":        food_positions,
            "skill_pct":             ind.tags.get("skill_pct", {}),
            "control_cost":          ind.tags.get("control_cost", 0.0),
            "yz_symmetry":           ind.tags.get("yz_symmetry", 0.0),
            "genome_hash":           ADAPTER.hash(ind.genotype[ADAPTER.genotype_key]),
            "eval_time_s":           ind.tags.get("eval_time_s", 0.0),
            "c_hinge":               ind.tags.get("c_hinge", 0),
            "initial_height":        ind.tags.get("initial_height", 0.0),
        }
        with JSONL_PATH.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    def _write_timing(self, wall_elapsed: float, eval_times: list[float]) -> None:
        n = len(eval_times)
        record = {
            "gen":                self.outer_gen,
            "n_individuals":      n,
            "wall_time_s":        round(wall_elapsed, 3),
            "mean_eval_time_s":   round(float(np.mean(eval_times)) if eval_times else 0.0, 3),
            "min_eval_time_s":    round(float(np.min(eval_times)) if eval_times else 0.0, 3),
            "max_eval_time_s":    round(float(np.max(eval_times)) if eval_times else 0.0, 3),
            "total_eval_time_s":  round(float(np.sum(eval_times)) if eval_times else 0.0, 3),
        }
        with TIMING_PATH.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    def _save_checkpoint(self, tag: str, genotype: dict, weights: dict[str, Optional[np.ndarray]],
                         fitness: float, waypoints: Optional[list] = None) -> None:
        if weights.get("loco") is None or not np.isfinite(fitness):
            return
        sub = CHECKPOINTS / tag
        sub.mkdir(exist_ok=True, parents=True)
        for name, w in weights.items():
            if w is not None:
                np.save(sub / f"{name}_weights.npy", w)
        if waypoints is not None:
            np.save(sub / "best_waypoints.npy", np.array(waypoints))
        with (sub / "best_genome.json").open("w") as fh:
            json.dump(genotype, fh, indent=2)
        with (sub / "meta.json").open("w") as fh:
            json.dump({
                "gen":         self.outer_gen,
                "fitness":     fitness,
                "yz_symmetry": ADAPTER.symmetry_score(genotype),
                "num_modules": ADAPTER.module_count(genotype),
                "genome_type": GENOME_TYPE,
            }, fh, indent=2)

    # -- main -----------------------------------------------------------------

    def evolve(self) -> Optional[Individual]:
        console.log("[yellow]Initialising population...[/yellow]")
        population = Population([
            ADAPTER.create_individual(RNG, NUM_MODULES, MAX_DEPTH) for _ in range(MU)
        ])

        population = self.evaluate(population)

        ops = [
            EAOperation(self.parent_selection),
            EAOperation(self.reproduction),
            EAOperation(self.evaluate),
            EAOperation(self.survivor_selection),
        ]
        ea = EA(
            population,
            operations=ops,
            num_steps=BUDGET,
            db_file_path=self.config.db_file_path,
            db_handling=self.config.db_handling,
            quiet=self.config.quiet,
        )
        t_start = time.time()
        for _ in range(BUDGET):
            if TIME_LIMIT is not None and time.time() - t_start >= TIME_LIMIT:
                console.log(f"[yellow]Time limit {TIME_LIMIT:.0f}s reached — stopping early[/yellow]")
                break
            ea.step()
        return ea.get_solution("best", only_alive=False)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    torch.manual_seed(BASE_SEED)

    console.rule("[bold magenta]Symmetry-Pressure — Food Collection (Skill Controller)[/bold magenta]")
    console.log(
        f"genome_type={GENOME_TYPE}  strategy={STRATEGY}  (mu+lam)=({MU}+{LAM})  budget={BUDGET}  "
        f"loco_only={LOCO_ONLY}  loco_gate_thresh={LOCO_GATE_THRESH}  "
        f"loco_budget={LOCO_BUDGET}  turn_budget={TURN_BUDGET}  commit_steps={COMMIT_STEPS}  "
        f"centre_fwd_thresh={CENTRE_FWD_THRESH}  brain_workers={BRAIN_WORKERS}  "
        f"repeat_evals={REPEAT_EVALS}  waypoints={NUM_WAYPOINTS}  seed={BASE_SEED}"
    )

    start = time.time()
    evo = BodyBrainEvolution()
    best = evo.evolve()
    elapsed = time.time() - start

    console.rule("[bold green]Done[/bold green]")
    if best is not None and best.fitness_ is not None:
        console.log(f"Best fitness: {best.fitness:.3f}")
    console.log(f"Best seen:    {evo.best_seen_fitness:.3f}")
    console.log(f"Elapsed:      {elapsed / 60:.1f} min")
    console.log(f"Data →        {DATA}")


if __name__ == "__main__":
    main()
    gc.disable()

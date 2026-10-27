"""
Food-collection benchmark for the distributed ANN on the fixed gecko body,
trained from scratch: one DANN drives the body directly, with no skills and no
state-based SkillController (compare gecko_food_skills.py).

Body:     kgd's canonical zoo gecko (examples/zoo_benchmark/canonical_bodies.py,
          zoo modules and hinge actuator), the body the zoo DANN benchmark
          used. No body evolution. Its bricks are green, the food's colour, so
          they are recoloured grey (only matters when rendering, e.g. videos).
          Its front is the core's free FRONT face (+X; the spine leaves the
          BACK face, the legs the sides), but the zoo core camera looks out of
          the right side (-Y), so here it is turned to look out of the front.
Brain:    shared.DannBrain (dann_pos_hinge_h16) with the camera's view of the
          food as [left, centre, right] fractions broadcast to every hinge as
          extra inputs (45 inputs, 753 params).
Vision:   RayVision: a VISION_RAYS grid of rays cast from the core camera
          (mj_multiRay, CPU only); each third of the image scores the fraction
          of its rays whose first hit is the food. Stands in for rendering the
          camera and counting green pixels (gecko_food_skills.py): within 0.044
          of the rendered 96x128 fractions (mean 0.003), and it scales with
          workers, while EGL renders serialise on the GPU. With --vision-norm
          sum (the default) the three fractions are divided by their sum, so
          the brain gets the food's direction only ([0, 1, 0] straight ahead
          at any distance; [0, 0, 0] out of view) rather than a signal that
          shrinks to ~0.1 at 2 m; raw passes the fractions as they are.
Task:     deterministic: --subtasks separate episodes, each starting from
          the same reset pose with one food --food-radius m away, the foods
          evenly spaced around the circle, the first --ring-offset deg from
          straight ahead of the gecko (world +X; positive = to its left), the
          rest clockwise. Each episode is a 3 s zero-action settle, then
          DSOCIAL_ZOO's 20 Hz control and alpha blending for at most
          --subtask-time s, ending early when the food is collected.
Fitness:  -total_score, minimised, the sum of two parts:
            ordered progress  the subtasks are ordered from straight ahead
                    clockwise. Each collected food scores 1 + --speed-weight x
                    speed, speed = 1 - (time to collect) / --subtask-time
                    (1 = instant); of the foods not collected, only the first
                    in that order adds --progress-weight x its progress,
                    1 - (closest approach to the food's edge) / (start
                    distance to it), in [0, 1) (--progress-mode all: every
                    missed food adds its progress).
            camera coverage  --coverage-weight x each subtask's coverage: the
                    food's share of the camera image averaged over
                    --subtask-time s, the time left after collecting the food
                    counting as full coverage (the camera is at the food then).
          With the default weights (all 1) the maximum is 3 per subtask. The
          optimisers rank candidates, so only the weights' ratios to the
          collection bonus (1) matter. Height and jerk penalties are logged
          but not part of fitness.
Training: --optimizer from x0 = 0 for --gens generations of --pop candidates:
            cma     full-covariance CMA-ES, sigma 0.5
            sepcma  separable (diagonal) CMA-ES, sigma 0.5: adapts faster in 753-D
            oes     OpenAI-ES (Salimans et al. 2017): antithetic pairs, centred
                    rank utilities, Adam on the mean (--oes-sigma, --oes-lr)
          Fitness is deterministic, so the best candidate ever seen is the
          run's controller (best_weights.npy).
Probe:    a fixed test of "collects every food", the same whatever the
          training setup: PROBE_N foods PROBE_RADIUS m away, PROBE_TIME s each,
          at angles halfway between eight evenly spaced directions (22.5 deg,
          67.5 deg, ...). Whenever the best score improves, the new best
          candidate is probed (not counted as training compute). Solved = it
          collects all PROBE_N.
Stopping: the run ends at --gens, or earlier on --stop-on-solve (probe
          solved), --max-sim-seconds (training compute: simulated seconds,
          settles included, summed over all training episodes), --hopeless-gen
          (no candidate has collected any training food by then) or
          --stall-gens (no new best for that many generations).
          summary.json records the stop reason and when the probe was first
          solved (generation, simulated seconds, wall time).
Final:    the controller is replayed on the training subtasks (recorded), on
          --subtasks held-out angles halfway between them and on the probe.

Outputs (cwd/__data__/gecko_food_dann/): run_config.json, learning_curve.jsonl,
best_weights.npy, checkpoints/gen_XXXX.npy, final_eval.json, summary.json.

Usage:
  python gecko_food_dann.py --gens 200 --pop 20 --subtasks 5 --subtask-time 20 --workers 20 --seed 42
"""

import os

# One BLAS/OpenMP thread per process, set before numpy loads (see
# gecko_skill_tasks.py: CMA-ES on a ~750-dim covariance otherwise starves the
# simulation workers).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import math
import sys
import time
import warnings
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cma
import mujoco
import numpy as np
import torch
from rich.console import Console
from shared import (
    DSOCIAL_ZOO,
    GATE_HALF_HEIGHT,
    SETTLE_DURATION,
    SPAWN_POSITION,
    DannBrain,
    floor_id,
    get_robot_state,
    rotor_geom_ids,
)

from ariel.simulation.environments import SimpleFlatWorld

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "zoo_benchmark"))
import brains  # noqa: E402
import canonical_bodies  # noqa: E402

warnings.filterwarnings("ignore", message="TPA: apparent inconsistency",
                        category=UserWarning, module="cma")

console = Console()

parser = argparse.ArgumentParser(description="DANN food benchmark on the fixed gecko")
parser.add_argument("--gens", type=int, default=200, help="Generations")
parser.add_argument("--pop", type=int, default=DSOCIAL_ZOO.cma_popsize, help="Candidates per generation")
parser.add_argument("--optimizer", choices=["cma", "sepcma", "oes"], default="cma")
parser.add_argument("--oes-sigma", type=float, default=0.5, help="OpenAI-ES perturbation scale")
parser.add_argument("--oes-lr", type=float, default=0.1, help="OpenAI-ES Adam learning rate")
parser.add_argument("--subtasks", type=int, default=5, help="Foods (one episode each), evenly spaced on the circle")
parser.add_argument("--food-radius", type=float, default=2.0, help="Distance of every food from the start (m)")
parser.add_argument("--subtask-time", type=float, default=20.0, help="Time limit per subtask after settling (s)")
parser.add_argument("--ring-offset", type=float, default=0.0,
                    help="Angle of the first training food from straight ahead (deg, positive = left); the rest follow clockwise")
parser.add_argument("--coverage-weight", type=float, default=1.0, help="Weight of the camera-coverage score per subtask")
parser.add_argument("--speed-weight", type=float, default=1.0, help="Weight of a collected food's speed score")
parser.add_argument("--progress-weight", type=float, default=1.0, help="Weight of the missed foods' progress")
parser.add_argument("--progress-mode", choices=["first", "all"], default="first",
                    help="Progress of the first missed food in order only, or of every missed food")
parser.add_argument("--sigma", type=float, default=DSOCIAL_ZOO.cma_sigma, help="CMA-ES initial step size")
parser.add_argument("--hidden", type=int, default=16, help="DANN hidden layer size")
parser.add_argument("--vision-norm", choices=["sum", "raw"], default="sum",
                    help="Brain's vision input: fractions divided by their sum (direction only) or raw")
parser.add_argument("--reach-radius", type=float, default=0.20)
parser.add_argument("--workers", type=int, default=os.cpu_count())
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint-every", type=int, default=25)
parser.add_argument("--time-limit", type=float, default=None,
                    help="Wall-clock seconds; stop after the generation in progress")
parser.add_argument("--max-sim-seconds", type=float, default=None,
                    help="Training compute budget in simulated seconds; stop after the generation that exceeds it")
parser.add_argument("--stop-on-solve", action="store_true", help="Stop once the best candidate solves the probe")
parser.add_argument("--hopeless-gen", type=int, default=0,
                    help="Stop if no candidate has collected a training food by this generation (0: never)")
parser.add_argument("--stall-gens", type=int, default=0,
                    help="Stop after this many generations without a new best score (0: never)")
# Defaults when imported (render_food_dann.py); the command line when run.
args = parser.parse_args() if __name__ == "__main__" else parser.parse_args([])

PROFILE = DSOCIAL_ZOO
BODY = "gecko"
BRICK_RGBA = (0.5, 0.5, 0.5, 1.0)
VISION_RAYS = (24, 32)  # rows x columns of the camera ray grid
N_VISION = 3
FORWARD_DEG = 0.0  # the gecko's front (core FRONT face) and its camera face world +X at spawn
PROBE_N, PROBE_RADIUS, PROBE_TIME = 8, 2.0, 30.0  # 30 s: the 300-generation policy needs 21 s for +67.5 deg

DATA = Path.cwd() / "__data__" / Path(__file__).stem
CHECKPOINTS = DATA / "checkpoints"


# ── World ─────────────────────────────────────────────────────────────────────


def build_food_world(reach_radius: float) -> tuple[mujoco.MjModel, mujoco.MjData, int, str | None]:
    """Zoo gecko (grey bricks) + green food marker + body-mounted camera lookup."""
    world = SimpleFlatWorld()
    world.spawn(canonical_bodies.get(BODY).spec, position=SPAWN_POSITION, correct_collision_with_floor=True)

    marker = world.spec.worldbody.add_body(
        name="green_target", mocap=True, pos=[0.0, 0.0, GATE_HALF_HEIGHT],
    )
    marker.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[reach_radius, GATE_HALF_HEIGHT],
        rgba=[0, 1, 0, 0.7],
        contype=0, conaffinity=0,
    )

    model = world.spec.compile()
    data = mujoco.MjData(model)
    for i in range(model.ngeom):
        if model.geom(i).name.startswith("robot1_") and model.geom(i).name.endswith("-brick"):
            model.geom_rgba[i] = BRICK_RGBA
    target_mocap_id = model.body("green_target").mocapid[0]

    cam_name: str | None = None
    for i in range(model.ncam):
        name = model.camera(i).name
        if ("camera" in name or "core" in name) and "overview" not in name:
            cam_name = name
            break
    if cam_name is not None:
        face_camera_forward(model, cam_name)
    return model, data, target_mocap_id, cam_name


def face_camera_forward(model: mujoco.MjModel, cam_name: str) -> None:
    """Turn the core camera to look out of the core's FRONT face, level, with
    the core's +z up (the zoo camera looks out of its right side).
    """
    cam = model.camera(cam_name).id
    front = np.array(model.site("robot1_core-front").pos, dtype=np.float64)  # core frame
    look = np.array([front[0], front[1], 0.0]) / np.linalg.norm(front[:2])
    up = np.array([0.0, 0.0, 1.0])
    z = -look  # a MuJoCo camera looks down its -z
    x = np.cross(up, z)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, up, z]).ravel())
    model.cam_quat[cam] = quat


def food_positions(n: int, radius: float, offset_deg: float = 0.0) -> list[list[float]]:
    """N foods evenly spaced on a circle around the spawn, the first at
    FORWARD_DEG + offset_deg (offset half a spacing for the held-out angles),
    the rest clockwise seen from above.
    """
    out = []
    for k in range(n):
        a = math.radians(FORWARD_DEG + offset_deg - 360.0 * k / n)
        out.append([SPAWN_POSITION[0] + radius * math.cos(a), SPAWN_POSITION[1] + radius * math.sin(a), GATE_HALF_HEIGHT])
    return out


class RayVision:
    """The camera's view of the food as [left, centre, right] fractions: rays
    through the pixel centres of an h x w image with the camera's fovy, each
    third of the columns scoring the share of its rays whose first hit is the
    food geom. The camera's own body is excluded (the camera sits on its
    surface); the rest of the robot occludes as in a render. normalise=True
    divides the three by their sum (direction only); coverage is always the
    raw share of the whole image.
    """

    def __init__(self, model: mujoco.MjModel, cam_name: str, target_geom: int, shape: tuple[int, int],
                 normalise: bool = False) -> None:
        self.model, self.target, (self.h, self.w) = model, target_geom, shape
        self.normalise = normalise
        self.cam = model.camera(cam_name).id
        self.body = int(model.cam_bodyid[self.cam])
        t = math.tan(math.radians(model.cam_fovy[self.cam]) / 2)
        u = (2 * (np.arange(self.w) + 0.5) / self.w - 1) * t * self.w / self.h
        v = (1 - 2 * (np.arange(self.h) + 0.5) / self.h) * t
        uu, vv = np.meshgrid(u, v)
        self.dirs = np.stack([uu, vv, -np.ones_like(uu)], -1).reshape(-1, 3)  # camera frame, looking down -z
        self.geomid = np.zeros(self.h * self.w, np.int32)
        self.dist = np.zeros(self.h * self.w)

    def __call__(self, data: mujoco.MjData) -> np.ndarray:
        vec = (self.dirs @ data.cam_xmat[self.cam].reshape(3, 3).T).ravel()
        mujoco.mj_multiRay(self.model, data, data.cam_xpos[self.cam], vec, None, 1, self.body,
                           self.geomid, self.dist, None, self.h * self.w, mujoco.mjMAXVAL)
        hit = (self.geomid == self.target).reshape(self.h, self.w)
        self.coverage = float(hit.mean())  # the food's share of the whole image
        sections = np.array([s.mean() for s in np.array_split(hit, 3, axis=1)])
        total = sections.sum()
        return sections / total if self.normalise and total > 0 else sections


# ── Episode ───────────────────────────────────────────────────────────────────


def run_subtask(
    ctx: dict[str, Any], food: np.ndarray, record: bool = False,
    recorder: Callable[[mujoco.MjData, np.ndarray, float], None] | None = None,
    time_limit: float | None = None,
) -> dict[str, Any]:
    """One subtask of ctx's brain, whose params are set: reset, settle, then
    walk to the single food until it is collected or the time is up
    (time_limit, default ctx's subtask_time). recorder, if given, is called
    as recorder(data, vision, progress) after every post-settle physics step
    (videos); it must not change data.
    """
    model, data, brain = ctx["model"], ctx["data"], ctx["brain"]
    vision_fn, reach_radius = ctx["vision"], ctx["reach_radius"]
    subtask_time = time_limit or ctx["subtask_time"]

    brain.reset()
    mujoco.mj_resetData(model, data)
    spawn_height = float(data.qpos[2])
    data.mocap_pos[ctx["target_mocap_id"]] = food

    def edge_dist() -> float:
        return max(0.0, math.hypot(data.qpos[0] - food[0], data.qpos[1] - food[1]) - reach_radius)

    mujoco.mj_step(model, data, nstep=round(SETTLE_DURATION / model.opt.timestep))
    start_dist = edge_dist()
    min_dist = start_dist

    step = 0
    current_action = np.zeros(model.nu)
    prev_ctrl: np.ndarray | None = None
    jerk_sum = 0.0
    ctrl_step = 0
    coverage_time = 0.0  # integral of the food's image share over time (s)
    ctrl_dt = PROFILE.control_step_freq * model.opt.timestep
    c_hinge = 0
    prev_rotor_contacts: set[int] = set()
    trajectory: list[list[float]] = []
    vision_log: list[list[float]] = []

    end_time = SETTLE_DURATION + subtask_time
    while data.time < end_time and min_dist > 0.0:
        if step % PROFILE.control_step_freq == 0:
            vision = vision_fn(data)
            state = get_robot_state(data).astype(np.float32)
            raw_action = brain.act(data, state, vision)
            current_action = np.clip(
                current_action * (1.0 - PROFILE.ctrl_alpha) + raw_action * PROFILE.ctrl_alpha,
                -math.pi / 2, math.pi / 2,
            )
            if prev_ctrl is not None:
                jerk_sum += float(np.mean(np.abs(current_action - prev_ctrl)))
            prev_ctrl = current_action.copy()
            ctrl_step += 1
            coverage_time += vision_fn.coverage * ctrl_dt
            if record:
                trajectory.append([float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2])])
                vision_log.append(vision.tolist())

        data.ctrl[:] = current_action
        mujoco.mj_step(model, data)
        step += 1

        # Rotor-ground contact events: diagnostics only (not in fitness), and
        # this per-step Python loop costs more than the physics, so only when recording.
        if record:
            curr_rotor_contacts: set[int] = set()
            for k in range(data.ncon):
                c = data.contact[k]
                g1, g2 = int(c.geom1), int(c.geom2)
                if g1 == ctx["floor_id"] and g2 in ctx["rotor_ids"]:
                    curr_rotor_contacts.add(g2)
                elif g2 == ctx["floor_id"] and g1 in ctx["rotor_ids"]:
                    curr_rotor_contacts.add(g1)
            c_hinge += len(curr_rotor_contacts - prev_rotor_contacts)
            prev_rotor_contacts = curr_rotor_contacts

        min_dist = min(min_dist, edge_dist())
        if recorder is not None:
            recorder(data, vision, float(np.clip(1.0 - min_dist / max(start_dist, 1e-6), 0.0, 1.0)))

    collected = float(min_dist <= 0.0)
    progress = float(np.clip(1.0 - min_dist / max(start_dist, 1e-6), 0.0, 1.0))
    sim_time = float(data.time) - SETTLE_DURATION
    speed = float(np.clip(1.0 - sim_time / subtask_time, 0.0, 1.0)) if collected else 0.0
    time_left = subtask_time - sim_time if collected else 0.0
    coverage = float(np.clip((coverage_time + time_left) / subtask_time, 0.0, 1.0))

    mean_jerk = jerk_sum / max(ctrl_step - 1, 1)
    jerk_penalty = PROFILE.jerk_weight * mean_jerk if mean_jerk >= PROFILE.jerk_threshold else 0.0
    initial_height = spawn_height if PROFILE.height_at_spawn else float(data.qpos[2])
    height_penalty = initial_height if initial_height > PROFILE.height_threshold else 0.0

    out: dict[str, Any] = {
        "score": 1.0 + speed if collected else 0.0, "coverage": coverage,
        "collected": collected, "progress": progress, "speed": speed,
        "jerk_penalty": jerk_penalty, "height_penalty": height_penalty,
        "sim_time": sim_time,
    }
    if record:
        out["c_hinge"] = c_hinge
        out["trajectory"] = trajectory
        out["vision"] = vision_log
        out["food_position"] = list(food)
    return out


# ── Workers ───────────────────────────────────────────────────────────────────

_ctx: dict[str, Any] | None = None


def make_ctx(reach_radius: float, subtask_time: float, vision_norm: str = "raw", hidden: int = 16) -> dict[str, Any]:
    """World, brain and ray-cast vision for one process."""
    model, data, target_mocap_id, cam_name = build_food_world(reach_radius)
    if cam_name is None:
        msg = f"{BODY} has no body-mounted camera"
        raise RuntimeError(msg)
    return {
        "model": model, "data": data, "target_mocap_id": target_mocap_id, "cam_name": cam_name,
        "brain": DannBrain(model, data, extra_inputs=N_VISION,
                           topology=brains.body_topology(BODY, parent_slot="back"), hidden=hidden),
        "vision": RayVision(model, cam_name, int(model.body("green_target").geomadr[0]), VISION_RAYS,
                            normalise=vision_norm == "sum"),
        "rotor_ids": rotor_geom_ids(model), "floor_id": floor_id(model),
        "reach_radius": reach_radius, "subtask_time": subtask_time,
    }


def _worker_init(reach_radius: float, subtask_time: float, vision_norm: str, hidden: int) -> None:
    global _ctx  # noqa: PLW0603
    torch.set_num_threads(1)
    _ctx = make_ctx(reach_radius, subtask_time, vision_norm, hidden)


def _worker_eval(job: tuple[list[float], list[float], bool, float | None]) -> dict[str, Any]:
    assert _ctx is not None
    params, food, record, time_limit = job
    _ctx["brain"].set_params(np.array(params))
    return run_subtask(_ctx, np.array(food), record, time_limit=time_limit)


# ── Optimisers (all minimise; ask -> candidates, tell(candidates, fitnesses)) ─


class CmaOptimizer:
    def __init__(self, n: int, pop: int, seed: int, separable: bool) -> None:
        self.es = cma.CMAEvolutionStrategy(
            [0.0] * n, args.sigma,
            {"popsize": pop, "seed": seed, "verbose": -9, "maxiter": 10**9, "CMA_diagonal": separable},
        )

    def ask(self) -> list[np.ndarray]:
        return self.es.ask()

    def tell(self, solutions: list[np.ndarray], fits: list[float]) -> None:
        self.es.tell(solutions, fits)

    @property
    def mean(self) -> np.ndarray:
        return np.array(self.es.mean)

    @property
    def sigma(self) -> float:
        return float(self.es.sigma)


class OpenAIES:
    """OpenAI-ES: antithetic Gaussian perturbations of a mean theta, centred
    rank utilities, and an Adam step on theta along the estimated gradient.
    """

    def __init__(self, n: int, pop: int, seed: int, sigma: float, lr: float) -> None:
        if pop % 2:
            msg = "OpenAI-ES needs an even --pop (antithetic pairs)"
            raise ValueError(msg)
        self.theta, self.sigma, self.lr, self.pop = np.zeros(n), sigma, lr, pop
        self.rng = np.random.default_rng(seed)
        self.m, self.v, self.t = np.zeros(n), np.zeros(n), 0
        self.eps = np.zeros((pop, n))

    def ask(self) -> list[np.ndarray]:
        half = self.rng.standard_normal((self.pop // 2, len(self.theta)))
        self.eps = np.vstack([half, -half])
        return list(self.theta + self.sigma * self.eps)

    def tell(self, solutions: list[np.ndarray], fits: list[float]) -> None:  # noqa: ARG002
        ranks = np.empty(self.pop)
        ranks[np.argsort(fits)] = np.arange(self.pop)
        utility = 0.5 - ranks / (self.pop - 1)  # best (lowest fitness) +0.5 .. worst -0.5
        grad = utility @ self.eps / (self.pop * self.sigma)  # ascent direction
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * grad**2
        m_hat, v_hat = self.m / (1 - 0.9**self.t), self.v / (1 - 0.999**self.t)
        self.theta += self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    @property
    def mean(self) -> np.ndarray:
        return self.theta.copy()


def make_optimizer(n: int) -> CmaOptimizer | OpenAIES:
    if args.optimizer == "oes":
        return OpenAIES(n, args.pop, args.seed, args.oes_sigma, args.oes_lr)
    return CmaOptimizer(n, args.pop, args.seed, separable=args.optimizer == "sepcma")


# ── Main ──────────────────────────────────────────────────────────────────────


def total_score(episodes: list[dict[str, Any]], coverage_weight: float, speed_weight: float = 1.0,
                progress_weight: float = 1.0, progress_mode: str = "first") -> float:
    """Collected subtasks' 1 + speed_weight x speed, plus progress_weight x
    the progress of the first subtask, in order, whose food was not collected
    (progress_mode "all": of every one), plus coverage_weight x every
    subtask's coverage (see the module docstring).
    """
    missed = [e["progress"] for e in episodes if not e["collected"]]
    progress = sum(missed) if progress_mode == "all" else (missed[0] if missed else 0.0)
    return (float(sum(1.0 + speed_weight * e["speed"] for e in episodes if e["collected"]))
            + progress_weight * progress + coverage_weight * float(sum(e["coverage"] for e in episodes)))


def max_score(n: int) -> float:
    return n * (1.0 + args.speed_weight + args.coverage_weight)


def score_of(episodes: list[dict[str, Any]]) -> float:
    return total_score(episodes, args.coverage_weight, args.speed_weight, args.progress_weight, args.progress_mode)


def evaluate(pool: ProcessPoolExecutor, params: list[np.ndarray], foods: list[list[float]],
             record: bool = False, time_limit: float | None = None) -> list[dict[str, Any]]:
    """Each parameter vector on every subtask (all episodes in one pool.map):
    total score, food collected, mean progress, and the subtask episodes.
    """
    episodes = list(pool.map(_worker_eval, [(p.tolist(), f, record, time_limit) for p in params for f in foods]))
    k = len(foods)
    out = []
    for i in range(len(params)):
        eps = episodes[i * k:(i + 1) * k]
        out.append({"score": score_of(eps), "collected": float(sum(e["collected"] for e in eps)),
                    "progress": float(np.mean([e["progress"] for e in eps])), "episodes": eps})
    return out


def main() -> None:
    CHECKPOINTS.mkdir(exist_ok=True, parents=True)
    train_foods = food_positions(args.subtasks, args.food_radius, offset_deg=args.ring_offset)
    heldout_foods = food_positions(args.subtasks, args.food_radius, offset_deg=args.ring_offset + 180.0 / args.subtasks)
    probe_foods = food_positions(PROBE_N, PROBE_RADIUS, offset_deg=180.0 / PROBE_N)
    with (DATA / "run_config.json").open("w") as fh:
        json.dump({**vars(args), "body": BODY, "profile": "dsocial_zoo", "n_vision": N_VISION,
                   "vision_rays": VISION_RAYS, "train_foods": train_foods, "heldout_foods": heldout_foods,
                   "probe_foods": probe_foods, "probe_time": PROBE_TIME}, fh, indent=2)

    probe = make_ctx(args.reach_radius, args.subtask_time, args.vision_norm, args.hidden)
    num_params = probe["brain"].num_params
    console.log(f"{BODY}: {probe['model'].nu} hinges, camera {probe['cam_name']!r}, DANN {num_params} params, "
                f"{args.subtasks} subtasks at {args.food_radius} m")

    opt = make_optimizer(num_params)
    best_score = -float("inf")
    best_w = np.zeros(num_params)
    t_start = time.perf_counter()
    curve_path = DATA / "learning_curve.jsonl"
    curve_path.unlink(missing_ok=True)
    sim_s = 0.0  # training compute: simulated seconds, settles included
    best_gen = 0
    first_food_gen: int | None = None
    probe_best, solved = 0.0, None
    stop_reason = "gens"

    with ProcessPoolExecutor(max_workers=min(args.pop * args.subtasks, args.workers),
                             initializer=_worker_init,
                             initargs=(args.reach_radius, args.subtask_time, args.vision_norm, args.hidden)) as pool:
        for gen in range(args.gens):
            t_gen = time.perf_counter()
            solutions = opt.ask()
            results = evaluate(pool, solutions, train_foods)
            scores = [r["score"] for r in results]
            opt.tell(solutions, [-sc for sc in scores])
            sim_s += sum(e["sim_time"] + SETTLE_DURATION for r in results for e in r["episodes"])
            if first_food_gen is None and any(r["collected"] for r in results):
                first_food_gen = gen

            i_best = int(np.argmax(scores))
            probe_collected = None
            if scores[i_best] > best_score:
                best_score, best_w, best_gen = scores[i_best], np.array(solutions[i_best]), gen
                np.save(DATA / "best_weights.npy", best_w)
                probe_collected = evaluate(pool, [best_w], probe_foods, time_limit=PROBE_TIME)[0]["collected"]
                probe_best = max(probe_best, probe_collected)
                if solved is None and probe_collected >= PROBE_N:
                    solved = {"gen": gen, "sim_s": sim_s, "elapsed_s": time.perf_counter() - t_start}

            row = {
                "gen": gen, "best_score": scores[i_best], "mean_score": float(np.mean(scores)),
                "median_score": float(np.median(scores)), "best_so_far": best_score,
                "best_collected": results[i_best]["collected"],
                "mean_collected": float(np.mean([r["collected"] for r in results])),
                "best_subtask_scores": [e["score"] for e in results[i_best]["episodes"]],
                "best_subtask_coverage": [e["coverage"] for e in results[i_best]["episodes"]],
                "sigma": float(opt.sigma), "probe_collected": probe_collected, "sim_s": sim_s,
                "gen_time_s": time.perf_counter() - t_gen, "elapsed_s": time.perf_counter() - t_start,
            }
            with curve_path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            probe_str = "" if probe_collected is None else f"  probe {probe_collected:.0f}/{PROBE_N}"
            console.log(f"gen {gen:4d}  best {row['best_score']:.3f}  mean {row['mean_score']:.3f}  "
                        f"food {row['best_collected']:.0f}/{args.subtasks}{probe_str}  ({row['gen_time_s']:.1f} s)")

            if (gen + 1) % args.checkpoint_every == 0:
                np.save(CHECKPOINTS / f"gen_{gen:04d}.npy", opt.mean)
            stop = None
            if args.stop_on_solve and solved is not None:
                stop = "solved"
            elif args.max_sim_seconds is not None and sim_s >= args.max_sim_seconds:
                stop = "budget"
            elif args.hopeless_gen and first_food_gen is None and gen + 1 >= args.hopeless_gen:
                stop = "hopeless"
            elif args.stall_gens and gen - best_gen >= args.stall_gens:
                stop = "stalled"
            elif args.time_limit is not None and time.perf_counter() - t_start > args.time_limit:
                stop = "time_limit"
            if stop:
                stop_reason = stop
                console.log(f"[yellow]stopping after gen {gen}: {stop}[/yellow]")
                break

        train = evaluate(pool, [best_w], train_foods, record=True)[0]
        heldout = evaluate(pool, [best_w], heldout_foods, record=True)[0]
        probe_final = evaluate(pool, [best_w], probe_foods, record=True, time_limit=PROBE_TIME)[0]

    final = {name: {"score": r["score"], "collected": r["collected"], "progress": r["progress"], "episodes": r["episodes"]}
             for name, r in (("train", train), ("heldout", heldout), ("probe", probe_final))}
    with (DATA / "final_eval.json").open("w") as fh:
        json.dump({"best_score": best_score, **final}, fh)
    summary = {"stop_reason": stop_reason, "gens": gen + 1, "sim_s": sim_s, "elapsed_s": time.perf_counter() - t_start,
               "best_score": best_score, "best_gen": best_gen, "first_food_gen": first_food_gen,
               "probe_best": probe_best, "probe_n": PROBE_N, "solved": solved, "num_params": num_params,
               "train_collected": train["collected"], "heldout_collected": heldout["collected"],
               "probe_final_collected": probe_final["collected"]}
    with (DATA / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    for name, r in final.items():
        n = len(r["episodes"])
        console.log(f"{name:8s} subtasks: score {r['score']:.3f} / {max_score(n):g}, food {r['collected']:.0f}/{n}, "
                    f"per subtask {[round(e['score'], 2) for e in r['episodes']]}")
    console.log(f"stop: {stop_reason}; probe solved: {solved}")


if __name__ == "__main__":
    main()

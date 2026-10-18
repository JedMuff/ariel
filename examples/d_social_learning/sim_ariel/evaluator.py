"""Picklable ARIEL evaluator: inner CMA-ES + DistributedMLP on MuJoCo sim."""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np

DURATION = 30.0
SETTLE_TIME = 3.0
SPAWN_POS = (-0.8, 0.0, 0.1)
CTRL_EVERY = 9  # 500Hz physics / 9 ≈ 55.6Hz control (closest integer divisor to 60Hz)
CTRL_ALPHA = 0.5   # control blending factor (0=no change, 1=instant) — damps resonant ground-jitter exploit
HEIGHT_PENALTY_THRESHOLD = 0.5  # m — only penalise spawn height above this
JERK_PENALTY_WEIGHT = 3.0  # penalty per unit of mean absolute ctrl delta, applied above JERK_THRESHOLD
JERK_THRESHOLD = 0.15  # hurdle: mean_jerk below this is free; at/above it, the full weight*mean_jerk applies
ZOO_MODULES = False  # build bodies with kgd's zoo modules (sim_ariel/zoo_modules.py)
DISPLACEMENT_FROM_START = False  # fitness uses x travelled since the settle phase, not the absolute x
# Start-state noise for perturbed episodes (run_episode's perturb_rng): Gaussian
# offsets added to the root's x/y position and to every hinge angle before the
# settle phase. Small enough to leave the task unchanged, large enough to
# decorrelate trajectories of contact-chaotic gaits (see evaluate.py
# --final-evals).
PERTURB_POS_STD = 0.001   # m
PERTURB_ANGLE_STD = 0.01  # rad

# Simulation presets (experiment.py --sim-preset). "legacy" is the module
# defaults above, used by every run before the zoo preset existed. "zoo" uses
# kgd's zoo module physics and timing (15 s episodes, 20 Hz control), keeps the
# settle phase, smoothing and jerk penalty, and measures displacement from the
# start. CTRL_ALPHA and the jerk constants act per control step, so they are
# rescaled from legacy's 9-step control period to 25 steps: same smoothing
# time constant, and the same penalty per second of control change.
_ZOO_STEP_RATIO = 25 / 9
PRESETS = {
    "legacy": {
        "DURATION": DURATION, "CTRL_EVERY": CTRL_EVERY, "CTRL_ALPHA": CTRL_ALPHA,
        "JERK_PENALTY_WEIGHT": JERK_PENALTY_WEIGHT, "JERK_THRESHOLD": JERK_THRESHOLD,
        "ZOO_MODULES": False, "DISPLACEMENT_FROM_START": False,
    },
    "zoo": {
        "DURATION": 15.0, "CTRL_EVERY": 25,  # 500Hz physics / 25 = 20Hz control
        "CTRL_ALPHA": 1.0 - (1.0 - CTRL_ALPHA) ** _ZOO_STEP_RATIO,
        "JERK_PENALTY_WEIGHT": JERK_PENALTY_WEIGHT / _ZOO_STEP_RATIO,
        "JERK_THRESHOLD": JERK_THRESHOLD * _ZOO_STEP_RATIO,
        "ZOO_MODULES": True, "DISPLACEMENT_FROM_START": True,
    },
}


def apply_preset(name: str) -> None:
    """Set this module's simulation constants to preset *name* for the rest of
    the process. The zoo preset also switches the module classes to zoo sizes
    and masses, so genome collision checks use the zoo geometry."""
    globals().update(PRESETS[name])
    if ZOO_MODULES:
        from sim_ariel.zoo_modules import use_zoo_modules_globally
        use_zoo_modules_globally()


def _scale_actions(raw: np.ndarray) -> np.ndarray:
    import math
    return raw * (math.pi / 2)

def initialize_world(genome_dict):
    import mujoco

    from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
    from ariel.ec.genotypes.tree.tree_genome import TreeGenome
    from ariel.simulation.controllers.morphology_adapter import MorphologyAdapter
    from ariel.simulation.environments import SimpleFlatWorld

    genome = TreeGenome.from_dict(genome_dict)
    graph = genome.to_networkx()

    core = construct_mjspec_from_graph(graph)
    if ZOO_MODULES:
        from sim_ariel.zoo_modules import apply_zoo_modules
        apply_zoo_modules(core.spec)
    world = SimpleFlatWorld()
    world.spawn(core.spec, position=SPAWN_POS, rotation=(0, 0, 90))
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Build rotor (not stator) and floor geom ID sets once per model
    hinge_geom_ids: set[int] = set()
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name and name.endswith("-rotor"):
            hinge_geom_ids.add(i)
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    adapter = MorphologyAdapter.from_graph(graph)

    return {
        "model": model,
        "data": data,
        "adapter": adapter,
        "floor_geom_id": floor_geom_id,
        "hinge_geom_ids": hinge_geom_ids
    }

def run_episode(theta: np.ndarray, brain, simulator_specifics: dict,
                perturb_rng: np.random.Generator | None = None, recorder=None) -> dict:
    """Run one episode; return fitness and diagnostics. With *perturb_rng*,
    the start state is jittered first (PERTURB_POS_STD, PERTURB_ANGLE_STD).
    *recorder*, if given, is called as recorder(sim_step, data) after every
    rollout physics step (not the settle phase), for replays that record
    behaviour (analysis/behavior_replay.py); it must not change *data*."""
    import mujoco

    model = simulator_specifics["model"]
    data = simulator_specifics["data"]
    adapter = simulator_specifics["adapter"]
    floor_geom_id = simulator_specifics["floor_geom_id"]
    hinge_geom_ids = simulator_specifics["hinge_geom_ids"]

    brain.set_theta(theta)
    mujoco.mj_resetData(model, data)
    if perturb_rng is not None:
        data.qpos[0:2] += perturb_rng.normal(0.0, PERTURB_POS_STD, 2)
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                data.qpos[model.jnt_qposadr[j]] += perturb_rng.normal(0.0, PERTURB_ANGLE_STD)

    # Record spawn height before any steps (penalises falling)
    core_height = float(data.qpos[2])

    # Settle phase: let robot fall into place, no control, no counting
    sim_step = 0
    while data.time < SETTLE_TIME:
        mujoco.mj_step(model, data)
        sim_step += 1
    start_x = float(data.qpos[0])

    # Rollout phase
    c_hinge = 0
    ctrl_step = 0
    active_hinge_contacts: set[frozenset[int]] = set()
    prev_ctrl = np.zeros(model.nu, dtype=np.float32)
    jerk_sum = 0.0
    rollout_end = SETTLE_TIME + DURATION
    while data.time < rollout_end:
        if sim_step % CTRL_EVERY == 0:
            node_inputs, t = adapter.get_node_inputs(model, data, ctrl_step)
            raw = brain.forward_all(node_inputs, t)
            target_ctrl = _scale_actions(raw)
            # Alpha-blend towards new action and clip to servo range
            # to prevent resonant ground-jitter exploitation.
            new_ctrl = np.clip(
                prev_ctrl * (1.0 - CTRL_ALPHA) + target_ctrl * CTRL_ALPHA,
                -np.pi / 2, np.pi / 2,
            ).astype(np.float32)
            if ctrl_step > 0 and model.nu > 0:
                jerk_sum += float(np.mean(np.abs(new_ctrl - prev_ctrl)))
            prev_ctrl = new_ctrl.copy()
            data.ctrl[:] = new_ctrl
            ctrl_step += 1

        mujoco.mj_step(model, data)
        if recorder is not None:
            recorder(sim_step, data)

        # Rising-edge hinge-floor contact events
        current: set[frozenset[int]] = set()
        for k in range(data.ncon):
            c = data.contact[k]
            if ((c.geom1 == floor_geom_id and c.geom2 in hinge_geom_ids) or
                    (c.geom2 == floor_geom_id and c.geom1 in hinge_geom_ids)):
                current.add(frozenset((c.geom1, c.geom2)))
        c_hinge += len(current - active_hinge_contacts)
        active_hinge_contacts = current
        sim_step += 1

    mean_jerk = jerk_sum / max(ctrl_step - 1, 1)

    d = float(data.qpos[0]) - (start_x if DISPLACEMENT_FROM_START else 0.0)
    height_penalty = core_height if core_height > HEIGHT_PENALTY_THRESHOLD else 0.0
    # Hurdle penalty (see learn_gecko_gait.py's gecko sweep): a linear
    # penalty from zero pushed CMA-ES toward a near-static "don't move
    # at all" optimum instead of just suppressing high-frequency
    # thrash. Giving a free jerk budget up to JERK_THRESHOLD before
    # any penalty kicks in let it find gaits with real net distance.
    jerk_penalty = JERK_PENALTY_WEIGHT * mean_jerk if mean_jerk >= JERK_THRESHOLD else 0.0
    fitness = d - height_penalty - jerk_penalty

    return {
        "fitness": fitness,
        "displacement": d,
        "mean_jerk": mean_jerk,
        "c_hinge": c_hinge,
    }

def extra_results(best_ep):
    return {"mean_jerk": best_ep["mean_jerk"],
            "c_hinge": best_ep["c_hinge"]}

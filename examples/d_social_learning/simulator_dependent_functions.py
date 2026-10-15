simulator = "NONE"
sim_preset = "legacy"

SIMULATOR_ARIEL = "ariel"
SIMULATOR_EVOGYM = "evogym"

def set_sim_preset(name):
    """Apply a simulation preset (sim_ariel/evaluator.py PRESETS) in this
    process. Must run in every worker too: they don't inherit module state."""
    global sim_preset
    if name != "legacy" and simulator != SIMULATOR_ARIEL:
        raise ValueError(f"sim preset {name!r} is ariel-only")
    sim_preset = name
    if simulator == SIMULATOR_ARIEL:
        from sim_ariel.evaluator import apply_preset
        apply_preset(name)

def get_descriptor(ind):
    if simulator == SIMULATOR_ARIEL:
        from sim_ariel.descriptor import tree_descriptor
        return tree_descriptor(ind)
    elif simulator == SIMULATOR_EVOGYM:
        from sim_evogym.descriptor import voxel_descriptor
        return voxel_descriptor(ind)
    else:
        return None

def mutate(parent):
    """Return (child morphology, mutation record) for *parent*."""
    if simulator == SIMULATOR_ARIEL:
        from sim_ariel.morphology_ops import mutate_with_info as ariel_mutate
        return ariel_mutate(parent.genotype_["morph"])
    elif simulator == SIMULATOR_EVOGYM:
        from sim_evogym.morphology_ops import mutate_with_info as evogym_mutate
        return evogym_mutate(parent.genotype_["morph"])
    else:
        return None, {}

def random_individual():
    if simulator == SIMULATOR_ARIEL:
        from sim_ariel.morphology_ops import random_individual as ariel_random_individual
        return ariel_random_individual()
    elif simulator == SIMULATOR_EVOGYM:
        from sim_evogym.morphology_ops import random_individual as evogym_random_individual
        return evogym_random_individual()
    else:
        return None

def extra_tags(result):
    if simulator == SIMULATOR_ARIEL:
        return {
            "mean_jerk": result.get("mean_jerk", 0.0),
            "c_hinge": result.get("c_hinge", 0)
        }
    else:
        return {}

def initialize_world(genome):
    if simulator == SIMULATOR_ARIEL:
        from sim_ariel.evaluator import initialize_world as initialize_world_ariel
        return initialize_world_ariel(genome)
    elif simulator == SIMULATOR_EVOGYM:
        from sim_evogym.evaluator import initialize_world as initialize_world_evogym
        return initialize_world_evogym(genome)
    else:
        return None

def run_episode(theta, brain, simulator_specifics):
    if simulator == SIMULATOR_ARIEL:
        from sim_ariel.evaluator import run_episode as run_episode_ariel
        return run_episode_ariel(theta, brain, simulator_specifics)
    elif simulator == SIMULATOR_EVOGYM:
        from sim_evogym.evaluator import run_episode as run_episode_evogym
        return run_episode_evogym(theta, brain, simulator_specifics)
    else:
        return None

def extra_results(best_episode):
    if simulator == SIMULATOR_ARIEL:
        from sim_ariel.evaluator import extra_results as extra_results_ariel
        return extra_results_ariel(best_episode)
    else:
        return {}

def stop_simulator(simulator_specifics):
    if simulator == SIMULATOR_EVOGYM:
        simulator_specifics["env"].close()

def similarity_function(metric="STRUCT"):
    """Distance between two raw stored morphologies for a structural metric:
    STRUCT = tree edit distance (ariel) / aligned Hamming distance (evogym);
    TOPDOWN = exact top-down tree edit distance (ariel only)."""
    if metric not in ("STRUCT", "TOPDOWN"):
        raise ValueError(f"unknown structural metric {metric!r}")
    if metric == "TOPDOWN":
        if simulator != SIMULATOR_ARIEL:
            raise ValueError("the TOPDOWN metric is only defined for ariel tree genomes")
        from sim_ariel.tree_edit_distance import top_down_tree_distance
        return top_down_tree_distance
    if simulator == SIMULATOR_ARIEL:
        from sim_ariel.tree_edit_distance import tree_edit_distance
        return tree_edit_distance
    if simulator == SIMULATOR_EVOGYM:
        import numpy as np
        from sim_evogym.evogym_body_descriptors import aligned_hamming_distance
        # aligned_hamming_distance needs np.ndarray inputs; callers pass raw
        # stored morphology (a list for evogym), so cast here.
        return lambda a, b: aligned_hamming_distance(
            np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        )
    else:
        return None

def n_neighbours():
    if simulator == SIMULATOR_ARIEL:
        return 6
    elif simulator == SIMULATOR_EVOGYM:
        return 8
    else:
        return None

def n_descriptors():
    if simulator == SIMULATOR_ARIEL:
        return 8
    elif simulator == SIMULATOR_EVOGYM:
        return 5
    else:
        return None
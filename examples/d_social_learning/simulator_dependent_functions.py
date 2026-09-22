simulator = "NONE"

SIMULATOR_ARIEL = "ariel"
SIMULATOR_EVOGYM = "evogym"

def get_descriptor(ind):
    from sim_ariel.descriptor import tree_descriptor
    from sim_evogym.descriptor import voxel_descriptor

    if simulator == SIMULATOR_ARIEL:
        return tree_descriptor(ind.genotype_["morph"])
    elif simulator == SIMULATOR_EVOGYM:
        return voxel_descriptor(ind.genotype_["morph"])
    else:
        return None

def mutate(parent):
    from sim_ariel.morphology_ops import mutate as ariel_mutate
    from sim_evogym.morphology_ops import mutate as evogym_mutate

    if simulator == SIMULATOR_ARIEL:
        return ariel_mutate(parent.genotype_["morph"])
    elif simulator == SIMULATOR_EVOGYM:
        return evogym_mutate(parent.genotype_["morph"])
    else:
        return None

def random_individual():
    from sim_ariel.morphology_ops import random_individual as ariel_random_individual
    from sim_evogym.morphology_ops import random_individual as evogym_random_individual

    if simulator == SIMULATOR_ARIEL:
        return ariel_random_individual()
    elif simulator == SIMULATOR_EVOGYM:
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
    from sim_ariel.evaluator import initialize_world as initialize_world_ariel
    from sim_evogym.evaluator import initialize_world as initialize_world_evogym

    if simulator == SIMULATOR_ARIEL:
        return initialize_world_ariel(genome)
    elif simulator == SIMULATOR_EVOGYM:
        return initialize_world_evogym(genome)
    else:
        return None

def run_episode(theta, brain, simulator_specifics):
    from sim_ariel.evaluator import run_episode as run_episode_ariel
    from sim_evogym.evaluator import run_episode as run_episode_evogym

    if simulator == SIMULATOR_ARIEL:
        return run_episode_ariel(theta, brain, simulator_specifics)
    elif simulator == SIMULATOR_EVOGYM:
        return run_episode_evogym(theta, brain, simulator_specifics)
    else:
        return None

def extra_results(best_episode):
    from sim_ariel.evaluator import extra_results as extra_results_ariel

    if simulator == SIMULATOR_ARIEL:
        return extra_results_ariel(best_episode)
    else:
        return {}

def stop_simulator(simulator_specifics):
    if simulator == SIMULATOR_EVOGYM:
        simulator_specifics["env"].close()

def similarity_function():
    from sim_evogym.evogym_body_descriptors import aligned_hamming_distance

    if simulator == SIMULATOR_ARIEL:
        return None #TODO: Add tree edit distance here
    if simulator == SIMULATOR_EVOGYM:
        return aligned_hamming_distance
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
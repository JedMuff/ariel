"""The experiment's protocol in one place.

Every script imports its settings from here, so a run's config.json (which
stores asdict(PROTOCOL)) says everything about how it was run.

  Simulation   flat floor (SimpleFlatWorld), zoo module sizes and masses and
               the zoo hinge (sim.py), physics step 0.002 s; 3 s of settling,
               then a 15 s episode with control every 25 physics steps (20 Hz),
               targets smoothed with alpha ~0.854.
  Fitness      forward displacement of the core in m along world -Y during the
               episode, unclipped, no penalties.
  Learning     CMA-ES, population 20, sigma 0.5, from a fixed x0 per brain
               (brains.BRAIN_X0), for a budget of 100 / 200 / 800 evaluations
               (5 / 10 / 40 CMA-ES generations).
  Evolution    (mu + lambda) = (20 + 20) over bodies for 50 generations,
               tournament of 4, crossover on 60% of offspring, mutation on all.
               Every survivor is retrained from scratch with a fresh seed each
               generation and keeps only its latest score (re-evaluation).
  Bodies       at most 25 modules besides the core, depth <= 25, >= 4 hinges.
"""

from __future__ import annotations

from dataclasses import dataclass

GENOME_TYPES: tuple[str, ...] = ("tree", "tree_symmetric", "tree_tied", "tree_tied_full", "grn")
GENOME_LABEL = {"tree": "Standard", "tree_symmetric": "Symmetric", "tree_tied": "Tied-matched",
                "tree_tied_full": "Tied-full", "grn": "Indirect"}
BRAIN_KINDS: tuple[str, ...] = ("bang_bang", "dann", "ann", "cpg")
BRAIN_LABEL = {"bang_bang": "Bang-bang", "dann": "DANN", "ann": "ANN", "cpg": "CPG"}
BUDGETS: tuple[int, ...] = (100, 200, 800)  # learning evaluations per body

# tree_tied: chance a placed module gets a randomly placed twin. 0.47 makes
# ~64% of modules tied (2p / (1 + p)), the mirrored share of tree_symmetric
# bodies. tree_tied_full ties every module.
TIE_PROB_MATCHED = 0.47
TIE_PROB_FULL = 1.0


@dataclass(frozen=True)
class Protocol:
    # Episode
    settle_s: float = 3.0
    episode_s: float = 15.0
    control_every: int = 25             # physics steps (0.002 s) between control updates: 20 Hz
    ctrl_alpha: float = 1.0 - 0.5 ** (25 / 9)  # target smoothing (the zoo preset's 9-step 0.5, rescaled)
    # Inner loop (controller learning)
    cma_popsize: int = 20
    cma_sigma: float = 0.5
    # Outer loop (body evolution)
    mu: int = 20
    lam: int = 20
    generations: int = 50
    tournament: int = 4
    p_crossover: float = 0.6
    offspring_attempts: int = 50        # fresh crossover + mutation tries per child before falling back
    # Bodies
    max_modules: int = 25               # besides the core
    max_depth: int = 25
    min_hinges: int = 4


PROTOCOL = Protocol()


def cma_generations(evals: int, protocol: Protocol = PROTOCOL) -> int:
    """CMA-ES generations for a budget of evals evaluations (rounded up)."""
    return -(-evals // protocol.cma_popsize)

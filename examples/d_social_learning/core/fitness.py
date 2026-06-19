"""Combined fitness function."""


def combined_fitness(distance: float, novelty: float, x: float) -> float:
    """f = distance * x + novelty * (1 - x)."""
    return distance * x + novelty * (1.0 - x)

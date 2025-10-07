from typing import List
import statistics

def mean_fitness(fitnesses: List[float]) -> float:
    """Compute average fitness."""
    return sum(fitnesses) / len(fitnesses) if fitnesses else 0.0


def diversity_fitness(fitnesses: List[float]) -> float:
    """Compute variance of fitness values."""
    if not fitnesses:
        return 0.0
    mean = mean_fitness(fitnesses)
    return sum((f - mean) ** 2 for f in fitnesses) / len(fitnesses)


# --- Option: use Python's built-in statistics ---
def diversity_fitness_alt(fitnesses: List[float]) -> float:
    return statistics.pvariance(fitnesses) if fitnesses else 0.0

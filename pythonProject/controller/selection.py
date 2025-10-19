import random
from typing import List, Tuple

def tournament_selection(population: List, fitness: List[float], k: int = 2, rng: random.Random = random) -> int:
    """
    Tournament selection: pick the best individual among k random samples.
    Returns index of the selected individual.
    """
    indices = rng.sample(range(len(population)), k)
    best = min(indices, key=lambda i: fitness[i])  # assuming minimization
    return best


def mating_pool(population: List, fitness: List[float], pool_size: int, rng: random.Random = random) -> List:
    """
    Build a mating pool using tournament selection.
    """
    return [population[tournament_selection(population, fitness, rng=rng)] for _ in range(pool_size)]


def environmental_selection(parents: List, offspring: List, problem, max_size: int) -> List:
    """
    Survivor selection: merge parents + offspring, keep best max_size.
    """
    combined = parents + offspring
    costs = [problem.evaluate(ind) for ind in combined]
    ranked = sorted(zip(combined, costs), key=lambda x: x[1])  # sort by cost/fitness
    return [ind for ind, _ in ranked[:max_size]]

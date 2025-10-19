import random
from typing import List

# --- Basic mutation operators ---

def swap_mutation(solution: List[List[int]], rng: random.Random = random) -> List[List[int]]:
    """Swap two customers between routes."""
    child = [route[:] for route in solution]  # deep copy
    flat_customers = [(ri, ci) for ri, route in enumerate(child) for ci in range(1, len(route)-1)]
    if len(flat_customers) < 2:
        return child
    i1, i2 = rng.sample(flat_customers, 2)
    r1, c1 = i1
    r2, c2 = i2
    child[r1][c1], child[r2][c2] = child[r2][c2], child[r1][c1]
    return child


def relocate_mutation(solution: List[List[int]], rng: random.Random = random) -> List[List[int]]:
    """Relocate a customer from one route to another."""
    child = [route[:] for route in solution]
    routes = [r for r in range(len(child)) if len(child[r]) > 2]
    if not routes:
        return child
    r1 = rng.choice(routes)
    c1 = rng.randint(1, len(child[r1]) - 2)  # not depot
    customer = child[r1].pop(c1)
    r2 = rng.choice(range(len(child)))
    pos = rng.randint(1, len(child[r2]))  # insert before depot if last
    child[r2].insert(pos, customer)
    return child


# --- Simple crossover ---

def route_based_crossover(parent1: List[List[int]], parent2: List[List[int]], rng: random.Random = random) -> List[List[int]]:
    """
    Route-based crossover:
    - Copy some routes from parent1
    - Fill missing customers with routes from parent2
    """
    child = []
    customers_used = set()

    # Take half routes from parent1
    for route in parent1[:len(parent1)//2]:
        new_route = route[:]
        child.append(new_route)
        customers_used.update(route[1:-1])  # exclude depots

    # Fill with customers from parent2 not already used
    remaining = [c for route in parent2 for c in route[1:-1] if c not in customers_used]
    if remaining:
        child.append([0] + remaining + [0])  # depot = 0 assumed

    return child


def apply_variation(parents: List[List[List[int]]], rng: random.Random = random) -> List[List[List[int]]]:
    """
    Apply crossover + mutation to generate offspring.
    """
    offspring = []
    for i in range(0, len(parents), 2):
        p1 = parents[i]
        p2 = parents[(i+1) % len(parents)]
        child = route_based_crossover(p1, p2, rng)
        # Apply random mutation
        if rng.random() < 0.5:
            child = swap_mutation(child, rng)
        else:
            child = relocate_mutation(child, rng)
        offspring.append(child)
    return offspring

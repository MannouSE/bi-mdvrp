# lower_level.py
import math

# mdvrp/lower_level.py

"""
def solve_lower_level(problem, y_alloc=None):
    # --- Basic data ---
    total_demand = sum(problem.demand.get(c, 0.0) for c in problem.customers)
    plants = problem.plants
    depots = problem.depots
    plant_caps = problem.plant_capacity
    K = len(plants)
    L = len(depots)

    # --- Initialize allocation matrix ---
    if y_alloc is None or not y_alloc:
        y_alloc = {k: {l: 0.0 for l in depots} for k in plants}

    # --- Approximate depot-level demand equally (simple heuristic) ---
    avg_demand = total_demand / L
    depot_demand = {l: avg_demand for l in depots}

    # --- Greedy proportional allocation ---
    for k in plants:
        remaining = plant_caps.get(k, total_demand / K)

        for l in depots:
            if remaining <= 0:
                break  # plant exhausted

            need = depot_demand[l]
            # allocate what this plant can supply
            qty = min(need, remaining)
            y_alloc[k][l] = qty

            # update counters
            remaining -= qty
            depot_demand[l] -= qty

    return y_alloc



"""

def solve_lower_level(problem, y_alloc=None):
    """
    Lower-level: allocate production from plants to depots.
    """
    D = problem.distance_matrix
    plants = problem.plants
    depots = problem.depots
    plant_caps = problem.plant_capacity

    # Initialize allocation dictionary
    if y_alloc is None or not y_alloc:
        y_alloc = {k: {l: 0.0 for l in depots} for k in plants}

    # Example simple proportional allocation
    total_demand = sum(problem.demand.values())
    depot_demand = {l: total_demand / len(depots) for l in depots}

    for l in depots:
        remaining = depot_demand[l]
        for k in plants:
            if remaining <= 0:
                break
            cap = plant_caps.get(k, 0.0)
            qty = min(cap, remaining)
            y_alloc[k][l] = qty
            remaining -= qty

    # Flatten allocation
    flat_alloc = {(k, l): q for k, depots_dict in y_alloc.items() for l, q in depots_dict.items()}

    # Compute cost
    alloc_cost = sum(q * D[k][l] for (k, l), q in flat_alloc.items())

    return y_alloc

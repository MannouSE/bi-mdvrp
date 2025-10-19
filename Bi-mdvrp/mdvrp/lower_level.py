# lower_level.py
import math

# mdvrp/lower_level.py
def solve_lower_level(problem, y_alloc=None):
    """
    Lower-level allocation function.
    Distributes production from plants to depots based on capacity and demand.

    Returns:
        y_alloc: dict {plant: {depot: quantity}}
    """
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


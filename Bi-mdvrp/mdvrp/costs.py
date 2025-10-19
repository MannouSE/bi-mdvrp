# costs.py
from typing import Dict, List
from mdvrp.data import Problem
from mdvrp.lower_level import solve_lower_level


# ==========================================================
# 🚚 1. Route-level Cost
# ==========================================================
def route_cost(route: List[int], D: List[List[float]]) -> float:
    """
    Compute the cost of a single route as the sum of arc distances.
    """
    total = 0.0
    for i in range(len(route) - 1):
        total += D[route[i]][route[i + 1]]
    return total


# ==========================================================
# 🏢 2. Depot-level Routing Cost
# ==========================================================
def routing_cost(solution: Dict[int, List[List[int]]], problem: Problem) -> float:
    """
    Compute the total routing cost:
        ∑_s ∑_(i,j) c_ij^a x_ij^s
    Loops over depots and their assigned routes.
    """
    total = 0.0
    for depot, routes in solution.items():
        for route in routes:
            total += route_cost(route, problem.distance_matrix)
    return total


# ==========================================================
# 🏭 3. Plant–Depot Linking Cost (lower-level term)
# ==========================================================
def plant_depot_link_cost(y: Dict[int, Dict[int, float]], problem: Problem) -> float:
    """
    Compute ∑_ℓ ∑_k c_kℓ^b * y_kℓ
    where y[k][ℓ] is the quantity shipped from plant k to depot ℓ.
    """
    total = 0.0
    for k in problem.plants:
        for l in problem.depots:
            q = y.get(k, {}).get(l, 0.0)
            total += problem.c_b.get(k, {}).get(l, 0.0) * q
    return total


# ==========================================================
# 💰 4. Total Cost Function
# ==========================================================
# mdvrp/costs.py
from typing import Dict, List
from mdvrp.data import Problem

from mdvrp.lower_level import solve_lower_level
"""
def total_cost(routes, y_alloc, problem):
    # Upper-level routing cost
    travel_cost = 0.0
    D = problem.distance_matrix
    for depot, route_list in routes.items():
        for route in route_list:
            for i in range(len(route) - 1):
                a, b = route[i], route[i + 1]
                travel_cost += problem.c_a.get((a, b), D[a][b])

    # Lower-level cost (now separate)
    y_alloc = solve_lower_level(problem, y_alloc)
    alloc_cost = 0.0
    for k, depots in y_alloc.items():
        for l, qty in depots.items():
            cb = problem.c_b.get(k, {}).get(l, 0.0)
            cc = problem.c_c.get(k, {}).get(l, 0.0)
            alloc_cost += (cb + cc) * qty

    return travel_cost + alloc_cost
"""
import math

def total_cost(routes, y_alloc, problem):
    """
    Compute total bi-MDVRP cost (routing + allocation),
    purely based on travel distances and allocation quantities.
    No SOC or charging logic.
    """

    D = problem.distance_matrix  # distance matrix
    routing_cost = 0.0

    # --- Upper-level: depot→customers routing ---
    for depot, route_list in routes.items():
        for route in route_list:
            for t in range(len(route) - 1):
                i, j = route[t], route[t + 1]
                routing_cost += D[i][j]

    # --- Lower-level: plant→depot allocation cost ---
    # You can adapt this if you have a transport cost per unit distance.
    allocation_cost = 0.0
    for k, depot_dict in y_alloc.items():
        for l, qty in depot_dict.items():
            if qty > 0:
                allocation_cost += qty * D[k][l]

    total = routing_cost + allocation_cost
    return routing_cost, allocation_cost, total


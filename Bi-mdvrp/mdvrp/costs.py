# costs.py
from typing import Dict, List, Tuple
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


def total_cost(routes: Dict[int, list],
               y_alloc: Dict[int, Dict[int, float]],
               problem) -> Tuple[float, float, float]:

    #Compute total bi-MDVRP cost (routing + allocation).
    #- routing: depot→customer→depot distances
    #- allocation: plant→depot * quantity

    D = problem.distance_matrix
    routing_cost, alloc_cost = 0.0, 0.0

    # --- upper level
    for depot, route_list in routes.items():
        for route in route_list:
            for i, j in zip(route[:-1], route[1:]):
                routing_cost += D[i][j]

    # --- lower level
    for k, depot_dict in y_alloc.items():
        for l, qty in depot_dict.items():
            alloc_cost += qty * D[k][l]

    total = routing_cost + alloc_cost
    return routing_cost, alloc_cost, total

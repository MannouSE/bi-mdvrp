# mdvrp/bilevel_solution.py
from typing import Dict, List
from mdvrp.costs import total_cost


def decode_permutation(perm: List[int]) -> Dict[int, List[List[int]]]:
    """
    Convert a permutation with negative depot markers
    into a dictionary of depot routes.
    Example:
        [-1, 2, 1, 3, -1, -2, 5, 4, -2]
        → {1: [[1,2,1,3,1]], 2: [[2,5,4,2]]}
    """
    routes: Dict[int, List[List[int]]] = {}
    current_depot = None
    current_route = []

    for val in perm:
        if val < 0:  # depot marker (start/end)
            depot_id = abs(val)
            if current_route and current_depot is not None:
                current_route.append(current_depot)
                routes.setdefault(current_depot, []).append(current_route)
            current_depot = depot_id
            current_route = [current_depot]
        else:
            current_route.append(val)

    # close last route
    if current_route and current_depot is not None:
        current_route.append(current_depot)
        routes.setdefault(current_depot, []).append(current_route)

    return routes




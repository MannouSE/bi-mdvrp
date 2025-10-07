from typing import List
from data.problem import Problem

def route_cost(route: List[int], problem: Problem) -> float:
    """Compute total distance of a route, checking battery feasibility."""
    total_distance = 0.0
    soc = problem.battery

    for i in range(len(route) - 1):
        u, v = route[i], route[i+1]
        d = problem.distance(u, v)
        e = problem.energy_consumption(u, v)

        total_distance += d
        soc -= e

        if soc < 0:
            return float("inf")  # infeasible due to energy exhaustion

        if v in problem.stations:
            soc = problem.battery  # recharge at station

    return total_distance


def full_cost(solution: List[List[int]], problem: Problem) -> float:
    """Sum cost over all vehicle routes."""
    return sum(route_cost(route, problem) for route in solution)

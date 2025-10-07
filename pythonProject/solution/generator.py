import random
from typing import List
from data.problem import Problem

def generate_initial_solution(problem: Problem, rng: random.Random = random) -> List[List[int]]:
    """
    Generate a naive solution:
    - Greedy shuffle of customers
    - Assign to vehicles until capacity is exceeded
    """
    customers = problem.customers[:]
    rng.shuffle(customers)

    solution = []
    route = []
    load = 0

    for c in customers:
        demand = problem.demand.get(c, 0)
        if load + demand <= problem.capacity:
            route.append(c)
            load += demand
        else:
            solution.append([problem.depot] + route + [problem.depot])
            route = [c]
            load = demand

    if route:
        solution.append([problem.depot] + route + [problem.depot])

    return solution

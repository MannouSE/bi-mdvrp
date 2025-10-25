from __future__ import annotations
import random
import math
from typing import Dict, List, Tuple
from mdvrp.data import Problem


def generate_initial_solution(problem: Problem, rng=None) -> dict:
    """Build initial MDVRP solution with routes and allocation."""
    if rng is None:
        rng = random.Random()

    def euclid(a, b):
        if a not in problem.coords or b not in problem.coords:
            return float('inf')
        ax, ay = problem.coords[a]
        bx, by = problem.coords[b]
        return math.hypot(ax - bx, ay - by)

    # Step 1: Assign customers to nearest depot
    depot_customers = {d: [] for d in problem.depots}
    for c in problem.customers:
        nearest = min(problem.depots, key=lambda d: euclid(c, d))
        depot_customers[nearest].append(c)

    # Step 2: Create routes respecting capacity
    routes = {}
    depot_demands = {}

    for depot in problem.depots:
        custs = depot_customers[depot]
        depot_demands[depot] = 0.0
        rng.shuffle(custs)

        depot_routes = []
        current_route = []
        current_load = 0.0

        for c in custs:
            demand = problem.demand.get(c, 0.0)
            depot_demands[depot] += demand

            if current_load + demand > problem.capacity and current_route:
                depot_routes.append([depot] + current_route + [depot])
                current_route = []
                current_load = 0.0

            current_route.append(c)
            current_load += demand

        if current_route:
            depot_routes.append([depot] + current_route + [depot])

        routes[depot] = depot_routes

    # Step 3: Generate allocation (plants → depots)
    alloc = generate_initial_allocation(problem, depot_demands, rng)

    return {"routes": routes, "alloc": alloc}


def generate_initial_allocation(
        problem: Problem,
        depot_demands: Dict[int, float],
        rng
) -> Dict[Tuple[int, int], float]:
    """Generate production allocation from plants to depots."""
    alloc = {}
    remaining_capacity = {k: problem.plant_capacity.get(k, 0.0) for k in problem.plants}

    for depot, demand in depot_demands.items():
        remaining_demand = demand

        # Sort plants by cost
        plants_sorted = sorted(
            problem.plants,
            key=lambda k: (
                    problem.c_b.get(k, {}).get(depot, 0.0) +
                    problem.c_c.get(k, {}).get(depot, 0.0)
            )
        )

        for plant in plants_sorted:
            if remaining_demand <= 0:
                break

            available = remaining_capacity[plant]
            quantity = min(remaining_demand, available)

            if quantity > 0:
                alloc[(plant, depot)] = quantity
                remaining_capacity[plant] -= quantity
                remaining_demand -= quantity

    return alloc


def quick_repair(solution: dict, problem: Problem) -> dict:
    """Repair infeasible routes by splitting overloaded routes."""

    # Normalize input: ensure dict with 'routes' and 'alloc'
    if not isinstance(solution, dict):
        solution = {"routes": {}, "alloc": {}}

    if "routes" not in solution:
        # Old format: entire dict is routes
        solution = {"routes": solution, "alloc": solution.get("alloc", {})}

    routes = solution.get("routes", {})
    alloc = solution.get("alloc", {})

    # Normalize routes to dict format
    if isinstance(routes, list):
        routes = {1: routes}

    # Repair overloaded routes
    repaired_routes = {d: [] for d in problem.depots}

    for depot, route_list in routes.items():
        for route in route_list:
            if len(route) < 3:
                continue

            current_route = [depot]
            current_load = 0.0

            for node in route[1:-1]:
                if node not in problem.customers:
                    continue

                demand = problem.demand.get(node, 0.0)

                if current_load + demand > problem.capacity and current_route != [depot]:
                    current_route.append(depot)
                    repaired_routes[depot].append(current_route)
                    current_route = [depot]
                    current_load = 0.0

                current_route.append(node)
                current_load += demand

            if len(current_route) > 1:
                current_route.append(depot)
                repaired_routes[depot].append(current_route)

    # Repair allocation
    repaired_alloc = repair_allocation(alloc, problem)

    return {"routes": repaired_routes, "alloc": repaired_alloc}


def repair_allocation(y_alloc: dict, problem) -> dict:
    """
    Repairs or initializes allocation matrix (plants → depots).
    Rules:
      - A plant may serve multiple depots.
      - Not all (plant, depot) pairs must exist.
      - Total depot demand ≤ total plant capacity.
    """

    # --- Initialize empty allocation if needed ---
    if not y_alloc:
        y_alloc = {p: {} for p in problem.plants}

    # --- Compute depot demands (if not provided) ---
    total_demand = sum(problem.demand.values())
    avg_demand = total_demand / max(len(problem.depots), 1)
    depot_demands = {d: avg_demand for d in problem.depots}

    # --- Track current plant usage ---
    plant_usage = {p: sum(y_alloc.get(p, {}).values()) for p in problem.plants}

    total_capacity = sum(problem.plant_capacity.values())
    if total_capacity <= 0 or total_demand <= 0:
        return y_alloc

    # --- Repair: add or adjust only feasible (plant, depot) pairs ---
    for plant in problem.plants:
        cap_share = problem.plant_capacity[plant] / total_capacity
        available = problem.plant_capacity[plant] - plant_usage[plant]
        if available <= 0:
            continue

        # Only touch depots this plant already serves OR a few random new ones
        depot_keys = list(y_alloc[plant].keys()) or [random.choice(problem.depots)]
        for depot in depot_keys:
            demand = depot_demands.get(depot, avg_demand)
            alloc_qty = min(available, demand * cap_share)
            y_alloc.setdefault(plant, {})[depot] = alloc_qty

    return y_alloc



def mutate_route(route: List[int], rng=None) -> List[int]:
    """Simple in-route 2-swap mutation."""
    if rng is None:
        rng = random.Random()

    if len(route) > 4:
        interior = list(range(1, len(route) - 1))
        if len(interior) >= 2:
            i, j = rng.sample(interior, 2)
            route[i], route[j] = route[j], route[i]

    return route


def apply_ul_operator(solution: dict, problem: Problem, rng=None) -> dict:
    """Apply random structural modification on routes."""
    if rng is None:
        rng = random.Random()

    routes = solution.get("routes", {})
    alloc = solution.get("alloc", {})

    # Deep copy routes
    new_routes = {d: [r[:] for r in route_list] for d, route_list in routes.items()}

    # Select a depot with routes
    depots_with_routes = [d for d in problem.depots if new_routes.get(d)]
    if not depots_with_routes:
        return solution

    depot = rng.choice(depots_with_routes)
    route_idx = rng.randint(0, len(new_routes[depot]) - 1)
    route = new_routes[depot][route_idx]

    if len(route) <= 3:
        return solution

    # Choose operation
    op = rng.choice(["swap", "reverse", "relocate"])

    if op == "swap":
        route = mutate_route(route, rng)

    elif op == "reverse" and len(route) > 4:
        i, j = sorted(rng.sample(range(1, len(route) - 1), 2))
        route[i:j + 1] = reversed(route[i:j + 1])

    elif op == "relocate" and len(route) > 3:
        customer_indices = list(range(1, len(route) - 1))
        if len(customer_indices) >= 2:
            i = rng.choice(customer_indices)
            node = route.pop(i)
            valid_positions = [p for p in range(1, len(route)) if p != i]
            if valid_positions:
                j = rng.choice(valid_positions)
                route.insert(j, node)

    new_routes[depot][route_idx] = route

    return {"routes": new_routes, "alloc": alloc}


def count_customers_served(solution: dict, problem: Problem) -> int:
    """Count unique customers in all routes."""
    routes = solution.get("routes", {})
    served = set()

    for route_list in routes.values():
        for route in route_list:
            for node in route:
                if node in problem.customers:
                    served.add(node)

    return len(served)


def validate_solution(solution: dict, problem: Problem) -> Tuple[bool, List[str]]:
    """Validate solution feasibility."""
    errors = []
    routes = solution.get("routes", {})
    alloc = solution.get("alloc", {})

    # Check all customers served
    served = set()
    for route_list in routes.values():
        for route in route_list:
            served.update(c for c in route if c in problem.customers)

    missing = set(problem.customers) - served
    if missing:
        errors.append(f"Missing customers: {missing}")

    # Check capacity constraints
    for depot, route_list in routes.items():
        for i, route in enumerate(route_list):
            load = sum(problem.demand.get(c, 0.0) for c in route if c in problem.customers)
            if load > problem.capacity:
                errors.append(f"Route {i} at depot {depot} exceeds capacity: {load:.2f} > {problem.capacity}")

    # Check plant capacity constraints
    plant_usage = {}
    for (plant, depot), qty in alloc.items():
        plant_usage[plant] = plant_usage.get(plant, 0.0) + qty

    for plant, usage in plant_usage.items():
        capacity = problem.plant_capacity.get(plant, 0.0)
        if usage > capacity + 1e-6:
            errors.append(f"Plant {plant} exceeds capacity: {usage:.2f} > {capacity:.2f}")

    return (len(errors) == 0, errors)
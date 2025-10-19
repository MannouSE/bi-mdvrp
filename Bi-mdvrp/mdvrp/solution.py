from __future__ import annotations
import random
import math
from typing import Dict, List, Tuple
from mdvrp.data import Problem


# ==========================================================
# Generate Initial Solution
# ==========================================================
def generate_initial_solution(problem: Problem, rng=None) -> dict:
    """
    Build an initial MDVRP solution with both upper and lower levels:
      Upper level: Routes from depots to customers
      Lower level: Production allocation from plants to depots

    Returns:
        dict with keys 'routes' and 'alloc'
    """
    if rng is None:
        rng = random.Random()

    depots = problem.depots
    customers = problem.customers
    capacity = problem.capacity
    coords = problem.coords
    vehicles_per_depot = getattr(problem, "vehicles_per_depot", {d: 1 for d in depots})

    # Distance helper
    def euclid(a, b):
        if a not in coords or b not in coords:
            return float('inf')
        ax, ay = coords[a]
        bx, by = coords[b]
        return math.hypot(ax - bx, ay - by)

    # Step 1: Assign each customer to nearest depot
    depot_customers = {d: [] for d in depots}
    for c in customers:
        nearest = min(depots, key=lambda d: euclid(c, d))
        depot_customers[nearest].append(c)

    # Step 2: Create routes respecting capacity
    routes = {}
    depot_demands = {}  # Track total demand per depot

    for depot in depots:
        custs = depot_customers[depot]
        depot_demands[depot] = 0.0

        if not custs:
            routes[depot] = []
            continue

        rng.shuffle(custs)
        depot_routes = []
        current_route = []
        current_load = 0.0

        for c in custs:
            demand = problem.demand.get(c, 0.0)
            depot_demands[depot] += demand

            # Check if adding this customer exceeds capacity
            if current_load + demand > capacity and current_route:
                # Close current route
                depot_routes.append([depot] + current_route + [depot])
                current_route = []
                current_load = 0.0

            current_route.append(c)
            current_load += demand

        # Add final route if exists
        if current_route:
            depot_routes.append([depot] + current_route + [depot])

        routes[depot] = depot_routes

    # Step 3: Generate lower-level allocation (plants → depots)
    alloc = generate_initial_allocation(problem, depot_demands, rng)

    return {
        "routes": routes,
        "alloc": alloc
    }


def generate_initial_allocation(
        problem: Problem,
        depot_demands: Dict[int, float],
        rng
) -> Dict[Tuple[int, int], float]:
    """
    Generate initial production allocation from plants to depots.
    Simple greedy approach: assign nearest plant first.

    Returns:
        {(plant_id, depot_id): quantity}
    """
    alloc = {}
    remaining_capacity = {k: problem.plant_capacity.get(k, 0.0) for k in problem.plants}

    for depot, demand in depot_demands.items():
        remaining_demand = demand

        # Sort plants by cost (c_b + c_c)
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

            # Allocate as much as possible from this plant
            available = remaining_capacity[plant]
            quantity = min(remaining_demand, available)

            if quantity > 0:
                alloc[(plant, depot)] = quantity
                remaining_capacity[plant] -= quantity
                remaining_demand -= quantity

        # If demand still not satisfied, use any remaining capacity
        if remaining_demand > 0:
            for plant in problem.plants:
                if remaining_capacity[plant] > 0:
                    quantity = min(remaining_demand, remaining_capacity[plant])
                    alloc[(plant, depot)] = alloc.get((plant, depot), 0.0) + quantity
                    remaining_capacity[plant] -= quantity
                    remaining_demand -= quantity
                    if remaining_demand <= 0:
                        break

    return alloc


# ==========================================================
# Quick Repair
# ==========================================================
def quick_repair(solution: dict, problem: Problem) -> dict:
    """
    Repair infeasible routes by splitting overloaded routes.
    Also validates allocation feasibility.

    Args:
        solution: dict with 'routes' and 'alloc'

    Returns:
        Repaired solution
    """
    if not isinstance(solution, dict) or "routes" not in solution:
        # Handle old format (just routes dict)
        routes = solution if isinstance(solution, dict) else {}
        solution = {"routes": routes, "alloc": {}}

    routes = solution.get("routes", {})
    alloc = solution.get("alloc", {})

    # Repair routes
    repaired_routes = {d: [] for d in problem.depots}

    for depot, route_list in routes.items():
        for route in route_list:
            if len(route) < 3:  # Empty route [depot, depot]
                continue

            current_route = [depot]
            current_load = 0.0

            # Skip first and last depot
            for node in route[1:-1]:
                if node not in problem.customers:
                    continue

                demand = problem.demand.get(node, 0.0)

                if current_load + demand > problem.capacity and current_route != [depot]:
                    # Close current route and start new one
                    current_route.append(depot)
                    repaired_routes[depot].append(current_route)
                    current_route = [depot]
                    current_load = 0.0

                current_route.append(node)
                current_load += demand

            # Add final route
            if len(current_route) > 1:
                current_route.append(depot)
                repaired_routes[depot].append(current_route)

    # Repair allocation (ensure plant capacities respected)
    repaired_alloc = repair_allocation(problem, alloc)

    return {
        "routes": repaired_routes,
        "alloc": repaired_alloc
    }


def repair_allocation(problem: Problem, alloc: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    """Ensure allocation respects plant capacity constraints."""
    repaired = {}
    plant_usage = {k: 0.0 for k in problem.plants}

    for (plant, depot), qty in alloc.items():
        if plant not in problem.plants or depot not in problem.depots:
            continue

        capacity = problem.plant_capacity.get(plant, 0.0)
        available = capacity - plant_usage[plant]

        if available > 0:
            actual_qty = min(qty, available)
            if actual_qty > 0:
                repaired[(plant, depot)] = actual_qty
                plant_usage[plant] += actual_qty

    return repaired


# ==========================================================
# Random Route Mutation
# ==========================================================
def mutate_route(route: List[int], rng=None) -> List[int]:
    """Simple in-route 2-swap mutation (avoids depot positions)."""
    if rng is None:
        rng = random.Random()

    if len(route) > 4:  # Need at least 2 customers
        interior = list(range(1, len(route) - 1))
        if len(interior) >= 2:
            i, j = rng.sample(interior, 2)
            route[i], route[j] = route[j], route[i]

    return route


# ==========================================================
# Apply Upper-Level Operator
# ==========================================================
def apply_ul_operator(solution: dict, problem: Problem, rng=None) -> dict:
    """
    Perform random structural modification on routes.
    Operations: swap, reverse, or relocate within a route.
    """
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

    if len(route) <= 3:  # Can't mutate [depot, customer, depot]
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
            # Reinsert at different position
            valid_positions = [p for p in range(1, len(route)) if p != i]
            if valid_positions:
                j = rng.choice(valid_positions)
                route.insert(j, node)

    # Update the modified route
    new_routes[depot][route_idx] = route

    return {
        "routes": new_routes,
        "alloc": alloc  # Keep allocation unchanged
    }


# ==========================================================
# Utilities
# ==========================================================
def count_customers_served(solution: dict, problem: Problem) -> int:
    """Count unique customers in all routes."""
    routes = solution.get("routes", {})
    served = set()

    for depot, route_list in routes.items():
        for route in route_list:
            for node in route:
                if node in problem.customers:
                    served.add(node)

    return len(served)


def validate_solution(solution: dict, problem: Problem) -> Tuple[bool, List[str]]:
    """
    Validate solution feasibility.
    Returns: (is_valid, list_of_errors)
    """
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
        if usage > capacity + 1e-6:  # Small tolerance for floating point
            errors.append(f"Plant {plant} exceeds capacity: {usage:.2f} > {capacity:.2f}")

    return (len(errors) == 0, errors)
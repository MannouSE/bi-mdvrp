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


import random


def generate_initial_allocation(problem, depot_demands=None, rng=None):
    """
    Génère une allocation plantes→dépôts où chaque plante a un coût de production
    aléatoire. L'allocation commence par les plantes les moins chères.

    Args:
        problem: instance du problème (doit avoir .plants, .depots, .plant_capacity)
        depot_demands: dict {depot: demand} (optionnel, 100.0 par défaut)
        rng: random.Random instance (optionnel)
    Returns:
        y_alloc: dict {plant: {depot: quantity}}
        plant_costs: dict {plant: production_cost}
    """
    if rng is None:
        rng = random.Random()

    plants = problem.plants
    depots = problem.depots
    plant_caps = problem.plant_capacity

    # Générer les coûts aléatoires par plante
    plant_costs = {p: round(rng.uniform(1.0, 5.0), 2) for p in plants}

    # Préparer les demandes
    if depot_demands is None:
        depot_demands = {d: 100.0 for d in depots}

    # Trier les plantes par coût croissant
    sorted_plants = sorted(plants, key=lambda p: plant_costs[p])

    # Initialiser l'allocation et les capacités restantes
    y_alloc = {p: {d: 0.0 for d in depots} for p in plants}
    remaining_cap = plant_caps.copy()

    # Allouer la demande de chaque dépôt
    for depot in depots:
        demand = depot_demands[depot]

        for plant in sorted_plants:
            if demand <= 0 or remaining_cap[plant] <= 0:
                continue

            assign = min(remaining_cap[plant], demand)
            y_alloc[plant][depot] = assign
            remaining_cap[plant] -= assign
            demand -= assign

            if demand <= 0:
                break

    return y_alloc, plant_costs
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
    print("Type of y_alloc before repair:", type(alloc))
    print("Content:", alloc)
    repaired_alloc = repair_allocation(problem, alloc)

    return {"routes": repaired_routes, "alloc": repaired_alloc}


def repair_allocation(problem, y_alloc, depot_demands=None):
    """
    Repair or rebuild the plant→depot allocation matrix.
    Works whether `problem` is an object or dict, and even if y_alloc is malformed (list/tuple).
    """

    # --- Normalize problem fields ---
    get = lambda obj, key, default=None: getattr(obj, key, obj.get(key, default)) if isinstance(obj, dict) else getattr(obj, key, default)

    plants = get(problem, "plants", [])
    depots = get(problem, "depots", [])
    plant_capacity = get(problem, "plant_capacity", {})

    # --- Ensure y_alloc is a dict ---
    if not isinstance(y_alloc, dict):
        print("[repair_allocation] ⚠️ Invalid alloc format, rebuilding from scratch")
        y_alloc = {p: {} for p in plants}

    repaired = {}
    plant_usage = {p: 0.0 for p in plants}

    # --- Demand / capacity stats ---
    total_cap = sum(plant_capacity.values()) or 1.0
    total_demand = sum(depot_demands.values()) if depot_demands else total_cap
    avg_demand = total_demand / max(1, len(depots))

    # --- Main allocation logic ---
    for plant in plants:
        capacity = plant_capacity.get(plant, 0.0)
        available = max(0.0, capacity - plant_usage[plant])

        # Get existing or initialize one depot
        existing = y_alloc.get(plant, {})
        depot_keys = list(existing.keys()) or [random.choice(depots)]
        repaired[plant] = {}

        for depot in depot_keys:
            demand = depot_demands.get(depot, avg_demand) if depot_demands else avg_demand
            cap_share = capacity / total_cap
            alloc_qty = min(available, demand * cap_share)
            repaired[plant][depot] = alloc_qty
            plant_usage[plant] += alloc_qty

    # --- Ensure all depots are served ---
    for depot in depots:
        if not any(depot in repaired[p] for p in repaired):
            plant = random.choice(plants)
            repaired.setdefault(plant, {})[depot] = avg_demand

    return repaired



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
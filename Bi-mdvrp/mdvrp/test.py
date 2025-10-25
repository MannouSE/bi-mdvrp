# optimize.py


ACTIONS = ["H1", "H2", "H3", "H4"]

# ==========================================================
# 🚀 Initialization
# ==========================================================
def initialize_algorithm(problem: Problem, pop_size: int, eps_start: float, rng):
    """
    Initialize population, elite archive, centroids, and Q-learning structures.
    Used for cost-minimization Q-learning hyper-heuristic.
    """
    # --- 1) Initial population ---
    P = []
    for _ in range(pop_size):
        sol = generate_initial_solution(problem)
        sol = quick_repair(sol, problem)
        P.append(sol)

    # --- 2) Elite & centroids ---
    elite = []       # list of (cost, sol, emb)
    centroids = []

    # --- 3) Best placeholders ---
    best_cost = float("inf")
    best_sol = None

    # --- 4) Q-learning setup ---
    Q = {}
    state = (0, "start")
    eps = eps_start

    for a in ACTIONS:
        Q[(state, a)] = 1e6  # optimistic initialization (minimization)

    return P, elite, centroids, best_cost, best_sol, Q, state, eps


# ==========================================================
# 🧠 Main Optimization Loop
# ==========================================================
def main_optimization(problem: Problem, cfg: SimpleNamespace, rng: random.Random):
    """
    Q-learning–driven hyper-heuristic EA for MDVRP.
    Expects cfg with:
      - pop_size, max_gens, tournament_size
      - alpha, gamma
      - eps_start, eps_min, decay
    """
    (
        P,
        elite,
        centroids,
        best_c,
        best_s,
        Q,
        state,
        eps,
    ) = initialize_algorithm(problem, cfg.pop_size, cfg.eps_start, rng)

    for gen in range(cfg.max_gens):

        # ---- 1) Fitness Evaluation ----
        costs_P = [safe_cost(sol, problem)[-1] for sol in P]  # extract total_cost only
        fitness = [1.0 / (1.0 + c) if math.isfinite(c) else 0.0 for c in costs_P]

        # ---- 2) Mating Pool (Tournament Selection) ----
        M: List[Any] = []
        tsize = max(1, min(getattr(cfg, "tournament_size", 2), len(P)))
        for _ in range(len(P)):
            idxs = rng.sample(range(len(P)), tsize)
            winner_idx = max(idxs, key=lambda i: fitness[i])
            M.append(P[winner_idx])

        # ---- 3) Action Selection (ε-greedy) ----
        action = rng.choice(ACTIONS) if rng.random() < eps else get_best_action(Q, state, ACTIONS)

        # ---- 4) Offspring Generation ----
        P_new: List[Any] = []
        use_h4 = action == "H4" and bool(elite) and len(centroids) > 0

        for parent in M:
            if action == "H1":
                child = heuristics.heuristic_h1_full_hierarchical(parent, elite, centroids, problem, rng)
            elif action == "H2":
                child = heuristics.heuristic_h2_selective_ll(parent, elite, problem, rng)
            elif action == "H3":
                child = heuristics.heuristic_h3_relaxed_ll(parent, problem, rng)
            elif use_h4:
                child = heuristics.heuristic_h4_similarity_based(parent, centroids, elite, problem, rng)
            else:
                child = heuristics.heuristic_h1_full_hierarchical(parent, elite, centroids, problem, rng)

            child = quick_repair(child, problem)
            P_new.append(child)

        # ---- 5) Offspring Evaluation ----
        costs_new = [safe_cost(c, problem)[-1] for c in P_new]

        # ---- 6) Best Offspring ----
        best_off = None
        best_off_cost = float("inf")
        # ---- 6 bis) Acceptation probabiliste (Simulated Annealing style) ----
        # Calcul de la différence de coût
        delta = best_off_cost - best_c

        # Température (peut décroître avec les générations)
        T = max(1e-3, 0.98 ** gen)  # tu peux ajuster le 0.98 pour ralentir ou accélérer la décroissance

        # Probabilité d'acceptation d'une solution pire
        if delta <= 0:
            accept_prob = 1.0  # toujours accepter si meilleure
        else:
            accept_prob = math.exp(-delta / T)

        # Décision d’acceptation
        if math.isfinite(best_off_cost) and (delta < 0 or random.random() < accept_prob):
            improved = True
            best_c, best_s = best_off_cost, best_off
            update_elite_archive(elite, best_off, best_off_cost)
            centroids = cluster_elite_archive(elite)
            print(f"[gen {gen}] ✅ Acceptée (Δ={delta:.2f}, p={accept_prob:.3f})")
        else:
            improved = False
            if not math.isfinite(best_off_cost):
                print(f"[gen {gen}] ⚠️ Aucune solution faisable.")
            else:
                print(f"[gen {gen}] Rejetée (Δ={delta:.2f}, p={accept_prob:.3f})")


        # ---- 7) Q-learning Update (cost-based reward) ----
        improved = math.isfinite(best_off_cost) and (best_off_cost < best_c)
        if not math.isfinite(best_off_cost) or best_off_cost <= 0:
            norm_cost = 1.0
        elif not math.isfinite(best_c) or best_c <= 0:
            norm_cost = best_off_cost

        else:
            norm_cost = min(best_off_cost / (1.0 + best_c), 1e6)

        next_state = (1 if improved else 0, action)
        update(Q, state, action, norm_cost, next_state, cfg.alpha, cfg.gamma, ACTIONS)

        # ---- 8) Survivor Selection (μ+λ) ----
        combined = P + P_new
        combined_costs = costs_P + costs_new
        order = sorted(range(len(combined)), key=lambda i: combined_costs[i])
        P = [combined[i] for i in order[: cfg.pop_size]]

        # ---- 9) Epsilon Decay ----
        eps = max(cfg.eps_min, decay_epsilon(eps, cfg.eps_min, cfg.decay))
        state = next_state

        # ---- 10) Logging ----
        bc = f"{best_c:.2f}" if math.isfinite(best_c) else "inf"
        print(f"[gen {gen}] best={bc} eps={eps:.3f} act={action}")

    # ---- 11) Summary ----
    print("\nFinal Q-table:")
    snapshot = {}
    for (s, a), val in Q.items():
        snapshot.setdefault(s, {})[a] = round(val, 3)
    for s, qs in snapshot.items():
        print(f"  state={s} -> {qs}")

    # Evaluate the final best solution
    #ok, ul_cost, ll_cost, total_cost = safe_cost(best_s, problem)
    #print(f"Upper (routing): {ul_cost:.2f} | Lower (plant): {ll_cost:.2f} | Total: {total_cost:.2f}")
    if best_s is None:
        print("[WARN] No feasible solution found — returning empty solution.")
        best_s = {"routes": {}, "alloc": {}}
        best_c = float("inf")  # or total_cost if you keep that name

    return best_s, best_c

#### solution.py
# solution.py
"""
from __future__ import annotations
import random
from typing import Dict, List
from mdvrp.data import Problem, assign_customers_to_depots
import math


# ==========================================================
# 🧩 1. Generate Initial Solution
# ==========================================================
import math
import random"""

def generate_initial_solution(problem, rng=random):
    """
    Build an initial MDVRP solution:
      - Assign each customer to its nearest depot (min distance)
      - Split customers into routes that respect vehicle capacity
      - Each route starts/ends at its depot
    """
    depots = problem.depots
    customers = problem.customers
    capacity = problem.capacity
    coords = problem.coords
    vehicles_per_depot = getattr(problem, "vehicles_per_depot", {d: 1 for d in depots})

    # --- Distance helper ---
    def euclid(a, b):
        ax, ay = coords[a]
        bx, by = coords[b]
        return math.hypot(ax - bx, ay - by)

    # --- Step 1: Assign each customer to nearest depot ---
    depot_customers = {d: [] for d in depots}
    for c in customers:
        nearest = min(depots, key=lambda d: euclid(c, d))
        depot_customers[nearest].append(c)

    # --- Step 2: Split customers into feasible routes per depot ---
    routes = {}
    for depot in depots:
        custs = depot_customers[depot]
        if not custs:
            routes[depot] = []
            continue

        rng.shuffle(custs)
        n_vehicles = vehicles_per_depot.get(depot, 1)
        depot_routes = []
        current_route, current_load = [], 0.0

        for c in custs:
            demand = problem.demand.get(c, 0.0)
            if current_load + demand > capacity and current_route:
                depot_routes.append([depot] + current_route + [depot])
                current_route, current_load = [], 0.0
            current_route.append(c)
            current_load += demand

        if current_route:
            depot_routes.append([depot] + current_route + [depot])

        routes[depot] = depot_routes

    # --- ✅ Diagnostic print block ---
    print("\n=== INITIAL CUSTOMER ASSIGNMENT PER DEPOT ===")
    for depot, route_list in routes.items():
        # Flatten all customers served by this depot (exclude depot IDs)
        custs = [c for r in route_list for c in r if c in problem.customers]
        print(f"Depot {depot}: {len(custs)} customers -> {custs}")
    print("=============================================\n")

    return routes


# ==========================================================
# 🧰 2. Quick Repair (capacity correction)
# ==========================================================
def quick_repair(solution: Dict[int, List[List[int]]], problem: Problem) -> Dict[int, List[List[int]]]:
    """
    Simple repair: if any route exceeds vehicle capacity,
    split it greedily into smaller feasible sub-routes.
    """
    repaired: Dict[int, List[List[int]]] = {d: [] for d in problem.depots}

    for depot, routes in solution.items():
        for route in routes:
            load = 0.00

            current: List[int] = [depot]
            for node in route[1:-1]:
                q = problem.demand.get(node, 0.0)
                if load + q <= problem.capacity:
                    current.append(node)
                    load += q
                else:
                    current.append(depot)
                    repaired[depot].append(current)
                    current = [depot, node]
                    load = q
            current.append(depot)
            repaired[depot].append(current)

    return repaired


# ==========================================================
# 🧮 3. Random Route Mutation (used by UL operators)
# ==========================================================
def mutate_route(route: List[int], rng: random.Random = random) -> List[int]:
    """Simple in-route 2-swap mutation."""
    if len(route) > 4:
        i, j = rng.sample(range(1, len(route) - 1), 2)
        route[i], route[j] = route[j], route[i]
    return route


# ==========================================================
# 🔄 4. Apply Upper-Level Operator
# ==========================================================
def apply_ul_operator(solution: Dict[int, List[List[int]]], problem: Problem, rng: random.Random = random):
    """
    Perform a random structural modification:
      - pick a depot
      - pick a route
      - mutate (swap, reverse, or relocate)
    """
    new_sol = {d: [r[:] for r in routes] for d, routes in solution.items()}
    if not new_sol:
        return solution

    depot = rng.choice(problem.depots)
    if not new_sol[depot]:
        return solution

    route = rng.choice(new_sol[depot])

    op = rng.choice(["swap", "reverse", "relocate"])

    if op == "swap":
        route = mutate_route(route, rng)
    elif op == "reverse" and len(route) > 4:
        i, j = sorted(rng.sample(range(1, len(route) - 1), 2))
        route[i:j] = reversed(route[i:j])
    elif op == "relocate" and len(route) > 3:
        i, j = rng.sample(range(1, len(route) - 1), 2)
        node = route.pop(i)
        route.insert(j, node)

    new_sol[depot][new_sol[depot].index(rng.choice(new_sol[depot]))] = route
    return new_sol


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
        problem:  Problem instance with depot, capacity, demand, etc.

    Returns:
        Repaired solution dict with feasible routes and allocation.
    """

    # --- Normalize input structure ---
    if not isinstance(solution, dict) or "routes" not in solution:
        # Handle old format (just routes dict)
        routes = solution if isinstance(solution, dict) else {}
        solution = {"routes": routes, "alloc": {}}

    routes = solution.get("routes", {})
    alloc = solution.get("alloc", {})

    # --- Initialize repaired routes ---
    repaired_routes = {d: [] for d in problem.depots}

    # --- Normalize routes structure ---
    if isinstance(routes, list):
        routes = {1: routes}  # single depot fallback

    # --- Repair overloaded routes ---
    for depot, route_list in routes.items():
        for route in route_list:
            if len(route) < 3:  # [depot, depot]
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

    # --- Repair allocation (ensure plant capacities respected) ---
    #repaired_alloc = repair_allocation(problem, alloc)

    return {
        "routes": repaired_routes,
        "alloc": repaired_alloc
    }

def repair_allocation(problem: Problem, alloc: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    #Ensure allocation respects plant capacity constraints.
    print("DEBUG ⚙️ alloc type:", type(alloc))
    print("DEBUG ⚙️ alloc value:", alloc if isinstance(alloc, list) else list(alloc.items())[:5])
    repaired = {}
    plant_usage = {k: 0.0 for k in problem.plants}

    for (plant, depot), qty in alloc.items():

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

#solution.py
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
#costs.py
def total_cost(routes, y_alloc, problem):
    D = problem.distance_matrix
    routing_cost = 0.0
    for depot, route_list in routes.items():
        for route in route_list:
            for i, j in zip(route[:-1], route[1:]):
                routing_cost += D[i][j]

    allocation_cost = 0.0
    for k, depot_dict in y_alloc.items():
        for l, qty in depot_dict.items():
            allocation_cost += qty * D[k][l]

    total = routing_cost + allocation_cost
    return routing_cost, allocation_cost, total

#utils.py
def safe_cost(solution, problem):
    """
    Safely compute cost breakdown (upper, lower, total).
    """
    try:
        routes = solution.get("routes", {})
        y_alloc = solution.get("alloc", {})

        # --- get cost breakdown
        ul, ll, total = total_cost(routes, y_alloc, problem)

        # Sanity: replace NaNs/infs nj
        if not math.isfinite(total):
            return False, float("inf"), float("inf"), float("inf")

        return True, float(ul), float(ll), float(total)

    except Exception as e:
        print("[safe_cost] error:", e)
        return False, float("inf"), float("inf"), float("inf")

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

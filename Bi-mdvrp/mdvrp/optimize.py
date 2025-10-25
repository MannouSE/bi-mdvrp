import math
import random
from types import SimpleNamespace
from typing import List, Dict, Any
from mdvrp.data import Problem
from mdvrp.lower_level import solve_lower_level
from mdvrp.solution import generate_initial_solution, quick_repair
from mdvrp import heuristics
from mdvrp.elite import update_elite_archive, cluster_elite_archive
from mdvrp.q_learning import get_best_action, update, decay_epsilon
from mdvrp.utils import safe_cost

ACTIONS = ["H1", "H2", "H3", "H4"]


# ==========================================================
# Initialization
# ==========================================================
def initialize(problem: Problem, pop_size: int, eps_start: float, rng):
    """Initialize population, elite archive, and Q-learning."""
    # Create initial population
    population = []
    for _ in range(pop_size):
        sol = generate_initial_solution(problem)
        sol = quick_repair(sol, problem)
        population.append(sol)

    # Initialize tracking variables
    elite = []
    centroids = []
    best_cost = float("inf")
    best_solution = None

    # Initialize Q-learning
    Q = {}
    state = (0, "start")
    for action in ACTIONS:
        Q[(state, action)] = 1e6  # optimistic initialization

    return population, elite, centroids, best_cost, best_solution, Q, state, eps_start


# ==========================================================
# Main Optimization
# ==========================================================
def optimize(problem, cfg: SimpleNamespace, rng: random.Random):
    """
    Q-learning hyper-heuristic for bi-level MDVRP.
    Minimizes total cost (routing + allocation).

    Config requires:
      pop_size, max_gens, tournament_size,
      alpha, gamma, eps_start, eps_min, decay
    """

    # 1️⃣ Initialization
    pop, elite, centroids, best_cost, best_sol, Q, state, eps = \
        initialize(problem, cfg.pop_size, cfg.eps_start, rng)

    # 2️⃣ Evolutionary loop
    for gen in range(cfg.max_gens):

        # === Evaluate population ===
        costs = [safe_cost(sol, problem)[-1] for sol in pop]

        # === Tournament selection (use costs directly) ===
        mating_pool = []
        t_size = max(1, min(getattr(cfg, "tournament_size", 2), len(pop)))
        for _ in range(len(pop)):
            competitors = rng.sample(range(len(pop)), t_size)
            # Minimization → pick lowest cost
            winner = min(competitors, key=lambda i: costs[i])
            mating_pool.append(pop[winner])

        # === Choose heuristic (ε-greedy Q-learning) ===
        if rng.random() < eps:
            action = rng.choice(ACTIONS)
        else:
            action = get_best_action(Q, state, ACTIONS)

        # === Apply heuristic to generate offspring ===
        offspring = []
        use_h4 = (action == "H4" and elite and centroids)

        for parent in mating_pool:
            if action == "H1":
                child = heuristics.heuristic_h1_full_hierarchical(
                    parent, elite, centroids, problem, rng)
            elif action == "H2":
                child = heuristics.heuristic_h2_selective_ll(
                    parent, elite, problem, rng)
            elif action == "H3":
                child = heuristics.heuristic_h3_relaxed_ll(
                    parent, problem, rng)
            elif use_h4:
                child = heuristics.heuristic_h4_similarity_based(
                    parent, centroids, elite, problem, rng)
            else:
                child = heuristics.heuristic_h1_full_hierarchical(
                    parent, elite, centroids, problem, rng)

            child = quick_repair(child, problem)
            offspring.append(child)

        # === Evaluate offspring ===
        offspring_costs = [safe_cost(c, problem)[-1] for c in offspring]
        best_off_idx = min(range(len(offspring_costs)),
                           key=lambda i: offspring_costs[i])
        best_off_cost = offspring_costs[best_off_idx]
        best_off = offspring[best_off_idx]

        # === Simulated annealing acceptance ===
        delta = best_off_cost - best_cost
        temperature = max(1e-3, 0.98 ** gen)

        if delta <= 0:
            accept_prob = 1.0
        else:
            accept_prob = math.exp(-delta / temperature)

        improved = False
        if math.isfinite(best_off_cost) and (delta < 0 or rng.random() < accept_prob):
            improved = True
            best_cost = best_off_cost
            best_sol = best_off
            update_elite_archive(elite, best_off, best_off_cost)
            centroids = cluster_elite_archive(elite)
            print(f"[gen {gen}] ✅ Accepted (Δ={delta:.2f}, p={accept_prob:.3f})")
        else:
            print(f"[gen {gen}] ❌ Rejected (Δ={delta:.2f}, p={accept_prob:.3f})")

        # === Update Q-learning reward ===
        # Reward is improvement (lower cost → higher reward)
        if delta < 0:
            reward = abs(delta) / (1.0 + best_off_cost)
        else:
            reward = -delta / (1.0 + best_off_cost)

        next_state = (1 if improved else 0, action)
        update(Q, state, action, reward, next_state,
               cfg.alpha, cfg.gamma, ACTIONS)
        state = next_state

        # === Survivor selection (μ+λ elitism) ===
        combined = pop + offspring
        combined_costs = costs + offspring_costs
        sorted_idx = sorted(range(len(combined)), key=lambda i: combined_costs[i])
        pop = [combined[i] for i in sorted_idx[:cfg.pop_size]]

        # === Decay exploration rate ===
        eps = max(cfg.eps_min, decay_epsilon(eps, cfg.eps_min, cfg.decay))

        # === Log progress ===
        bc = f"{best_cost:.2f}" if math.isfinite(best_cost) else "inf"
        print(f"[gen {gen}] best={bc} eps={eps:.3f} action={action}")

    # 3️⃣ End of optimization
    print("\n=== Final Q-Table ===")
    for (s, a), val in sorted(Q.items()):
        print(f"  state={s}, action={a}: {val:.3f}")

    # Handle infeasible case
    if best_sol is None:
        print("[WARN] No feasible solution found.")
        best_sol = {"routes": {}, "alloc": {}}
        best_cost = float("inf")

    return best_sol, best_cost
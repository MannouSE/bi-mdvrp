# operators.py
import math, random
from typing import Dict, List
from mdvrp.costs import total_cost


# ==========================================================
# 🧩 1. Utility
# ==========================================================
def clone_solution(sol: Dict[int, List[List[int]]]) -> Dict[int, List[List[int]]]:
    """Deep copy of a multi-depot solution, robust to malformed entries."""
    clean_sol = {}
    for d, routes in sol.items():
        if isinstance(routes, list) and all(isinstance(r, list) for r in routes):
            clean_sol[d] = [r[:] for r in routes]
        else:
            print(f"[WARN] Depot {d} has invalid routes format: {routes}")
            clean_sol[d] = []
    return clean_sol


def _valid_customer_pos(route):
    """Return valid indices (excluding depot endpoints)."""
    return range(1, max(1, len(route) - 1))


# ==========================================================
# 🔁 2. Intra-depot 2-opt
# ==========================================================
def _two_opt_once(sol, problem, rng):
    for depot, routes in sol.items():
        for r_idx, route in enumerate(routes):
            n = len(route)
            if n < 4:
                continue
            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    cand = clone_solution(sol)
                    cand[depot][r_idx] = route[:i] + list(reversed(route[i:j + 1])) + route[j + 1:]
                    yield cand


# ==========================================================
# 🚛 3. Relocate (intra + inter-depot)
# ==========================================================
def _relocate_once(sol, problem, rng):
    depots = list(sol.keys())
    for d_a in depots:
        for d_b in depots:
            for r_a_idx, ra in enumerate(sol[d_a]):
                if len(ra) < 3:
                    continue
                for i in _valid_customer_pos(ra):
                    node = ra[i]
                    for r_b_idx, rb in enumerate(sol[d_b]):
                        for j in range(1, len(rb)):  # insert before depot end
                            if d_a == d_b and (r_a_idx == r_b_idx) and (j == i or j == i + 1):
                                continue
                            cand = clone_solution(sol)
                            cand[d_a][r_a_idx].pop(i)
                            cand[d_b][r_b_idx].insert(j, node)
                            yield cand


# ==========================================================
# 🔄 4. Swap (intra + inter-depot)
# ==========================================================
def _swap_once(sol, problem, rng):
    depots = list(sol.keys())
    for d_a in depots:
        for d_b in depots:
            for r_a_idx, ra in enumerate(sol[d_a]):
                if len(ra) < 3:
                    continue
                for r_b_idx, rb in enumerate(sol[d_b]):
                    if len(rb) < 3:
                        continue
                    for i in _valid_customer_pos(ra):
                        for j in _valid_customer_pos(rb):
                            cand = clone_solution(sol)
                            cand[d_a][r_a_idx][i], cand[d_b][r_b_idx][j] = cand[d_b][r_b_idx][j], cand[d_a][r_a_idx][i]
                            yield cand


# ==========================================================
# 🎲 5. Simulated Annealing Acceptance
# ==========================================================
def _accept(old_cost, new_cost, T, rng):
    """
    Accept a new solution if it's better, or probabilistically worse (simulated annealing style).
    Handles both scalar and (ul, ll, total) tuple costs.
    """
    # If the inputs are tuples, take the last element (total)
    if isinstance(old_cost, tuple):
        old_cost = old_cost[-1]
    if isinstance(new_cost, tuple):
        new_cost = new_cost[-1]

    if new_cost <= old_cost:
        return True

    # Probabilistic acceptance when worse
    return T > 1e-12 and (rng.random() < math.exp(-(new_cost - old_cost) / T))


# ==========================================================
# 🧠 6. Variable Neighborhood Descent with SA
# ==========================================================
def _vnd_with_sa(parent, problem, rng, T0=0.02, max_passes=2):
    current = clone_solution(parent)

    # quick lower-level allocation to compute cost
    y_alloc = {
        0: {
            d: sum(problem.demand.get(c, 0.0)
                   for route in current.get(d, [])
                   for c in route
                   if c in problem.customers)
            for d in problem.depots
        }
    }

    cur_cost = total_cost(current, y_alloc, problem)
    neighborhoods = (_two_opt_once, _relocate_once, _swap_once)

    T = T0
    for _ in range(max_passes):
        improved = False
        for gen in neighborhoods:
            best_local, best_cost = current, cur_cost
            for cand in gen(current, problem, rng):
                y_alloc = {
                    0: {
                        d: sum(problem.demand.get(c, 0.0)
                               for route in cand.get(d, [])
                               for c in route
                               if c in problem.customers)
                        for d in problem.depots
                    }
                }
                c = total_cost(cand, y_alloc, problem)
                if c < best_cost or _accept(cur_cost, c, T, rng):
                    best_local, best_cost = cand, c
            if best_cost < cur_cost:
                current, cur_cost = best_local, best_cost
                improved = True
            T *= 0.8
        if not improved:
            break
    return current


# ==========================================================
# 🏁 7. Public API
# ==========================================================
def apply_ul_operator(parent, problem, rng=random, n_candidates: int = 8):
    """Apply standard structural variation."""
    return _vnd_with_sa(parent, problem, rng, T0=0.02, max_passes=2)


def apply_ul_operator_guided(parent, ll_hint_cost, problem, rng=random, n_candidates: int = 8):
    """Guided variant (placeholder)."""
    return _vnd_with_sa(parent, problem, rng, T0=0.02, max_passes=2)

import math
from mdvrp.costs import total_cost, routing_cost

import math

def _is_empty_routes(routes):
    return (not routes) or all((not v) for v in routes.values())

def _is_empty_alloc(y_alloc):
    return (not y_alloc) or all((not v) for v in y_alloc.values())

def safe_cost(solution, problem):
    """
    Return (ok, ul_cost, ll_cost, total_cost).
    """
    try:
        routes = solution.get("routes", {})
        y_alloc = solution.get("alloc", {})

        # --- if empty → infeasible
        if not routes or not any(routes.values()):
            return False, float("inf"), float("inf"), float("inf")

        ul, ll, total = total_cost(routes, y_alloc, problem)

        if not math.isfinite(total):
            return False, float("inf"), float("inf"), float("inf")

        return True, float(ul), float(ll), float(total)

    except Exception as e:
        print(f"[safe_cost] error: {e}")
        return False, float("inf"), float("inf"), float("inf")

def evaluate_bi_solution(sol, problem):
    """
    Decode a Bi-MDVRP solution into its core components:
      routes: list[list[int]]
      y_alloc: dict[int, dict[int, float]]  # depot -> {customer -> qty}
    """
    # Object form
    if hasattr(sol, "perm"):
        routes = decode_permutation(sol.perm)
        y_alloc = getattr(sol, "alloc", None)

    # Dict form with perm/alloc
    elif isinstance(sol, dict) and "perm" in sol:
        routes = decode_permutation(sol["perm"])
        y_alloc = sol.get("alloc")

    # Dict form with routes/alloc (classic)
    else:
        routes = sol.get("routes")
        y_alloc = sol.get("alloc")

    return routes, y_alloc
def ensure_alloc(sol, problem, compute_alloc_fn):
    """
    Guarantee that sol has 'alloc' consistent with its routes.
    compute_alloc_fn(routes, problem) -> y_alloc
    Returns (routes, y_alloc, sol_with_alloc)
    """
    routes, y_alloc = evaluate_bi_solution(sol, problem)
    if y_alloc is None:
        y_alloc = compute_alloc_fn(routes, problem)
        # cache back into the solution for later (object or dict)
        if hasattr(sol, "alloc"):
            sol.alloc = y_alloc
        elif isinstance(sol, dict):
            sol["alloc"] = y_alloc
    return routes, y_alloc, sol

"""
def safe_cost(sol, problem, compute_alloc_fn=None):
    
    #Returns: (ok: bool, ul_cost: float, ll_cost: float, total_cost: float)
    #- UL = routing_cost(routes, problem)
    #- Total = total_cost(routes, y_alloc, problem)
    #- LL = Total - UL
    
    try:
        # If you pass a compute_alloc_fn, we’ll make sure y_alloc exists
        if compute_alloc_fn:
            routes, y_alloc, sol = ensure_alloc(sol, problem, compute_alloc_fn)
        else:
            routes, y_alloc = evaluate_bi_solution(sol, problem)
            if y_alloc is None:
                # LL unknown → treat as infeasible unless you demand compute_alloc_fn
                return False, float("inf"), float("inf"), float("inf")

        ul = routing_cost(routes, problem)
        tot = total_cost(routes, y_alloc, problem)  # NOTE: 3-arg signature
        ll = tot - ul
        ok = (tot < float("inf"))
        return ok, float(ul), float(ll), float(tot)

    except Exception as e:
        print(f"[safe_cost] error: {e}")
        return False, float("inf"), float("inf"), float("inf")
"""

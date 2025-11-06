# scripts/run_instance.py
import argparse
import random
from types import SimpleNamespace
from mdvrp.data import load_cordeau_mdvrp
from mdvrp.lower_level import solve_lower_level
from mdvrp.optimize import optimize
from mdvrp.utils import safe_cost
import math


def main():
    # ---- 1. Parse arguments ----
    parser = argparse.ArgumentParser(description="Run MDVRP-Qlearning optimization.")
    parser.add_argument("--instance", type=str, required=True, help="Path to instance file")
    parser.add_argument("--gens", type=int, default=100, help="Number of generations")
    parser.add_argument("--pop", type=int, default=10, help="Population size")
    args = parser.parse_args()

    # ---- 2. Load instance ----
    problem = load_cordeau_mdvrp(args.instance)
    print(f"✅ Loaded {len(problem.customers)} customers and {len(problem.depots)} depots from {args.instance}")

    # ---- 3. Config object ----
    cfg = SimpleNamespace(
        pop_size=args.pop,
        max_gens=args.gens,
        tournament_size=3,
        alpha=0.1,
        gamma=0.9,
        eps_start=0.3,
        eps_min=0.05,
        decay=0.995,
    )

    # ---- 4. Run optimizer ----
    rng = random.Random(42)
    best_sol, best_cost = optimize(problem, cfg, rng)

    # ---- 5. Print results ----
    print("=====EXECUTION=====")

    print("\n=== FINAL RESULT ===")
    print(f"Best cost: {best_cost:.2f}")
    # --- Final report ---
    routes = best_sol.get("routes", {})
    alloc = best_sol.get("alloc", {})  # expected: {depot: {customer: qty}}

    print("\n=== FINAL TRAJECTORIES ===")
    if not routes:
        print("[WARN] No routes found.")
    else:
        for depot_id, route_list in routes.items():
            if not route_list:
                print(f"Depot {depot_id}: (no vehicles/routes)")
                continue
            for v_idx, route in enumerate(route_list, start=1):
                path = " → ".join(map(str, route))
                print(f"Depot {depot_id} - Vehicle {v_idx}: {path}")

    print("\n=== ALLOCATION (y_alloc) PER DEPOT ===")
    if not alloc:
        print("[WARN] No allocation decisions found.")
    else:
        for depot_id, cust_qty in alloc.items():
            print(f"Depot {depot_id}:")
            if not cust_qty:
                print("   (no customers allocated)")
                continue
            for cust_id, qty in cust_qty.items():
                print(f"   → Customer {cust_id}: {qty:.2f} units")
    print("\n=== Final Costs ===")
    ok, ul_c, ll_c, tot_c = safe_cost(best_sol, problem, compute_alloc_fn=solve_lower_level)
    print("\n=== Final Costs ===")
    print(f"Upper-Level cost = {ul_c:.2f}")
    print(f"Lower-Level cost = {ll_c:.2f}")
    print(f"Total cost       = {tot_c:.2f}")
"""
    best_sol, best_cost = optimize(problem, cfg, rng)

    # Coerce to safe structure if needed
    if not isinstance(best_sol, dict):
        print("[WARN] best_sol is None or invalid — using empty fallback.")
        best_sol = {"routes": {}, "alloc": {}}

    routes = best_sol.get("routes", {})
    alloc = best_sol.get("alloc", {})

    print("\n=== FINAL TRAJECTORIES ===")
    if routes and any(routes.values()):
        for depot_id, route_list in routes.items():
            if not route_list:
                print(f"Depot {depot_id}: (no vehicles/routes)")
                continue
            for v_idx, route in enumerate(route_list, start=1):
                path = " → ".join(map(str, route))
                print(f"Depot {depot_id} - Vehicle {v_idx}: {path}")
    else:
        print("(no routes)")

    print("\n=== ALLOCATION (y_alloc) PER DEPOT ===")
    if alloc and any(alloc.values()):
        for depot_id, cust_qty in alloc.items():
            print(f"Depot {depot_id}:")
            if not cust_qty:
                print("   (no customers allocated)")
                continue
            for cust_id, qty in cust_qty.items():
                try:
                    print(f"   → Customer {cust_id}: {float(qty):.2f} units")
                except Exception:
                    print(f"   → Customer {cust_id}: {qty} units")
    else:
        print("(no allocation)")

    # Cost breakdown
    ok, ul, ll, tot = safe_cost(best_sol, problem)
    ul_s = f"{ul:.2f}" if math.isfinite(ul) else "inf"
    ll_s = f"{ll:.2f}" if math.isfinite(ll) else "inf"
    tot_s = f"{tot:.2f}" if math.isfinite(tot) else "inf"
    print(f"\nCosts — Upper (routing): {ul_s} | Lower (plant): {ll_s} | Total: {tot_s}")
"""
if __name__ == "__main__":
    main()
# python -m script.run_instance --instance "instances/p01" --gens 5 --pop 5

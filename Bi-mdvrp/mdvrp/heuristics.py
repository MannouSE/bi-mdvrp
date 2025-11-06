# heuristics.py
from __future__ import annotations
import random
from typing import List, Dict, Optional
from mdvrp.costs import total_cost
from mdvrp.operators import apply_ul_operator
from mdvrp.elite import update_elite_archive, cluster_elite_archive
from mdvrp.cluster import embed_solution, nearest_centroid_idx, sqdist
from mdvrp.solution import quick_repair
from mdvrp.utils import safe_cost

# ==========================================================
# 🧩 Centroid / Embedding Helpers
# ==========================================================
def _ensure_centroids(elite, centroids, rng: Optional[random.Random] = None):
    """Compute centroids if missing."""
    if centroids:
        return centroids, False
    return cluster_elite_archive(elite, rng=rng), True


def find_nearest_centroid(solution, centroids: List[List[float]], problem) -> int:
    """Embed solution and find the nearest centroid index."""
    if not centroids:
        return -1
    dim = getattr(problem, "embed_dim", 256)
    v = embed_solution(solution, dim)
    return nearest_centroid_idx(v, centroids)


# ==========================================================
# 💡 Heuristic H1 — Full hierarchical (exploit elite)
# ==========================================================
def heuristic_h1_full_hierarchical(parent, elite, centroids, problem, rng):
    """
    H1 – Full Hierarchical:
    Solve LL exactly for each UL solution, align with nearest centroid,
    and apply a UL operator. Full exploitation mode.
    """
    c_parent = full_cost(parent, problem)  # Evaluate full cost once
    centroids, _ = _ensure_centroids(elite, centroids, rng)

    start = parent
    if centroids:
        ci = find_nearest_centroid(parent, centroids, problem)
        if ci >= 0:
            target = centroids[ci]
            best_e, best_d = None, float("inf")
            for cost, sol, emb in elite:
                d = sqdist(emb, target)
                if d < best_d:
                    best_d, best_e = d, sol
            if best_e is not None:
                start = best_e

    child = apply_ul_operator(start, problem, rng)
    return quick_repair(child, problem)


def heuristic_h2_selective_ll(parent, elite, problem, rng):
    """
    H2 – Selective LL Evaluation:
    Evaluate LL only for promising ULs, then apply UL operator for exploration.
    """
    if is_promising_ul(parent, problem):
        ok, ll_sol, ll_cost = _solve_ll_exact3(parent, problem, rng)
        if ok:
            update_elite_archive(
                elite, ll_sol, full_cost(ll_sol, problem),
                dim=getattr(problem, "embed_dim", 256),
                max_size=getattr(problem, "elite_max", 120)
            )

    child = apply_ul_operator(parent, problem, rng)
    return quick_repair(child, problem)


def heuristic_h3_relaxed_ll(parent, problem, rng):
    """
    H3 – Relaxed LL:
    Skip LL solving; perform quick UL exploration only.
    """
    child = apply_ul_operator(parent, problem, rng)
    return quick_repair(child, problem)


def heuristic_h4_similarity_based(parent, centroids, elite, problem, rng):
    """
    H4 – Similarity-based:
    Start from the elite solution closest to the parent’s nearest centroid.
    """
    centroids, _ = _ensure_centroids(elite, centroids, rng)
    if not centroids:
        child = apply_ul_operator(parent, problem, rng)
        return quick_repair(child, problem)

    ci = find_nearest_centroid(parent, centroids, problem)
    if ci < 0:
        child = apply_ul_operator(parent, problem, rng)
        return quick_repair(child, problem)

    target = centroids[ci]
    best_sol, best_d = None, float("inf")
    for cost, sol, emb in elite:
        d = sqdist(emb, target)
        if d < best_d:
            best_d, best_sol = d, sol

    start = best_sol if best_sol is not None else parent
    child = apply_ul_operator(start, problem, rng)
    return quick_repair(child, problem)


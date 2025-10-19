# heuristics.py
from __future__ import annotations
import random
from typing import List, Dict, Optional
from mdvrp.costs import total_cost
from mdvrp.operators import apply_ul_operator
from mdvrp.elite import update_elite_archive, cluster_elite_archive
from mdvrp.cluster import embed_solution, nearest_centroid_idx, sqdist
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
    H1: Full hierarchical.
      1. Evaluate parent cost.
      2. Add to elite archive with embedding.
      3. Find nearest elite by centroid similarity.
      4. Mutate and return child.
    """
    c_parent = safe_cost(parent, problem)
    update_elite_archive(
        elite,
        parent,
        c_parent,
        dim=getattr(problem, "embed_dim", 256),
        max_size=getattr(problem, "elite_max", 120),
    )

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

    return apply_ul_operator(start, problem, rng)


# ==========================================================
# 💡 Heuristic H2 — Selective LL evaluation
# ==========================================================
def heuristic_h2_selective_ll(parent, elite, problem, rng: random.Random = random):
    """
    H2: Selective lower-level evaluation.
      - Evaluate cost and update elite if improved.
      - Always return a perturbed child for exploration.
    """
    c_parent = safe_cost(parent, problem)
    if c_parent < float("inf"):
        update_elite_archive(
            elite,
            parent,
            c_parent,
            dim=getattr(problem, "embed_dim", 256),
            max_size=getattr(problem, "elite_max", 120),
        )
    return apply_ul_operator(parent, problem, rng)


# ==========================================================
# 💡 Heuristic H3 — Relaxed (exploratory mutation)
# ==========================================================
def heuristic_h3_relaxed_ll(parent, problem, rng: random.Random = random):
    """H3: Random exploration without evaluation."""
    return apply_ul_operator(parent, problem, rng)


# ==========================================================
# 💡 Heuristic H4 — Similarity-based
# ==========================================================
def heuristic_h4_similarity_based(parent, centroids, elite, problem, rng: random.Random = random):
    """
    H4: Similarity-based start.
      - Use embedding distance to start from the most similar elite.
      - Then apply operator for mutation.
    """
    centroids, _ = _ensure_centroids(elite, centroids, rng)
    if not centroids:
        return apply_ul_operator(parent, problem, rng)

    ci = find_nearest_centroid(parent, centroids, problem)
    if ci < 0:
        return apply_ul_operator(parent, problem, rng)

    target = centroids[ci]
    best_sol, best_d = None, float("inf")
    for cost, sol, emb in elite:
        d = sqdist(emb, target)
        if d < best_d:
            best_d, best_sol = d, sol

    start = best_sol if best_sol is not None else parent
    return apply_ul_operator(start, problem, rng)



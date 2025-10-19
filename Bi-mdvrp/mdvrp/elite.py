# elite.py
from __future__ import annotations
from typing import List, Tuple, Iterable, Optional, Union
from random import Random
from mdvrp.cluster import embed_solution, kmeans


# ==========================================================
# 🧩 1. Elite Entry
# ==========================================================
# Each entry: (cost, solution, embedding)
EliteEntry = Tuple[float, dict, List[float]]  # <-- changed: sol = dict {depot: [routes]}


# ==========================================================
# 🏆 2. Update Elite Archive
# ==========================================================
def update_elite_archive(
    elite: List[EliteEntry],
    sol: dict,
    cost: float,
    dim: int = 32,
    max_size: int = 100
):
    """
    Add a solution to the elite archive, sorted by cost.
    Keeps best 'max_size' solutions.
    """
    emb = embed_solution(sol, dim)
    if isinstance(cost, tuple):
        cost_value = cost[-1]  # use total cost
    else:
        cost_value = cost

    elite.append((cost_value, sol, emb))
    elite.sort(key=lambda e: e[0])  # e[0] is now always a float
    if len(elite) > max_size:
        del elite[max_size:]


# ==========================================================
# 🔁 3. Iterator for flexible elite structure
# ==========================================================
def _iter_entries(elite: Union[List[EliteEntry], dict]) -> Iterable[EliteEntry]:
    """Support both list and dict elite formats."""
    return elite.values() if isinstance(elite, dict) else elite


# ==========================================================
# 🧮 4. Normalize Embeddings
def _normalize_points(points: List[List[float]]) -> List[List[float]]:
    """
    Pad or truncate embedding vectors to a uniform length.

    - If a vector is shorter than the longest, pad with zeros.
    - If a vector is longer, truncate it.
    """
    if not points:
        return points

    target = max(len(p) for p in points)
    norm = []
    for p in points:
        if len(p) < target:
            # Pad with zeros
            padded = p + [0.0] * (target - len(p))
            norm.append(padded)
        else:
            # Truncate to the target length
            norm.append(p[:target])

    return norm
def cluster_elite_archive(
    elite: Union[List[EliteEntry], dict],
    k: int = 3,
    rounds: int = 10,
    rng: Optional[Random] = None
) -> List[List[float]]:
    """
    Cluster elite archive embeddings into k centroids.
    Safe version: handles empty or malformed elite entries.
    """
    entries = list(_iter_entries(elite))
    if not entries:
        return []

    # extract all embeddings (e[2]) that are not None
    points = [e[2] for e in entries if e[2] is not None]

    # if still empty, nothing to cluster
    if not points:
        return []

    # normalize shape
    points = _normalize_points(points)

    # 🔒 safety guard: make sure normalization didn’t return None
    if not points or points is None:
        return []

    k = min(k, len(points))
    if k <= 0:
        return []

    centers, _ = kmeans(points, k, rounds, rng)
    return centers if centers is not None else []
# cluster.py
from __future__ import annotations
from typing import List, Tuple, Optional
import random



# ==========================================================
# 🧠 1. Embedding Function
# ==========================================================
def embed_solution(solution, dim: int = 32) -> List[float]:
    """
    Convert a multi-depot solution into a simple numeric embedding vector.
    For MDVRP: concatenate route lengths, depot loads, etc.
    Used for similarity and clustering.
    """
    values = []

    # Handle multi-depot dictionary {depot: [routes]}
    if isinstance(solution, dict):
        for depot, routes in solution.items():
            for route in routes:
                # Encode route length and load roughly
                values.append(len(route))
                values.append(sum(route))
    else:
        # Single list of routes
        for route in solution:
            values.append(len(route))
            values.append(sum(route))

    # Normalize size to 'dim'
    flat = values[:dim] + [0.0] * max(0, dim - len(values))
    return flat


# ==========================================================
# 📍 2. Distance Function
# ==========================================================
def sqdist(a: List[float], b: List[float]) -> float:
    """Squared Euclidean distance."""
    return sum((ai - bi) ** 2 for ai, bi in zip(a, b))


# ==========================================================
# 🔍 3. Nearest Centroid
# ==========================================================
def nearest_centroid_idx(v: List[float], centroids: List[List[float]]) -> int:
    """Return the index of the nearest centroid."""
    if not centroids:
        return -1
    best_i, best_d = -1, float("inf")
    for i, c in enumerate(centroids):
        d = sqdist(v, c)
        if d < best_d:
            best_i, best_d = i, d
    return best_i


# ==========================================================
# 📊 4. Simple K-Means Clustering
# ==========================================================
def kmeans(points: List[List[float]], k: int, rounds: int = 10,
           rng: Optional[random.Random] = None) -> Tuple[List[List[float]], List[int]]:
    """
    Lightweight K-Means for elite clustering.
    Returns (centroids, assignments).
    """
    if rng is None:
        rng = random

    if not points:
        return [], []

    n = len(points)
    dim = len(points[0])
    k = min(k, n)
    centroids = [points[i][:] for i in rng.sample(range(n), k)]
    assign = [0] * n

    for _ in range(rounds):
        # Assign
        for i, p in enumerate(points):
            assign[i] = nearest_centroid_idx(p, centroids)

        # Recompute
        new_centroids = [[0.0] * dim for _ in range(k)]
        counts = [0] * k
        for idx, p in zip(assign, points):
            for j in range(dim):
                new_centroids[idx][j] += p[j]
            counts[idx] += 1

        for i in range(k):
            if counts[i] > 0:
                new_centroids[i] = [x / counts[i] for x in new_centroids[i]]
            else:
                new_centroids[i] = centroids[i][:]

        centroids = new_centroids

    return centroids, assign

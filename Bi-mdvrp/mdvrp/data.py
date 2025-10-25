from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from math import sqrt
import random


@dataclass
class Problem:
    """Container for an MDVRP instance (multi-depot VRP)."""

    name: str = "unnamed"
    depots: List[int] = field(default_factory=list)
    customers: List[int] = field(default_factory=list)
    plants: List[int] = field(default_factory=list)

    coords: Dict[int, Tuple[float, float]] = field(default_factory=dict)  # Changed to Dict!
    distance_matrix: List[List[float]] = field(default_factory=list)

    capacity: float = 0.0
    demand: Dict[int, float] = field(default_factory=dict)
    vehicles_per_depot: Dict[int, int] = field(default_factory=dict)

    # Bi-level costs
    c_a: Dict[Tuple[int, int], float] = field(default_factory=dict)  # transport cost
    c_b: Dict[int, Dict[int, float]] = field(default_factory=dict)  # buy/unload cost
    c_c: Dict[int, Dict[int, float]] = field(default_factory=dict)  # production cost
    plant_capacity: Dict[int, float] = field(default_factory=dict)  # plant capacity

    # Optional
    route_limit: float = 0.0
    ready_time: Dict[int, float] = field(default_factory=dict)
    due_time: Dict[int, float] = field(default_factory=dict)
    service_time: Dict[int, float] = field(default_factory=dict)

    plants: List[int] = field(default_factory=list)
    depots: List[int] = field(default_factory=list)
    plant_capacity: Dict[int, float] = field(default_factory=dict)
    plant_cost_matrix: Dict[int, Dict[int, float]] = field(default_factory=dict)

def _euc2d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance."""
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def build_distance_matrix(coords: Dict[int, Tuple[float, float]]) -> List[List[float]]:
    """
    Build a 1-based dense distance matrix.
    Matrix[i][j] = distance from node i to node j.
    Index 0 is unused (to match 1-based node IDs).
    """
    if not coords:
        return [[]]

    max_node = max(coords.keys())
    D = [[0.0] * (max_node + 1) for _ in range(max_node + 1)]

    for i in coords:
        for j in coords:
            if i != j:
                D[i][j] = _euc2d(coords[i], coords[j])

    return D


def load_cordeau_mdvrp(path: str, seed: int = 42) -> Problem:
    """
    Parse a Cordeau/Gendreau MDVRP file robustly.
    Treats depots as both distribution centers AND production plants.
    """
    rng = random.Random(seed)

    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Parse header
    try:
        n_depots, veh_per_depot, capacity, route_limit = map(int, lines[0].split())
    except Exception as e:
        raise ValueError(f"Error reading header from {path}: {e}")

    # Skip depot constraint lines
    node_lines = lines[n_depots + 1:]

    # Parse nodes
    coords: Dict[int, Tuple[float, float]] = {}
    demand: Dict[int, float] = {}
    depots_list = []
    customers_list = []

    for raw in node_lines:
        parts = raw.split()
        if len(parts) < 5:
            print(f"⚠️ Skipping malformed line: '{raw}'")
            continue

        try:
            idx = int(parts[0])
            x, y = float(parts[1]), float(parts[2])
            q = float(parts[4])  # demand
        except Exception as e:
            print(f"⚠️ Could not parse line: '{raw}' ({e})")
            continue

        coords[idx] = (x, y)
        demand[idx] = q

        if q == 0:
            depots_list.append(idx)
        else:
            customers_list.append(idx)

    if not coords:
        raise ValueError(f"No coordinates parsed from {path}")

    # Select active depots (last n_depots)
    active_depots = depots_list[-n_depots:] if len(depots_list) >= n_depots else depots_list

    if not active_depots:
        raise ValueError(f"No depots found in {path}")

    # Build distance matrix
    D = build_distance_matrix(coords)

    # ============================================
    # CRITICAL: Use depots as plants
    # ============================================
    plants = active_depots.copy()  # Depots act as manufacturing plants

    # Calculate total demand
    total_demand = sum(demand.get(c, 0.0) for c in customers_list)
    K = len(plants)

    # Generate plant capacities: each plant can produce between [D/K, D]
    plant_capacity = {}
    for k in plants:
        plant_capacity[k] = rng.uniform(
            total_demand / max(K, 1),
            total_demand
        )

    # Generate production cost c_c[k][l]: cost to manufacture at plant k for depot l
    c_c = {}
    for k in plants:
        c_c[k] = {}
        for l in active_depots:
            if k == l:
                c_c[k][l] = rng.uniform(1, 2)  # cheaper to produce at own location
            else:
                c_c[k][l] = rng.uniform(2, 4)  # more expensive for other depots

    # Generate buy/unload cost c_b[k][l]: cost to transfer from plant k to depot l
    c_b = {}
    for k in plants:
        c_b[k] = {}
        for l in active_depots:
            if k == l:
                c_b[k][l] = 0.0  # no transfer cost if same location
            else:
                c_b[k][l] = rng.uniform(0.36, 5.36)

    # Transportation costs c_a[(i,j)]: based on distance
    c_a = {}
    max_node = len(D) - 1
    for i in range(1, max_node + 1):
        for j in range(1, max_node + 1):
            if i != j and i in coords and j in coords:
                c_a[(i, j)] = D[i][j]

    # Create problem instance
    problem = Problem(
        name=path.split("/")[-1].split("\\")[-1],  # Handle both / and \
        depots=active_depots,
        customers=customers_list,
        plants=plants,
        coords=coords,  # Keep as dict - easier to use!
        distance_matrix=D,
        demand=demand,
        capacity=capacity,
        vehicles_per_depot={d: veh_per_depot for d in active_depots},
        c_a=c_a,
        c_b=c_b,
        c_c=c_c,
        plant_capacity=plant_capacity,
        route_limit=route_limit,
    )

    return problem


def assign_customers_to_depots(problem: Problem) -> Dict[int, List[int]]:
    """
    Simple heuristic: assign each customer to the nearest depot.
    Returns: {depot: [customers]}
    """
    clusters = {d: [] for d in problem.depots}
    for c in problem.customers:
        nearest = min(
            problem.depots,
            key=lambda d: problem.distance_matrix[d][c]
        )
        clusters[nearest].append(c)
    return clusters
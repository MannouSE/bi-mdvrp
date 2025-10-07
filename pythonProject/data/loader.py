from math import sqrt
from typing import List, Tuple, Optional
from data.problem import Problem

def _euc2d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance in 2D."""
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def build_distance_matrix(coords: List[Tuple[float, float]]) -> List[List[float]]:
    """
    Build a 1-based dense distance matrix (index 0 row/col left as zeros).
    """
    n = len(coords) - 1
    D = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        ci = coords[i]
        for j in range(1, n + 1):
            if i != j:
                D[i][j] = _euc2d(ci, coords[j])
    return D


def load_evrp(path: str) -> Problem:
    """
    Load an EVRP instance from file.

    Expected headers:
      NAME, VEHICLES, DIMENSION, STATIONS, CAPACITY, ENERGY_CAPACITY,
      NODE_COORD_SECTION with coordinates (index x y).

    Assumptions:
      - Depot index = 1
      - Stations = last `STATIONS` nodes
      - Customers = between depot and stations
    """
    name = ""
    vehicles = capacity = dimension = stations_cnt = None
    energy_capacity = None

    coords: List[Tuple[float, float]] = [(-1.0, -1.0)]  # pad index 0 (unused)
    reading_coords = False

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if reading_coords:
                if line and line[0].isdigit():
                    idx, x, y = line.split()[:3]
                    idx = int(idx)
                    x, y = float(x), float(y)
                    if idx >= len(coords):
                        coords.extend([(0.0, 0.0)] * (idx - len(coords) + 1))
                    coords[idx] = (x, y)
                    continue
                else:
                    reading_coords = False

            if line.startswith("NAME:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("VEHICLES:"):
                vehicles = int(line.split(":", 1)[1].strip())
            elif line.startswith("DIMENSION:"):
                dimension = int(line.split(":", 1)[1].strip())
            elif line.startswith("STATIONS:"):
                stations_cnt = int(line.split(":", 1)[1].strip())
            elif line.startswith("CAPACITY:"):
                capacity = int(line.split(":", 1)[1].strip())
            elif line.startswith("ENERGY_CAPACITY:"):
                energy_capacity = float(line.split(":", 1)[1].strip())
            elif line.startswith("NODE_COORD_SECTION"):
                reading_coords = True

    if dimension is None or stations_cnt is None:
        raise ValueError("Missing DIMENSION or STATIONS in instance header.")
    if vehicles is None or capacity is None or energy_capacity is None:
        raise ValueError("Missing VEHICLES, CAPACITY, or ENERGY_CAPACITY.")
    if len(coords) - 1 != dimension:
        raise ValueError(f"Expected {dimension} coords, got {len(coords) - 1}.")

    depot = 1
    num_customers = dimension - stations_cnt - 1
    customers = list(range(2, 2 + num_customers))
    stations = list(range(dimension - stations_cnt + 1, dimension + 1))

    problem = Problem(
        name=name,
        vehicles=vehicles,
        capacity=capacity,
        battery=energy_capacity,
        depot=depot,
        customers=customers,
        stations=stations,
        coords={i: coords[i] for i in range(1, len(coords))}
    )

    # Precompute distances
    D = build_distance_matrix(coords)
    problem.distances = {(i, j): D[i][j] for i in range(1, dimension+1)
                                        for j in range(1, dimension+1) if i != j}

    # Default demands = 0 (can be extended if DEMAND_SECTION is in the file)
    problem.demand = {i: 0 for i in range(1, dimension+1)}

    return problem


def apply_defaults(
    problem: Problem,
    *,
    charge_rate: Optional[float] = None,
    energy_cost: Optional[float] = None,
    waiting_cost: Optional[float] = None,
    speed: Optional[float] = None,
) -> Problem:
    """Optional runtime overrides for parameters."""
    if charge_rate is not None:
        problem.charge_rate = charge_rate
    if energy_cost is not None:
        problem.energy_cost = energy_cost
    if waiting_cost is not None:
        problem.waiting_cost = waiting_cost
    if speed is not None:
        problem.speed = speed
    return problem

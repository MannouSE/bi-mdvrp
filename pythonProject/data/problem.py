from dataclasses import dataclass, field
from typing import List, Dict, Tuple

@dataclass
class Problem:
    name: str = "unnamed"
    vehicles: int = 1
    capacity: int = 1000                # vehicle payload capacity
    battery: float = 100.0              # max battery capacity (SoC)
    depot: int = 0

    customers: List[int] = field(default_factory=list)
    stations: List[int] = field(default_factory=list)

    demand: Dict[int, float] = field(default_factory=dict)      # q_i
    coords: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    distances: Dict[Tuple[int, int], float] = field(default_factory=dict)
    energy: Dict[Tuple[int, int], float] = field(default_factory=dict)

    # Optional: time window info
    ready_time: Dict[int, float] = field(default_factory=dict)
    due_time: Dict[int, float] = field(default_factory=dict)
    service_time: Dict[int, float] = field(default_factory=dict)

    def distance(self, i: int, j: int) -> float:
        return self.distances.get((i, j), float("inf"))

    def energy_consumption(self, i: int, j: int) -> float:
        return self.energy.get((i, j), float("inf"))

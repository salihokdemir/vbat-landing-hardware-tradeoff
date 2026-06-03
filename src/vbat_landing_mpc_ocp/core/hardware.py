from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class HardwarePoint:
    """Four-metric actuator-authority vector used by MPC search and OCP refinement."""

    tw_ratio: float
    T_dot_max: float
    delta_dot_max: float
    delta_max: float

    @property
    def delta_dot_max_deg(self) -> float:
        return math.degrees(self.delta_dot_max)

    @property
    def delta_max_deg(self) -> float:
        return math.degrees(self.delta_max)

    def key(self, digits: int = 8) -> Tuple[float, float, float, float]:
        return (
            round(self.tw_ratio, digits),
            round(self.T_dot_max, digits),
            round(self.delta_dot_max, digits),
            round(self.delta_max, digits),
        )

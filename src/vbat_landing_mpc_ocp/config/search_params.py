from __future__ import annotations

import math
from dataclasses import dataclass, replace

from config.vehicle_params import (
    HardwareOverrides,
    MPCParams as _BaseMPCParams,
    UAVParams as _BaseUAVParams,
    build_stabilized_bundle,
)
from core.hardware import HardwarePoint


# Re-export the stabilized baseline dataclasses so the copied MPC controller can live in its own file
# without touching the original baseline imports.
UAVParams = _BaseUAVParams
MPCParams = _BaseMPCParams


@dataclass(frozen=True)
class MPCFrequencySpec:
    frequency_hz: float
    preview_h_s: float = 2.0

    @property
    def dt(self) -> float:
        return 1.0 / float(self.frequency_hz)

    @property
    def N(self) -> int:
        return max(1, int(round(float(self.preview_h_s) * float(self.frequency_hz))))

    @property
    def preview_h_actual_s(self) -> float:
        return self.N * self.dt


@dataclass(frozen=True)
class MPCFinalSearchCaps:
    # These are search safety caps, not physical truth claims.
    tw_ratio_cap: float = 2.0
    T_dot_cap_weight_per_sec: float = 2.0
    delta_cap_deg: float = 60.0
    delta_dot_cap_deg_s: float = 360.0


def make_hardware_overrides_from_point(hw: HardwarePoint) -> HardwareOverrides:
    return HardwareOverrides(
        tw_ratio=float(hw.tw_ratio),
        T_dot_max=float(hw.T_dot_max),
        delta_max_deg=float(hw.delta_max_deg),
        delta_dot_max_deg_s=float(hw.delta_dot_max_deg),
    )


def build_frequency_bundle(
    *,
    base_wind: float,
    frequency_hz: float,
    preview_h_s: float = 2.0,
    hardware_overrides: HardwareOverrides | None = None,
):
    uav_p, mpc_p_base, _ = build_stabilized_bundle(
        base_wind=float(base_wind),
        hardware_overrides=hardware_overrides,
    )
    freq = MPCFrequencySpec(frequency_hz=float(frequency_hz), preview_h_s=float(preview_h_s))
    mpc_p = replace(mpc_p_base, N=freq.N, dt=freq.dt)
    return uav_p, mpc_p, freq


def build_cap_hardware(ref_hw: HardwarePoint, *, weight: float, caps: MPCFinalSearchCaps) -> HardwarePoint:
    return HardwarePoint(
        tw_ratio=max(float(ref_hw.tw_ratio), float(caps.tw_ratio_cap)),
        T_dot_max=max(float(ref_hw.T_dot_max), float(caps.T_dot_cap_weight_per_sec) * float(weight)),
        delta_dot_max=max(float(ref_hw.delta_dot_max), math.radians(float(caps.delta_dot_cap_deg_s))),
        delta_max=max(float(ref_hw.delta_max), math.radians(float(caps.delta_cap_deg))),
    )

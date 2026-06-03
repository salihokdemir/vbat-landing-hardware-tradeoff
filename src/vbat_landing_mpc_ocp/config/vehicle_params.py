from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class UAVParams:
    # Physical
    m: float = 4.0
    g: float = 9.81
    I_yy: float = 0.5
    L_cg_prop: float = 0.70
    L_cp: float = 0.05

    # Aero / environment
    rho: float = 1.225
    S_wing: float = 0.4
    C_D90: float = 1.15
    V_wind: float = 0.0

    # Safe / generous hardware defaults used for baseline MPC warm-start generation.
    # 4D search already overrides these per OCP attempt.
    TW_ratio_max: float = 2.0
    delta_max: float = math.radians(60.0)
    T_dot_max: float = 100.0
    delta_dot_max: float = 3.0

    @property
    def delta_max_deg(self) -> float:
        return math.degrees(self.delta_max)

    @property
    def delta_dot_max_deg_s(self) -> float:
        return math.degrees(self.delta_dot_max)


@dataclass
class MPCParams:
    # Horizon
    N: int = 40
    dt: float = 0.05

    # Tracking / landing shaping
    W_track: float = 150.0
    W_theta: float = 400.0
    W_v_landing: float = 0.0
    W_effort: float = 0.0
    W_pitch_limit: float = 5000.0

    # Mining-mode regularization
    W_q: float = 2.0
    W_delta_state: float = 0.0
    W_T_state: float = 0.0

    # Shared landing / flare profile
    flare_start_alt_m: float = 12.0
    flare_settle_alt_m: float = 5.0
    vx_rel_cap_high: float = 5.0
    vx_rel_cap_low: float = 1.2
    vz_rel_cap_high: float = 4.0
    vz_rel_cap_low: float = 0.6
    ref_velocity_smoothing_window: int = 5
    ref_velocity_smoothing_passes: int = 2
    ref_vx_accel_limit: float = 2.5
    ref_vz_accel_limit: float = 2.0

    # Soft pitch window used inside objective
    theta_min_deg: float = 50.0
    theta_max_deg: float = 130.0

    # Funnel geometry
    funnel_low_m: float = 1.5
    funnel_mid_m: float = 1.5
    funnel_high_base_m: float = 2.5
    funnel_mid_slope: float = (1.0 / 7.0)
    funnel_high_slope: float = 5.0

    # Solver / warm-start behaviour
    ipopt_max_iter: int = 3000
    ipopt_tol: float = 1e-4
    ipopt_acceptable_tol: float = 5e-4
    warmstart_shift: bool = True
    retry_on_fail: bool = True
    retry_with_default_guess: bool = True


@dataclass
class OCPParams:
    # Shared landing / flare profile
    flare_start_alt_m: float = 12.0
    flare_settle_alt_m: float = 5.0
    vx_rel_cap_high: float = 5.0
    vx_rel_cap_low: float = 1.2
    vz_rel_cap_high: float = 4.0
    vz_rel_cap_low: float = 0.6
    ref_velocity_smoothing_window: int = 5
    ref_velocity_smoothing_passes: int = 2
    ref_vx_accel_limit: float = 2.5
    ref_vz_accel_limit: float = 2.0

    # Local refinement shaping / anchoring
    W_track_stage: float = 5.0
    W_effort: float = 0.0
    W_q: float = 0.2
    W_delta_state: float = 0.0
    W_T_state: float = 0.0
    W_theta_soft: float = 5.0
    W_vx_rel_soft: float = 0.0

    # Near-saturation regularization (candidate-normalized)
    sat_penalty_start_ratio: float = 0.85
    W_sat_T_excess: float = 0.15
    W_sat_delta: float = 0.05
    W_sat_T_dot: float = 0.10
    W_sat_delta_dot: float = 0.10

    # Hard / semi-hard geometry
    theta_min_deg: float = 30.0
    theta_max_deg: float = 150.0
    tube_low_m: float = 1.5
    tube_mid_m: float = 1.5
    tube_high_m: float = 2.5

    terminal_x_tol_m: float = 1.5
    terminal_z_below_tol_m: float = 2.0
    terminal_z_above_tol_m: float = 0.5
    terminal_vx_rel_tol: float = 1.0
    terminal_vz_rel_min: float = -1.0
    terminal_vz_rel_max: float = 0.1

    stage_vz_lower_margin: float = 5.0
    stage_vz_upper_margin: float = 0.25

    # Solver / retry
    ipopt_max_iter: int = 2000
    ipopt_tol: float = 1e-4
    ipopt_acceptable_tol: float = 5e-4
    retry_on_fail: bool = True
    retry_with_default_guess: bool = True


@dataclass(frozen=True)
class HardwareOverrides:
    tw_ratio: Optional[float] = None
    T_dot_max: Optional[float] = None
    delta_max_deg: Optional[float] = None
    delta_dot_max_deg_s: Optional[float] = None


def apply_hardware_overrides(
    uav_p: UAVParams,
    overrides: Optional[HardwareOverrides],
) -> UAVParams:
    if overrides is None:
        return uav_p

    return UAVParams(
        m=uav_p.m,
        g=uav_p.g,
        I_yy=uav_p.I_yy,
        L_cg_prop=uav_p.L_cg_prop,
        L_cp=uav_p.L_cp,
        rho=uav_p.rho,
        S_wing=uav_p.S_wing,
        C_D90=uav_p.C_D90,
        V_wind=uav_p.V_wind,
        TW_ratio_max=float(overrides.tw_ratio) if overrides.tw_ratio is not None else uav_p.TW_ratio_max,
        delta_max=math.radians(float(overrides.delta_max_deg))
        if overrides.delta_max_deg is not None
        else uav_p.delta_max,
        T_dot_max=float(overrides.T_dot_max) if overrides.T_dot_max is not None else uav_p.T_dot_max,
        delta_dot_max=math.radians(float(overrides.delta_dot_max_deg_s))
        if overrides.delta_dot_max_deg_s is not None
        else uav_p.delta_dot_max,
    )


def build_stabilized_bundle(
    *,
    base_wind: float = 0.0,
    hardware_overrides: Optional[HardwareOverrides] = None,
) -> Tuple[UAVParams, MPCParams, OCPParams]:
    uav_p = UAVParams(V_wind=float(base_wind))
    uav_p = apply_hardware_overrides(uav_p, hardware_overrides)
    return uav_p, MPCParams(), OCPParams()

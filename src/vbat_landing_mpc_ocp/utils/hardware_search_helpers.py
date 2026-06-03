from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from config.search_params import MPCFinalSearchCaps
from core.hardware import HardwarePoint
from run_mpc_replay import FrequencySweepConfig


@dataclass(frozen=True)
class UtopiaBatchConfig:
    output_dir: str = 'outputs_main4_hardware_search'
    seed: int = 42
    num_scenarios: int = 20
    num_workers: int = 8

    anchor_freq_hz: float = 20.0
    preview_h_s: float = 2.0
    sim_time_max_s: float = 30.0

    touchdown_x_tol: float = 1.0
    touchdown_z_tol: float = 0.10
    touchdown_vx_rel_tol: float = 1.0
    touchdown_vz_rel_min: float = -2.0
    touchdown_vz_rel_max: float = 0.5

    disable_deck_heave: bool = False
    disable_wind_gusts: bool = False
    utopia_only: bool = False
    sanity_check_cap: bool = True

    second_utopia_enabled: bool = True
    second_utopia_pad_frac: float = 0.10
    second_utopia_seed_tag: str = 'closest'


def _hardware_to_row_dict(hw: Optional[HardwarePoint], prefix: str, weight: float) -> Dict[str, Optional[float]]:
    if hw is None:
        return {
            f'{prefix}_tw_ratio': None,
            f'{prefix}_T_dot_max': None,
            f'{prefix}_T_dot_weight_per_sec': None,
            f'{prefix}_delta_max_deg': None,
            f'{prefix}_delta_dot_max_deg_s': None,
        }
    return {
        f'{prefix}_tw_ratio': float(hw.tw_ratio),
        f'{prefix}_T_dot_max': float(hw.T_dot_max),
        f'{prefix}_T_dot_weight_per_sec': float(hw.T_dot_max) / float(weight),
        f'{prefix}_delta_max_deg': float(hw.delta_max_deg),
        f'{prefix}_delta_dot_max_deg_s': float(hw.delta_dot_max_deg),
    }


def _scenario_base_row(scenario: dict) -> dict:
    return {
        'scenario_id': int(scenario['id']),
        'deck_vx': float(scenario['deck_vx']),
        'deck_z0': float(scenario['deck_z0']),
        'deck_z_amp': float(scenario['deck_z_amp']),
        'deck_z_freq': float(scenario['deck_z_freq']),
        'deck_z_phase': float(scenario['deck_z_phase']),
        'deck_z_peak_to_peak': float(scenario.get('deck_z_peak_to_peak', 0.0)),
        'deck_z_peak_vz': float(scenario.get('deck_z_peak_vz', 0.0)),
        'deck_z_period_s': scenario.get('deck_z_period_s'),
        'base_wind_x': float(scenario['base_wind_x']),
        'rel_x0': float(scenario['rel_x0']),
        'rel_z0': float(scenario['rel_z0']),
        'v_x_rel0': float(scenario['v_x_rel0']),
        'v_z0': float(scenario['v_z']),
        'theta_deg': float(scenario['theta_deg']),
        'q': float(scenario['q']),
    }


def _apply_debug_overrides(scenario: dict, *, disable_deck_heave: bool, disable_wind_gusts: bool) -> dict:
    out = dict(scenario)
    if disable_deck_heave:
        out['deck_z_amp'] = 0.0
        out['deck_z_freq'] = 0.0
        out['deck_z_phase'] = 0.0
        out['deck_z_peak_to_peak'] = 0.0
        out['deck_z_peak_vz'] = 0.0
        out['deck_z_period_s'] = None
    if disable_wind_gusts:
        out['wind_gust_amp_1'] = 0.0
        out['wind_gust_amp_2'] = 0.0
    return out


def _annotate_attempt_rows(rows: Iterable[dict], *, role: str, ref_name: str) -> List[dict]:
    out = []
    for row in rows:
        rr = dict(row)
        rr['search_role'] = str(role)
        rr['reference_name'] = str(ref_name)
        out.append(rr)
    return out


def _make_sweep_cfg(batch_cfg: UtopiaBatchConfig) -> FrequencySweepConfig:
    return FrequencySweepConfig(
        minima_csv='__unused__',
        output_dir=str(batch_cfg.output_dir),
        seed=int(batch_cfg.seed),
        preview_h_s=float(batch_cfg.preview_h_s),
        sim_time_max_s=float(batch_cfg.sim_time_max_s),
        num_workers=int(batch_cfg.num_workers),
        validate_generated_scenarios=False,
        touchdown_x_tol=float(batch_cfg.touchdown_x_tol),
        touchdown_z_tol=float(batch_cfg.touchdown_z_tol),
        touchdown_vx_rel_tol=float(batch_cfg.touchdown_vx_rel_tol),
        touchdown_vz_rel_min=float(batch_cfg.touchdown_vz_rel_min),
        touchdown_vz_rel_max=float(batch_cfg.touchdown_vz_rel_max),
    )


def _make_cap_ref_hw(weight: float, caps: MPCFinalSearchCaps) -> HardwarePoint:
    return HardwarePoint(
        tw_ratio=float(caps.tw_ratio_cap),
        T_dot_max=float(caps.T_dot_cap_weight_per_sec) * float(weight),
        delta_dot_max=math.radians(float(caps.delta_dot_cap_deg_s)),
        delta_max=math.radians(float(caps.delta_cap_deg)),
    )

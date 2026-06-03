from __future__ import annotations

import os

for _env_name in (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
):
    os.environ.setdefault(_env_name, '1')

import argparse
import csv
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config.vehicle_params import OCPParams, build_stabilized_bundle
from config.search_params import MPCFinalSearchCaps, make_hardware_overrides_from_point
from dynamics.environment import SeaEnvironment
from dynamics.planar_vbat import VbatModel
from run_mpc_replay import FrequencySweepConfig, MPCFrequencySearchContext, MPCSearchResult
from core.hardware import HardwarePoint
from ocp.path_pool_ocp import OCPPathPoolShooting25
from utils.adaptive_search import AdaptiveSearchConfig, METRIC_NAMES, get_metric, with_metric
from utils.batch_tools import evaluate_relative_metrics, touchdown_like_success, write_csv
from utils.scenario_generator import generate_scenarios


@dataclass(frozen=True)
class OCPRefineBatchConfig:
    main4_output_dir: str = 'outputs_hardware_search'
    output_dir: str = 'outputs_ocp_path_pool_refine'
    seed: int = 42
    num_scenarios: int = 20
    num_workers: int = 2

    preview_h_s: float = 2.0
    sim_time_max_s: float = 30.0

    touchdown_x_tol: float = 1.0
    touchdown_z_tol: float = 0.10
    touchdown_vx_rel_tol: float = 1.0
    touchdown_vz_rel_min: float = -2.0
    touchdown_vz_rel_max: float = 0.5

    disable_deck_heave: bool = False
    disable_wind_gusts: bool = False

    candidate_source: str = 'shortlist'
    candidate_tag: str = 'closest'

    # 0.0 = normalized usage penalty works over the whole interval.
    # 0.85 = earlier near-saturation-only behaviour.
    sat_penalty_start_ratio: float = 0.0
    sat_weight_T_excess: Optional[float] = None
    sat_weight_delta: Optional[float] = None
    sat_weight_T_dot: Optional[float] = None
    sat_weight_delta_dot: Optional[float] = None

    # OCP hardware lowering floor.
    # 'zero' lets OCP search below the MPC utopia values.
    # 'utopia' keeps the older behaviour and does not go below main4 utopia.
    # 'custom' uses the four floor_* command-line values below.
    ocp_floor_source: str = 'zero'
    floor_tw_ratio: float = 0.0
    floor_tdot_weight_per_sec: float = 0.0
    floor_delta_deg: float = 0.0
    floor_delta_dot_deg_s: float = 0.0

    # OCP local path refinement controls
    # The MPC rollout path is used as a soft centerline; the deck-relative landing
    # state is enforced at the terminal pool.
    ocp_terminal_pool_nodes: int = 1
    ocp_path_tube_x_m: float = 2.5
    ocp_path_tube_z_m: float = 3.0
    ocp_path_tube_weight: float = 20.0
    ocp_hard_stage_vz_corridor: bool = False
    ocp_deck_terminal_ref_nodes: int = 1
    ocp_terminal_x_tol_m: float = 1.0
    ocp_terminal_z_tol_m: float = 0.10
    ocp_terminal_vx_rel_tol: float = 1.0
    ocp_terminal_vz_rel_min: float = -2.0
    ocp_terminal_vz_rel_max: float = 0.5


@dataclass(frozen=True)
class OCPRefineSearchConfig:
    bisect_iters: int = 18
    rel_tol: float = 0.005
    refine_metrics: str = 'all'  # all or active
    max_permutations: int = 24


@dataclass
class OCPRefineResult:
    success: bool
    solver_success: bool
    stage: str
    hardware: HardwarePoint
    return_status: str
    iter_count: Optional[int]
    solve_time_ms: Optional[float]
    t_wall_total: Optional[float]
    max_T_usage: float
    max_delta_usage: float
    max_T_dot_usage: float
    max_delta_dot_usage: float
    final_rel_x: Optional[float]
    final_rel_z: Optional[float]
    final_rel_vx: Optional[float]
    final_rel_vz: Optional[float]
    X_opt: Optional[np.ndarray] = None
    U_opt: Optional[np.ndarray] = None


def _parse_int_list(text: Optional[str]) -> Optional[List[int]]:
    if text is None or str(text).strip() == '':
        return None
    out = []
    for part in str(text).split(','):
        part = part.strip()
        if part:
            out.append(int(part))
    return out or None


def _parse_float_list(text: Optional[str]) -> Optional[List[float]]:
    if text is None or str(text).strip() == '':
        return None
    out = []
    for part in str(text).split(','):
        part = part.strip()
        if part:
            out.append(float(part))
    return out or None


def _parse_str_list(text: Optional[str]) -> Optional[List[str]]:
    if text is None or str(text).strip() == '':
        return None
    out = []
    for part in str(text).split(','):
        part = part.strip()
        if part:
            out.append(part)
    return out or None


def _to_float(value, default=None):
    try:
        if value is None or value == '':
            return default
        return float(value)
    except Exception:
        return default


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


def _build_env(scenario: dict) -> SeaEnvironment:
    return SeaEnvironment(
        base_wind_x=scenario['base_wind_x'],
        wind_gust_amp_1=scenario['wind_gust_amp_1'],
        wind_gust_freq_1=scenario['wind_gust_freq_1'],
        wind_gust_phase_1=scenario['wind_gust_phase_1'],
        wind_gust_amp_2=scenario['wind_gust_amp_2'],
        wind_gust_freq_2=scenario['wind_gust_freq_2'],
        wind_gust_phase_2=scenario['wind_gust_phase_2'],
        deck_x0=scenario['deck_x0'],
        deck_vx=scenario['deck_vx'],
        deck_z0=scenario['deck_z0'],
        deck_z_amp=scenario['deck_z_amp'],
        deck_z_freq=scenario['deck_z_freq'],
        deck_z_phase=scenario['deck_z_phase'],
    )


def _make_sweep_cfg(batch_cfg: OCPRefineBatchConfig) -> FrequencySweepConfig:
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


def _row_hw(row: dict, prefix: str, *, weight: float) -> HardwarePoint:
    tw = _to_float(row.get(f'{prefix}_tw_ratio'))
    tdot = _to_float(row.get(f'{prefix}_T_dot_max'))
    if tdot is None:
        tdot_wps = _to_float(row.get(f'{prefix}_T_dot_weight_per_sec'))
        tdot = None if tdot_wps is None else float(tdot_wps) * float(weight)
    delta_deg = _to_float(row.get(f'{prefix}_delta_max_deg'))
    delta_dot_deg_s = _to_float(row.get(f'{prefix}_delta_dot_max_deg_s'))
    if tw is None or tdot is None or delta_deg is None or delta_dot_deg_s is None:
        raise ValueError(f'{prefix} hardware kolonları eksik veya okunamadı.')
    return HardwarePoint(
        tw_ratio=float(tw),
        T_dot_max=float(tdot),
        delta_max=math.radians(float(delta_deg)),
        delta_dot_max=math.radians(float(delta_dot_deg_s)),
    )




def _floor_hw_from_config(batch_cfg: OCPRefineBatchConfig, *, weight: float) -> HardwarePoint:
    return HardwarePoint(
        tw_ratio=float(batch_cfg.floor_tw_ratio),
        T_dot_max=float(batch_cfg.floor_tdot_weight_per_sec) * float(weight),
        delta_max=math.radians(float(batch_cfg.floor_delta_deg)),
        delta_dot_max=math.radians(float(batch_cfg.floor_delta_dot_deg_s)),
    )


def _metricwise_min_hw(a: HardwarePoint, b: HardwarePoint) -> HardwarePoint:
    return HardwarePoint(
        tw_ratio=min(float(a.tw_ratio), float(b.tw_ratio)),
        T_dot_max=min(float(a.T_dot_max), float(b.T_dot_max)),
        delta_max=min(float(a.delta_max), float(b.delta_max)),
        delta_dot_max=min(float(a.delta_dot_max), float(b.delta_dot_max)),
    )


def _select_ocp_floor_hw(
    *,
    row_utopia_hw: HardwarePoint,
    candidate_hw: HardwarePoint,
    batch_cfg: OCPRefineBatchConfig,
    weight: float,
) -> HardwarePoint:
    source = str(batch_cfg.ocp_floor_source).strip().lower()
    if source == 'utopia':
        floor = row_utopia_hw
    elif source in ('zero', 'custom'):
        floor = _floor_hw_from_config(batch_cfg, weight=weight)
    else:
        raise ValueError(f'Bilinmeyen ocp_floor_source: {batch_cfg.ocp_floor_source}')
    # Guard against accidental custom floors above the selected candidate.
    return _metricwise_min_hw(floor, candidate_hw)


def _delta_hw_row(refined: HardwarePoint, reference: HardwarePoint, prefix: str, weight: float) -> Dict[str, float]:
    return {
        f'{prefix}_tw_ratio': float(refined.tw_ratio) - float(reference.tw_ratio),
        f'{prefix}_T_dot_max': float(refined.T_dot_max) - float(reference.T_dot_max),
        f'{prefix}_T_dot_weight_per_sec': (float(refined.T_dot_max) - float(reference.T_dot_max)) / float(weight),
        f'{prefix}_delta_max_deg': math.degrees(float(refined.delta_max) - float(reference.delta_max)),
        f'{prefix}_delta_dot_max_deg_s': math.degrees(float(refined.delta_dot_max) - float(reference.delta_dot_max)),
    }

def _candidate_key(row: dict) -> str:
    existing = str(row.get('candidate_key') or '').strip()
    if existing:
        return existing
    sid = int(float(row['scenario_id']))
    freq = float(row['frequency_hz'])
    freq_str = f'{freq:g}'.replace('.', 'p')
    tag = str(row.get('shortlist_tag') or '').strip()
    if tag:
        return f's{sid}_f{freq_str}_{tag}'
    subset = str(row.get('subset') or 'subset').replace('+', '-')
    perm = str(row.get('best_permutation') or '').replace(' -> ', '-')
    return f's{sid}_f{freq_str}_{subset}_{perm}'


def _read_candidate_rows(main4_output_dir: Path, source: str) -> List[dict]:
    filename = 'main4_shortlist.csv' if str(source).strip().lower() == 'shortlist' else 'main4_subset_candidates.csv'
    path = main4_output_dir / filename
    if not path.exists():
        raise FileNotFoundError(f'Candidate CSV bulunamadı: {path}')
    with path.open('r', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'Candidate CSV boş: {path}')
    for row in rows:
        row['candidate_key'] = _candidate_key(row)
    return rows


def _filter_candidate_rows(
    rows: Sequence[dict],
    *,
    scenario_ids: Optional[Sequence[int]],
    freq_list: Optional[Sequence[float]],
    candidate_tag: str,
    candidate_keys: Optional[Sequence[str]],
) -> List[dict]:
    wanted_sids = None if scenario_ids is None else set(int(x) for x in scenario_ids)
    wanted_freqs = None if freq_list is None else [float(x) for x in freq_list]
    wanted_keys = None if candidate_keys is None else set(str(x) for x in candidate_keys)
    tag = str(candidate_tag).strip().lower() or 'closest'
    out = []
    for row in rows:
        sid = int(float(row['scenario_id']))
        freq = float(row['frequency_hz'])
        if wanted_keys is not None:
            if row['candidate_key'] in wanted_keys:
                out.append(row)
            continue
        if wanted_sids is not None and sid not in wanted_sids:
            continue
        if wanted_freqs is not None and not any(abs(freq - f) <= 1e-9 for f in wanted_freqs):
            continue
        if row.get('shortlist_tag'):
            if tag != 'all' and str(row.get('shortlist_tag')).strip().lower() != tag:
                continue
        out.append(row)
    return out


def _make_ocp_params(batch_cfg: OCPRefineBatchConfig) -> OCPParams:
    p = OCPParams()
    p.sat_penalty_start_ratio = float(batch_cfg.sat_penalty_start_ratio)
    if batch_cfg.sat_weight_T_excess is not None:
        p.W_sat_T_excess = float(batch_cfg.sat_weight_T_excess)
    if batch_cfg.sat_weight_delta is not None:
        p.W_sat_delta = float(batch_cfg.sat_weight_delta)
    if batch_cfg.sat_weight_T_dot is not None:
        p.W_sat_T_dot = float(batch_cfg.sat_weight_T_dot)
    if batch_cfg.sat_weight_delta_dot is not None:
        p.W_sat_delta_dot = float(batch_cfg.sat_weight_delta_dot)

    # Align OCP terminal hard pool with the MPC touchdown success window.
    p.terminal_x_tol_m = float(batch_cfg.ocp_terminal_x_tol_m)
    p.terminal_z_above_tol_m = float(batch_cfg.ocp_terminal_z_tol_m)
    p.terminal_z_below_tol_m = 0.0
    p.terminal_vx_rel_tol = float(batch_cfg.ocp_terminal_vx_rel_tol)
    p.terminal_vz_rel_min = float(batch_cfg.ocp_terminal_vz_rel_min)
    p.terminal_vz_rel_max = float(batch_cfg.ocp_terminal_vz_rel_max)
    return p


class CandidateOCPRefinementContext:
    def __init__(
        self,
        *,
        scenario: dict,
        freq_hz: float,
        candidate_key: str,
        candidate_hw: HardwarePoint,
        floor_hw: HardwarePoint,
        mpc_result: MPCSearchResult,
        batch_cfg: OCPRefineBatchConfig,
    ):
        if mpc_result.X_opt is None or mpc_result.U_opt is None:
            raise ValueError('MPC rollout X/U saklanmamış. store_solution=True olmalı.')
        if not bool(mpc_result.success):
            raise ValueError('OCP refine için MPC candidate rollout başarılı olmalı.')

        self.scenario = dict(scenario)
        self.freq_hz = float(freq_hz)
        self.candidate_key = str(candidate_key)
        self.candidate_hw = candidate_hw
        self.floor_hw = floor_hw
        self.batch_cfg = batch_cfg
        self.weight = 4.0 * 9.81
        self.dt = 1.0 / self.freq_hz

        self.X_mpc = np.asarray(mpc_result.X_opt, dtype=float)
        self.U_mpc = np.asarray(mpc_result.U_opt, dtype=float)
        self.N_total = int(self.U_mpc.shape[1])
        if self.X_mpc.shape[1] != self.N_total + 1:
            raise ValueError(f'MPC X/U shape uyumsuz: X={self.X_mpc.shape}, U={self.U_mpc.shape}')

        self.env = _build_env(self.scenario)
        env_full = self.env.sample_horizon(0.0, self.dt, self.N_total + 1)
        self.deck_x_full = np.asarray(env_full['ship_x'], dtype=float)
        self.deck_vx_full = np.asarray(env_full['ship_vx'], dtype=float)
        self.deck_vz_full = np.asarray(env_full['ship_vz'], dtype=float)
        self.deck_z_full = np.asarray(env_full['ship_z'], dtype=float)
        self.wind_x_stage = np.asarray(env_full['wind_x'][:-1], dtype=float)

        self.x0_ocp = self.X_mpc[:, 0].copy()
        # Purposeful bias: OCP refines the MPC family, so the MPC closed-loop path is the OCP centerline.
        # The last few nodes can be overwritten by the deck path so the terminal constraints are tied to
        # deck-relative landing state instead of the last MPC path point.
        self.x_ref_ocp = self.X_mpc[0, :].copy()
        self.z_ref_ocp = self.X_mpc[1, :].copy()
        tail_nodes = max(1, min(int(batch_cfg.ocp_deck_terminal_ref_nodes), self.N_total + 1))
        self.x_ref_ocp[-tail_nodes:] = self.deck_x_full[-tail_nodes:]
        self.z_ref_ocp[-tail_nodes:] = self.deck_z_full[-tail_nodes:]

        uav_p, _, _ = build_stabilized_bundle(
            base_wind=float(self.scenario['base_wind_x']),
            hardware_overrides=make_hardware_overrides_from_point(candidate_hw),
        )
        self.model = VbatModel(uav_p)
        self.ocp_p = _make_ocp_params(batch_cfg)
        self.ocp = OCPPathPoolShooting25(
            self.model,
            self.dt,
            self.N_total,
            self.ocp_p,
            terminal_pool_nodes=int(batch_cfg.ocp_terminal_pool_nodes),
            path_tube_x_m=float(batch_cfg.ocp_path_tube_x_m),
            path_tube_z_m=float(batch_cfg.ocp_path_tube_z_m),
            W_path_tube=float(batch_cfg.ocp_path_tube_weight),
            hard_stage_vz_corridor=bool(batch_cfg.ocp_hard_stage_vz_corridor),
        )
        self.cache: Dict[Tuple[float, float, float, float], OCPRefineResult] = {}
        self.attempt_log: List[dict] = []

    def _log_attempt(self, result: OCPRefineResult) -> None:
        row = {
            **_scenario_base_row(self.scenario),
            'candidate_key': self.candidate_key,
            'frequency_hz': float(self.freq_hz),
            'N_total': int(self.N_total),
            'dt': float(self.dt),
            'stage': str(result.stage),
            'success': bool(result.success),
            'solver_success': bool(result.solver_success),
            'return_status': str(result.return_status),
            'iter_count': result.iter_count,
            'solve_time_ms': result.solve_time_ms,
            't_wall_total_s': result.t_wall_total,
            'final_rel_x': result.final_rel_x,
            'final_rel_z': result.final_rel_z,
            'final_rel_vx': result.final_rel_vx,
            'final_rel_vz': result.final_rel_vz,
            'usage_T': result.max_T_usage,
            'usage_T_dot': result.max_T_dot_usage,
            'usage_delta': result.max_delta_usage,
            'usage_delta_dot': result.max_delta_dot_usage,
        }
        row.update(_hardware_to_row_dict(result.hardware, 'hardware', self.weight))
        self.attempt_log.append(row)

    def evaluate(self, hardware: HardwarePoint, stage: str, guess_X=None, guess_U=None) -> OCPRefineResult:
        key = hardware.key()
        if key in self.cache:
            return self.cache[key]

        if guess_X is None:
            guess_X = self.X_mpc
        if guess_U is None:
            guess_U = self.U_mpc

        X_opt, U_opt, solver_success, summary = self.ocp.solve(
            x0=self.x0_ocp,
            x_ref_full=self.x_ref_ocp,
            z_ref_full=self.z_ref_ocp,
            hardware=hardware,
            deck_x_full=self.deck_x_full,
            deck_vx_full=self.deck_vx_full,
            deck_vz_full=self.deck_vz_full,
            deck_z_full=self.deck_z_full,
            wind_x_stage=self.wind_x_stage,
            guess_X=guess_X,
            guess_U=guess_U,
        )

        t_final = self.N_total * self.dt
        final_metrics = evaluate_relative_metrics(X_opt[:, -1], self.env, t_final)
        strict_success = bool(solver_success) and touchdown_like_success(
            final_metrics,
            x_tol=self.batch_cfg.touchdown_x_tol,
            z_tol=self.batch_cfg.touchdown_z_tol,
            vx_tol=self.batch_cfg.touchdown_vx_rel_tol,
            vz_rel_min=self.batch_cfg.touchdown_vz_rel_min,
            vz_rel_max=self.batch_cfg.touchdown_vz_rel_max,
        )

        result = OCPRefineResult(
            success=bool(strict_success),
            solver_success=bool(solver_success),
            stage=str(stage),
            hardware=hardware,
            return_status=str(summary.get('return_status', 'UNKNOWN')),
            iter_count=summary.get('iter_count'),
            solve_time_ms=summary.get('solve_time_ms_manual'),
            t_wall_total=summary.get('t_wall_total'),
            max_T_usage=float(summary.get('max_T_usage', 0.0)),
            max_delta_usage=float(summary.get('max_delta_usage', 0.0)),
            max_T_dot_usage=float(summary.get('max_T_dot_usage', 0.0)),
            max_delta_dot_usage=float(summary.get('max_delta_dot_usage', 0.0)),
            final_rel_x=float(final_metrics['rel_x']),
            final_rel_z=float(final_metrics['rel_z']),
            final_rel_vx=float(final_metrics['rel_vx']),
            final_rel_vz=float(final_metrics['rel_vz']),
            X_opt=X_opt,
            U_opt=U_opt,
        )
        self.cache[key] = result
        self._log_attempt(result)
        return result


def _metric_order_from_candidate_subset(row: dict, refine_metrics: str) -> Tuple[str, ...]:
    mode = str(refine_metrics).strip().lower()
    if mode == 'active':
        parts = set(part.strip() for part in str(row.get('subset') or '').split('+') if part.strip())
        active = tuple(metric for metric in METRIC_NAMES if metric in parts)
        return active or tuple(METRIC_NAMES)
    return tuple(METRIC_NAMES)


def _interval_small(low: float, high: float, *, rel_tol: float) -> bool:
    return abs(float(high) - float(low)) <= max(1e-12, float(rel_tol) * max(abs(float(high)), abs(float(low)), 1.0))


def lower_metric_toward_floor(
    *,
    metric: str,
    base_res: OCPRefineResult,
    floor_hw: HardwarePoint,
    ctx: CandidateOCPRefinementContext,
    stage_prefix: str,
    search_cfg: OCPRefineSearchConfig,
) -> OCPRefineResult:
    current_hw = base_res.hardware
    current_value = float(get_metric(current_hw, metric))
    floor_value = float(get_metric(floor_hw, metric))
    if current_value <= floor_value + max(1e-12, search_cfg.rel_tol * max(abs(current_value), 1.0)):
        return base_res

    floor_candidate = with_metric(current_hw, metric, floor_value)
    floor_res = ctx.evaluate(
        floor_candidate,
        f'{stage_prefix}_{metric}_floor_probe',
        guess_X=base_res.X_opt,
        guess_U=base_res.U_opt,
    )
    if bool(floor_res.success):
        return floor_res

    low_value = float(floor_value)
    high_value = float(current_value)
    best_res = base_res

    for _ in range(int(search_cfg.bisect_iters)):
        if _interval_small(low_value, high_value, rel_tol=search_cfg.rel_tol):
            break
        mid_value = 0.5 * (low_value + high_value)
        if abs(mid_value - low_value) <= 1e-15 or abs(mid_value - high_value) <= 1e-15:
            break
        mid_hw = with_metric(current_hw, metric, mid_value)
        mid_res = ctx.evaluate(
            mid_hw,
            f'{stage_prefix}_{metric}_bisect',
            guess_X=best_res.X_opt,
            guess_U=best_res.U_opt,
        )
        if bool(mid_res.success):
            best_res = mid_res
            high_value = float(mid_value)
            current_hw = best_res.hardware
        else:
            low_value = float(mid_value)

    return best_res


def _slack_and_reduction(hw: HardwarePoint, candidate_hw: HardwarePoint, floor_hw: HardwarePoint):
    remaining: Dict[str, float] = {}
    reduction: Dict[str, float] = {}
    for metric in METRIC_NAMES:
        start = float(get_metric(candidate_hw, metric))
        floor = float(get_metric(floor_hw, metric))
        value = float(get_metric(hw, metric))
        denom = max(start - floor, 0.0)
        if denom <= 1e-12:
            remaining[metric] = 0.0
            reduction[metric] = 0.0
        else:
            rem = float(np.clip((value - floor) / denom, 0.0, 1.0))
            remaining[metric] = rem
            reduction[metric] = float(1.0 - rem)
    return remaining, reduction


def _parse_score_weights(text: Optional[str]) -> Dict[str, float]:
    weights = {m: 1.0 for m in METRIC_NAMES}
    if not text:
        return weights
    for part in str(text).split(','):
        if not part.strip():
            continue
        if '=' not in part:
            raise ValueError('score weights formatı: tw=1,tdot=1,delta=1,delta_dot=1')
        key, value = part.split('=', 1)
        key = key.strip()
        if key not in weights:
            raise ValueError(f'Bilinmeyen metric weight: {key}')
        weights[key] = float(value)
    return weights


def _score_refined(hw: HardwarePoint, candidate_hw: HardwarePoint, floor_hw: HardwarePoint, weights: Dict[str, float]) -> dict:
    remaining, reduction = _slack_and_reduction(hw, candidate_hw, floor_hw)
    remaining_l2_sq = float(sum(float(weights[m]) * remaining[m] ** 2 for m in METRIC_NAMES))
    remaining_l1 = float(sum(float(weights[m]) * remaining[m] for m in METRIC_NAMES))
    reduction_l1 = float(sum(float(weights[m]) * reduction[m] for m in METRIC_NAMES))
    reduction_l2_sq = float(sum(float(weights[m]) * reduction[m] ** 2 for m in METRIC_NAMES))
    return {
        'remaining_l2_sq': remaining_l2_sq,
        'remaining_l1': remaining_l1,
        'reduction_l1': reduction_l1,
        'reduction_l2_sq': reduction_l2_sq,
        **{f'remaining_{m}': remaining[m] for m in METRIC_NAMES},
        **{f'reduction_{m}': reduction[m] for m in METRIC_NAMES},
    }


def _best_perm_key(item: dict):
    score = item['score']
    return (
        -float(score['remaining_l2_sq']),
        float(score['reduction_l1']),
        float(score['reduction_l2_sq']),
    )


def _save_npz(path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, metric_order=np.array(['tw', 'tdot', 'delta', 'delta_dot']), **kwargs)




def _npz_scalar(data, name: str, default=None):
    try:
        if name not in data.files:
            return default
        value = data[name]
        if getattr(value, 'shape', ()) == ():
            return value.item()
        if value.size == 0:
            return default
        return value.reshape(-1)[0].item()
    except Exception:
        return default


def _safe_optional_float(value, default=None):
    try:
        if value is None:
            return default
        val = float(value)
        if not np.isfinite(val):
            return default
        return val
    except Exception:
        return default


def _candidate_saved_path(row: dict, batch_cfg: OCPRefineBatchConfig) -> Optional[Path]:
    raw = row.get('mpc_path_npz') or row.get('candidate_mpc_path_npz')
    if raw is None or str(raw).strip() == '' or str(raw).strip().lower() in {'none', 'nan', 'not_saved'}:
        return None
    path = Path(str(raw).strip())
    if not path.is_absolute():
        path = Path(batch_cfg.main4_output_dir) / path
    if not path.exists():
        return None
    return path


def _load_saved_mpc_candidate_rollout(
    *,
    row: dict,
    scenario: dict,
    freq_hz: float,
    candidate_hw: HardwarePoint,
    candidate_key: str,
    batch_cfg: OCPRefineBatchConfig,
) -> Optional[Tuple[MPCSearchResult, List[dict]]]:
    """Load exact main4 MPC X/U path if available for this candidate.

    This baseline variant preserves retry policy and hardware search decisions.
    It only avoids regenerating an already-found closed-loop MPC trajectory at
    the beginning of main5.
    """
    path = _candidate_saved_path(row, batch_cfg)
    if path is None:
        return None

    with np.load(path, allow_pickle=False) as data:
        X = np.asarray(data['X_mpc'], dtype=float)
        U = np.asarray(data['U_mpc'], dtype=float)
        if X.ndim != 2 or U.ndim != 2 or X.shape[1] != U.shape[1] + 1:
            raise ValueError(f'Saved MPC X/U shape uyumsuz: {path} | X={X.shape}, U={U.shape}')

        saved_hw = np.asarray(data['candidate_hw'], dtype=float).flatten() if 'candidate_hw' in data.files else None
        if saved_hw is not None and saved_hw.size >= 4:
            expected = np.array([
                candidate_hw.tw_ratio,
                candidate_hw.T_dot_max,
                candidate_hw.delta_max,
                candidate_hw.delta_dot_max,
            ], dtype=float)
            if not np.allclose(saved_hw[:4], expected, rtol=1e-8, atol=1e-8):
                raise ValueError(f'Saved MPC path hardware mismatch: {path} | saved={saved_hw[:4]} expected={expected}')

        retry_count = int(_npz_scalar(data, 'retry_count', row.get('retry_count', 0)) or 0)
        solver_success_count = int(_npz_scalar(data, 'solver_success_count', U.shape[1]) or U.shape[1])
        return_status = str(_npz_scalar(data, 'return_status', 'loaded_from_main4_saved_mpc_path'))
        failure_mode = str(_npz_scalar(data, 'failure_mode', 'none'))

        result = MPCSearchResult(
            success=True,
            stage='path_pool_selected_candidate_mpc_rollout_loaded_from_main4',
            hardware=candidate_hw,
            return_status=f'loaded_from_main4_saved_mpc_path:{return_status}',
            failure_mode=failure_mode if failure_mode not in {'', 'None'} else 'none',
            solver_success_count=solver_success_count,
            retry_count=retry_count,
            avg_solve_ms=_safe_optional_float(_npz_scalar(data, 'avg_solve_ms', row.get('avg_solve_ms'))),
            max_solve_ms=_safe_optional_float(_npz_scalar(data, 'max_solve_ms', row.get('max_solve_ms'))),
            max_T_usage=_safe_optional_float(_npz_scalar(data, 'usage_T', row.get('usage_T')), 0.0),
            max_delta_usage=_safe_optional_float(_npz_scalar(data, 'usage_delta', row.get('usage_delta')), 0.0),
            max_T_dot_usage=_safe_optional_float(_npz_scalar(data, 'usage_T_dot', row.get('usage_T_dot')), 0.0),
            max_delta_dot_usage=_safe_optional_float(_npz_scalar(data, 'usage_delta_dot', row.get('usage_delta_dot')), 0.0),
            final_rel_x=_safe_optional_float(_npz_scalar(data, 'final_rel_x', row.get('final_rel_x'))),
            final_rel_z=_safe_optional_float(_npz_scalar(data, 'final_rel_z', row.get('final_rel_z'))),
            final_rel_vx=_safe_optional_float(_npz_scalar(data, 'final_rel_vx', row.get('final_rel_vx'))),
            final_rel_vz=_safe_optional_float(_npz_scalar(data, 'final_rel_vz', row.get('final_rel_vz'))),
            touch_time_s=_safe_optional_float(_npz_scalar(data, 'touch_time_s', row.get('touch_time_s'))),
            steps_simulated=int(U.shape[1]),
            X_opt=X,
            U_opt=U,
        )

    attempt_row = {
        **_scenario_base_row(scenario),
        'candidate_key': str(candidate_key),
        'attempt_kind': 'mpc_candidate_rollout_loaded_from_main4_path',
        'frequency_hz': float(freq_hz),
        'stage': result.stage,
        'success': True,
        'return_status': result.return_status,
        'failure_mode': result.failure_mode,
        'solver_success_count': int(result.solver_success_count),
        'retry_count': int(result.retry_count),
        'avg_solve_ms': result.avg_solve_ms,
        'max_solve_ms': result.max_solve_ms,
        'touch_time_s': result.touch_time_s,
        'steps_simulated': result.steps_simulated,
        'final_rel_x': result.final_rel_x,
        'final_rel_z': result.final_rel_z,
        'final_rel_vx': result.final_rel_vx,
        'final_rel_vz': result.final_rel_vz,
        'usage_T': result.max_T_usage,
        'usage_T_dot': result.max_T_dot_usage,
        'usage_delta': result.max_delta_usage,
        'usage_delta_dot': result.max_delta_dot_usage,
        'mpc_path_npz': str(path),
    }
    attempt_row.update(_hardware_to_row_dict(candidate_hw, 'hardware', 4.0 * 9.81))
    return result, [attempt_row]

def _run_mpc_candidate_rollout(
    *,
    scenario: dict,
    freq_hz: float,
    candidate_hw: HardwarePoint,
    batch_cfg: OCPRefineBatchConfig,
    caps: MPCFinalSearchCaps,
) -> MPCSearchResult:
    ctx = MPCFrequencySearchContext(
        scenario=scenario,
        ref_hw=candidate_hw,
        freq_hz=float(freq_hz),
        sweep_cfg=_make_sweep_cfg(batch_cfg),
        search_cfg=AdaptiveSearchConfig(),
        caps=caps,
        store_solution=True,
    )
    return ctx.evaluate(candidate_hw, 'path_pool_selected_candidate_mpc_rollout', None, None)


def _run_single_candidate_worker(
    row: dict,
    scenario: dict,
    batch_cfg: OCPRefineBatchConfig,
    search_cfg: OCPRefineSearchConfig,
    caps: MPCFinalSearchCaps,
    score_weights: Dict[str, float],
):
    scenario = _apply_debug_overrides(
        scenario,
        disable_deck_heave=batch_cfg.disable_deck_heave,
        disable_wind_gusts=batch_cfg.disable_wind_gusts,
    )
    weight = 4.0 * 9.81
    candidate_key = str(row['candidate_key'])
    freq_hz = float(row['frequency_hz'])
    candidate_hw = _row_hw(row, 'candidate', weight=weight)
    utopia_hw = _row_hw(row, 'utopia', weight=weight)
    floor_hw = _select_ocp_floor_hw(
        row_utopia_hw=utopia_hw,
        candidate_hw=candidate_hw,
        batch_cfg=batch_cfg,
        weight=weight,
    )
    cap_hw = _row_hw(row, 'cap', weight=weight)

    summary_base = {
        **_scenario_base_row(scenario),
        'candidate_key': candidate_key,
        'frequency_hz': float(freq_hz),
        'shortlist_tag': row.get('shortlist_tag'),
        'subset': row.get('subset'),
        'cardinality': row.get('cardinality'),
        'best_permutation': row.get('best_permutation'),
        'sat_penalty_start_ratio': float(batch_cfg.sat_penalty_start_ratio),
        'ocp_floor_source': str(batch_cfg.ocp_floor_source),
        **{f'score_weight_{m}': float(score_weights[m]) for m in METRIC_NAMES},
    }
    summary_base.update(_hardware_to_row_dict(utopia_hw, 'utopia', weight))
    summary_base.update(_hardware_to_row_dict(floor_hw, 'search_floor', weight))
    # Backward-readable alias: in old main5 this field was the utopia floor.
    # Now it is the actual lower-search floor.
    summary_base.update(_hardware_to_row_dict(floor_hw, 'floor_utopia', weight))
    summary_base.update(_hardware_to_row_dict(candidate_hw, 'candidate', weight))
    summary_base.update(_hardware_to_row_dict(cap_hw, 'cap', weight))

    loaded = _load_saved_mpc_candidate_rollout(
        row=row,
        scenario=scenario,
        freq_hz=freq_hz,
        candidate_hw=candidate_hw,
        candidate_key=candidate_key,
        batch_cfg=batch_cfg,
    )
    if loaded is not None:
        mpc_res, mpc_attempt_rows = loaded
        mpc_rollout_source = 'main4_saved_mpc_path'
    else:
        mpc_res = _run_mpc_candidate_rollout(
            scenario=scenario,
            freq_hz=freq_hz,
            candidate_hw=candidate_hw,
            batch_cfg=batch_cfg,
            caps=caps,
        )
        mpc_attempt_rows = []
        mpc_rollout_source = 'fresh_main5_replay'

    output_dir = Path(batch_cfg.output_dir)
    npz_dir = output_dir / 'npz'
    mpc_npz = npz_dir / f'{candidate_key}_mpc_rollout.npz'
    if mpc_res.X_opt is not None and mpc_res.U_opt is not None:
        _save_npz(
            mpc_npz,
            X_mpc=mpc_res.X_opt,
            U_mpc=mpc_res.U_opt,
            candidate_hw=np.array([candidate_hw.tw_ratio, candidate_hw.T_dot_max, candidate_hw.delta_max, candidate_hw.delta_dot_max]),
            utopia_hw=np.array([utopia_hw.tw_ratio, utopia_hw.T_dot_max, utopia_hw.delta_max, utopia_hw.delta_dot_max]),
            floor_hw=np.array([floor_hw.tw_ratio, floor_hw.T_dot_max, floor_hw.delta_max, floor_hw.delta_dot_max]),
            frequency_hz=np.array([freq_hz]),
            scenario_id=np.array([int(scenario['id'])]),
        )

    if not bool(mpc_res.success):
        out = dict(summary_base)
        out.update({
            'mpc_rollout_success': False,
            'mpc_rollout_source': mpc_rollout_source,
            'mpc_failure_mode': mpc_res.failure_mode,
            'mpc_return_status': mpc_res.return_status,
            'ocp_start_success': False,
            'ocp_refine_success': False,
            'failed_stage': 'mpc_candidate_rollout',
            'mpc_npz': str(mpc_npz),
            'best_ocp_npz': None,
        })
        return {'summary_rows': [out], 'permutation_rows': [], 'attempt_rows': mpc_attempt_rows}

    ocp_ctx = CandidateOCPRefinementContext(
        scenario=scenario,
        freq_hz=freq_hz,
        candidate_key=candidate_key,
        candidate_hw=candidate_hw,
        floor_hw=floor_hw,
        mpc_result=mpc_res,
        batch_cfg=batch_cfg,
    )

    start_res = ocp_ctx.evaluate(candidate_hw, 'ocp_candidate_start', guess_X=mpc_res.X_opt, guess_U=mpc_res.U_opt)
    if not bool(start_res.success):
        out = dict(summary_base)
        out.update({
            'mpc_rollout_success': True,
            'mpc_rollout_source': mpc_rollout_source,
            'mpc_failure_mode': mpc_res.failure_mode,
            'mpc_return_status': mpc_res.return_status,
            'ocp_start_success': False,
            'ocp_start_return_status': start_res.return_status,
            'ocp_refine_success': False,
            'failed_stage': 'ocp_candidate_start',
            'mpc_npz': str(mpc_npz),
            'best_ocp_npz': None,
        })
        return {'summary_rows': [out], 'permutation_rows': [], 'attempt_rows': mpc_attempt_rows + ocp_ctx.attempt_log}

    metrics = _metric_order_from_candidate_subset(row, search_cfg.refine_metrics)
    orders = list(permutations(metrics))
    if len(orders) > int(search_cfg.max_permutations):
        orders = orders[: int(search_cfg.max_permutations)]

    perm_outputs = []
    for order in orders:
        current_res = start_res
        for metric in order:
            current_res = lower_metric_toward_floor(
                metric=str(metric),
                base_res=current_res,
                floor_hw=floor_hw,
                ctx=ocp_ctx,
                stage_prefix=f'perm_{"-".join(order)}',
                search_cfg=search_cfg,
            )
        score = _score_refined(current_res.hardware, candidate_hw, floor_hw, score_weights)
        perm_outputs.append({'order': tuple(order), 'result': current_res, 'score': score})

    best = max(perm_outputs, key=_best_perm_key) if perm_outputs else {'order': metrics, 'result': start_res, 'score': _score_refined(start_res.hardware, candidate_hw, floor_hw, score_weights)}
    best_res = best['result']
    best_score = best['score']
    best_order = best['order']

    best_ocp_npz = npz_dir / f'{candidate_key}_ocp_best.npz'
    if best_res.X_opt is not None and best_res.U_opt is not None:
        _save_npz(
            best_ocp_npz,
            X_mpc=mpc_res.X_opt,
            U_mpc=mpc_res.U_opt,
            X_ocp=best_res.X_opt,
            U_ocp=best_res.U_opt,
            candidate_hw=np.array([candidate_hw.tw_ratio, candidate_hw.T_dot_max, candidate_hw.delta_max, candidate_hw.delta_dot_max]),
            utopia_hw=np.array([utopia_hw.tw_ratio, utopia_hw.T_dot_max, utopia_hw.delta_max, utopia_hw.delta_dot_max]),
            floor_hw=np.array([floor_hw.tw_ratio, floor_hw.T_dot_max, floor_hw.delta_max, floor_hw.delta_dot_max]),
            refined_hw=np.array([best_res.hardware.tw_ratio, best_res.hardware.T_dot_max, best_res.hardware.delta_max, best_res.hardware.delta_dot_max]),
            frequency_hz=np.array([freq_hz]),
            scenario_id=np.array([int(scenario['id'])]),
        )

    permutation_rows = []
    for item in perm_outputs:
        res = item['result']
        prow = {
            **_scenario_base_row(scenario),
            'candidate_key': candidate_key,
            'frequency_hz': float(freq_hz),
            'order': ' -> '.join(item['order']),
            'success': bool(res.success),
            'return_status': res.return_status,
            **item['score'],
        }
        prow.update(_hardware_to_row_dict(res.hardware, 'refined', weight))
        prow.update(_delta_hw_row(res.hardware, utopia_hw, 'refined_minus_utopia', weight))
        permutation_rows.append(prow)

    out = dict(summary_base)
    out.update({
        'mpc_rollout_success': True,
            'mpc_rollout_source': mpc_rollout_source,
        'mpc_failure_mode': mpc_res.failure_mode,
        'mpc_return_status': mpc_res.return_status,
        'mpc_touch_time_s': mpc_res.touch_time_s,
        'mpc_steps_simulated': mpc_res.steps_simulated,
        'ocp_start_success': True,
        'ocp_start_return_status': start_res.return_status,
        'ocp_refine_success': bool(best_res.success),
        'failed_stage': None if bool(best_res.success) else 'ocp_refine',
        'best_order': ' -> '.join(best_order),
        **best_score,
        'final_rel_x': best_res.final_rel_x,
        'final_rel_z': best_res.final_rel_z,
        'final_rel_vx': best_res.final_rel_vx,
        'final_rel_vz': best_res.final_rel_vz,
        'usage_T': best_res.max_T_usage,
        'usage_T_dot': best_res.max_T_dot_usage,
        'usage_delta': best_res.max_delta_usage,
        'usage_delta_dot': best_res.max_delta_dot_usage,
        'mpc_npz': str(mpc_npz),
        'best_ocp_npz': str(best_ocp_npz),
    })
    out.update(_hardware_to_row_dict(best_res.hardware, 'refined', weight))
    out.update(_delta_hw_row(best_res.hardware, utopia_hw, 'refined_minus_utopia', weight))
    return {'summary_rows': [out], 'permutation_rows': permutation_rows, 'attempt_rows': mpc_attempt_rows + ocp_ctx.attempt_log}


def parse_args():
    parser = argparse.ArgumentParser(description='MPC candidate -> OCP path-pool local refinement')
    parser.add_argument('--main4-output-dir', type=str, default='outputs_hardware_search')
    parser.add_argument('--output-dir', type=str, default='outputs_ocp_path_pool_refine')
    parser.add_argument('--num-scenarios', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scenario-ids', type=str, default='')
    parser.add_argument('--freq-list', type=str, default='20')
    parser.add_argument('--candidate-source', type=str, choices=['shortlist', 'candidates'], default='shortlist')
    parser.add_argument('--candidate-tag', type=str, default='closest', help='shortlist için: closest, sparse, balanced veya all')
    parser.add_argument('--candidate-keys', type=str, default='', help='örn: s1_f20_closest,s3_f20_sparse')
    parser.add_argument('--preview-h-s', type=float, default=2.0)
    parser.add_argument('--sim-time-max-s', type=float, default=30.0)
    parser.add_argument('--cap-tw-ratio', type=float, default=2.0)
    parser.add_argument('--cap-tdot-weight-per-sec', type=float, default=2.0)
    parser.add_argument('--cap-delta-deg', type=float, default=60.0)
    parser.add_argument('--cap-delta-dot-deg-s', type=float, default=360.0)
    parser.add_argument('--ocp-floor-source', type=str, choices=['zero', 'utopia', 'custom'], default='zero',
                        help='zero/custom: OCP utopia altına inebilir. utopia: eski davranış, utopia altına inmez.')
    parser.add_argument('--floor-tw-ratio', type=float, default=0.0)
    parser.add_argument('--floor-tdot-weight-per-sec', type=float, default=0.0)
    parser.add_argument('--floor-delta-deg', type=float, default=0.0)
    parser.add_argument('--floor-delta-dot-deg-s', type=float, default=0.0)
    parser.add_argument('--ocp-terminal-pool-nodes', type=int, default=1)
    parser.add_argument('--ocp-deck-terminal-ref-nodes', type=int, default=1)
    parser.add_argument('--ocp-path-tube-x-m', type=float, default=2.5)
    parser.add_argument('--ocp-path-tube-z-m', type=float, default=3.0)
    parser.add_argument('--ocp-path-tube-weight', type=float, default=20.0)
    parser.add_argument('--ocp-hard-stage-vz-corridor', action='store_true')
    parser.add_argument('--ocp-terminal-x-tol-m', type=float, default=1.0)
    parser.add_argument('--ocp-terminal-z-tol-m', type=float, default=0.10)
    parser.add_argument('--ocp-terminal-vx-rel-tol', type=float, default=1.0)
    parser.add_argument('--ocp-terminal-vz-rel-min', type=float, default=-2.0)
    parser.add_argument('--ocp-terminal-vz-rel-max', type=float, default=0.5)
    parser.add_argument('--disable-deck-heave', action='store_true')
    parser.add_argument('--disable-wind-gusts', action='store_true')
    parser.add_argument('--sat-penalty-start-ratio', type=float, default=0.0)
    parser.add_argument('--sat-weight-T-excess', type=float, default=None)
    parser.add_argument('--sat-weight-delta', type=float, default=None)
    parser.add_argument('--sat-weight-T-dot', type=float, default=None)
    parser.add_argument('--sat-weight-delta-dot', type=float, default=None)
    parser.add_argument('--refine-metrics', choices=['all', 'active'], default='all')
    parser.add_argument('--ocp-bisect-iters', type=int, default=18)
    parser.add_argument('--ocp-rel-tol-pct', type=float, default=0.5)
    parser.add_argument('--max-permutations', type=int, default=24)
    parser.add_argument('--score-weights', type=str, default='tw=1,tdot=1,delta=1,delta_dot=1')
    parser.add_argument('--list-candidates-only', action='store_true')
    return parser.parse_args()


def run_batch():
    args = parse_args()
    scenario_ids = _parse_int_list(args.scenario_ids)
    freq_list = _parse_float_list(args.freq_list)
    candidate_keys = _parse_str_list(args.candidate_keys)
    score_weights = _parse_score_weights(args.score_weights)

    batch_cfg = OCPRefineBatchConfig(
        main4_output_dir=str(args.main4_output_dir),
        output_dir=str(args.output_dir),
        seed=int(args.seed),
        num_scenarios=int(args.num_scenarios),
        num_workers=int(args.num_workers),
        preview_h_s=float(args.preview_h_s),
        sim_time_max_s=float(args.sim_time_max_s),
        disable_deck_heave=bool(args.disable_deck_heave),
        disable_wind_gusts=bool(args.disable_wind_gusts),
        candidate_source=str(args.candidate_source),
        candidate_tag=str(args.candidate_tag),
        sat_penalty_start_ratio=float(args.sat_penalty_start_ratio),
        sat_weight_T_excess=args.sat_weight_T_excess,
        sat_weight_delta=args.sat_weight_delta,
        sat_weight_T_dot=args.sat_weight_T_dot,
        sat_weight_delta_dot=args.sat_weight_delta_dot,
        ocp_floor_source=str(args.ocp_floor_source),
        floor_tw_ratio=float(args.floor_tw_ratio),
        floor_tdot_weight_per_sec=float(args.floor_tdot_weight_per_sec),
        floor_delta_deg=float(args.floor_delta_deg),
        floor_delta_dot_deg_s=float(args.floor_delta_dot_deg_s),
        ocp_terminal_pool_nodes=int(args.ocp_terminal_pool_nodes),
        ocp_path_tube_x_m=float(args.ocp_path_tube_x_m),
        ocp_path_tube_z_m=float(args.ocp_path_tube_z_m),
        ocp_path_tube_weight=float(args.ocp_path_tube_weight),
        ocp_hard_stage_vz_corridor=bool(args.ocp_hard_stage_vz_corridor),
        ocp_deck_terminal_ref_nodes=int(args.ocp_deck_terminal_ref_nodes),
        ocp_terminal_x_tol_m=float(args.ocp_terminal_x_tol_m),
        ocp_terminal_z_tol_m=float(args.ocp_terminal_z_tol_m),
        ocp_terminal_vx_rel_tol=float(args.ocp_terminal_vx_rel_tol),
        ocp_terminal_vz_rel_min=float(args.ocp_terminal_vz_rel_min),
        ocp_terminal_vz_rel_max=float(args.ocp_terminal_vz_rel_max),
    )
    search_cfg = OCPRefineSearchConfig(
        bisect_iters=int(args.ocp_bisect_iters),
        rel_tol=float(args.ocp_rel_tol_pct) / 100.0,
        refine_metrics=str(args.refine_metrics),
        max_permutations=int(args.max_permutations),
    )
    caps = MPCFinalSearchCaps(
        tw_ratio_cap=float(args.cap_tw_ratio),
        T_dot_cap_weight_per_sec=float(args.cap_tdot_weight_per_sec),
        delta_cap_deg=float(args.cap_delta_deg),
        delta_dot_cap_deg_s=float(args.cap_delta_dot_deg_s),
    )

    all_rows = _read_candidate_rows(Path(batch_cfg.main4_output_dir), batch_cfg.candidate_source)
    rows = _filter_candidate_rows(
        all_rows,
        scenario_ids=scenario_ids,
        freq_list=freq_list,
        candidate_tag=batch_cfg.candidate_tag,
        candidate_keys=candidate_keys,
    )
    if not rows:
        raise ValueError('Seçilen filtrelerle candidate bulunamadı.')

    if bool(args.list_candidates_only):
        output_dir = Path(batch_cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        list_csv = output_dir / 'main5_candidate_key_list.csv'
        write_csv(list_csv, rows)
        print('Seçilebilen candidate key listesi:')
        for row in rows:
            print(f"  {row['candidate_key']} | scenario={row['scenario_id']} | freq={row['frequency_hz']} | tag={row.get('shortlist_tag')} | subset={row.get('subset')}")
        print(f'CSV: {list_csv.resolve()}')
        return

    scenarios = generate_scenarios(num_scenarios=batch_cfg.num_scenarios, seed=batch_cfg.seed)
    scenario_by_id = {int(sc['id']): sc for sc in scenarios}

    print('path-pool OCP refine başlıyor...')
    print(f'main4_output_dir       : {batch_cfg.main4_output_dir}')
    print(f'output_dir             : {batch_cfg.output_dir}')
    print(f'candidate_source       : {batch_cfg.candidate_source}')
    print(f'candidate_tag          : {batch_cfg.candidate_tag}')
    print(f'candidate_count        : {len(rows)}')
    print(f'freq_filter            : {freq_list}')
    print(f'scenario_filter        : {scenario_ids}')
    print(f'sat_penalty_start_ratio: {batch_cfg.sat_penalty_start_ratio:.3f}')
    print(f'ocp_floor_source       : {batch_cfg.ocp_floor_source}')
    print(f'floor_tw_ratio         : {batch_cfg.floor_tw_ratio:.6g}')
    print(f'floor_tdot_wps         : {batch_cfg.floor_tdot_weight_per_sec:.6g}')
    print(f'floor_delta_deg        : {batch_cfg.floor_delta_deg:.6g}')
    print(f'floor_delta_dot_deg_s  : {batch_cfg.floor_delta_dot_deg_s:.6g}')
    print(f'ocp terminal pool     : nodes={batch_cfg.ocp_terminal_pool_nodes}, z_tol={batch_cfg.ocp_terminal_z_tol_m}, vx_tol={batch_cfg.ocp_terminal_vx_rel_tol}')
    print(f'ocp path tube         : x={batch_cfg.ocp_path_tube_x_m}, z={batch_cfg.ocp_path_tube_z_m}, W={batch_cfg.ocp_path_tube_weight}')
    print(f'refine_metrics         : {search_cfg.refine_metrics}')
    print(f'ocp_rel_tol_pct        : {100.0 * search_cfg.rel_tol:.3f}')
    print(f'score_weights          : {score_weights}')

    summary_rows: List[dict] = []
    permutation_rows: List[dict] = []
    attempt_rows: List[dict] = []

    mp_ctx = mp.get_context('spawn')
    max_workers = min(int(batch_cfg.num_workers), len(rows))
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as executor:
        futures = []
        for row in rows:
            sid = int(float(row['scenario_id']))
            if sid not in scenario_by_id:
                raise ValueError(f'scenario_id={sid} generate_scenarios çıktısında yok. num_scenarios/seed aynı mı?')
            futures.append(executor.submit(_run_single_candidate_worker, row, scenario_by_id[sid], batch_cfg, search_cfg, caps, score_weights))
        for future in as_completed(futures):
            item = future.result()
            summary_rows.extend(item['summary_rows'])
            permutation_rows.extend(item['permutation_rows'])
            attempt_rows.extend(item['attempt_rows'])
            row0 = item['summary_rows'][0]
            print(
                f"candidate {row0['candidate_key']} | "
                f"MPC={bool(row0.get('mpc_rollout_success'))} | "
                f"OCP={bool(row0.get('ocp_refine_success'))} | "
                f"best={row0.get('best_order')}"
            )

    summary_rows.sort(key=lambda r: (int(r['scenario_id']), float(r['frequency_hz']), str(r['candidate_key'])))
    permutation_rows.sort(key=lambda r: (int(r['scenario_id']), float(r['frequency_hz']), str(r['candidate_key']), str(r['order'])))
    attempt_rows.sort(key=lambda r: (int(r['scenario_id']), float(r['frequency_hz']), str(r['candidate_key']), str(r['stage'])))

    output_dir = Path(batch_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / 'main5_ocp_refine_summary.csv'
    permutation_csv = output_dir / 'main5_ocp_refine_permutations.csv'
    attempts_csv = output_dir / 'main5_ocp_attempts.csv'
    selected_csv = output_dir / 'main5_selected_candidates.csv'

    write_csv(summary_csv, summary_rows)
    write_csv(permutation_csv, permutation_rows)
    write_csv(attempts_csv, attempt_rows)
    write_csv(selected_csv, rows)

    ok = sum(1 for r in summary_rows if bool(r.get('ocp_refine_success', False)))
    print('\n' + '=' * 84)
    print('MAIN5 OCP REFINE ÖZETİ')
    print('=' * 84)
    print(f'ocp refine successes : {ok}/{len(summary_rows)}')
    print(f'summary csv          : {summary_csv.resolve()}')
    print(f'permutation csv      : {permutation_csv.resolve()}')
    print(f'attempts csv         : {attempts_csv.resolve()}')
    print(f'npz dir              : {(output_dir / "npz").resolve()}')
    print('=' * 84 + '\n')


if __name__ == '__main__':
    mp.freeze_support()
    run_batch()

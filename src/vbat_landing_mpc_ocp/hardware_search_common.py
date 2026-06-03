from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

for _env_name in (
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
):
    os.environ.setdefault(_env_name, '1')

from config.search_params import MPCFinalSearchCaps
from run_mpc_replay import FrequencySweepConfig, MPCFrequencySearchContext
from utils.hardware_search_helpers import (
    UtopiaBatchConfig,
    _annotate_attempt_rows,
    _apply_debug_overrides,
    _hardware_to_row_dict,
    _make_cap_ref_hw,
    _make_sweep_cfg,
    _scenario_base_row,
)
from core.hardware import HardwarePoint
from utils.adaptive_search import AdaptiveSearchConfig, METRIC_NAMES, get_metric
from utils.contraction_search import (
    Search20Config,
    SingleMetricRecord20,
    FeasiblePointRecord20,
    FinalPermutationRecord20,
    build_utopia_from_records20,
    choose_shortlist20,
    compute_utopia_pass20,
    dominance_prune,
    enumerate_metric_subsets,
    find_first_feasible_global_inflation20,
    normalize_between,
    permutation_name,
    run_final_family20,
    subset_name,
)
from utils.scenario_generator import generate_scenarios


# -----------------------------------------------------------------------------
# Small file helpers
# -----------------------------------------------------------------------------


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False, dir=str(path.parent)) as f:
        f.write(text)
        tmp = Path(f.name)
    tmp.replace(path)


def write_json(path: Path, data: dict) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv_rows(path: Path) -> List[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def write_csv_union(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(v, default=None):
    try:
        if v is None or v == '':
            return default
        out = float(v)
        if not math.isfinite(out):
            return default
        return out
    except Exception:
        return default


def _safe_int(v, default=None):
    try:
        if v is None or v == '':
            return default
        return int(float(v))
    except Exception:
        return default


# -----------------------------------------------------------------------------
# Hardware serialization
# -----------------------------------------------------------------------------


def hw_to_dict(hw: Optional[HardwarePoint]) -> Optional[dict]:
    if hw is None:
        return None
    return {
        'tw_ratio': float(hw.tw_ratio),
        'T_dot_max': float(hw.T_dot_max),
        'delta_max': float(hw.delta_max),
        'delta_dot_max': float(hw.delta_dot_max),
    }


def hw_from_dict(d: Optional[dict]) -> Optional[HardwarePoint]:
    if not d:
        return None
    return HardwarePoint(
        tw_ratio=float(d['tw_ratio']),
        T_dot_max=float(d['T_dot_max']),
        delta_max=float(d['delta_max']),
        delta_dot_max=float(d['delta_dot_max']),
    )


def metric_record_to_dict(rec: SingleMetricRecord20) -> dict:
    return {
        'pass_name': rec.pass_name,
        'metric': rec.metric,
        'success': bool(rec.success),
        'low_value': float(rec.low_value),
        'high_value': float(rec.high_value),
        'hardware': hw_to_dict(rec.hardware),
        'iterations': int(rec.iterations),
    }


def metric_record_from_dict(d: dict) -> SingleMetricRecord20:
    return SingleMetricRecord20(
        pass_name=str(d.get('pass_name', '')),
        metric=str(d['metric']),
        success=bool(d.get('success', False)),
        low_value=float(d.get('low_value', 0.0)),
        high_value=float(d.get('high_value', 0.0)),
        hardware=hw_from_dict(d.get('hardware')),
        iterations=int(d.get('iterations', 0)),
    )


def fp_to_dict(fp: Optional[FeasiblePointRecord20]) -> Optional[dict]:
    if fp is None:
        return None
    return {
        'pass_name': fp.pass_name,
        'success': bool(fp.success),
        'alpha': fp.alpha,
        'hardware': hw_to_dict(fp.hardware),
        'attempts': int(fp.attempts),
        'return_status': None if fp.eval_result is None else getattr(fp.eval_result, 'return_status', None),
        'failure_mode': None if fp.eval_result is None else getattr(fp.eval_result, 'failure_mode', None),
    }


def fp_from_dict(d: Optional[dict]) -> Optional[FeasiblePointRecord20]:
    if not d:
        return None
    return FeasiblePointRecord20(
        pass_name=str(d.get('pass_name', '')),
        success=bool(d.get('success', False)),
        alpha=None if d.get('alpha') is None else float(d['alpha']),
        hardware=hw_from_dict(d.get('hardware')),
        eval_result=None,
        attempts=int(d.get('attempts', 0)),
    )


def records_from_state(state: dict, pass_key: str) -> Dict[str, SingleMetricRecord20]:
    raw = state.get(pass_key, {}) or {}
    out: Dict[str, SingleMetricRecord20] = {}
    for m, d in raw.items():
        if d:
            out[m] = metric_record_from_dict(d)
    return out


# -----------------------------------------------------------------------------
# Args/config construction
# -----------------------------------------------------------------------------


def parse_int_list(text: Optional[str]) -> List[int]:
    if text is None or str(text).strip() == '':
        return []
    return [int(p.strip()) for p in str(text).split(',') if p.strip()]


def parse_float_list(text: Optional[str]) -> List[float]:
    if text is None or str(text).strip() == '':
        return []
    return [float(p.strip()) for p in str(text).split(',') if p.strip()]


def make_batch_cfg(args) -> UtopiaBatchConfig:
    return UtopiaBatchConfig(
        output_dir=str(args.output_dir),
        seed=int(args.seed),
        num_scenarios=int(args.num_scenarios),
        num_workers=int(getattr(args, 'num_workers', 1)),
        anchor_freq_hz=float(args.anchor_freq_hz),
        preview_h_s=float(args.preview_h_s),
        sim_time_max_s=float(args.sim_time_max_s),
        disable_deck_heave=bool(args.disable_deck_heave),
        disable_wind_gusts=bool(args.disable_wind_gusts),
        sanity_check_cap=not bool(args.skip_cap_sanity),
    )


def make_search_cfg(args) -> Search20Config:
    return Search20Config(
        single_metric_rel_tol=float(args.single_rel_tol_pct) / 100.0,
        single_metric_bisect_iters=int(args.single_bisect_iters),
        contraction_alpha_step=float(args.contraction_alpha_step),
        final_grid_step=float(args.final_grid_step),
        reclaim_grid_step=float(args.reclaim_grid_step),
        min_metric_rel_change=float(args.min_metric_rel_change),
        active_threshold=float(args.active_threshold),
        subset_max_cardinality=int(args.subset_max_cardinality),
        final_entry_mode=str(args.final_entry_mode),
    )


def make_caps(args) -> MPCFinalSearchCaps:
    return MPCFinalSearchCaps(
        tw_ratio_cap=float(args.cap_tw_ratio),
        T_dot_cap_weight_per_sec=float(args.cap_tdot_weight_per_sec),
        delta_cap_deg=float(args.cap_delta_deg),
        delta_dot_cap_deg_s=float(args.cap_delta_dot_deg_s),
    )


def make_ctx(*, scenario: dict, freq_hz: float, batch_cfg: UtopiaBatchConfig, caps: MPCFinalSearchCaps, ref_hw: HardwarePoint, store_solution: bool = False):
    return MPCFrequencySearchContext(
        scenario=scenario,
        ref_hw=ref_hw,
        freq_hz=float(freq_hz),
        sweep_cfg=_make_sweep_cfg(batch_cfg),
        search_cfg=AdaptiveSearchConfig(),
        caps=caps,
        store_solution=bool(store_solution),
    )


def scenario_dir(output_dir: Path, scenario_id: int) -> Path:
    return output_dir / 'checkpoints' / f's{int(scenario_id):03d}'


def state_path(output_dir: Path, scenario_id: int) -> Path:
    return scenario_dir(output_dir, scenario_id) / 'search_state.json'


def load_state(output_dir: Path, scenario_id: int) -> dict:
    path = state_path(output_dir, scenario_id)
    if path.exists():
        return read_json(path)
    return {}


def save_state(output_dir: Path, scenario_id: int, state: dict) -> None:
    write_json(state_path(output_dir, scenario_id), state)


def make_scenario_and_common(args, scenario_id: int):
    scenarios = generate_scenarios(num_scenarios=int(args.num_scenarios), seed=int(args.seed))
    by_id = {int(sc['id']): sc for sc in scenarios}
    if int(scenario_id) not in by_id:
        raise ValueError(f'scenario_id={scenario_id} yok. --num-scenarios değerini kontrol et.')
    scenario = _apply_debug_overrides(
        by_id[int(scenario_id)],
        disable_deck_heave=bool(args.disable_deck_heave),
        disable_wind_gusts=bool(args.disable_wind_gusts),
    )
    batch_cfg = make_batch_cfg(args)
    search_cfg = make_search_cfg(args)
    caps = make_caps(args)
    weight = 4.0 * 9.81
    cap_ref_hw = _make_cap_ref_hw(weight, caps)
    ctx0 = make_ctx(scenario=scenario, freq_hz=float(args.anchor_freq_hz), batch_cfg=batch_cfg, caps=caps, ref_hw=cap_ref_hw)
    cap_hw = ctx0.cap_hw
    return scenario, batch_cfg, search_cfg, caps, weight, cap_ref_hw, cap_hw


# -----------------------------------------------------------------------------
# Row builders
# -----------------------------------------------------------------------------


def single_metric_rows(*, scenario: dict, weight: float, pass_name: str, support_name: str, upper_name: str, records: Dict[str, SingleMetricRecord20], support_hw: HardwarePoint, upper_hw: HardwarePoint, utopia_hw: Optional[HardwarePoint]) -> List[dict]:
    rows = []
    base = _scenario_base_row(scenario)
    for m in METRIC_NAMES:
        rec = records.get(m)
        row = dict(base)
        row['pass_name'] = pass_name
        row['metric'] = m
        row['support_name'] = support_name
        row['upper_name'] = upper_name
        row['success'] = False if rec is None else bool(rec.success)
        row['low_value'] = None if rec is None else rec.low_value
        row['high_value'] = None if rec is None else rec.high_value
        row['iterations'] = None if rec is None else rec.iterations
        row.update(_hardware_to_row_dict(support_hw, 'support', weight))
        row.update(_hardware_to_row_dict(upper_hw, 'upper', weight))
        row.update(_hardware_to_row_dict(None if rec is None else rec.hardware, 'single', weight))
        row.update(_hardware_to_row_dict(utopia_hw, 'utopia_vector', weight))
        rows.append(row)
    return rows


def feasible_point_row(*, scenario: dict, weight: float, pass_name: str, low_name: str, target_name: str, low_hw: HardwarePoint, target_hw: HardwarePoint, fp: FeasiblePointRecord20) -> dict:
    row = _scenario_base_row(scenario)
    row['pass_name'] = pass_name
    row['low_name'] = low_name
    row['target_name'] = target_name
    row['success'] = bool(fp.success)
    row['alpha'] = fp.alpha
    row['attempts'] = int(fp.attempts)
    row['return_status'] = None if fp.eval_result is None else getattr(fp.eval_result, 'return_status', None)
    row['failure_mode'] = None if fp.eval_result is None else getattr(fp.eval_result, 'failure_mode', None)
    row.update(_hardware_to_row_dict(low_hw, 'low', weight))
    row.update(_hardware_to_row_dict(target_hw, 'target', weight))
    row.update(_hardware_to_row_dict(fp.hardware, 'feasible', weight))
    return row


def utopia_summary_row(*, scenario: dict, weight: float, cap_hw: HardwarePoint, state: dict) -> dict:
    row = _scenario_base_row(scenario)
    row.update(_hardware_to_row_dict(cap_hw, 'C0', weight))
    for name in ('hu1', 'hf1', 'hu2', 'hf2', 'hu3', 'hf3', 'hu4'):
        hw = hw_from_dict(state.get(f'{name}_hw'))
        row.update(_hardware_to_row_dict(hw, name, weight))
    for name in ('hf1', 'hf2', 'hf3'):
        fp = fp_from_dict(state.get(f'{name}_record'))
        row[f'{name}_success'] = None if fp is None else bool(fp.success)
        row[f'{name}_alpha'] = None if fp is None else fp.alpha
        row[f'{name}_attempts'] = None if fp is None else fp.attempts
    row['final_box_ready'] = hw_from_dict(state.get('hf3_hw')) is not None and hw_from_dict(state.get('hu4_hw')) is not None
    row['method'] = 'recursive_C0_contraction_plus_dual_direction_final_search'
    return row


def final_screen_rows(*, scenario: dict, freq_hz: float, raw_rows: List[dict]) -> List[dict]:
    base = _scenario_base_row(scenario)
    out = []
    for item in raw_rows:
        row = dict(base)
        row['frequency_hz'] = float(freq_hz)
        row.update(item)
        out.append(row)
    return out




def _short_hash(obj, *, length: int = 16) -> str:
    text = json.dumps(obj, sort_keys=True, default=str, separators=(',', ':'))
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:int(length)]


def _float_or_nan(value) -> float:
    try:
        if value is None:
            return float('nan')
        return float(value)
    except Exception:
        return float('nan')


def save_candidate_mpc_path_npz(
    *,
    output_dir: Path,
    sdir: Path,
    scenario: dict,
    freq_hz: float,
    direction: str,
    subset: Sequence[str],
    rec: FinalPermutationRecord20,
    candidate_index: int,
    weight: float,
) -> Optional[str]:
    """Persist the exact MPC X/U trajectory found by main4 for an accepted final candidate.

    This baseline variant does not change the search oracle or retry policy.  It only
    saves the MPC trajectory that was actually produced in main4 so main5 can warm-start
    OCP from that path instead of regenerating the closed-loop MPC rollout.
    """
    del weight
    result = getattr(rec, 'eval_result', None)
    X = getattr(result, 'X_opt', None)
    U = getattr(result, 'U_opt', None)
    if X is None or U is None:
        return None
    X = np.asarray(X, dtype=float)
    U = np.asarray(U, dtype=float)
    if X.ndim != 2 or U.ndim != 2 or X.shape[1] != U.shape[1] + 1:
        return None
    subset_txt = subset_name(subset)
    tag = _short_hash({
        'scenario_id': int(scenario['id']),
        'frequency_hz': float(freq_hz),
        'direction': str(direction),
        'subset': subset_txt,
        'candidate_index': int(candidate_index),
        'entry_param': float(rec.entry_param),
        'permutation': permutation_name(rec.permutation),
        'hardware': rec.hardware.key(digits=12),
    })
    path = sdir / 'candidate_mpc_paths' / f's{int(scenario["id"]):03d}_f{float(freq_hz):g}_{tag}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X_mpc=X,
        U_mpc=U,
        candidate_hw=np.array([
            float(rec.hardware.tw_ratio),
            float(rec.hardware.T_dot_max),
            float(rec.hardware.delta_max),
            float(rec.hardware.delta_dot_max),
        ], dtype=float),
        scenario_id=np.array([int(scenario['id'])], dtype=int),
        frequency_hz=np.array([float(freq_hz)], dtype=float),
        stage=np.array(str(getattr(result, 'stage', 'final_search'))),
        return_status=np.array(str(getattr(result, 'return_status', 'UNKNOWN'))),
        failure_mode=np.array(str(getattr(result, 'failure_mode', 'none'))),
        solver_success_count=np.array([int(getattr(result, 'solver_success_count', U.shape[1]) or U.shape[1])], dtype=int),
        retry_count=np.array([int(getattr(result, 'retry_count', 0) or 0)], dtype=int),
        avg_solve_ms=np.array([_float_or_nan(getattr(result, 'avg_solve_ms', None))], dtype=float),
        max_solve_ms=np.array([_float_or_nan(getattr(result, 'max_solve_ms', None))], dtype=float),
        touch_time_s=np.array([_float_or_nan(getattr(result, 'touch_time_s', None))], dtype=float),
        final_rel_x=np.array([_float_or_nan(getattr(result, 'final_rel_x', None))], dtype=float),
        final_rel_z=np.array([_float_or_nan(getattr(result, 'final_rel_z', None))], dtype=float),
        final_rel_vx=np.array([_float_or_nan(getattr(result, 'final_rel_vx', None))], dtype=float),
        final_rel_vz=np.array([_float_or_nan(getattr(result, 'final_rel_vz', None))], dtype=float),
        usage_T=np.array([_float_or_nan(getattr(result, 'max_T_usage', None))], dtype=float),
        usage_T_dot=np.array([_float_or_nan(getattr(result, 'max_T_dot_usage', None))], dtype=float),
        usage_delta=np.array([_float_or_nan(getattr(result, 'max_delta_usage', None))], dtype=float),
        usage_delta_dot=np.array([_float_or_nan(getattr(result, 'max_delta_dot_usage', None))], dtype=float),
    )
    return path.relative_to(output_dir).as_posix()

def final_perm_row(*, scenario: dict, freq_hz: float, weight: float, low_hw: HardwarePoint, high_hw: HardwarePoint, rec: FinalPermutationRecord20) -> dict:
    row = _scenario_base_row(scenario)
    row['frequency_hz'] = float(freq_hz)
    row['direction'] = rec.direction
    row['subset'] = subset_name(rec.subset)
    row['cardinality'] = int(len(rec.subset))
    row['entry_param'] = float(rec.entry_param)
    row['permutation'] = permutation_name(rec.permutation)
    row['best_permutation'] = permutation_name(rec.permutation)
    row['family_permutation'] = permutation_name(getattr(rec, 'family_permutation', ()) or ())
    row['outside_permutation'] = permutation_name(getattr(rec, 'outside_permutation', ()) or ())
    row['cleanup_mode'] = getattr(rec, 'cleanup_mode', 'family_only')
    row['reclaim_score'] = float(rec.reclaim_score)
    row['l1_sum'] = float(rec.l1_sum)
    row['l2_sq'] = float(rec.l2_sq)
    row['linf'] = float(rec.linf)
    row['active_count'] = int(rec.active_count)
    row.update({f'slack_{m}': float(rec.slack_by_metric[m]) for m in METRIC_NAMES})
    row['return_status'] = getattr(rec.eval_result, 'return_status', None)
    row['failure_mode'] = getattr(rec.eval_result, 'failure_mode', None)
    row['solver_success_count'] = getattr(rec.eval_result, 'solver_success_count', None)
    row['retry_count'] = getattr(rec.eval_result, 'retry_count', None)
    row['avg_solve_ms'] = getattr(rec.eval_result, 'avg_solve_ms', None)
    row['max_solve_ms'] = getattr(rec.eval_result, 'max_solve_ms', None)
    row['touch_time_s'] = getattr(rec.eval_result, 'touch_time_s', None)
    row['steps_simulated'] = getattr(rec.eval_result, 'steps_simulated', None)
    row['final_rel_x'] = getattr(rec.eval_result, 'final_rel_x', None)
    row['final_rel_z'] = getattr(rec.eval_result, 'final_rel_z', None)
    row['final_rel_vx'] = getattr(rec.eval_result, 'final_rel_vx', None)
    row['final_rel_vz'] = getattr(rec.eval_result, 'final_rel_vz', None)
    row['usage_T'] = getattr(rec.eval_result, 'max_T_usage', None)
    row['usage_T_dot'] = getattr(rec.eval_result, 'max_T_dot_usage', None)
    row['usage_delta'] = getattr(rec.eval_result, 'max_delta_usage', None)
    row['usage_delta_dot'] = getattr(rec.eval_result, 'max_delta_dot_usage', None)
    row.update(_hardware_to_row_dict(low_hw, 'utopia', weight))
    row.update(_hardware_to_row_dict(high_hw, 'cap', weight))
    row.update(_hardware_to_row_dict(rec.hardware, 'candidate', weight))
    return row


def shortlist_rows_from_candidate_rows(rows: List[dict]) -> List[dict]:
    if not rows:
        return []
    by_key: Dict[Tuple[int, float], List[dict]] = {}
    for r in rows:
        by_key.setdefault((int(float(r['scenario_id'])), float(r['frequency_hz'])), []).append(r)
    out: List[dict] = []
    for _, group in sorted(by_key.items()):
        def f(row, name, default=1e18):
            return _safe_float(row.get(name), default)
        def i(row, name, default=999):
            return _safe_int(row.get(name), default)
        choices = {
            'closest': min(group, key=lambda r: (f(r, 'l2_sq'), f(r, 'l1_sum'), f(r, 'linf'), i(r, 'active_count'))),
            'low_total': min(group, key=lambda r: (f(r, 'l1_sum'), f(r, 'l2_sq'), f(r, 'linf'), i(r, 'active_count'))),
            'balanced': min(group, key=lambda r: (f(r, 'linf'), f(r, 'l2_sq'), f(r, 'l1_sum'), i(r, 'active_count'))),
            'sparse': min(group, key=lambda r: (i(r, 'active_count'), f(r, 'l2_sq'), f(r, 'l1_sum'), f(r, 'linf'))),
            'best_reclaim': max(group, key=lambda r: (f(r, 'reclaim_score', -1e18), -f(r, 'l2_sq'), -f(r, 'l1_sum'))),
        }
        bu = [r for r in group if str(r.get('direction')) == 'bottom_up']
        td = [r for r in group if str(r.get('direction')) == 'top_down']
        if bu:
            choices['best_bottom_up'] = min(bu, key=lambda r: (f(r, 'l2_sq'), f(r, 'l1_sum'), f(r, 'linf'), i(r, 'active_count')))
        if td:
            choices['best_top_down'] = min(td, key=lambda r: (f(r, 'l2_sq'), f(r, 'l1_sum'), f(r, 'linf'), i(r, 'active_count')))
        for tag, src in choices.items():
            rr = dict(src)
            rr['shortlist_tag'] = tag
            rr['candidate_key'] = f"s{int(float(rr['scenario_id']))}_f{int(float(rr['frequency_hz']))}_{tag}"
            out.append(rr)
    return out


# -----------------------------------------------------------------------------
# Rebuild outputs
# -----------------------------------------------------------------------------


def rebuild_global_outputs(output_dir: Path) -> None:
    ckpt_root = output_dir / 'checkpoints'
    if not ckpt_root.exists():
        return
    utopia_rows: List[dict] = []
    single_rows: List[dict] = []
    feasible_rows: List[dict] = []
    screen_rows: List[dict] = []
    perm_rows: List[dict] = []
    cand_rows: List[dict] = []
    attempts: List[dict] = []
    for sdir in sorted(p for p in ckpt_root.glob('s*') if p.is_dir()):
        utopia_rows.extend(read_csv_rows(sdir / 'utopia_summary.csv'))
        single_rows.extend(read_csv_rows(sdir / 'single_metric_records.csv'))
        feasible_rows.extend(read_csv_rows(sdir / 'feasible_contraction_points.csv'))
        screen_rows.extend(read_csv_rows(sdir / 'final_subset_screen.csv'))
        perm_rows.extend(read_csv_rows(sdir / 'final_subset_permutations.csv'))
        cand_rows.extend(read_csv_rows(sdir / 'final_subset_candidates.csv'))
        for afile in sorted(sdir.glob('attempts_*.csv')):
            attempts.extend(read_csv_rows(afile))
    shortlist = shortlist_rows_from_candidate_rows(cand_rows)

    def key_scen(row):
        return int(float(row.get('scenario_id', 0)))
    def key_final(row):
        return (int(float(row.get('scenario_id', 0))), float(row.get('frequency_hz', 0) or 0), str(row.get('direction', '')), str(row.get('subset', '')), float(row.get('entry_param', row.get('param', 0)) or 0), str(row.get('permutation', '')))

    utopia_rows.sort(key=key_scen)
    single_rows.sort(key=lambda r: (key_scen(r), str(r.get('pass_name', '')), str(r.get('metric', ''))))
    feasible_rows.sort(key=lambda r: (key_scen(r), str(r.get('pass_name', ''))))
    screen_rows.sort(key=key_final)
    perm_rows.sort(key=key_final)
    cand_rows.sort(key=key_final)
    shortlist.sort(key=lambda r: (key_scen(r), float(r.get('frequency_hz', 0) or 0), str(r.get('shortlist_tag', ''))))
    attempts.sort(key=lambda r: (key_scen(r), float(r.get('frequency_hz', 0) or 0), str(r.get('stage', ''))))

    write_csv_union(output_dir / 'main4_utopia_summary.csv', utopia_rows)
    write_csv_union(output_dir / 'main4_single_metric_records.csv', single_rows)
    write_csv_union(output_dir / 'main4_feasible_contraction_points.csv', feasible_rows)
    write_csv_union(output_dir / 'main4_subset_screen.csv', screen_rows)
    write_csv_union(output_dir / 'main4_subset_permutations.csv', perm_rows)
    write_csv_union(output_dir / 'main4_subset_candidates.csv', cand_rows)
    write_csv_union(output_dir / 'main4_shortlist.csv', shortlist)
    write_csv_union(output_dir / 'main4_attempts.csv', attempts)


# -----------------------------------------------------------------------------
# Stage logic
# -----------------------------------------------------------------------------


def run_utopia_pass(args, scenario_id: int, *, pass_index: int) -> None:
    output_dir = Path(args.output_dir)
    sdir = scenario_dir(output_dir, scenario_id)
    sdir.mkdir(parents=True, exist_ok=True)
    scenario, batch_cfg, search_cfg, caps, weight, cap_ref_hw, C0 = make_scenario_and_common(args, scenario_id)
    state = load_state(output_dir, scenario_id)
    state.setdefault('scenario', scenario)
    state['C0_hw'] = hw_to_dict(C0)
    pass_name = f'hu{pass_index}'

    if pass_index == 1:
        support_hw = C0
        metric_upper_hw = C0
        support_name = 'C0'
        upper_name = 'C0'
    elif pass_index == 2:
        support_hw = hw_from_dict(state.get('hf1_hw'))
        # hu2 is searched under the feasible support vector hf1. C0 is still
        # used later as the broad all-metric inflation target for f2, but it
        # must not be used as the single-metric upper point here. Otherwise the
        # searched metric can jump to C0 while the other three stay at hf1, and
        # the MPC can fail because of aggressive/high-authority behavior even
        # though hf1 itself is feasible.
        metric_upper_hw = support_hw
        support_name = 'hf1'
        upper_name = 'hf1'
    elif pass_index == 3:
        support_hw = hw_from_dict(state.get('hf2_hw'))
        # Same rule: hf2 is the feasible support for hu3; C0 remains only the
        # broad all-metric inflation target for f3.
        metric_upper_hw = support_hw
        support_name = 'hf2'
        upper_name = 'hf2'
    elif pass_index == 4:
        support_hw = hw_from_dict(state.get('hf3_hw'))
        metric_upper_hw = support_hw
        support_name = 'hf3'
        upper_name = 'hf3'
    else:
        raise ValueError(pass_index)
    if support_hw is None or metric_upper_hw is None:
        print(f's{scenario_id}: {pass_name} için gerekli support/upper hardware yok; stage atlandı.')
        return

    records_key = f'{pass_name}_records'
    if state.get(f'{pass_name}_hw') is not None and not bool(args.overwrite):
        print(f's{scenario_id}: {pass_name} zaten var, geçiliyor.')
        rebuild_global_outputs(output_dir)
        return

    ctx = make_ctx(scenario=scenario, freq_hz=float(args.anchor_freq_hz), batch_cfg=batch_cfg, caps=caps, ref_hw=metric_upper_hw)
    state.setdefault(records_key, {})
    for metric in METRIC_NAMES:
        existing_raw = state.get(records_key, {}).get(metric)
        if existing_raw and not bool(args.overwrite):
            try:
                existing_rec = metric_record_from_dict(existing_raw)
            except Exception:
                existing_rec = None
            if existing_rec is not None and existing_rec.success and existing_rec.hardware is not None:
                print(f's{scenario_id}: {pass_name} {metric} zaten başarılı, geçiliyor.')
                continue
            print(f's{scenario_id}: {pass_name} {metric} için eski kayıt başarısız/eksik; yeniden deneniyor.')
        ctx.attempt_log.clear()
        recs, _ = compute_utopia_pass20(
            pass_name=pass_name,
            support_hw=support_hw,
            metric_upper_hw=metric_upper_hw,
            evaluate_fn=ctx.evaluate,
            stage_prefix=f'{pass_name}_single_metric',
            search_cfg=search_cfg,
        ) if False else ({}, None)
        # Run one metric at a time so checkpointing can keep partial pass results.
        from utils.contraction_search import find_single_metric_minimum_with_support
        rec = find_single_metric_minimum_with_support(
            pass_name=pass_name,
            metric=metric,
            support_hw=support_hw,
            metric_upper_hw=metric_upper_hw,
            evaluate_fn=ctx.evaluate,
            stage_prefix=f'{pass_name}_single_metric',
            search_cfg=search_cfg,
        )
        state[records_key][metric] = metric_record_to_dict(rec)
        write_csv_union(sdir / f'attempts_{pass_name}_{metric}.csv', _annotate_attempt_rows(ctx.attempt_log, role=pass_name, ref_name=f'{support_name}_support_{upper_name}_upper'))
        save_state(output_dir, scenario_id, state)
        print(f's{scenario_id}: {pass_name} {metric} kaydedildi | success={rec.success}')
        if not rec.success:
            break

    records = records_from_state(state, records_key)
    hu_hw = build_utopia_from_records20(records)
    if hu_hw is not None:
        state[f'{pass_name}_hw'] = hw_to_dict(hu_hw)
        state[f'{pass_name}_complete'] = True
        save_state(output_dir, scenario_id, state)
    else:
        # Do not leave a false completed pass behind when only partial or failed
        # metric records exist.
        state.pop(f'{pass_name}_hw', None)
        state[f'{pass_name}_complete'] = False
        save_state(output_dir, scenario_id, state)
    rows = single_metric_rows(
        scenario=scenario,
        weight=weight,
        pass_name=pass_name,
        support_name=support_name,
        upper_name=upper_name,
        records=records,
        support_hw=support_hw,
        upper_hw=metric_upper_hw,
        utopia_hw=hu_hw,
    )
    write_csv_union(sdir / 'single_metric_records.csv', _merge_pass_metric_rows(read_csv_rows(sdir / 'single_metric_records.csv'), rows, scenario_id, pass_name))
    write_csv_union(sdir / 'utopia_summary.csv', [utopia_summary_row(scenario=scenario, weight=weight, cap_hw=C0, state=state)])
    rebuild_global_outputs(output_dir)


def _merge_pass_metric_rows(existing: List[dict], new_rows: List[dict], scenario_id: int, pass_name: str) -> List[dict]:
    def same(r):
        return int(float(r.get('scenario_id', -999))) == int(scenario_id) and str(r.get('pass_name', '')) == str(pass_name)
    return [r for r in existing if not same(r)] + list(new_rows)


def _merge_pass_rows(existing: List[dict], new_row: dict, scenario_id: int, pass_name: str) -> List[dict]:
    def same(r):
        return int(float(r.get('scenario_id', -999))) == int(scenario_id) and str(r.get('pass_name', '')) == str(pass_name)
    return [r for r in existing if not same(r)] + [new_row]


def run_feasible_pass(args, scenario_id: int, *, pass_index: int) -> None:
    output_dir = Path(args.output_dir)
    sdir = scenario_dir(output_dir, scenario_id)
    sdir.mkdir(parents=True, exist_ok=True)
    scenario, batch_cfg, search_cfg, caps, weight, cap_ref_hw, C0 = make_scenario_and_common(args, scenario_id)
    state = load_state(output_dir, scenario_id)
    fp_name = f'hf{pass_index}'
    hu_name = f'hu{pass_index}'
    if state.get(f'{fp_name}_hw') is not None and not bool(args.overwrite):
        print(f's{scenario_id}: {fp_name} zaten var, geçiliyor.')
        rebuild_global_outputs(output_dir)
        return
    low_hw = hw_from_dict(state.get(f'{hu_name}_hw'))
    if low_hw is None:
        print(f's{scenario_id}: {fp_name} atlandı; önce başarılı {hu_name} gerekir.')
        return
    target_hw = C0  # current rule: hf1/hf2/hf3 are all inflated toward original C0.
    ctx = make_ctx(scenario=scenario, freq_hz=float(args.anchor_freq_hz), batch_cfg=batch_cfg, caps=caps, ref_hw=target_hw)
    ctx.attempt_log.clear()
    fp = find_first_feasible_global_inflation20(
        pass_name=fp_name,
        low_hw=low_hw,
        target_hw=target_hw,
        evaluate_fn=ctx.evaluate,
        stage_prefix=f'{fp_name}_C0_global',
        search_cfg=search_cfg,
    )
    state[f'{fp_name}_record'] = fp_to_dict(fp)
    if fp.success and fp.hardware is not None:
        state[f'{fp_name}_hw'] = hw_to_dict(fp.hardware)
    save_state(output_dir, scenario_id, state)
    write_csv_union(sdir / f'attempts_{fp_name}.csv', _annotate_attempt_rows(ctx.attempt_log, role=fp_name, ref_name=f'{hu_name}_to_C0'))
    row = feasible_point_row(scenario=scenario, weight=weight, pass_name=fp_name, low_name=hu_name, target_name='C0', low_hw=low_hw, target_hw=target_hw, fp=fp)
    write_csv_union(sdir / 'feasible_contraction_points.csv', _merge_pass_rows(read_csv_rows(sdir / 'feasible_contraction_points.csv'), row, scenario_id, fp_name))
    write_csv_union(sdir / 'utopia_summary.csv', [utopia_summary_row(scenario=scenario, weight=weight, cap_hw=C0, state=state)])
    rebuild_global_outputs(output_dir)
    print(f's{scenario_id}: {fp_name} kaydedildi | success={fp.success} | alpha={fp.alpha}')


def run_final_search_for_scenario(args, scenario_id: int) -> None:
    output_dir = Path(args.output_dir)
    sdir = scenario_dir(output_dir, scenario_id)
    sdir.mkdir(parents=True, exist_ok=True)
    scenario, batch_cfg, search_cfg, caps, weight, cap_ref_hw, C0 = make_scenario_and_common(args, scenario_id)
    state = load_state(output_dir, scenario_id)
    low_hw = hw_from_dict(state.get('hu4_hw'))
    high_hw = hw_from_dict(state.get('hf3_hw'))
    if low_hw is None or high_hw is None:
        print(f's{scenario_id}: final-search atlandı; final search için hu4 ve hf3 gerekir.')
        return
    freqs = parse_float_list(args.freq_list) or [float(args.anchor_freq_hz)]
    subsets = parse_subset_names(args.subset_names, max_cardinality=int(args.subset_max_cardinality))
    directions = parse_directions(args.directions)
    done = set(state.get('final_search_done_keys', []))

    screen_all_existing = read_csv_rows(sdir / 'final_subset_screen.csv')
    perm_all_existing = read_csv_rows(sdir / 'final_subset_permutations.csv')
    cand_all_existing = read_csv_rows(sdir / 'final_subset_candidates.csv')

    for freq in freqs:
        ctx = make_ctx(scenario=scenario, freq_hz=float(freq), batch_cfg=batch_cfg, caps=caps, ref_hw=high_hw, store_solution=True)
        for direction in directions:
            for subset in subsets:
                ss = subset_name(subset)
                key = f'f{float(freq):g}:{direction}:{ss}'
                if key in done and not bool(args.overwrite):
                    print(f's{scenario_id}: final {key} zaten var, geçiliyor.')
                    continue
                ctx.attempt_log.clear()
                screen_raw, perm_records, best_records = run_final_family20(
                    direction=direction,
                    subset=subset,
                    low_hw=low_hw,
                    high_hw=high_hw,
                    evaluate_fn=ctx.evaluate,
                    stage_prefix=f'final20_{float(freq):g}',
                    search_cfg=search_cfg,
                )
                screen_rows = final_screen_rows(scenario=scenario, freq_hz=float(freq), raw_rows=screen_raw)
                perm_rows = [final_perm_row(scenario=scenario, freq_hz=float(freq), weight=weight, low_hw=low_hw, high_hw=high_hw, rec=r) for r in perm_records]
                cand_rows = []
                for idx, r in enumerate(best_records):
                    row = final_perm_row(
                        scenario=scenario,
                        freq_hz=float(freq),
                        weight=weight,
                        low_hw=low_hw,
                        high_hw=high_hw,
                        rec=r,
                    )
                    mpc_path = save_candidate_mpc_path_npz(
                        output_dir=output_dir,
                        sdir=sdir,
                        scenario=scenario,
                        freq_hz=float(freq),
                        direction=direction,
                        subset=subset,
                        rec=r,
                        candidate_index=idx,
                        weight=weight,
                    )
                    row['mpc_path_npz'] = mpc_path
                    row['mpc_path_source'] = 'main4_final_search_path' if mpc_path else 'not_saved'
                    row['mpc_path_saved'] = bool(mpc_path)
                    cand_rows.append(row)
                screen_all_existing = _merge_final_rows(screen_all_existing, screen_rows, scenario_id, freq, direction, ss, by_entry=False)
                perm_all_existing = _merge_final_rows(perm_all_existing, perm_rows, scenario_id, freq, direction, ss, by_entry=True)
                cand_all_existing = _merge_final_rows(cand_all_existing, cand_rows, scenario_id, freq, direction, ss, by_entry=True)
                write_csv_union(sdir / 'final_subset_screen.csv', screen_all_existing)
                write_csv_union(sdir / 'final_subset_permutations.csv', perm_all_existing)
                write_csv_union(sdir / 'final_subset_candidates.csv', cand_all_existing)
                write_csv_union(sdir / f'attempts_final_{float(freq):g}_{direction}_{ss.replace("+", "-")}.csv', _annotate_attempt_rows(ctx.attempt_log, role='final_search', ref_name='hu4_hf3_box'))
                done.add(key)
                state['final_search_done_keys'] = sorted(done)
                save_state(output_dir, scenario_id, state)
                # Rebuild local shortlist after every family.
                write_csv_union(sdir / 'final_shortlist.csv', shortlist_rows_from_candidate_rows(cand_all_existing))
                rebuild_global_outputs(output_dir)
                print(f's{scenario_id}: final {key} kaydedildi | best_entries={len(best_records)} | perm_records={len(perm_records)}')


def _merge_final_rows(existing: List[dict], new_rows: List[dict], scenario_id: int, freq: float, direction: str, subset: str, by_entry: bool) -> List[dict]:
    def same_base(r):
        return (
            int(float(r.get('scenario_id', -999))) == int(scenario_id)
            and abs(float(r.get('frequency_hz', freq)) - float(freq)) < 1e-9
            and str(r.get('direction', '')) == str(direction)
            and str(r.get('subset', '')) == str(subset)
        )
    return [r for r in existing if not same_base(r)] + list(new_rows)


def parse_subset_names(text: str, max_cardinality: int = 4) -> List[Tuple[str, ...]]:
    all_subsets = enumerate_metric_subsets(max_cardinality=max_cardinality)
    by_name = {subset_name(s): tuple(s) for s in all_subsets}
    if not text or str(text).strip().lower() in ('all', '*'):
        return all_subsets
    out = []
    for raw in str(text).split(','):
        name = raw.strip().replace(' ', '')
        if not name:
            continue
        name = name.replace('-', '+')
        parts = tuple(p for p in name.split('+') if p)
        norm = subset_name(parts)
        if norm not in by_name:
            raise ValueError(f'Bilinmeyen subset: {raw}. Örnek: tw, tw+tdot, tw+delta+delta_dot')
        out.append(by_name[norm])
    return out


def parse_directions(text: str) -> List[str]:
    t = (text or 'both').strip().lower()
    if t in ('both', 'all', '*'):
        return ['bottom_up', 'top_down']
    out = []
    for raw in t.split(','):
        r = raw.strip().replace('-', '_')
        if r in ('bu', 'bottom', 'bottom_up'):
            out.append('bottom_up')
        elif r in ('td', 'top', 'top_down'):
            out.append('top_down')
        elif r:
            raise ValueError(f'Bilinmeyen direction: {raw}')
    return out or ['bottom_up', 'top_down']


def print_status(args) -> None:
    output_dir = Path(args.output_dir)
    scenario_ids = parse_int_list(args.scenario_ids) or [0]
    for sid in scenario_ids:
        state = load_state(output_dir, sid)
        print(f'\ns{sid}:')
        for name in ('hu1', 'hf1', 'hu2', 'hf2', 'hu3', 'hf3', 'hu4'):
            print(f'  {name:<4}: {state.get(name + "_hw") is not None}')
        print(f'  final done families: {len(state.get("final_search_done_keys", []))}')


def add_common_args(parser) -> None:
    parser.add_argument('--num-scenarios', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scenario-ids', type=str, default='0')
    parser.add_argument('--output-dir', type=str, default='outputs_hardware_search')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--num-workers', type=int, default=1)

    parser.add_argument('--anchor-freq-hz', type=float, default=20.0)
    parser.add_argument('--freq-list', type=str, default='20')
    parser.add_argument('--subset-names', type=str, default='all')
    parser.add_argument('--directions', type=str, default='both')
    parser.add_argument('--preview-h-s', type=float, default=2.0)
    parser.add_argument('--sim-time-max-s', type=float, default=30.0)

    parser.add_argument('--single-rel-tol-pct', type=float, default=0.5)
    parser.add_argument('--single-bisect-iters', type=int, default=18)
    parser.add_argument('--contraction-alpha-step', type=float, default=0.01)
    parser.add_argument('--final-grid-step', type=float, default=0.001)
    parser.add_argument('--reclaim-grid-step', type=float, default=0.001)
    parser.add_argument('--min-metric-rel-change', type=float, default=0.001)
    parser.add_argument('--active-threshold', type=float, default=0.01)
    parser.add_argument('--subset-max-cardinality', type=int, default=4)
    parser.add_argument('--final-entry-mode', type=str, default='bisection', choices=['bisection', 'all_feasible', 'first'])

    parser.add_argument('--cap-tw-ratio', type=float, default=2.0)
    parser.add_argument('--cap-tdot-weight-per-sec', type=float, default=1.0)
    parser.add_argument('--cap-delta-deg', type=float, default=60.0)
    parser.add_argument('--cap-delta-dot-deg-s', type=float, default=180.0)

    parser.add_argument('--disable-deck-heave', action='store_true')
    parser.add_argument('--disable-wind-gusts', action='store_true')
    parser.add_argument('--skip-cap-sanity', action='store_true')


def run_full_scenario(args, scenario_id: int) -> None:
    # Scenario-local full run with strict dependencies. Each internal stage
    # writes checkpoints, but downstream stages are skipped if the required
    # upstream vector was not actually produced.
    output_dir = Path(args.output_dir)

    run_utopia_pass(args, scenario_id, pass_index=1)
    if hw_from_dict(load_state(output_dir, scenario_id).get('hu1_hw')) is None:
        return

    run_feasible_pass(args, scenario_id, pass_index=1)
    if hw_from_dict(load_state(output_dir, scenario_id).get('hf1_hw')) is None:
        return

    run_utopia_pass(args, scenario_id, pass_index=2)
    if hw_from_dict(load_state(output_dir, scenario_id).get('hu2_hw')) is None:
        return

    run_feasible_pass(args, scenario_id, pass_index=2)
    if hw_from_dict(load_state(output_dir, scenario_id).get('hf2_hw')) is None:
        return

    run_utopia_pass(args, scenario_id, pass_index=3)
    if hw_from_dict(load_state(output_dir, scenario_id).get('hu3_hw')) is None:
        return

    run_feasible_pass(args, scenario_id, pass_index=3)
    if hw_from_dict(load_state(output_dir, scenario_id).get('hf3_hw')) is None:
        return

    run_utopia_pass(args, scenario_id, pass_index=4)
    if hw_from_dict(load_state(output_dir, scenario_id).get('hu4_hw')) is None:
        return

    run_final_search_for_scenario(args, scenario_id)


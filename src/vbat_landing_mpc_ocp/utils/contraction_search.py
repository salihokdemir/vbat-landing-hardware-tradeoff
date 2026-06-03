from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from core.hardware import HardwarePoint
from utils.adaptive_search import METRIC_NAMES, get_metric, with_metric

EvaluateFn = Callable[[HardwarePoint, str, Optional[Any], Optional[Any]], Any]


@dataclass(frozen=True)
class Search20Config:
    """Knobs for the recursive contraction + final dual-direction search."""

    single_metric_rel_tol: float = 0.005
    single_metric_bisect_iters: int = 18

    contraction_alpha_step: float = 0.05
    # For final search, these are bisection tolerances when
    # final_entry_mode='bisection'. They are still grid steps for legacy modes.
    final_grid_step: float = 0.001
    reclaim_grid_step: float = 0.001
    # Do not spend solver calls for changes smaller than this relative physical
    # change. For group moves, at least one moved metric must change by this
    # fraction for the trial to be worth evaluating.
    min_metric_rel_change: float = 0.001
    active_threshold: float = 0.05

    subset_max_cardinality: int = 4
    final_entry_mode: str = 'bisection'  # bisection | all_feasible | first


@dataclass(frozen=True)
class SingleMetricRecord20:
    pass_name: str
    metric: str
    success: bool
    low_value: float
    high_value: float
    hardware: Optional[HardwarePoint]
    iterations: int


@dataclass(frozen=True)
class FeasiblePointRecord20:
    pass_name: str
    success: bool
    alpha: Optional[float]
    hardware: Optional[HardwarePoint]
    eval_result: Optional[Any]
    attempts: int


@dataclass(frozen=True)
class FinalEntryRecord20:
    direction: str  # bottom_up or top_down
    subset: Tuple[str, ...]
    param: float
    hardware: HardwarePoint
    eval_result: Any


@dataclass(frozen=True)
class FinalPermutationRecord20:
    direction: str
    subset: Tuple[str, ...]
    permutation: Tuple[str, ...]
    entry_param: float
    hardware: HardwarePoint
    slack_by_metric: Dict[str, float]
    l1_sum: float
    l2_sq: float
    linf: float
    active_count: int
    reclaim_score: float
    eval_result: Any
    # Diagnostics for the two-layer top-down cleanup. These are optional and
    # older CSV writers can ignore them.
    family_permutation: Tuple[str, ...] = ()
    outside_permutation: Tuple[str, ...] = ()
    cleanup_mode: str = 'family_only'


# -----------------------------------------------------------------------------
# Basic hardware helpers
# -----------------------------------------------------------------------------


def subset_name(subset: Sequence[str]) -> str:
    ordered = [metric for metric in METRIC_NAMES if metric in set(subset)]
    return '+'.join(ordered)


def permutation_name(order: Sequence[str]) -> str:
    return ' -> '.join(order)


def enumerate_metric_subsets(max_cardinality: int = 4) -> List[Tuple[str, ...]]:
    out: List[Tuple[str, ...]] = []
    max_cardinality = max(1, min(int(max_cardinality), len(METRIC_NAMES)))
    for card in range(1, max_cardinality + 1):
        out.extend(tuple(combo) for combo in combinations(METRIC_NAMES, card))
    return out


def _hardware_from_values(values: Dict[str, float]) -> HardwarePoint:
    return HardwarePoint(
        tw_ratio=float(values['tw']),
        T_dot_max=float(values['tdot']),
        delta_dot_max=float(values['delta_dot']),
        delta_max=float(values['delta']),
    )


def interpolate_all(low_hw: HardwarePoint, high_hw: HardwarePoint, alpha: float) -> HardwarePoint:
    a = float(np.clip(float(alpha), 0.0, 1.0))
    return HardwarePoint(
        tw_ratio=float(get_metric(low_hw, 'tw') + a * (get_metric(high_hw, 'tw') - get_metric(low_hw, 'tw'))),
        T_dot_max=float(get_metric(low_hw, 'tdot') + a * (get_metric(high_hw, 'tdot') - get_metric(low_hw, 'tdot'))),
        delta_max=float(get_metric(low_hw, 'delta') + a * (get_metric(high_hw, 'delta') - get_metric(low_hw, 'delta'))),
        delta_dot_max=float(get_metric(low_hw, 'delta_dot') + a * (get_metric(high_hw, 'delta_dot') - get_metric(low_hw, 'delta_dot'))),
    )


def interpolate_bottom_up(low_hw: HardwarePoint, high_hw: HardwarePoint, subset: Sequence[str], alpha: float) -> HardwarePoint:
    subset_set = set(subset)
    a = float(np.clip(float(alpha), 0.0, 1.0))
    vals: Dict[str, float] = {}
    for m in METRIC_NAMES:
        lo = get_metric(low_hw, m)
        hi = get_metric(high_hw, m)
        vals[m] = float(lo + a * (hi - lo)) if m in subset_set else float(lo)
    return _hardware_from_values(vals)


def interpolate_top_down(low_hw: HardwarePoint, high_hw: HardwarePoint, subset: Sequence[str], beta: float) -> HardwarePoint:
    subset_set = set(subset)
    b = float(np.clip(float(beta), 0.0, 1.0))
    vals: Dict[str, float] = {}
    for m in METRIC_NAMES:
        lo = get_metric(low_hw, m)
        hi = get_metric(high_hw, m)
        vals[m] = float(hi - b * (hi - lo)) if m in subset_set else float(hi)
    return _hardware_from_values(vals)


def normalize_between(hw: HardwarePoint, low_hw: HardwarePoint, high_hw: HardwarePoint) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for m in METRIC_NAMES:
        lo = float(get_metric(low_hw, m))
        hi = float(get_metric(high_hw, m))
        v = float(get_metric(hw, m))
        denom = hi - lo
        if denom <= 1e-12:
            out[m] = 0.0 if abs(v - lo) <= 1e-12 else 1.0
        else:
            out[m] = float(np.clip((v - lo) / denom, 0.0, 1.0))
    return out


def slack_l1(slack: Dict[str, float]) -> float:
    return float(sum(float(slack[m]) for m in METRIC_NAMES))


def slack_l2_sq(slack: Dict[str, float]) -> float:
    return float(sum(float(slack[m]) ** 2 for m in METRIC_NAMES))


def slack_linf(slack: Dict[str, float]) -> float:
    return float(max((float(slack[m]) for m in METRIC_NAMES), default=0.0))


def active_count(slack: Dict[str, float], threshold: float) -> int:
    return int(sum(1 for m in METRIC_NAMES if float(slack[m]) > float(threshold)))


def build_utopia_from_records20(records_by_metric: Dict[str, SingleMetricRecord20]) -> Optional[HardwarePoint]:
    if not all(m in records_by_metric and records_by_metric[m].success and records_by_metric[m].hardware is not None for m in METRIC_NAMES):
        return None
    return HardwarePoint(
        tw_ratio=float(records_by_metric['tw'].hardware.tw_ratio),
        T_dot_max=float(records_by_metric['tdot'].hardware.T_dot_max),
        delta_dot_max=float(records_by_metric['delta_dot'].hardware.delta_dot_max),
        delta_max=float(records_by_metric['delta'].hardware.delta_max),
    )


# -----------------------------------------------------------------------------
# Utopia and contraction passes
# -----------------------------------------------------------------------------


def find_single_metric_minimum_with_support(
    *,
    pass_name: str,
    metric: str,
    support_hw: HardwarePoint,
    metric_upper_hw: HardwarePoint,
    evaluate_fn: EvaluateFn,
    stage_prefix: str,
    search_cfg: Search20Config,
) -> SingleMetricRecord20:
    """Single-metric bisection.

    The three non-searched metrics stay at support_hw. The searched metric is
    bracketed between 0 and metric_upper_hw[metric]. This is intentionally more
    flexible than the old 'cap_hw for everything' version: h_f20-1/h_f20-2 can
    act as support vectors without becoming strict caps, while C0 can remain the
    metric upper target during early contraction passes.
    """
    metric = str(metric)
    upper_value = float(get_metric(metric_upper_hw, metric))
    high_hw = with_metric(support_hw, metric, upper_value)
    high_res = evaluate_fn(high_hw, f'{stage_prefix}_{metric}_upper', None, None)
    if not bool(getattr(high_res, 'success', False)):
        return SingleMetricRecord20(
            pass_name=str(pass_name),
            metric=metric,
            success=False,
            low_value=0.0,
            high_value=upper_value,
            hardware=None,
            iterations=0,
        )

    low_value = 0.0
    high_value = upper_value
    best_res = high_res
    ref_scale = max(abs(upper_value), 1.0)
    iters = 0
    for _ in range(int(search_cfg.single_metric_bisect_iters)):
        if abs(high_value - low_value) <= max(1e-12, float(search_cfg.single_metric_rel_tol) * ref_scale):
            break
        mid_value = 0.5 * (low_value + high_value)
        if abs(mid_value - low_value) <= 1e-15 or abs(mid_value - high_value) <= 1e-15:
            break
        candidate_hw = with_metric(support_hw, metric, mid_value)
        res = evaluate_fn(candidate_hw, f'{stage_prefix}_{metric}_bisect', None, None)
        iters += 1
        if bool(getattr(res, 'success', False)):
            best_res = res
            high_value = mid_value
        else:
            low_value = mid_value

    return SingleMetricRecord20(
        pass_name=str(pass_name),
        metric=metric,
        success=True,
        low_value=float(low_value),
        high_value=float(high_value),
        hardware=best_res.hardware,
        iterations=int(iters),
    )


def compute_utopia_pass20(
    *,
    pass_name: str,
    support_hw: HardwarePoint,
    metric_upper_hw: HardwarePoint,
    evaluate_fn: EvaluateFn,
    stage_prefix: str,
    search_cfg: Search20Config,
) -> Tuple[Dict[str, SingleMetricRecord20], Optional[HardwarePoint]]:
    records: Dict[str, SingleMetricRecord20] = {}
    for metric in METRIC_NAMES:
        rec = find_single_metric_minimum_with_support(
            pass_name=pass_name,
            metric=metric,
            support_hw=support_hw,
            metric_upper_hw=metric_upper_hw,
            evaluate_fn=evaluate_fn,
            stage_prefix=stage_prefix,
            search_cfg=search_cfg,
        )
        records[metric] = rec
        if not rec.success:
            break
    return records, build_utopia_from_records20(records)


def _alpha_grid(step: float, include_zero: bool = True) -> List[float]:
    step = max(float(step), 1e-6)
    vals = []
    if include_zero:
        vals.append(0.0)
    k = 1
    while True:
        a = round(k * step, 12)
        if a >= 1.0 - 1e-12:
            vals.append(1.0)
            break
        vals.append(a)
        k += 1
    return vals


def find_first_feasible_global_inflation20(
    *,
    pass_name: str,
    low_hw: HardwarePoint,
    target_hw: HardwarePoint,
    evaluate_fn: EvaluateFn,
    stage_prefix: str,
    search_cfg: Search20Config,
) -> FeasiblePointRecord20:
    attempts = 0
    for idx, alpha in enumerate(_alpha_grid(search_cfg.contraction_alpha_step, include_zero=True)):
        hw = interpolate_all(low_hw, target_hw, alpha)
        res = evaluate_fn(hw, f'{stage_prefix}_{pass_name}_alpha_{idx:03d}', None, None)
        attempts += 1
        if bool(getattr(res, 'success', False)):
            return FeasiblePointRecord20(
                pass_name=str(pass_name),
                success=True,
                alpha=float(alpha),
                hardware=res.hardware,
                eval_result=res,
                attempts=int(attempts),
            )
    return FeasiblePointRecord20(
        pass_name=str(pass_name),
        success=False,
        alpha=None,
        hardware=None,
        eval_result=None,
        attempts=int(attempts),
    )


# -----------------------------------------------------------------------------
# Final dual-direction search
# -----------------------------------------------------------------------------


def _make_final_entry_hw(direction: str, low_hw: HardwarePoint, high_hw: HardwarePoint, subset: Sequence[str], param: float) -> HardwarePoint:
    if direction == 'bottom_up':
        return interpolate_bottom_up(low_hw, high_hw, subset, param)
    if direction == 'top_down':
        return interpolate_top_down(low_hw, high_hw, subset, param)
    raise ValueError(direction)


def _relative_metric_change(hw_a: HardwarePoint, hw_b: HardwarePoint, metric: str) -> float:
    a = float(get_metric(hw_a, metric))
    b = float(get_metric(hw_b, metric))
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def _max_relative_change(hw_a: HardwarePoint, hw_b: HardwarePoint, metrics: Sequence[str]) -> float:
    vals = [_relative_metric_change(hw_a, hw_b, m) for m in metrics]
    return max(vals) if vals else 0.0


def _is_worth_testing(hw_a: HardwarePoint, hw_b: HardwarePoint, metrics: Sequence[str], search_cfg: Search20Config) -> bool:
    """Return True if at least one moved metric changes enough.

    Implements the one-thousandth relative-change rule. If a proposed trial
    changes every moved metric by less than min_metric_rel_change, the solver
    call is skipped because it is below useful resolution.
    """
    return _max_relative_change(hw_a, hw_b, metrics) >= float(search_cfg.min_metric_rel_change)


def find_final_entries20(
    *,
    direction: str,
    subset: Sequence[str],
    low_hw: HardwarePoint,
    high_hw: HardwarePoint,
    evaluate_fn: EvaluateFn,
    stage_prefix: str,
    search_cfg: Search20Config,
) -> Tuple[List[FinalEntryRecord20], List[dict]]:
    """Find final-search entry using bisection by default.

    bottom_up: find the smallest feasible alpha from hu4 toward hf3.
    top_down : find the largest feasible beta from hf3 toward hu4.

    This avoids evaluating all 100 grid points when the final box is already
    very narrow.
    """
    mode = str(search_cfg.final_entry_mode).strip().lower()

    if mode in {'all_feasible', 'first'}:
        entries: List[FinalEntryRecord20] = []
        screen_rows: List[dict] = []
        grid = _alpha_grid(search_cfg.final_grid_step, include_zero=True)
        eval_grid = list(reversed(grid)) if (mode == 'first' and direction == 'top_down') else list(grid)
        for eval_idx, param in enumerate(eval_grid):
            hw = _make_final_entry_hw(direction, low_hw, high_hw, subset, float(param))
            try:
                grid_index = int(grid.index(param))
            except ValueError:
                grid_index = int(eval_idx)
            res = evaluate_fn(hw, f'{stage_prefix}_{direction}_{subset_name(subset)}_grid_{grid_index:03d}', None, None)
            success = bool(getattr(res, 'success', False))
            screen_rows.append({
                'direction': direction,
                'subset': subset_name(subset),
                'cardinality': len(tuple(subset)),
                'grid_index': int(grid_index),
                'param': float(param),
                'entry_success': bool(success),
                'return_status': getattr(res, 'return_status', None),
                'failure_mode': getattr(res, 'failure_mode', None),
                'search_mode': mode,
            })
            if success:
                entries.append(FinalEntryRecord20(direction=direction, subset=tuple(subset), param=float(param), hardware=res.hardware, eval_result=res))
                if mode == 'first':
                    break
        if mode != 'first' and direction == 'top_down' and len(entries) > 1:
            entries = [e for e in entries if e.param > 1e-12] or entries
        return entries, screen_rows

    if mode != 'bisection':
        raise ValueError(f'Unknown final_entry_mode={mode!r}')

    subset_tuple = tuple(subset)
    screen_rows: List[dict] = []
    tol = max(float(search_cfg.final_grid_step), 1e-6)

    def eval_param(param: float, label: str):
        hw = _make_final_entry_hw(direction, low_hw, high_hw, subset_tuple, float(param))
        res = evaluate_fn(hw, f'{stage_prefix}_{direction}_{subset_name(subset_tuple)}_bisect_{label}', None, None)
        success = bool(getattr(res, 'success', False))
        screen_rows.append({
            'direction': direction,
            'subset': subset_name(subset_tuple),
            'cardinality': len(subset_tuple),
            'grid_index': None,
            'param': float(param),
            'entry_success': bool(success),
            'return_status': getattr(res, 'return_status', None),
            'failure_mode': getattr(res, 'failure_mode', None),
            'search_mode': 'bisection',
        })
        return res, success

    if direction == 'bottom_up':
        low_res, low_ok = eval_param(0.0, 'lo')
        if low_ok:
            return [FinalEntryRecord20(direction=direction, subset=subset_tuple, param=0.0, hardware=low_res.hardware, eval_result=low_res)], screen_rows

        high_res, high_ok = eval_param(1.0, 'hi')
        if not high_ok:
            return [], screen_rows

        lo = 0.0
        hi = 1.0
        best_res = high_res
        best_param = 1.0
        it = 0
        while (hi - lo) > tol:
            mid = 0.5 * (lo + hi)
            cand_hw = _make_final_entry_hw(direction, low_hw, high_hw, subset_tuple, mid)
            hi_hw = _make_final_entry_hw(direction, low_hw, high_hw, subset_tuple, hi)
            if not _is_worth_testing(cand_hw, hi_hw, subset_tuple, search_cfg):
                break
            res, ok = eval_param(mid, f'{it:03d}')
            if ok:
                hi = mid
                best_res = res
                best_param = mid
            else:
                lo = mid
            it += 1
        return [FinalEntryRecord20(direction=direction, subset=subset_tuple, param=float(best_param), hardware=best_res.hardware, eval_result=best_res)], screen_rows

    if direction == 'top_down':
        zero_res, zero_ok = eval_param(0.0, 'lo')
        if not zero_ok:
            return [], screen_rows

        one_res, one_ok = eval_param(1.0, 'hi')
        if one_ok:
            return [FinalEntryRecord20(direction=direction, subset=subset_tuple, param=1.0, hardware=one_res.hardware, eval_result=one_res)], screen_rows

        lo = 0.0  # feasible beta
        hi = 1.0  # infeasible beta
        best_res = zero_res
        best_param = 0.0
        it = 0
        while (hi - lo) > tol:
            mid = 0.5 * (lo + hi)
            cand_hw = _make_final_entry_hw(direction, low_hw, high_hw, subset_tuple, mid)
            lo_hw = _make_final_entry_hw(direction, low_hw, high_hw, subset_tuple, lo)
            if not _is_worth_testing(cand_hw, lo_hw, subset_tuple, search_cfg):
                break
            res, ok = eval_param(mid, f'{it:03d}')
            if ok:
                lo = mid
                best_res = res
                best_param = mid
            else:
                hi = mid
            it += 1
        return [FinalEntryRecord20(direction=direction, subset=subset_tuple, param=float(best_param), hardware=best_res.hardware, eval_result=best_res)], screen_rows

    raise ValueError(direction)


def _metric_normalized_value(hw: HardwarePoint, low_hw: HardwarePoint, high_hw: HardwarePoint, metric: str) -> float:
    return normalize_between(hw, low_hw, high_hw)[metric]


def reclaim_one_metric_grid20(
    *,
    metric: str,
    base_res: Any,
    low_hw: HardwarePoint,
    high_hw: HardwarePoint,
    evaluate_fn: EvaluateFn,
    stage_prefix: str,
    search_cfg: Search20Config,
) -> Any:
    """Lower one metric toward low_hw using bisection.

    base_res is known feasible. We search for the lowest normalized z in
    [0, current_z] that remains feasible. The search stops at reclaim_grid_step
    or when the next physical relative change is below min_metric_rel_change.
    """
    current_res = base_res
    current_hw = base_res.hardware
    current_z = _metric_normalized_value(current_hw, low_hw, high_hw, metric)
    tol = max(float(search_cfg.reclaim_grid_step), 1e-6)
    if current_z <= max(1e-12, tol):
        return current_res

    low_metric_hw = with_metric(current_hw, metric, get_metric(low_hw, metric))
    if not _is_worth_testing(low_metric_hw, current_hw, [metric], search_cfg):
        return current_res

    res0 = evaluate_fn(low_metric_hw, f'{stage_prefix}_{metric}_reclaim_lo', None, None)
    if bool(getattr(res0, 'success', False)):
        return res0

    lo = 0.0        # infeasible normalized z
    hi = current_z  # feasible normalized z
    best_res = current_res
    it = 0
    while (hi - lo) > tol:
        mid = 0.5 * (lo + hi)
        low = get_metric(low_hw, metric)
        high = get_metric(high_hw, metric)
        value = float(low + mid * (high - low))
        cand_hw = with_metric(current_hw, metric, value)
        hi_hw = with_metric(current_hw, metric, float(low + hi * (high - low)))
        if not _is_worth_testing(cand_hw, hi_hw, [metric], search_cfg):
            break
        res = evaluate_fn(cand_hw, f'{stage_prefix}_{metric}_reclaim_bisect_{it:03d}', None, None)
        if bool(getattr(res, 'success', False)):
            hi = mid
            best_res = res
        else:
            lo = mid
        it += 1
    return best_res


def _make_final_permutation_record20(
    *,
    direction: str,
    subset: Sequence[str],
    permutation_order: Sequence[str],
    entry_param: float,
    start_res: Any,
    final_res: Any,
    low_hw: HardwarePoint,
    high_hw: HardwarePoint,
    search_cfg: Search20Config,
    family_permutation: Sequence[str] = (),
    outside_permutation: Sequence[str] = (),
    cleanup_mode: str = 'family_only',
) -> FinalPermutationRecord20:
    """Build one final candidate record.

    reclaim_score is measured from the feasible entry point to the final
    candidate over all four metrics. This makes top-down outside cleanup
    comparable with the older family-only reclaim records.
    """
    start_slack = normalize_between(start_res.hardware, low_hw, high_hw)
    final_slack = normalize_between(final_res.hardware, low_hw, high_hw)
    reclaim_score = float(sum(float(start_slack[m]) - float(final_slack[m]) for m in METRIC_NAMES))
    return FinalPermutationRecord20(
        direction=str(direction),
        subset=tuple(subset),
        permutation=tuple(permutation_order),
        entry_param=float(entry_param),
        hardware=final_res.hardware,
        slack_by_metric=final_slack,
        l1_sum=slack_l1(final_slack),
        l2_sq=slack_l2_sq(final_slack),
        linf=slack_linf(final_slack),
        active_count=active_count(final_slack, search_cfg.active_threshold),
        reclaim_score=float(reclaim_score),
        eval_result=final_res,
        family_permutation=tuple(family_permutation),
        outside_permutation=tuple(outside_permutation),
        cleanup_mode=str(cleanup_mode),
    )


def run_permutation_reclaim20(
    *,
    direction: str,
    subset: Sequence[str],
    permutation_order: Sequence[str],
    entry_param: float,
    entry_res: Any,
    low_hw: HardwarePoint,
    high_hw: HardwarePoint,
    evaluate_fn: EvaluateFn,
    stage_prefix: str,
    search_cfg: Search20Config,
) -> FinalPermutationRecord20:
    """Old family-only reclaim.

    For bottom-up this is usually enough because non-subset metrics are already
    at the lower reference. For top-down, run_final_family20 now calls this as
    the first layer and then optionally performs an outside-subset cleanup.
    """
    current_res = entry_res
    for metric in permutation_order:
        current_res = reclaim_one_metric_grid20(
            metric=str(metric),
            base_res=current_res,
            low_hw=low_hw,
            high_hw=high_hw,
            evaluate_fn=evaluate_fn,
            stage_prefix=f'{stage_prefix}_{direction}_{subset_name(subset)}_family_{metric}',
            search_cfg=search_cfg,
        )

    return _make_final_permutation_record20(
        direction=direction,
        subset=subset,
        permutation_order=permutation_order,
        entry_param=entry_param,
        start_res=entry_res,
        final_res=current_res,
        low_hw=low_hw,
        high_hw=high_hw,
        search_cfg=search_cfg,
        family_permutation=tuple(permutation_order),
        outside_permutation=(),
        cleanup_mode='family_only',
    )


def run_outside_cleanup20(
    *,
    direction: str,
    subset: Sequence[str],
    family_order: Sequence[str],
    outside_order: Sequence[str],
    entry_param: float,
    entry_res: Any,
    family_res: Any,
    low_hw: HardwarePoint,
    high_hw: HardwarePoint,
    evaluate_fn: EvaluateFn,
    stage_prefix: str,
    search_cfg: Search20Config,
) -> FinalPermutationRecord20:
    """Second-layer top-down cleanup.

    In a top-down family, metrics outside the selected subset stay at the upper
    reference H during the entry search. After the selected subset has been
    reclaimed, this function tries to lower those outside metrics as well.

    This implements the user's intended logic:
        top-down entry -> subset permutation reclaim -> outside-subset
        permutation cleanup.
    """
    current_res = family_res
    for metric in outside_order:
        current_res = reclaim_one_metric_grid20(
            metric=str(metric),
            base_res=current_res,
            low_hw=low_hw,
            high_hw=high_hw,
            evaluate_fn=evaluate_fn,
            stage_prefix=f'{stage_prefix}_{direction}_{subset_name(subset)}_outside_{metric}',
            search_cfg=search_cfg,
        )

    combined_order = tuple(family_order) + tuple(outside_order)
    return _make_final_permutation_record20(
        direction=direction,
        subset=subset,
        permutation_order=combined_order,
        entry_param=entry_param,
        start_res=entry_res,
        final_res=current_res,
        low_hw=low_hw,
        high_hw=high_hw,
        search_cfg=search_cfg,
        family_permutation=tuple(family_order),
        outside_permutation=tuple(outside_order),
        cleanup_mode='family_plus_outside',
    )


def _best_final_perm_key(rec: FinalPermutationRecord20):
    return (
        float(rec.reclaim_score),
        -float(rec.l2_sq),
        -float(rec.l1_sum),
        -float(rec.linf),
        -float(rec.active_count),
    )


def run_final_family20(
    *,
    direction: str,
    subset: Sequence[str],
    low_hw: HardwarePoint,
    high_hw: HardwarePoint,
    evaluate_fn: EvaluateFn,
    stage_prefix: str,
    search_cfg: Search20Config,
) -> Tuple[List[dict], List[FinalPermutationRecord20], List[FinalPermutationRecord20]]:
    entries, screen_rows = find_final_entries20(
        direction=direction,
        subset=subset,
        low_hw=low_hw,
        high_hw=high_hw,
        evaluate_fn=evaluate_fn,
        stage_prefix=stage_prefix,
        search_cfg=search_cfg,
    )
    all_perm_records: List[FinalPermutationRecord20] = []
    best_records: List[FinalPermutationRecord20] = []
    subset_tuple = tuple(subset)
    outside_tuple = tuple(m for m in METRIC_NAMES if m not in set(subset_tuple))

    for entry in entries:
        entry_records: List[FinalPermutationRecord20] = []

        for family_order in permutations(subset_tuple):
            # Layer 1: same as before. Reclaim the selected subset metrics.
            family_rec = run_permutation_reclaim20(
                direction=direction,
                subset=subset_tuple,
                permutation_order=family_order,
                entry_param=entry.param,
                entry_res=entry.eval_result,
                low_hw=low_hw,
                high_hw=high_hw,
                evaluate_fn=evaluate_fn,
                stage_prefix=stage_prefix,
                search_cfg=search_cfg,
            )

            if direction == 'top_down' and outside_tuple:
                # Layer 2 for top-down only: after the selected subset has been
                # lowered/reclaimed, try to lower the metrics that were outside
                # the subset and therefore stayed at the upper reference H.
                #
                # This is intentionally NOT applied to bottom-up because outside
                # metrics are already at L there.
                for outside_order in permutations(outside_tuple):
                    rec = run_outside_cleanup20(
                        direction=direction,
                        subset=subset_tuple,
                        family_order=family_order,
                        outside_order=outside_order,
                        entry_param=entry.param,
                        entry_res=entry.eval_result,
                        family_res=family_rec.eval_result,
                        low_hw=low_hw,
                        high_hw=high_hw,
                        evaluate_fn=evaluate_fn,
                        stage_prefix=stage_prefix,
                        search_cfg=search_cfg,
                    )
                    all_perm_records.append(rec)
                    entry_records.append(rec)
            else:
                all_perm_records.append(family_rec)
                entry_records.append(family_rec)

        if entry_records:
            best_records.append(max(entry_records, key=_best_final_perm_key))

    return screen_rows, all_perm_records, best_records


def dominance_prune(records: Sequence[FinalPermutationRecord20], eps: float = 1e-9) -> List[FinalPermutationRecord20]:
    out: List[FinalPermutationRecord20] = []
    for i, rec in enumerate(records):
        dominated = False
        for j, other in enumerate(records):
            if i == j:
                continue
            le_all = all(float(other.slack_by_metric[m]) <= float(rec.slack_by_metric[m]) + eps for m in METRIC_NAMES)
            lt_any = any(float(other.slack_by_metric[m]) < float(rec.slack_by_metric[m]) - eps for m in METRIC_NAMES)
            if le_all and lt_any:
                dominated = True
                break
        if not dominated:
            out.append(rec)
    return out


def choose_shortlist20(records: Sequence[FinalPermutationRecord20]) -> Dict[str, FinalPermutationRecord20]:
    rows = list(records)
    if not rows:
        return {}
    closest = min(rows, key=lambda r: (float(r.l2_sq), float(r.l1_sum), float(r.linf), int(r.active_count)))
    low_total = min(rows, key=lambda r: (float(r.l1_sum), float(r.l2_sq), float(r.linf), int(r.active_count)))
    balanced = min(rows, key=lambda r: (float(r.linf), float(r.l2_sq), float(r.l1_sum), int(r.active_count)))
    sparse = min(rows, key=lambda r: (int(r.active_count), float(r.l2_sq), float(r.l1_sum), float(r.linf)))
    best_reclaim = max(rows, key=lambda r: (float(r.reclaim_score), -float(r.l2_sq), -float(r.l1_sum)))
    out: Dict[str, FinalPermutationRecord20] = {
        'closest': closest,
        'low_total': low_total,
        'balanced': balanced,
        'sparse': sparse,
        'best_reclaim': best_reclaim,
    }
    bu = [r for r in rows if r.direction == 'bottom_up']
    td = [r for r in rows if r.direction == 'top_down']
    if bu:
        out['best_bottom_up'] = min(bu, key=lambda r: (float(r.l2_sq), float(r.l1_sum), float(r.linf), int(r.active_count)))
    if td:
        out['best_top_down'] = min(td, key=lambda r: (float(r.l2_sq), float(r.l1_sum), float(r.linf), int(r.active_count)))
    return out

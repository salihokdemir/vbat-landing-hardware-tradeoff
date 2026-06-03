from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from core.hardware import HardwarePoint


METRIC_NAMES = ('tw', 'tdot', 'delta', 'delta_dot')


@dataclass(frozen=True)
class AdaptiveSearchConfig:
    """
    Generic multiplicative feasibility-boundary search config.

    Interpretation
    --------------
    - base_step = 0.01  -> first outward / inward move is 1%
    - step_growth = 0.01 -> the *step size itself* grows by 1% each iteration
      (1.00%, 1.01%, 1.0201%, ...)
    - bisect_rel_tol = 0.005 -> stop once the feasible/infeasible bracket is below 0.5%
    """

    base_step: float = 0.01
    step_growth: float = 0.01
    max_expand_iters: int = 400
    max_bisect_iters: int = 24
    bisect_rel_tol: float = 0.005
    tighten_cycles: int = 2


def get_metric(hw: HardwarePoint, metric: str) -> float:
    if metric == 'tw':
        return float(hw.tw_ratio)
    if metric == 'tdot':
        return float(hw.T_dot_max)
    if metric == 'delta':
        return float(hw.delta_max)
    if metric == 'delta_dot':
        return float(hw.delta_dot_max)
    raise ValueError(metric)


def with_metric(hw: HardwarePoint, metric: str, value: float) -> HardwarePoint:
    if metric == 'tw':
        return HardwarePoint(float(value), hw.T_dot_max, hw.delta_dot_max, hw.delta_max)
    if metric == 'tdot':
        return HardwarePoint(hw.tw_ratio, float(value), hw.delta_dot_max, hw.delta_max)
    if metric == 'delta':
        return HardwarePoint(hw.tw_ratio, hw.T_dot_max, hw.delta_dot_max, float(value))
    if metric == 'delta_dot':
        return HardwarePoint(hw.tw_ratio, hw.T_dot_max, float(value), hw.delta_max)
    raise ValueError(metric)


def combine_metric_results(results_by_metric: Dict[str, Any]) -> HardwarePoint:
    return HardwarePoint(
        tw_ratio=results_by_metric['tw'].hardware.tw_ratio,
        T_dot_max=results_by_metric['tdot'].hardware.T_dot_max,
        delta_dot_max=results_by_metric['delta_dot'].hardware.delta_dot_max,
        delta_max=results_by_metric['delta'].hardware.delta_max,
    )


def margin_pct(value: float, ref_value: float) -> Optional[float]:
    if ref_value <= 0.0:
        return None
    return 100.0 * (float(value) / float(ref_value) - 1.0)


def fold_increase(value: float, ref_value: float) -> Optional[float]:
    if ref_value <= 0.0:
        return None
    return float(value) / float(ref_value)


def hardware_margin_dict(hw: HardwarePoint, ref_hw: HardwarePoint, prefix: str = 'margin') -> Dict[str, Optional[float]]:
    return {
        f'{prefix}_tw_pct': margin_pct(hw.tw_ratio, ref_hw.tw_ratio),
        f'{prefix}_T_dot_pct': margin_pct(hw.T_dot_max, ref_hw.T_dot_max),
        f'{prefix}_delta_pct': margin_pct(hw.delta_max, ref_hw.delta_max),
        f'{prefix}_delta_dot_pct': margin_pct(hw.delta_dot_max, ref_hw.delta_dot_max),
        f'{prefix}_tw_fold': fold_increase(hw.tw_ratio, ref_hw.tw_ratio),
        f'{prefix}_T_dot_fold': fold_increase(hw.T_dot_max, ref_hw.T_dot_max),
        f'{prefix}_delta_fold': fold_increase(hw.delta_max, ref_hw.delta_max),
        f'{prefix}_delta_dot_fold': fold_increase(hw.delta_dot_max, ref_hw.delta_dot_max),
    }


def scale_all_up(hw: HardwarePoint, cap_hw: HardwarePoint, step: float) -> HardwarePoint:
    factor = 1.0 + float(step)
    return HardwarePoint(
        tw_ratio=min(cap_hw.tw_ratio, hw.tw_ratio * factor),
        T_dot_max=min(cap_hw.T_dot_max, hw.T_dot_max * factor),
        delta_dot_max=min(cap_hw.delta_dot_max, hw.delta_dot_max * factor),
        delta_max=min(cap_hw.delta_max, hw.delta_max * factor),
    )


def _geometric_mean(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        return 0.5 * (a + b)
    return math.sqrt(float(a) * float(b))


def _interp_log_scalar(low: float, high: float, alpha: float) -> float:
    if low <= 0.0 or high <= 0.0:
        return (1.0 - alpha) * float(low) + alpha * float(high)
    return math.exp((1.0 - alpha) * math.log(float(low)) + alpha * math.log(float(high)))


def interpolate_hardware_log(low_hw: HardwarePoint, high_hw: HardwarePoint, alpha: float) -> HardwarePoint:
    return HardwarePoint(
        tw_ratio=_interp_log_scalar(low_hw.tw_ratio, high_hw.tw_ratio, alpha),
        T_dot_max=_interp_log_scalar(low_hw.T_dot_max, high_hw.T_dot_max, alpha),
        delta_dot_max=_interp_log_scalar(low_hw.delta_dot_max, high_hw.delta_dot_max, alpha),
        delta_max=_interp_log_scalar(low_hw.delta_max, high_hw.delta_max, alpha),
    )


def max_relative_gap(low_hw: HardwarePoint, high_hw: HardwarePoint) -> float:
    ratios = []
    for metric in METRIC_NAMES:
        low = get_metric(low_hw, metric)
        high = get_metric(high_hw, metric)
        if low <= 0.0:
            continue
        ratios.append(high / low - 1.0)
    if not ratios:
        return 0.0
    return float(max(ratios))


def _next_step(step: float, growth: float) -> float:
    return float(step) * (1.0 + float(growth))


EvaluateFn = Callable[[HardwarePoint, str, Optional[Any], Optional[Any]], Any]




def raise_single_metric_until_feasible(
    *,
    metric: str,
    seed_hw: HardwarePoint,
    cap_hw: HardwarePoint,
    evaluate_fn: EvaluateFn,
    stage_prefix: str,
    search_cfg: AdaptiveSearchConfig,
    guess_X=None,
    guess_U=None,
):
    seed_res = evaluate_fn(seed_hw, f'{stage_prefix}_{metric}_seed', guess_X, guess_U)
    if bool(seed_res.success):
        return seed_res

    current_value = get_metric(seed_hw, metric)
    cap_value = get_metric(cap_hw, metric)
    last_infeasible_value = current_value
    last_feasible_res = None
    step = float(search_cfg.base_step)

    for _ in range(int(search_cfg.max_expand_iters)):
        candidate_value = min(cap_value, current_value * (1.0 + step))
        if abs(candidate_value - current_value) <= 1e-15:
            return None
        candidate_hw = with_metric(seed_hw, metric, candidate_value)
        res = evaluate_fn(candidate_hw, f'{stage_prefix}_{metric}_expand', guess_X, guess_U)
        if bool(res.success):
            last_feasible_res = res
            break
        last_infeasible_value = candidate_value
        current_value = candidate_value
        step = _next_step(step, search_cfg.step_growth)

    if last_feasible_res is None:
        return None

    best_res = last_feasible_res
    low_value = float(last_infeasible_value)
    high_value = float(get_metric(last_feasible_res.hardware, metric))

    for _ in range(int(search_cfg.max_bisect_iters)):
        if high_value <= 0.0 or low_value <= 0.0:
            rel_gap = abs(high_value - low_value)
            if rel_gap <= 1e-12:
                break
        else:
            rel_gap = high_value / low_value - 1.0
            if rel_gap <= float(search_cfg.bisect_rel_tol):
                break

        mid_value = _geometric_mean(low_value, high_value)
        if abs(mid_value - low_value) <= 1e-15 or abs(mid_value - high_value) <= 1e-15:
            break

        candidate_hw = with_metric(seed_hw, metric, mid_value)
        res = evaluate_fn(
            candidate_hw,
            f'{stage_prefix}_{metric}_bisect',
            getattr(best_res, 'X_opt', None),
            getattr(best_res, 'U_opt', None),
        )
        if bool(res.success):
            best_res = res
            high_value = mid_value
        else:
            low_value = mid_value

    return best_res

def raise_all_until_feasible(
    *,
    seed_hw: HardwarePoint,
    cap_hw: HardwarePoint,
    evaluate_fn: EvaluateFn,
    stage_prefix: str,
    search_cfg: AdaptiveSearchConfig,
    guess_X=None,
    guess_U=None,
):
    """
    Joint outward search from `seed_hw`.

    Flow:
    1) Check seed
    2) Increase all 4 metrics together with accelerating multiplicative steps
    3) Once a feasible point is found, tighten the bracket with log-space bisection
    """
    seed_res = evaluate_fn(seed_hw, f'{stage_prefix}_seed', guess_X, guess_U)
    if bool(seed_res.success):
        return seed_res

    current_hw = seed_hw
    last_infeasible_hw = seed_hw
    last_feasible_res = None
    step = float(search_cfg.base_step)

    for _ in range(int(search_cfg.max_expand_iters)):
        candidate_hw = scale_all_up(current_hw, cap_hw, step)
        if candidate_hw.key() == current_hw.key():
            return None

        res = evaluate_fn(candidate_hw, f'{stage_prefix}_expand', guess_X, guess_U)
        if bool(res.success):
            last_feasible_res = res
            break

        last_infeasible_hw = candidate_hw
        current_hw = candidate_hw
        step = _next_step(step, search_cfg.step_growth)

    if last_feasible_res is None:
        return None

    best_res = last_feasible_res
    low_hw = last_infeasible_hw
    high_hw = last_feasible_res.hardware

    for _ in range(int(search_cfg.max_bisect_iters)):
        if max_relative_gap(low_hw, high_hw) <= float(search_cfg.bisect_rel_tol):
            break
        mid_hw = interpolate_hardware_log(low_hw, high_hw, 0.5)
        if mid_hw.key() == low_hw.key() or mid_hw.key() == high_hw.key():
            break
        mid_res = evaluate_fn(
            mid_hw,
            f'{stage_prefix}_bisect',
            getattr(best_res, 'X_opt', None),
            getattr(best_res, 'U_opt', None),
        )
        if bool(mid_res.success):
            best_res = mid_res
            high_hw = mid_res.hardware
        else:
            low_hw = mid_hw

    return best_res


def tighten_metric_from_feasible(
    *,
    metric: str,
    joint_res,
    ref_hw: HardwarePoint,
    evaluate_fn: EvaluateFn,
    stage_prefix: str,
    search_cfg: AdaptiveSearchConfig,
):
    """
    Coordinate-wise inward tightening.

    Other 3 metrics stay fixed at the joint-feasible point.
    The target metric is reduced with accelerating multiplicative steps until the first infeasible point,
    then refined with scalar log-space bisection.
    """
    base_hw = joint_res.hardware
    ref_value = get_metric(ref_hw, metric)
    best_res = joint_res
    best_value = get_metric(best_res.hardware, metric)

    if best_value <= ref_value * (1.0 + float(search_cfg.bisect_rel_tol)):
        return best_res

    step = float(search_cfg.base_step)
    first_infeasible_value = None

    for _ in range(int(search_cfg.max_expand_iters)):
        candidate_value = max(ref_value, best_value * (1.0 - step))
        if abs(candidate_value - best_value) <= 1e-15:
            return best_res

        candidate_hw = with_metric(base_hw, metric, candidate_value)
        res = evaluate_fn(
            candidate_hw,
            f'{stage_prefix}_{metric}_expand',
            getattr(best_res, 'X_opt', None),
            getattr(best_res, 'U_opt', None),
        )
        if bool(res.success):
            best_res = res
            best_value = candidate_value
            if best_value <= ref_value * (1.0 + float(search_cfg.bisect_rel_tol)):
                return best_res
            step = _next_step(step, search_cfg.step_growth)
            continue

        first_infeasible_value = candidate_value
        break

    if first_infeasible_value is None:
        return best_res

    low_value = float(first_infeasible_value)
    high_value = float(best_value)
    base_joint_hw = base_hw

    for _ in range(int(search_cfg.max_bisect_iters)):
        if high_value <= 0.0 or low_value <= 0.0:
            rel_gap = abs(high_value - low_value)
            if rel_gap <= 1e-12:
                break
        else:
            rel_gap = high_value / low_value - 1.0
            if rel_gap <= float(search_cfg.bisect_rel_tol):
                break

        mid_value = _geometric_mean(low_value, high_value)
        if abs(mid_value - low_value) <= 1e-15 or abs(mid_value - high_value) <= 1e-15:
            break

        candidate_hw = with_metric(base_joint_hw, metric, mid_value)
        res = evaluate_fn(
            candidate_hw,
            f'{stage_prefix}_{metric}_bisect',
            getattr(best_res, 'X_opt', None),
            getattr(best_res, 'U_opt', None),
        )
        if bool(res.success):
            best_res = res
            high_value = mid_value
        else:
            low_value = mid_value

    return best_res

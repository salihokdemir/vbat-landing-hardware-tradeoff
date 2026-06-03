from __future__ import annotations

import math
import os

# Keep per-worker linear algebra thread counts low.
for _env_name in (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
):
    os.environ.setdefault(_env_name, '1')

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from config.search_params import (
    MPCFinalSearchCaps,
    build_cap_hardware,
    build_frequency_bundle,
    make_hardware_overrides_from_point,
)
from control.mpc_controller import MPCController
from dynamics.environment import SeaEnvironment
from dynamics.planar_vbat import VbatModel
from guidance.trajectory import TrajectoryPlanner
from core.hardware import HardwarePoint
from utils.adaptive_search import AdaptiveSearchConfig, hardware_margin_dict
from utils.batch_tools import (
    build_world_reference_window,
    evaluate_relative_metrics,
    max_optional,
    mean_optional,
    safe_float,
    touchdown_like_success,
)


@dataclass(frozen=True)
class FrequencySweepConfig:
    minima_csv: str
    output_dir: str = 'outputs_run_mpc_replay'
    seed: int = 42
    preview_h_s: float = 2.0
    sim_time_max_s: float = 30.0
    num_workers: int = 8
    validate_generated_scenarios: bool = True

    touchdown_x_tol: float = 1.0
    touchdown_z_tol: float = 0.10
    touchdown_vx_rel_tol: float = 1.0
    touchdown_vz_rel_min: float = -2.0
    touchdown_vz_rel_max: float = 0.5


@dataclass
class MPCSearchResult:
    success: bool
    stage: str
    hardware: HardwarePoint
    return_status: str
    failure_mode: str
    solver_success_count: int
    retry_count: int
    avg_solve_ms: Optional[float]
    max_solve_ms: Optional[float]
    max_T_usage: float
    max_delta_usage: float
    max_T_dot_usage: float
    max_delta_dot_usage: float
    final_rel_x: Optional[float]
    final_rel_z: Optional[float]
    final_rel_vx: Optional[float]
    final_rel_vz: Optional[float]
    touch_time_s: Optional[float]
    steps_simulated: int
    X_opt: Optional[np.ndarray] = None
    U_opt: Optional[np.ndarray] = None


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


class MPCFrequencySearchContext:
    """Minimal search/evaluation context used by utils.hardware_search_helpers.

    This is intentionally slimmed down from the older main3 file so the clean package
    contains only the pieces main4 actually depends on.
    """

    def __init__(
        self,
        *,
        scenario: dict,
        ref_hw: HardwarePoint,
        freq_hz: float,
        sweep_cfg: FrequencySweepConfig,
        search_cfg: AdaptiveSearchConfig,
        caps: MPCFinalSearchCaps,
        store_solution: bool = False,
    ):
        self.scenario = dict(scenario)
        self.ref_hw = ref_hw
        self.freq_hz = float(freq_hz)
        self.sweep_cfg = sweep_cfg
        self.search_cfg = search_cfg
        self.caps = caps
        self.store_solution = bool(store_solution)

        self.weight = 4.0 * 9.81
        self.cap_hw = build_cap_hardware(ref_hw, weight=self.weight, caps=caps)
        self.attempt_log: List[dict] = []
        self.cache: Dict[tuple, MPCSearchResult] = {}

        _, _, self.freq_spec = build_frequency_bundle(
            base_wind=float(self.scenario['base_wind_x']),
            frequency_hz=self.freq_hz,
            preview_h_s=self.sweep_cfg.preview_h_s,
            hardware_overrides=None,
        )
        self.N = int(self.freq_spec.N)
        self.dt = float(self.freq_spec.dt)
        self.preview_h_actual_s = float(self.freq_spec.preview_h_actual_s)
        self.sim_steps_max = int(math.ceil(float(self.sweep_cfg.sim_time_max_s) / max(self.dt, 1e-12)))

    def _log_attempt(self, result: MPCSearchResult) -> None:
        row = {
            'scenario_id': int(self.scenario['id']),
            'frequency_hz': float(self.freq_hz),
            'N': int(self.N),
            'dt': float(self.dt),
            'preview_h_s': float(self.preview_h_actual_s),
            'stage': str(result.stage),
            'success': bool(result.success),
            'return_status': str(result.return_status),
            'failure_mode': str(result.failure_mode),
            'solver_success_count': int(result.solver_success_count),
            'retry_count': int(result.retry_count),
            'avg_solve_ms': result.avg_solve_ms,
            'max_solve_ms': result.max_solve_ms,
            'touch_time_s': result.touch_time_s,
            'steps_simulated': int(result.steps_simulated),
            'final_rel_x': result.final_rel_x,
            'final_rel_z': result.final_rel_z,
            'final_rel_vx': result.final_rel_vx,
            'final_rel_vz': result.final_rel_vz,
            'usage_T': float(result.max_T_usage),
            'usage_T_dot': float(result.max_T_dot_usage),
            'usage_delta': float(result.max_delta_usage),
            'usage_delta_dot': float(result.max_delta_dot_usage),
        }
        row.update(_hardware_to_row_dict(result.hardware, 'hardware', self.weight))
        row.update(hardware_margin_dict(result.hardware, self.ref_hw, prefix='margin'))
        self.attempt_log.append(row)

    def _run_single_attempt(self, hardware: HardwarePoint, stage: str) -> MPCSearchResult:
        hardware_overrides = make_hardware_overrides_from_point(hardware)
        uav_p, mpc_p, _ = build_frequency_bundle(
            base_wind=float(self.scenario['base_wind_x']),
            frequency_hz=self.freq_hz,
            preview_h_s=self.sweep_cfg.preview_h_s,
            hardware_overrides=hardware_overrides,
        )
        model = VbatModel(uav_p)
        env = _build_env(self.scenario)
        planner = TrajectoryPlanner(dt=mpc_p.dt, profile=mpc_p)
        controller = MPCController(model, mpc_p)
        if hasattr(controller, "reset_warm_start"):
            controller.reset_warm_start()

        x_ref_rel, z_ref_rel = planner.generate_landing_curve(
            self.scenario['rel_x0'],
            self.scenario['rel_z0'],
            0.0,
            0.0,
        )
        pad_len = mpc_p.N + self.sim_steps_max + 5
        x_ref_rel_full = np.pad(x_ref_rel, (0, pad_len), 'constant', constant_values=0.0)
        z_ref_rel_full = np.pad(z_ref_rel, (0, pad_len), 'constant', constant_values=0.0)

        curr_x = np.array(
            [
                self.scenario['x0'],
                self.scenario['z0'],
                self.scenario['theta'],
                self.scenario['v_x'],
                self.scenario['v_z'],
                self.scenario['q'],
                uav_p.m * uav_p.g,
                0.0,
            ],
            dtype=float,
        )

        # Closed-loop histories are optional. main4 calls this evaluator thousands of
        # times, so keeping X/U for every search attempt would consume too much RAM.
        # main5 enables store_solution only for selected candidates.
        hist_X = [curr_x.copy()] if self.store_solution else None
        hist_U = [] if self.store_solution else None

        max_T = float(uav_p.TW_ratio_max * uav_p.m * uav_p.g)
        max_delta = float(uav_p.delta_max)
        max_T_dot = float(uav_p.T_dot_max)
        max_delta_dot = float(uav_p.delta_dot_max)
        eps = 1e-9

        solve_times_ms: List[Optional[float]] = []
        solver_success_count = 0
        retry_count = 0
        last_return_status = 'NOT_STARTED'

        max_T_seen = float(curr_x[6])
        max_delta_seen = float(abs(curr_x[7]))
        max_T_dot_seen = 0.0
        max_delta_dot_seen = 0.0

        failure_mode = 'timeout'
        touch_time_s = None
        last_metrics = evaluate_relative_metrics(curr_x, env, 0.0)

        for step in range(self.sim_steps_max):
            t = step * mpc_p.dt
            x_ref_window, z_ref_window, env_window = build_world_reference_window(
                env,
                x_ref_rel_full,
                z_ref_rel_full,
                step,
                mpc_p.dt,
                mpc_p.N,
            )

            u_opt, solve_info = controller.solve(
                curr_x,
                x_ref_window,
                z_ref_window,
                deck_vx_window=env_window['ship_vx'],
                deck_vz_window=env_window['ship_vz'],
                deck_z_window=env_window['ship_z'],
                wind_x_window=env_window['wind_x'],
            )
            solve_times_ms.append(safe_float(solve_info.get('solve_time_ms_manual'), None))
            last_return_status = str(solve_info.get('return_status', 'UNKNOWN'))
            retry_count += int(bool(solve_info.get('retry_used', False)))

            if not bool(solve_info.get('success', False)):
                failure_mode = 'mpc_solver_failure'
                break

            solver_success_count += 1
            if hist_U is not None:
                hist_U.append(np.asarray(u_opt, dtype=float).copy())
            max_T_dot_seen = max(max_T_dot_seen, float(abs(u_opt[0])))
            max_delta_dot_seen = max(max_delta_dot_seen, float(abs(u_opt[1])))
            max_T_seen = max(max_T_seen, float(curr_x[6]))
            max_delta_seen = max(max_delta_seen, float(abs(curr_x[7])))

            wind_now = env.get_wind_x(t)
            res_f = model.f_env(curr_x, u_opt, wind_now)
            curr_x = curr_x + np.array(res_f).flatten() * mpc_p.dt
            if hist_X is not None:
                hist_X.append(curr_x.copy())

            t_next = (step + 1) * mpc_p.dt
            last_metrics = evaluate_relative_metrics(curr_x, env, t_next)

            if last_metrics['rel_z'] <= self.sweep_cfg.touchdown_z_tol:
                touch_time_s = t_next
                if touchdown_like_success(
                    last_metrics,
                    x_tol=self.sweep_cfg.touchdown_x_tol,
                    z_tol=self.sweep_cfg.touchdown_z_tol,
                    vx_tol=self.sweep_cfg.touchdown_vx_rel_tol,
                    vz_rel_min=self.sweep_cfg.touchdown_vz_rel_min,
                    vz_rel_max=self.sweep_cfg.touchdown_vz_rel_max,
                ):
                    failure_mode = 'none'
                else:
                    failure_mode = 'touchdown_out_of_bounds'
                break

        success = failure_mode == 'none'
        X_hist = None
        U_hist = None
        if hist_X is not None and hist_U is not None and len(hist_X) >= 1:
            X_hist = np.array(hist_X, dtype=float).T
            U_hist = np.array(hist_U, dtype=float).T if len(hist_U) else np.zeros((model.n_controls, 0), dtype=float)

        return MPCSearchResult(
            success=bool(success),
            stage=str(stage),
            hardware=hardware,
            return_status=str(last_return_status),
            failure_mode=str(failure_mode),
            solver_success_count=int(solver_success_count),
            retry_count=int(retry_count),
            avg_solve_ms=mean_optional(solve_times_ms),
            max_solve_ms=max_optional(solve_times_ms),
            max_T_usage=float(max_T_seen / max(max_T, eps)),
            max_delta_usage=float(max_delta_seen / max(max_delta, eps)),
            max_T_dot_usage=float(max_T_dot_seen / max(max_T_dot, eps)),
            max_delta_dot_usage=float(max_delta_dot_seen / max(max_delta_dot, eps)),
            final_rel_x=None if last_metrics is None else float(last_metrics['rel_x']),
            final_rel_z=None if last_metrics is None else float(last_metrics['rel_z']),
            final_rel_vx=None if last_metrics is None else float(last_metrics['rel_vx']),
            final_rel_vz=None if last_metrics is None else float(last_metrics['rel_vz']),
            touch_time_s=touch_time_s,
            steps_simulated=int(solver_success_count),
            X_opt=X_hist,
            U_opt=U_hist,
        )

    def evaluate(self, hardware: HardwarePoint, stage: str, guess_X=None, guess_U=None) -> MPCSearchResult:
        del guess_X, guess_U
        key = hardware.key()
        if key in self.cache:
            return self.cache[key]
        res = self._run_single_attempt(hardware, stage)
        self.cache[key] = res
        self._log_attempt(res)
        return res

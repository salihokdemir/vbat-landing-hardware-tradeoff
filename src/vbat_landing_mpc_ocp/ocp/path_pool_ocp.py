from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import casadi as ca
import numpy as np

from dynamics.planar_vbat import VbatModel
from config.vehicle_params import OCPParams
from guidance.landing_profile import (
    clip_reference_velocity_to_caps,
    derive_reference_world_velocities,
    flare_blend_ca,
    flare_blend_np,
    relative_speed_caps_ca,
)

try:
    from core.hardware import HardwarePoint
except Exception:  # pragma: no cover
    @dataclass(frozen=True)
    class HardwarePoint:
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


class OCPPathPoolShooting25:
    """
    Path-pool local-refinement OCP.

    Intent
    ------
    - The centerline is the selected MPC rollout path, not the original time path.
    - The path is mostly a soft anchor; the OCP is allowed to deviate from it.
    - The landing requirements are enforced at the terminal pool.
    - By default the pool is one node, i.e. the final node. You can increase
      terminal_pool_nodes later if the MPC/OCP horizon tail is extended.

    This avoids the previous behaviour where the final state was constrained around
    the MPC final path point with a loose z tolerance. Here the final/pool reference
    is the deck itself: rel_x, rel_z, rel_vx, rel_vz are constrained directly.
    """

    def __init__(
        self,
        model: VbatModel,
        dt: float,
        N_total: int,
        ocp_p: OCPParams,
        *,
        terminal_pool_nodes: int = 1,
        path_tube_x_m: float = 2.5,
        path_tube_z_m: float = 3.0,
        W_path_tube: float = 20.0,
        hard_stage_vz_corridor: bool = False,
    ):
        self.model = model
        self.dt = float(dt)
        self.N = int(N_total)
        self.p = ocp_p
        self.terminal_pool_nodes = max(1, min(int(terminal_pool_nodes), self.N + 1))
        self.path_tube_x_m = float(path_tube_x_m)
        self.path_tube_z_m = float(path_tube_z_m)
        self.W_path_tube = float(W_path_tube)
        self.hard_stage_vz_corridor = bool(hard_stage_vz_corridor)
        self._setup_nlp()

    def _sat_penalty_expr(self, ratio):
        rho = float(getattr(self.p, 'sat_penalty_start_ratio', 0.0))
        denom = max(1.0 - rho, 1e-6)
        return ca.fmax(0.0, (ratio - rho) / denom) ** 2

    def _setup_nlp(self):
        n_x = self.model.n_states
        n_u = self.model.n_controls
        N = self.N
        n_profile = N + 1

        self.X = ca.MX.sym('X', n_x, N + 1)
        self.U = ca.MX.sym('U', n_u, N)

        # P layout:
        # [x0]
        # [x_ref_full N+1]
        # [z_ref_full N+1]
        # [deck_x_full N+1]
        # [deck_vx_full N+1]
        # [deck_z_full N+1]
        # [deck_vz_full N+1]
        # [wind_x_stage N]
        # [max_T, max_delta, max_T_dot, max_delta_dot]
        self.n_profile = n_profile
        self.off_x_ref = n_x
        self.off_z_ref = self.off_x_ref + n_profile
        self.off_deck_x = self.off_z_ref + n_profile
        self.off_deck_vx = self.off_deck_x + n_profile
        self.off_deck_z = self.off_deck_vx + n_profile
        self.off_deck_vz = self.off_deck_z + n_profile
        self.off_wind = self.off_deck_vz + n_profile
        self.off_hw = self.off_wind + N
        self.P = ca.MX.sym('P', self.off_hw + 4)

        obj = 0
        g_eq = []
        g_ineq = []

        g_eq.append(self.X[:, 0] - self.P[0:n_x])

        hover_T = float(self.model.p.m * self.model.p.g)
        max_T_p = self.P[self.off_hw + 0]
        max_delta_p = self.P[self.off_hw + 1]
        max_T_dot_p = self.P[self.off_hw + 2]
        max_delta_dot_p = self.P[self.off_hw + 3]

        for k in range(N):
            st = self.X[:, k]
            con = self.U[:, k]

            x_ref_k = self.P[self.off_x_ref + k]
            z_ref_k = self.P[self.off_z_ref + k]
            deck_vx_k = self.P[self.off_deck_vx + k]
            deck_z_k = self.P[self.off_deck_z + k]
            deck_vz_k = self.P[self.off_deck_vz + k]
            wind_x_k = self.P[self.off_wind + k]

            st_next = st + self.model.f_env(st, con, wind_x_k) * self.dt
            g_eq.append(self.X[:, k + 1] - st_next)

            x, z, theta, v_x, v_z, q, T, delta = (
                st[0], st[1], st[2], st[3], st[4], st[5], st[6], st[7]
            )
            T_dot, delta_dot = con[0], con[1]

            # Soft path anchor: OCP can deviate from the MPC path if it needs to.
            x_err = x - x_ref_k
            z_err = z - z_ref_k
            obj += float(self.p.W_track_stage) * (x_err ** 2 + z_err ** 2)
            obj += self.W_path_tube * ca.fmax(0.0, ca.fabs(x_err) - self.path_tube_x_m) ** 2
            obj += self.W_path_tube * ca.fmax(0.0, ca.fabs(z_err) - self.path_tube_z_m) ** 2
            obj += float(self.p.W_q) * (q ** 2)

            z_rel = z - deck_z_k
            vx_rel = v_x - deck_vx_k
            vz_rel = v_z - deck_vz_k
            flare_blend = flare_blend_ca(z_rel, self.p)
            obj += float(self.p.W_vx_rel_soft) * flare_blend * (vx_rel ** 2)
            obj += float(self.p.W_theta_soft) * flare_blend * (theta - ca.pi / 2.0) ** 2

            max_T_excess = ca.fmax(max_T_p - hover_T, 1e-6)
            T_excess_ratio = ca.fmax(0.0, T - hover_T) / max_T_excess
            delta_ratio = ca.fabs(delta) / ca.fmax(max_delta_p, 1e-6)
            T_dot_ratio = ca.fabs(T_dot) / ca.fmax(max_T_dot_p, 1e-6)
            delta_dot_ratio = ca.fabs(delta_dot) / ca.fmax(max_delta_dot_p, 1e-6)

            obj += float(self.p.W_sat_T_excess) * self._sat_penalty_expr(T_excess_ratio)
            obj += float(self.p.W_sat_delta) * self._sat_penalty_expr(delta_ratio)
            obj += float(self.p.W_sat_T_dot) * self._sat_penalty_expr(T_dot_ratio)
            obj += float(self.p.W_sat_delta_dot) * self._sat_penalty_expr(delta_dot_ratio)

            g_ineq.append(z_rel)  # never go below the deck during the OCP rollout
            if self.hard_stage_vz_corridor:
                _vx_cap, vz_cap = relative_speed_caps_ca(z_rel, self.p)
                g_ineq.append(vz_rel + vz_cap)

        # Terminal pool constraints. For default pool_nodes=1 this is just final node.
        self.pool_start = max(0, (N + 1) - self.terminal_pool_nodes)
        for k in range(self.pool_start, N + 1):
            st = self.X[:, k]
            deck_x_k = self.P[self.off_deck_x + k]
            deck_vx_k = self.P[self.off_deck_vx + k]
            deck_z_k = self.P[self.off_deck_z + k]
            deck_vz_k = self.P[self.off_deck_vz + k]
            rel_x = st[0] - deck_x_k
            rel_z = st[1] - deck_z_k
            rel_vx = st[3] - deck_vx_k
            rel_vz = st[4] - deck_vz_k
            g_ineq.extend([rel_x, rel_z, rel_vx, rel_vz])

        opt_variables = ca.vertcat(ca.reshape(self.X, -1, 1), ca.reshape(self.U, -1, 1))
        g_cons_final = ca.vertcat(*(g_eq + g_ineq))
        nlp_prob = {'f': obj, 'x': opt_variables, 'g': g_cons_final, 'p': self.P}
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.sb': 'yes',
            'ipopt.max_iter': int(self.p.ipopt_max_iter),
            'ipopt.tol': float(self.p.ipopt_tol),
            'ipopt.acceptable_tol': float(self.p.ipopt_acceptable_tol),
            'ipopt.warm_start_init_point': 'yes',
            'error_on_fail': False,
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    @staticmethod
    def _normalize_profile(arr: Optional[np.ndarray], expected_len: int, name: str, fill_value: float = 0.0):
        if arr is None:
            return np.full(expected_len, float(fill_value), dtype=float)
        arr = np.asarray(arr, dtype=float).flatten()
        if len(arr) == expected_len:
            return arr
        if len(arr) == expected_len - 1:
            return np.concatenate([arr, np.array([arr[-1]], dtype=float)])
        raise ValueError(f'{name} uzunluğu hatalı. Beklenen: {expected_len} veya {expected_len - 1}, gelen: {len(arr)}')

    def _build_default_guess(self, n_opt, x_ref_full, z_ref_full, deck_vx_full, deck_vz_full, deck_z_full):
        n_x = self.model.n_states
        X_guess = np.zeros((n_x, self.N + 1), dtype=float)
        U_guess = np.zeros((self.model.n_controls, self.N), dtype=float)
        vx_ref_world, vz_ref_world = derive_reference_world_velocities(x_ref_full, z_ref_full, dt=self.dt, profile=self.p)
        vx_guess_world, vz_guess_world = clip_reference_velocity_to_caps(
            vx_world=vx_ref_world,
            vz_world=vz_ref_world,
            deck_vx=deck_vx_full,
            deck_vz=deck_vz_full,
            deck_z=deck_z_full,
            z_ref_world=z_ref_full,
            profile=self.p,
        )
        hover_T = float(self.model.p.m * self.model.p.g)
        for k in range(self.N + 1):
            X_guess[0, k] = x_ref_full[k]
            X_guess[1, k] = z_ref_full[k]
            z_rel_ref = float(z_ref_full[k] - deck_z_full[k])
            flare_blend = float(flare_blend_np(z_rel_ref, self.p))
            X_guess[2, k] = (1.0 - flare_blend) * np.deg2rad(80.0) + flare_blend * (np.pi / 2.0)
            X_guess[3, k] = float(vx_guess_world[k])
            X_guess[4, k] = float(vz_guess_world[k])
            X_guess[5, k] = 0.0
            X_guess[6, k] = hover_T
            X_guess[7, k] = 0.0
        return self._pack_guess(X_guess, U_guess, n_opt)

    def _pack_guess(self, X_guess, U_guess, n_opt):
        n_x = self.model.n_states
        n_X = n_x * (self.N + 1)
        vec = np.zeros(n_opt, dtype=float)
        vec[:n_X] = np.asarray(X_guess, dtype=float).reshape(-1, order='F')
        vec[n_X:] = np.asarray(U_guess, dtype=float).reshape(-1, order='F')
        return vec

    def _pack_warm_start(self, guess_X, guess_U, n_opt, x_ref_full, z_ref_full, deck_vx_full, deck_vz_full, deck_z_full):
        if guess_X is None or guess_U is None:
            return self._build_default_guess(n_opt, x_ref_full, z_ref_full, deck_vx_full, deck_vz_full, deck_z_full)
        guess_X = np.asarray(guess_X, dtype=float)
        guess_U = np.asarray(guess_U, dtype=float)
        if guess_X.shape != (self.model.n_states, self.N + 1):
            raise ValueError(f'Warm-start X shape uyumsuz. Beklenen: {(self.model.n_states, self.N + 1)}, gelen: {guess_X.shape}')
        if guess_U.shape != (self.model.n_controls, self.N):
            raise ValueError(f'Warm-start U shape uyumsuz. Beklenen: {(self.model.n_controls, self.N)}, gelen: {guess_U.shape}')
        return self._pack_guess(guess_X, guess_U, n_opt)

    def _unpack_solution(self, sol_x, n_X, n_x, n_u):
        sol_vec = sol_x.full().flatten()
        X_out = sol_vec[:n_X].reshape((n_x, self.N + 1), order='F')
        U_out = sol_vec[n_X:].reshape((n_u, self.N), order='F')
        return X_out, U_out

    @staticmethod
    def _to_float_or_none(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _solve_once(self, x0_guess, lbx, ubx, lbg, ubg, p_val):
        tic = time.perf_counter()
        sol = self.solver(x0=x0_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p_val)
        solve_time_s = time.perf_counter() - tic
        stats = self.solver.stats()
        return sol, stats, solve_time_s

    def solve(
        self,
        x0,
        x_ref_full,
        z_ref_full,
        hardware: HardwarePoint,
        deck_x_full=None,
        deck_vx_full=None,
        deck_vz_full=None,
        deck_z_full=None,
        wind_x_stage=None,
        guess_X=None,
        guess_U=None,
    ):
        n_x = self.model.n_states
        n_u = self.model.n_controls
        N = self.N
        n_profile = N + 1

        x0 = np.asarray(x0, dtype=float).flatten()
        x_ref_full = np.asarray(x_ref_full, dtype=float).flatten()
        z_ref_full = np.asarray(z_ref_full, dtype=float).flatten()
        if x0.shape != (n_x,):
            raise ValueError(f'x0 boyutu hatalı. Beklenen: ({n_x},), gelen: {x0.shape}')
        if len(x_ref_full) != n_profile:
            raise ValueError(f'x_ref_full uzunluğu hatalı. Beklenen: {n_profile}, gelen: {len(x_ref_full)}')
        if len(z_ref_full) != n_profile:
            raise ValueError(f'z_ref_full uzunluğu hatalı. Beklenen: {n_profile}, gelen: {len(z_ref_full)}')

        deck_x_full = self._normalize_profile(deck_x_full, n_profile, 'deck_x_full', 0.0)
        deck_vx_full = self._normalize_profile(deck_vx_full, n_profile, 'deck_vx_full', 0.0)
        deck_vz_full = self._normalize_profile(deck_vz_full, n_profile, 'deck_vz_full', 0.0)
        deck_z_full = self._normalize_profile(deck_z_full, n_profile, 'deck_z_full', 0.0)
        if wind_x_stage is None:
            wind_x_stage = np.full(N, float(self.model.p.V_wind), dtype=float)
        else:
            wind_x_stage = np.asarray(wind_x_stage, dtype=float).flatten()
            if len(wind_x_stage) == n_profile:
                wind_x_stage = wind_x_stage[:-1]
            if len(wind_x_stage) != N:
                raise ValueError(f'wind_x_stage uzunluğu hatalı. Beklenen: {N} veya {n_profile}, gelen: {len(wind_x_stage)}')

        max_T = float(hardware.tw_ratio) * float(self.model.p.m * self.model.p.g)
        max_delta = float(hardware.delta_max)
        max_T_dot = float(hardware.T_dot_max)
        max_delta_dot = float(hardware.delta_dot_max)

        n_opt = n_x * n_profile + n_u * N
        lbx = -np.inf * np.ones(n_opt)
        ubx = np.inf * np.ones(n_opt)
        n_X = n_x * n_profile

        theta_min = float(self.p.theta_min_deg) * np.pi / 180.0
        theta_max = float(self.p.theta_max_deg) * np.pi / 180.0
        for k in range(n_profile):
            idx = k * n_x
            lbx[idx + 2] = theta_min
            ubx[idx + 2] = theta_max
            lbx[idx + 4] = deck_vz_full[k] - float(self.p.stage_vz_lower_margin)
            ubx[idx + 4] = deck_vz_full[k] + float(self.p.stage_vz_upper_margin)
            lbx[idx + 6] = 0.0
            ubx[idx + 6] = max_T
            lbx[idx + 7] = -max_delta
            ubx[idx + 7] = max_delta

        for k in range(N):
            idx = n_X + k * n_u
            lbx[idx + 0] = -max_T_dot
            ubx[idx + 0] = max_T_dot
            lbx[idx + 1] = -max_delta_dot
            ubx[idx + 1] = max_delta_dot

        n_eq = n_x * n_profile
        n_stage_pos = N
        n_stage_vz = N if self.hard_stage_vz_corridor else 0
        n_pool = 4 * self.terminal_pool_nodes
        n_cons = n_eq + n_stage_pos + n_stage_vz + n_pool
        lbg = np.zeros(n_cons)
        ubg = np.zeros(n_cons)
        cursor = n_eq
        for _ in range(n_stage_pos):
            lbg[cursor] = 0.0
            ubg[cursor] = np.inf
            cursor += 1
        for _ in range(n_stage_vz):
            lbg[cursor] = 0.0
            ubg[cursor] = np.inf
            cursor += 1

        # pool order per node: rel_x, rel_z, rel_vx, rel_vz
        x_tol = float(self.p.terminal_x_tol_m)
        z_tol = float(self.p.terminal_z_above_tol_m)
        vx_tol = float(self.p.terminal_vx_rel_tol)
        vz_min = float(self.p.terminal_vz_rel_min)
        vz_max = float(self.p.terminal_vz_rel_max)
        for _ in range(self.terminal_pool_nodes):
            lbg[cursor + 0] = -x_tol
            ubg[cursor + 0] = x_tol
            lbg[cursor + 1] = 0.0
            ubg[cursor + 1] = z_tol
            lbg[cursor + 2] = -vx_tol
            ubg[cursor + 2] = vx_tol
            lbg[cursor + 3] = vz_min
            ubg[cursor + 3] = vz_max
            cursor += 4

        p_val = np.concatenate([
            x0,
            x_ref_full,
            z_ref_full,
            deck_x_full,
            deck_vx_full,
            deck_z_full,
            deck_vz_full,
            wind_x_stage,
            np.array([max_T, max_delta, max_T_dot, max_delta_dot], dtype=float),
        ])

        x0_guess = self._pack_warm_start(guess_X, guess_U, n_opt, x_ref_full, z_ref_full, deck_vx_full, deck_vz_full, deck_z_full)
        sol, stats, solve_time_s = self._solve_once(x0_guess, lbx, ubx, lbg, ubg, p_val)
        is_success = bool(stats.get('success', False))
        retry_used = False
        retry_status = None
        if (not is_success) and bool(getattr(self.p, 'retry_on_fail', True)) and bool(getattr(self.p, 'retry_with_default_guess', True)):
            retry_used = True
            default_guess = self._build_default_guess(n_opt, x_ref_full, z_ref_full, deck_vx_full, deck_vz_full, deck_z_full)
            sol_retry, stats_retry, solve_time_retry_s = self._solve_once(default_guess, lbx, ubx, lbg, ubg, p_val)
            retry_status = str(stats_retry.get('return_status', 'UNKNOWN'))
            solve_time_s += solve_time_retry_s
            if bool(stats_retry.get('success', False)):
                sol = sol_retry
                stats = stats_retry
                is_success = True

        X_out, U_out = self._unpack_solution(sol['x'], n_X, n_x, n_u)
        eps = 1e-9
        final_rel_x = float(X_out[0, -1] - deck_x_full[-1])
        final_rel_z = float(X_out[1, -1] - deck_z_full[-1])
        final_rel_vx = float(X_out[3, -1] - deck_vx_full[-1])
        final_rel_vz = float(X_out[4, -1] - deck_vz_full[-1])
        summary = {
            'success': is_success,
            'return_status': str(stats.get('return_status', 'UNKNOWN')),
            'retry_used': bool(retry_used),
            'retry_status': retry_status,
            'iter_count': int(stats.get('iter_count', -1)) if 'iter_count' in stats else None,
            't_wall_total': self._to_float_or_none(stats.get('t_wall_total', None)),
            't_proc_total': self._to_float_or_none(stats.get('t_proc_total', None)),
            'solve_time_ms_manual': 1000.0 * float(solve_time_s),
            'max_T_usage': float(np.max(X_out[6, :]) / max(max_T, eps)),
            'max_delta_usage': float(np.max(np.abs(X_out[7, :])) / max(max_delta, eps)),
            'max_T_dot_usage': float(np.max(np.abs(U_out[0, :])) / max(max_T_dot, eps)),
            'max_delta_dot_usage': float(np.max(np.abs(U_out[1, :])) / max(max_delta_dot, eps)),
            'final_rel_x': final_rel_x,
            'final_rel_z': final_rel_z,
            'final_rel_vx': final_rel_vx,
            'final_rel_vz': final_rel_vz,
            'final_theta_deg': float(np.degrees(X_out[2, -1])),
            'max_abs_wind_x': float(np.max(np.abs(wind_x_stage))) if len(wind_x_stage) else 0.0,
            'max_abs_q': float(np.max(np.abs(X_out[5, :]))),
            'max_theta_deg': float(np.max(np.degrees(X_out[2, :]))) if X_out.shape[1] else 0.0,
            'min_theta_deg': float(np.min(np.degrees(X_out[2, :]))) if X_out.shape[1] else 0.0,
            'terminal_pool_nodes': int(self.terminal_pool_nodes),
            'path_tube_x_m': float(self.path_tube_x_m),
            'path_tube_z_m': float(self.path_tube_z_m),
            'hard_stage_vz_corridor': bool(self.hard_stage_vz_corridor),
        }
        return X_out, U_out, is_success, summary

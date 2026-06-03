import time
from typing import Optional, Tuple

import casadi as ca
import numpy as np

from config.search_params import MPCParams
from dynamics.planar_vbat import VbatModel
from guidance.landing_profile import (
    clip_reference_velocity_to_caps,
    derive_reference_world_velocities,
    flare_blend_ca,
    flare_blend_np,
    relative_speed_caps_ca,
)


class MPCController:
    """
    Copy of the active stabilized_v1 MPC controller for frequency / subset search.

    This version keeps the user's intended role split:
    - MPC uses the available hardware aggressively to answer feasibility
    - flare timing lives inside the x-z reference and matching hard corridors
    - actuator-avoidance costs are removed, q regularization remains
    """

    def __init__(self, model: VbatModel, mpc_p: MPCParams):
        self.model = model
        self.p = mpc_p
        self.last_solution_vec: Optional[np.ndarray] = None
        self.last_u = np.zeros(self.model.n_controls, dtype=float)

        self._setup_bounds()
        self._setup_nlp()

    @staticmethod
    def _to_float_or_none(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _setup_nlp(self):
        N = self.p.N
        n_x = self.model.n_states
        n_u = self.model.n_controls

        self.X = ca.MX.sym('X', n_x, N + 1)
        self.U = ca.MX.sym('U', n_u, N)

        # P layout:
        # [current_x (n_x)]
        # [x_ref (N)]
        # [z_ref (N)]
        # [deck_vx (N)]
        # [deck_vz (N)]
        # [deck_z (N)]
        # [wind_x (N)]
        self.P = ca.MX.sym('P', n_x + 6 * N)

        obj = 0
        g_eq = []
        g_ineq = []

        g_eq.append(self.X[:, 0] - self.P[0:n_x])

        theta_min = float(self.p.theta_min_deg) * np.pi / 180.0
        theta_max = float(self.p.theta_max_deg) * np.pi / 180.0

        for k in range(N):
            st = self.X[:, k]
            con = self.U[:, k]

            x_ref_k = self.P[n_x + k]
            z_ref_k = self.P[n_x + N + k]
            deck_vx_k = self.P[n_x + 2 * N + k]
            deck_vz_k = self.P[n_x + 3 * N + k]
            deck_z_k = self.P[n_x + 4 * N + k]
            wind_x_k = self.P[n_x + 5 * N + k]

            st_next = st + (self.model.f_env(st, con, wind_x_k) * self.p.dt)
            g_eq.append(self.X[:, k + 1] - st_next)

            x, z, theta, v_x, v_z, q, T, delta = (
                st[0], st[1], st[2], st[3], st[4], st[5], st[6], st[7]
            )
            _T_dot, _delta_dot = con[0], con[1]

            obj += self.p.W_track * ((x - x_ref_k) ** 2 + (z - z_ref_k) ** 2)

            v_x_rel = v_x - deck_vx_k
            v_z_rel = v_z - deck_vz_k
            z_rel = z - deck_z_k

            flare_blend = flare_blend_ca(z_rel, self.p)
            obj += self.p.W_theta * flare_blend * (theta - (ca.pi / 2.0)) ** 2
            obj += self.p.W_q * (q ** 2)

            obj += self.p.W_pitch_limit * ca.fmax(0, theta_min - theta) ** 2
            obj += self.p.W_pitch_limit * ca.fmax(0, theta - theta_max) ** 2

            _vx_cap, vz_cap = relative_speed_caps_ca(z_rel, self.p)

            g_ineq.append(z_rel)

            funnel_limit = ca.if_else(
                z_rel > 10.0,
                self.p.funnel_high_base_m + (z_rel - 10.0) * self.p.funnel_high_slope,
                ca.if_else(
                    z_rel >= 3.0,
                    self.p.funnel_mid_m + (z_rel - 3.0) * self.p.funnel_mid_slope,
                    self.p.funnel_low_m,
                ),
            )
            x_error = x - x_ref_k
            g_ineq.append(x_error - funnel_limit)
            g_ineq.append(x_error + funnel_limit)
            # Important: do NOT impose the hard descent-rate corridor on the measured current state (k==0).
            # If the real state slips slightly below the corridor because of model mismatch / integration / the
            # previous saturated move, then X[:,0] is fixed by equality and the whole NLP becomes instantly
            # infeasible even though the controller could recover in the next move. Keep the corridor hard for
            # predicted future states only.
            if k == 0:
                g_ineq.append(1.0)
            else:
                g_ineq.append(v_z_rel + vz_cap)

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
            'ipopt.warm_start_init_point': 'no',
            #'ipopt.bound_push': 1e-8,
            #'ipopt.bound_frac': 1e-8,
            'error_on_fail': False,
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    def _setup_bounds(self):
        n_x = self.model.n_states
        n_u = self.model.n_controls
        N = self.p.N

        max_T = self.model.p.TW_ratio_max * self.model.p.m * self.model.p.g
        max_delta = self.model.p.delta_max
        max_T_dot = self.model.p.T_dot_max
        max_delta_dot = self.model.p.delta_dot_max

        n_opt = n_x * (N + 1) + n_u * N
        self.lbx = -ca.inf * np.ones(n_opt)
        self.ubx = ca.inf * np.ones(n_opt)
        n_X = n_x * (N + 1)

        for k in range(N + 1):
            idx = k * n_x
            self.lbx[idx + 6] = 0.0
            self.ubx[idx + 6] = max_T
            self.lbx[idx + 7] = -max_delta
            self.ubx[idx + 7] = max_delta

        for k in range(N):
            idx = n_X + k * n_u
            self.lbx[idx + 0] = -max_T_dot
            self.ubx[idx + 0] = max_T_dot
            self.lbx[idx + 1] = -max_delta_dot
            self.ubx[idx + 1] = max_delta_dot

        n_eq = n_x * (N + 1)
        n_ineq = 4 * N
        self.lbg = np.zeros(n_eq + n_ineq)
        self.ubg = np.zeros(n_eq + n_ineq)

        for i in range(N):
            idx = n_eq + i * 4
            self.lbg[idx] = 0.0
            self.ubg[idx] = ca.inf
            self.lbg[idx + 1] = -ca.inf
            self.ubg[idx + 1] = 0.0
            self.lbg[idx + 2] = 0.0
            self.ubg[idx + 2] = ca.inf
            self.lbg[idx + 3] = 0.0
            self.ubg[idx + 3] = ca.inf

    def _pack_guess(self, X_guess: np.ndarray, U_guess: np.ndarray) -> np.ndarray:
        n_x = self.model.n_states
        n_u = self.model.n_controls
        n_X = n_x * (self.p.N + 1)
        n_opt = n_X + n_u * self.p.N

        vec = np.zeros(n_opt, dtype=float)
        vec[:n_X] = X_guess.reshape(-1, order='F')
        vec[n_X:] = U_guess.reshape(-1, order='F')
        return vec

    def _build_default_guess(
        self,
        current_x: np.ndarray,
        x_ref_window: np.ndarray,
        z_ref_window: np.ndarray,
        deck_vx_window: np.ndarray,
        deck_vz_window: np.ndarray,
        deck_z_window: np.ndarray,
    ) -> np.ndarray:
        n_x = self.model.n_states
        n_u = self.model.n_controls
        N = self.p.N

        X_guess = np.zeros((n_x, N + 1), dtype=float)
        U_guess = np.zeros((n_u, N), dtype=float)

        x_nodes = np.concatenate([np.asarray(x_ref_window, dtype=float).flatten(), np.array([float(x_ref_window[-1])])])
        z_nodes = np.concatenate([np.asarray(z_ref_window, dtype=float).flatten(), np.array([float(z_ref_window[-1])])])
        deck_vx_nodes = np.concatenate([np.asarray(deck_vx_window, dtype=float).flatten(), np.array([float(deck_vx_window[-1])])])
        deck_vz_nodes = np.concatenate([np.asarray(deck_vz_window, dtype=float).flatten(), np.array([float(deck_vz_window[-1])])])
        deck_z_nodes = np.concatenate([np.asarray(deck_z_window, dtype=float).flatten(), np.array([float(deck_z_window[-1])])])

        vx_ref_world, vz_ref_world = derive_reference_world_velocities(
            x_nodes,
            z_nodes,
            dt=float(self.p.dt),
            profile=self.p,
        )
        vx_guess_world, vz_guess_world = clip_reference_velocity_to_caps(
            vx_world=vx_ref_world,
            vz_world=vz_ref_world,
            deck_vx=deck_vx_nodes,
            deck_vz=deck_vz_nodes,
            deck_z=deck_z_nodes,
            z_ref_world=z_nodes,
            profile=self.p,
        )

        X_guess[:, 0] = current_x
        max_T = float(self.model.p.TW_ratio_max * self.model.p.m * self.model.p.g)
        for k in range(1, N + 1):
            node_idx = min(k, N)
            X_guess[0, k] = x_nodes[node_idx]
            X_guess[1, k] = z_nodes[node_idx]
            z_rel_ref = float(z_nodes[node_idx] - deck_z_nodes[node_idx])
            flare_blend = float(flare_blend_np(z_rel_ref, self.p))
            X_guess[2, k] = (1.0 - flare_blend) * float(current_x[2]) + flare_blend * (np.pi / 2.0)
            X_guess[3, k] = float(vx_guess_world[node_idx])
            X_guess[4, k] = float(vz_guess_world[node_idx])
            X_guess[5, k] = 0.0
            X_guess[6, k] = float(np.clip(current_x[6], 0.0, max_T))
            X_guess[7, k] = 0.0
        return self._pack_guess(X_guess, U_guess)

    def _shift_last_solution(self, current_x: np.ndarray) -> Optional[np.ndarray]:
        if self.last_solution_vec is None:
            return None

        n_x = self.model.n_states
        n_u = self.model.n_controls
        N = self.p.N
        n_X = n_x * (N + 1)

        try:
            sol_vec = np.asarray(self.last_solution_vec, dtype=float).flatten()
            X_prev = sol_vec[:n_X].reshape((n_x, N + 1), order='F')
            U_prev = sol_vec[n_X:].reshape((n_u, N), order='F')
        except Exception:
            return None

        X_guess = np.zeros_like(X_prev)
        U_guess = np.zeros_like(U_prev)
        X_guess[:, 0] = current_x
        if N > 1:
            X_guess[:, 1:N] = X_prev[:, 2:N + 1]
        X_guess[:, N] = X_prev[:, N]
        if N > 1:
            U_guess[:, 0:N - 1] = U_prev[:, 1:N]
        U_guess[:, N - 1] = U_prev[:, N - 1]
        return self._pack_guess(X_guess, U_guess)

    def _solve_once(self, p_val: np.ndarray, x0_guess: np.ndarray):
        tic = time.perf_counter()
        sol = self.solver(
            x0=x0_guess,
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=self.lbg,
            ubg=self.ubg,
            p=p_val,
        )
        solve_time_s = time.perf_counter() - tic
        stats = self.solver.stats()
        return sol, stats, solve_time_s

    def solve(
        self,
        current_x,
        x_ref_window,
        z_ref_window,
        deck_vx_window=None,
        deck_vz_window=None,
        deck_z_window=None,
        wind_x_window=None,
    ) -> Tuple[np.ndarray, dict]:
        current_x = np.asarray(current_x, dtype=float).flatten()
        x_ref_window = np.asarray(x_ref_window, dtype=float).flatten()
        z_ref_window = np.asarray(z_ref_window, dtype=float).flatten()

        if current_x.shape != (self.model.n_states,):
            raise ValueError(
                f'current_x boyutu hatalı. Beklenen: ({self.model.n_states},), gelen: {current_x.shape}'
            )
        if x_ref_window.shape != (self.p.N,):
            raise ValueError(f'x_ref_window boyutu hatalı. Beklenen: ({self.p.N},), gelen: {x_ref_window.shape}')
        if z_ref_window.shape != (self.p.N,):
            raise ValueError(f'z_ref_window boyutu hatalı. Beklenen: ({self.p.N},), gelen: {z_ref_window.shape}')

        if deck_vx_window is None:
            deck_vx_window = np.zeros(self.p.N, dtype=float)
        else:
            deck_vx_window = np.asarray(deck_vx_window, dtype=float).flatten()

        if deck_vz_window is None:
            deck_vz_window = np.zeros(self.p.N, dtype=float)
        else:
            deck_vz_window = np.asarray(deck_vz_window, dtype=float).flatten()

        if deck_z_window is None:
            deck_z_window = np.zeros(self.p.N, dtype=float)
        else:
            deck_z_window = np.asarray(deck_z_window, dtype=float).flatten()

        if wind_x_window is None:
            wind_x_window = np.full(self.p.N, float(self.model.p.V_wind), dtype=float)
        else:
            wind_x_window = np.asarray(wind_x_window, dtype=float).flatten()

        for name, arr in (
            ('deck_vx_window', deck_vx_window),
            ('deck_vz_window', deck_vz_window),
            ('deck_z_window', deck_z_window),
            ('wind_x_window', wind_x_window),
        ):
            if arr.shape != (self.p.N,):
                raise ValueError(f'{name} boyutu hatalı. Beklenen: ({self.p.N},), gelen: {arr.shape}')

        p_val = np.concatenate(
            [current_x, x_ref_window, z_ref_window, deck_vx_window, deck_vz_window, deck_z_window, wind_x_window]
        )

        # Strict no-warm-start MPC replay:
        # Always build the same default initial guess from the current state and reference window.
        # Do NOT use the previous MPC solution as raw or shifted x0.
        #
        # Note: IPOPT option 'ipopt.warm_start_init_point=no' only disables IPOPT's own
        # warm-start mode. It does not prevent us from passing a previous solution as x0.
        # This block removes that x0-level continuation effect completely.
        guess_kind = 'default_cold_no_internal_ws'
        x0_guess = self._build_default_guess(
            current_x,
            x_ref_window,
            z_ref_window,
            deck_vx_window,
            deck_vz_window,
            deck_z_window,
        )

        sol, stats, solve_time_s = self._solve_once(p_val, x0_guess)
        is_success = bool(stats.get('success', False))
        retry_used = False
        retry_status = None

        if (not is_success) and self.p.retry_on_fail and self.p.retry_with_default_guess:
            retry_used = True
            default_guess = self._build_default_guess(
                current_x,
                x_ref_window,
                z_ref_window,
                deck_vx_window,
                deck_vz_window,
                deck_z_window,
            )
            sol_retry, stats_retry, solve_time_retry_s = self._solve_once(p_val, default_guess)
            retry_status = str(stats_retry.get('return_status', 'UNKNOWN'))
            if bool(stats_retry.get('success', False)):
                sol = sol_retry
                stats = stats_retry
                solve_time_s += solve_time_retry_s
                is_success = True
                guess_kind = 'default_retry'
            else:
                solve_time_s += solve_time_retry_s

        n_X = self.model.n_states * (self.p.N + 1)
        try:
            sol_vec = sol['x'].full().flatten()
            # Strict no-warm-start mode: do not store a solution for the next MPC step.
            # Keeping this None guarantees that a future edit cannot accidentally reuse it.
            self.last_solution_vec = None
            u_out = sol_vec[n_X:n_X + self.model.n_controls]
        except Exception:
            # On failed extraction do not fall back to the previous control; this avoids
            # hidden carry-over behaviour. The rollout caller stops on solver failure anyway.
            u_out = np.zeros(self.model.n_controls, dtype=float)

        self.last_u = np.asarray(u_out, dtype=float).copy()

        summary = {
            'success': is_success,
            'return_status': str(stats.get('return_status', 'UNKNOWN')),
            'retry_used': bool(retry_used),
            'retry_status': retry_status,
            'iter_count': int(stats.get('iter_count', -1)) if 'iter_count' in stats else None,
            't_wall_total': self._to_float_or_none(stats.get('t_wall_total', None)),
            't_proc_total': self._to_float_or_none(stats.get('t_proc_total', None)),
            'solve_time_ms_manual': 1000.0 * float(solve_time_s),
            'guess_kind': guess_kind,
        }
        return np.asarray(u_out, dtype=float), summary

from __future__ import annotations

import casadi as ca

from config.vehicle_params import UAVParams


class VbatModel:
    """
    3-DOF simplified V-BAT model.

    Notes
    -----
    - State  : [x, z, theta, v_x, v_z, q, T, delta]
    - Control: [T_dot, delta_dot]
    - `f_env` uses a time-varying wind sample provided by the environment.
    - `f` keeps the legacy fixed-wind call shape for convenience.
    """

    def __init__(self, params: UAVParams):
        self.p = params
        self._build_model()

    def _build_model(self) -> None:
        self.x_pos = ca.MX.sym('x_pos')
        self.z_pos = ca.MX.sym('z_pos')
        self.theta = ca.MX.sym('theta')
        self.v_x = ca.MX.sym('v_x')
        self.v_z = ca.MX.sym('v_z')
        self.q = ca.MX.sym('q')
        self.T = ca.MX.sym('T')
        self.delta = ca.MX.sym('delta')

        self.states = ca.vertcat(
            self.x_pos,
            self.z_pos,
            self.theta,
            self.v_x,
            self.v_z,
            self.q,
            self.T,
            self.delta,
        )
        self.n_states = int(self.states.size1())

        self.T_dot = ca.MX.sym('T_dot')
        self.delta_dot = ca.MX.sym('delta_dot')
        self.controls = ca.vertcat(self.T_dot, self.delta_dot)
        self.n_controls = int(self.controls.size1())

        wind_x = ca.MX.sym('wind_x')

        # Signed quadratic wind load. The old fixed-wind model used V_wind**2;
        # here the sign is preserved because harsh-env scenarios explicitly vary wind direction.
        F_wind_x = 0.5 * self.p.rho * wind_x * ca.fabs(wind_x) * self.p.S_wing * self.p.C_D90 * ca.sin(self.theta)
        M_wind = F_wind_x * self.p.L_cp

        rhs = ca.vertcat(
            self.v_x,
            self.v_z,
            self.q,
            (self.T * ca.cos(self.theta + self.delta) + F_wind_x) / self.p.m,
            (self.T * ca.sin(self.theta + self.delta)) / self.p.m - self.p.g,
            (-self.T * ca.sin(self.delta) * self.p.L_cg_prop + M_wind) / self.p.I_yy,
            self.T_dot,
            self.delta_dot,
        )

        self.f_env = ca.Function('f_env', [self.states, self.controls, wind_x], [rhs])
        self.f = ca.Function('f', [self.states, self.controls], [self.f_env(self.states, self.controls, self.p.V_wind)])

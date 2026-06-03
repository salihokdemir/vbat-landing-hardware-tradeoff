from __future__ import annotations

import numpy as np

from guidance.landing_profile import LandingProfileParams, relative_speed_caps_np


class TrajectoryPlanner:
    def __init__(self, dt: float, profile: LandingProfileParams | None = None):
        self.dt = float(dt)
        self.profile = profile or LandingProfileParams()

    def _build_componentwise_dt(self, x_path: np.ndarray, z_path: np.ndarray) -> np.ndarray:
        dx = np.diff(x_path)
        dz = np.diff(z_path)
        z_mid = 0.5 * (z_path[:-1] + z_path[1:])
        vx_cap, vz_cap = relative_speed_caps_np(z_mid, self.profile)
        dt_from_x = np.abs(dx) / np.maximum(vx_cap, 1e-6)
        dt_from_z = np.abs(dz) / np.maximum(vz_cap, 1e-6)
        dt_steps = np.maximum(dt_from_x, dt_from_z)
        return np.maximum(dt_steps, 1e-4)

    def generate_landing_curve(self, x0: float, z0: float, x_f: float, z_f: float):
        P0 = np.array([x0, z0], dtype=float)
        vert_dist = float(z0 - z_f)
        P1 = np.array([x0 * 0.5, z0 - vert_dist * 0.2], dtype=float)
        P2 = np.array([x_f, z_f + vert_dist * 0.5], dtype=float)
        P3 = np.array([x_f, z_f], dtype=float)

        t_bez = np.linspace(0.0, 1.0, 600)
        x_bez = (
            (1 - t_bez) ** 3 * P0[0]
            + 3 * (1 - t_bez) ** 2 * t_bez * P1[0]
            + 3 * (1 - t_bez) * t_bez ** 2 * P2[0]
            + t_bez ** 3 * P3[0]
        )
        z_bez = (
            (1 - t_bez) ** 3 * P0[1]
            + 3 * (1 - t_bez) ** 2 * t_bez * P1[1]
            + 3 * (1 - t_bez) * t_bez ** 2 * P2[1]
            + t_bez ** 3 * P3[1]
        )

        dt_steps = self._build_componentwise_dt(x_bez, z_bez)
        t_curve = np.insert(np.cumsum(dt_steps), 0, 0.0)
        t_total = float(t_curve[-1])
        if t_total <= 0.0:
            return np.array([x_f], dtype=float), np.array([z_f], dtype=float)

        t_uniform = np.arange(0.0, t_total + 0.5 * self.dt, self.dt)
        x_ref = np.interp(t_uniform, t_curve, x_bez)
        z_ref = np.interp(t_uniform, t_curve, z_bez)
        x_ref[-1] = float(x_f)
        z_ref[-1] = float(z_f)
        return x_ref, z_ref

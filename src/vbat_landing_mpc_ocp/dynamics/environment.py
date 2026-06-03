from __future__ import annotations

import numpy as np


class SeaEnvironment:
    """
    Harsh-env helper.

    Supported effects:
    - moving deck in X with constant speed
    - optional deck heave in Z
    - time-varying X-wind with two sinusoidal gust components
    """

    def __init__(
        self,
        *,
        base_wind_x: float = 0.0,
        wind_gust_amp_1: float = 0.0,
        wind_gust_freq_1: float = 0.0,
        wind_gust_phase_1: float = 0.0,
        wind_gust_amp_2: float = 0.0,
        wind_gust_freq_2: float = 0.0,
        wind_gust_phase_2: float = 0.0,
        deck_x0: float = 0.0,
        deck_vx: float = 0.0,
        deck_z0: float = 0.0,
        deck_z_amp: float = 0.0,
        deck_z_freq: float = 0.0,
        deck_z_phase: float = 0.0,
    ):
        self.base_wind_x = float(base_wind_x)
        self.wind_gust_amp_1 = float(wind_gust_amp_1)
        self.wind_gust_freq_1 = float(wind_gust_freq_1)
        self.wind_gust_phase_1 = float(wind_gust_phase_1)
        self.wind_gust_amp_2 = float(wind_gust_amp_2)
        self.wind_gust_freq_2 = float(wind_gust_freq_2)
        self.wind_gust_phase_2 = float(wind_gust_phase_2)

        self.deck_x0 = float(deck_x0)
        self.deck_vx = float(deck_vx)
        self.deck_z0 = float(deck_z0)
        self.deck_z_amp = float(deck_z_amp)
        self.deck_z_freq = float(deck_z_freq)
        self.deck_z_phase = float(deck_z_phase)

    def get_ship_x(self, t: float) -> float:
        return self.deck_x0 + self.deck_vx * float(t)

    def get_ship_vx(self, t: float) -> float:
        _ = t
        return self.deck_vx

    def get_ship_z(self, t: float) -> float:
        t = float(t)
        return self.deck_z0 + self.deck_z_amp * np.sin(self.deck_z_freq * t + self.deck_z_phase)

    def get_ship_vz(self, t: float) -> float:
        t = float(t)
        return self.deck_z_amp * self.deck_z_freq * np.cos(self.deck_z_freq * t + self.deck_z_phase)

    def get_wind_x(self, t: float) -> float:
        t = float(t)
        return (
            self.base_wind_x
            + self.wind_gust_amp_1 * np.sin(self.wind_gust_freq_1 * t + self.wind_gust_phase_1)
            + self.wind_gust_amp_2 * np.sin(self.wind_gust_freq_2 * t + self.wind_gust_phase_2)
        )

    def sample_horizon(self, t0: float, dt: float, n: int):
        if n <= 0:
            raise ValueError('n must be positive.')
        t = float(t0) + float(dt) * np.arange(int(n), dtype=float)
        return {
            't': t,
            'ship_x': np.array([self.get_ship_x(tt) for tt in t], dtype=float),
            'ship_z': np.array([self.get_ship_z(tt) for tt in t], dtype=float),
            'ship_vx': np.array([self.get_ship_vx(tt) for tt in t], dtype=float),
            'ship_vz': np.array([self.get_ship_vz(tt) for tt in t], dtype=float),
            'wind_x': np.array([self.get_wind_x(tt) for tt in t], dtype=float),
        }

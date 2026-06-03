from __future__ import annotations

import math
from typing import List

import numpy as np


def generate_scenarios(
    num_scenarios: int = 20,
    seed: int = 42,
    *,
    enable_deck_heave: bool = True,
    deck_z_amp_range: tuple[float, float] = (0.20, 0.45),
    deck_z_freq_range: tuple[float, float] = (0.18, 0.35),
) -> List[dict]:
    """
    Harsh-env scenario generator.

    Design intent:
    - deck moves in X with constant speed
    - default batch now uses a *calm* deck-heave profile in Z
    - wind is time-varying in X through two sinusoidal gust terms
    - initial conditions are created in deck-relative frame, then shifted to world

    Deck-heave note
    ----------------
    This version intentionally avoids a rough-sea profile.
    The default Z motion is a gentle sine wave with small amplitude and low frequency,
    so touchdown stays challenging but not artificially violent.
    """
    rng = np.random.default_rng(seed)
    scenarios = []

    for i in range(int(num_scenarios)):
        deck_x0 = 0.0
        deck_vx = float(rng.uniform(-1.5, 1.5))

        # Gentle deck-heave in Z.
        # We keep the wave calm on purpose: small amplitude, low frequency.
        if enable_deck_heave:
            deck_z0 = 0.0
            deck_z_amp = float(rng.uniform(*deck_z_amp_range))
            deck_z_freq = float(rng.uniform(*deck_z_freq_range))
            deck_z_phase = float(rng.uniform(0.0, 2.0 * math.pi))
        else:
            deck_z0 = 0.0
            deck_z_amp = 0.0
            deck_z_freq = 0.0
            deck_z_phase = 0.0

        rel_x0 = float(rng.uniform(-55.0, -25.0))
        rel_z0 = float(rng.uniform(35.0, 70.0))
        v_x_rel0 = float(rng.uniform(-1.5, 1.5))
        v_z = float(rng.uniform(-2.5, -0.2))

        base_wind_x = float(rng.uniform(-6.0, 10.0))
        wind_gust_amp_1 = float(rng.uniform(0.0, 4.0))
        wind_gust_freq_1 = float(rng.uniform(0.2, 1.2))
        wind_gust_phase_1 = float(rng.uniform(0.0, 2.0 * math.pi))
        wind_gust_amp_2 = float(rng.uniform(0.0, 2.0))
        wind_gust_freq_2 = float(rng.uniform(1.2, 3.5))
        wind_gust_phase_2 = float(rng.uniform(0.0, 2.0 * math.pi))

        theta_deg = float(rng.uniform(75.0, 85.0))
        theta = math.radians(theta_deg)
        q = float(rng.uniform(-0.1, 0.1))

        x0 = deck_x0 + rel_x0
        z0 = deck_z0 + rel_z0
        v_x = deck_vx + v_x_rel0

        scenarios.append(
            {
                'id': int(i),
                'deck_x0': deck_x0,
                'deck_vx': deck_vx,
                'deck_z0': deck_z0,
                'deck_z_amp': deck_z_amp,
                'deck_z_freq': deck_z_freq,
                'deck_z_phase': deck_z_phase,
                'deck_z_peak_to_peak': 2.0 * deck_z_amp,
                'deck_z_peak_vz': deck_z_amp * deck_z_freq,
                'deck_z_period_s': (2.0 * math.pi / deck_z_freq) if deck_z_freq > 1e-9 else None,
                'base_wind_x': base_wind_x,
                'wind_gust_amp_1': wind_gust_amp_1,
                'wind_gust_freq_1': wind_gust_freq_1,
                'wind_gust_phase_1': wind_gust_phase_1,
                'wind_gust_amp_2': wind_gust_amp_2,
                'wind_gust_freq_2': wind_gust_freq_2,
                'wind_gust_phase_2': wind_gust_phase_2,
                'rel_x0': rel_x0,
                'rel_z0': rel_z0,
                'v_x_rel0': v_x_rel0,
                'x0': x0,
                'z0': z0,
                'v_x': v_x,
                'v_z': v_z,
                'theta_deg': theta_deg,
                'theta': theta,
                'q': q,
            }
        )

    return scenarios

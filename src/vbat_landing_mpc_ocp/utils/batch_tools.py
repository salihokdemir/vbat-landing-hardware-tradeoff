from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

import numpy as np

from dynamics.environment import SeaEnvironment


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return

    fieldnames = list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_world_reference_window(
    env: SeaEnvironment,
    x_ref_rel_full: np.ndarray,
    z_ref_rel_full: np.ndarray,
    step: int,
    dt: float,
    horizon: int,
):
    env_window = env.sample_horizon(step * dt, dt, horizon)
    sl = slice(step, step + horizon)
    x_ref_window = x_ref_rel_full[sl] + env_window['ship_x']
    z_ref_window = z_ref_rel_full[sl] + env_window['ship_z']
    return x_ref_window, z_ref_window, env_window


def build_full_profiles(
    env: SeaEnvironment,
    x_ref_rel_full: np.ndarray,
    z_ref_rel_full: np.ndarray,
    landed_step: int,
    dt: float,
):
    env_full = env.sample_horizon(0.0, dt, landed_step + 1)
    x_ref_world = x_ref_rel_full[: landed_step + 1] + env_full['ship_x']
    z_ref_world = z_ref_rel_full[: landed_step + 1] + env_full['ship_z']
    return {
        't_full': env_full['t'],
        'ship_x_full': env_full['ship_x'],
        'ship_z_full': env_full['ship_z'],
        'x_ref_world': x_ref_world,
        'z_ref_world': z_ref_world,
        'deck_vx_full': env_full['ship_vx'],
        'deck_vz_full': env_full['ship_vz'],
        'wind_x_stage': env_full['wind_x'][:-1],
        'max_abs_wind_x': float(np.max(np.abs(env_full['wind_x']))) if len(env_full['wind_x']) else 0.0,
    }


def evaluate_relative_metrics(curr_x: np.ndarray, env: SeaEnvironment, t: float) -> Dict[str, float]:
    ship_x = env.get_ship_x(t)
    ship_z = env.get_ship_z(t)
    ship_vx = env.get_ship_vx(t)
    ship_vz = env.get_ship_vz(t)
    return {
        'rel_x': float(curr_x[0] - ship_x),
        'rel_z': float(curr_x[1] - ship_z),
        'rel_vx': float(curr_x[3] - ship_vx),
        'rel_vz': float(curr_x[4] - ship_vz),
    }


def touchdown_like_success(metrics: Dict[str, float], *, x_tol: float, z_tol: float, vx_tol: float, vz_rel_min: float, vz_rel_max: float) -> bool:
    x_ok = abs(float(metrics['rel_x'])) <= float(x_tol)
    z_ok = float(metrics['rel_z']) <= float(z_tol)
    vx_ok = abs(float(metrics['rel_vx'])) <= float(vx_tol)
    vz_ok = float(vz_rel_min) <= float(metrics['rel_vz']) <= float(vz_rel_max)
    return bool(x_ok and z_ok and vx_ok and vz_ok)


def safe_float(value, default=None):
    try:
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


def mean_optional(values):
    valid = [safe_float(v, None) for v in values]
    valid = [v for v in valid if v is not None]
    if not valid:
        return None
    return float(np.mean(valid))


def max_optional(values):
    valid = [safe_float(v, None) for v in values]
    valid = [v for v in valid if v is not None]
    if not valid:
        return None
    return float(np.max(valid))

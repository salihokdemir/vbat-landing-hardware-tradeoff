from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

try:  # pragma: no cover - optional during syntax-only checks
    import casadi as ca  # type: ignore
except Exception:  # pragma: no cover
    ca = None


@dataclass(frozen=True)
class LandingProfileParams:
    flare_start_alt_m: float = 12.0
    flare_settle_alt_m: float = 5.0
    vx_rel_cap_high: float = 5.0
    vx_rel_cap_low: float = 1.2
    vz_rel_cap_high: float = 4.0
    vz_rel_cap_low: float = 0.6
    ref_velocity_smoothing_window: int = 5
    ref_velocity_smoothing_passes: int = 2
    ref_vx_accel_limit: float = 2.5
    ref_vz_accel_limit: float = 2.0


def _require_profile_attr(profile: Any, name: str) -> float:
    value = getattr(profile, name, None)
    if value is None:
        raise AttributeError(f'Landing profile eksik parametre: {name}')
    return float(value)


def _optional_profile_attr(profile: Any, name: str, default: float) -> float:
    value = getattr(profile, name, None)
    if value is None:
        return float(default)
    return float(value)


def _smoothstep_np(z_rel: np.ndarray | float, z_hi: float, z_lo: float) -> np.ndarray:
    z = np.asarray(z_rel, dtype=float)
    denom = max(float(z_hi) - float(z_lo), 1e-9)
    xi = np.clip((float(z_hi) - z) / denom, 0.0, 1.0)
    return xi * xi * (3.0 - 2.0 * xi)


def _smoothstep_ca(z_rel, z_hi: float, z_lo: float):
    if ca is None:  # pragma: no cover
        raise RuntimeError('casadi yokken _smoothstep_ca kullanılamaz.')
    denom = max(float(z_hi) - float(z_lo), 1e-9)
    xi = (float(z_hi) - z_rel) / denom
    xi = ca.fmin(1.0, ca.fmax(0.0, xi))
    return xi * xi * (3.0 - 2.0 * xi)


def flare_blend_np(z_rel: np.ndarray | float, profile: Any) -> np.ndarray:
    z_hi = _require_profile_attr(profile, 'flare_start_alt_m')
    z_lo = _require_profile_attr(profile, 'flare_settle_alt_m')
    return _smoothstep_np(z_rel, z_hi, z_lo)


def flare_blend_ca(z_rel, profile: Any):
    z_hi = _require_profile_attr(profile, 'flare_start_alt_m')
    z_lo = _require_profile_attr(profile, 'flare_settle_alt_m')
    return _smoothstep_ca(z_rel, z_hi, z_lo)


def relative_speed_caps_np(z_rel: np.ndarray | float, profile: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two-stage landing-speed envelope.

    Stage 1: above 12 m -> 5 m
        Slow down from the far approach caps toward an intermediate settle band.

    Stage 2: 5 m -> touchdown
        Continue slowing further toward the true touchdown-soft caps.

    Why this version exists
    -----------------------
    v1 hit the final low vz cap already at 5 m. That made the whole last 5 m creep
    with touchdown-soft sink rate, stretched the nominal reference close to the full
    30 s sim budget, and likely caused widespread early cap-sanity failures. Here we
    keep the user's intent (12 m'de yavaşlamaya başla, 5 m'ye gelince zaten yavaşlamış
    ol, ama tam touchdown softness en sonda gelsin) without freezing the whole 5->0
    segment at the terminal sink rate.
    """
    alpha_far_to_settle = flare_blend_np(z_rel, profile)

    z_settle = _require_profile_attr(profile, 'flare_settle_alt_m')
    z_touch = _optional_profile_attr(profile, 'touchdown_soft_alt_m', 0.0)
    alpha_settle_to_touch = _smoothstep_np(z_rel, z_settle, z_touch)

    vx_high = _require_profile_attr(profile, 'vx_rel_cap_high')
    vx_low = _require_profile_attr(profile, 'vx_rel_cap_low')
    vz_high = _require_profile_attr(profile, 'vz_rel_cap_high')
    vz_low = _require_profile_attr(profile, 'vz_rel_cap_low')

    # Sensible defaults for the "already slowed by 5 m" band.
    # If the user later adds explicit params, they override these automatically.
    vx_settle = _optional_profile_attr(profile, 'vx_rel_cap_settle', min(vx_high, max(vx_low, 2.0)))
    vz_settle = _optional_profile_attr(profile, 'vz_rel_cap_settle', min(vz_high, max(vz_low, 1.2)))

    vx_stage1 = (1.0 - alpha_far_to_settle) * vx_high + alpha_far_to_settle * vx_settle
    vz_stage1 = (1.0 - alpha_far_to_settle) * vz_high + alpha_far_to_settle * vz_settle

    vx_cap = (1.0 - alpha_settle_to_touch) * vx_stage1 + alpha_settle_to_touch * vx_low
    vz_cap = (1.0 - alpha_settle_to_touch) * vz_stage1 + alpha_settle_to_touch * vz_low
    return np.asarray(vx_cap, dtype=float), np.asarray(vz_cap, dtype=float)


def relative_speed_caps_ca(z_rel, profile: Any):
    if ca is None:  # pragma: no cover
        raise RuntimeError('casadi yokken relative_speed_caps_ca kullanılamaz.')

    alpha_far_to_settle = flare_blend_ca(z_rel, profile)

    z_settle = _require_profile_attr(profile, 'flare_settle_alt_m')
    z_touch = _optional_profile_attr(profile, 'touchdown_soft_alt_m', 0.0)
    alpha_settle_to_touch = _smoothstep_ca(z_rel, z_settle, z_touch)

    vx_high = _require_profile_attr(profile, 'vx_rel_cap_high')
    vx_low = _require_profile_attr(profile, 'vx_rel_cap_low')
    vz_high = _require_profile_attr(profile, 'vz_rel_cap_high')
    vz_low = _require_profile_attr(profile, 'vz_rel_cap_low')

    vx_settle = _optional_profile_attr(profile, 'vx_rel_cap_settle', min(vx_high, max(vx_low, 2.0)))
    vz_settle = _optional_profile_attr(profile, 'vz_rel_cap_settle', min(vz_high, max(vz_low, 1.2)))

    vx_stage1 = (1.0 - alpha_far_to_settle) * vx_high + alpha_far_to_settle * vx_settle
    vz_stage1 = (1.0 - alpha_far_to_settle) * vz_high + alpha_far_to_settle * vz_settle

    vx_cap = (1.0 - alpha_settle_to_touch) * vx_stage1 + alpha_settle_to_touch * vx_low
    vz_cap = (1.0 - alpha_settle_to_touch) * vz_stage1 + alpha_settle_to_touch * vz_low
    return vx_cap, vz_cap


def finite_difference_profile(values: np.ndarray, dt: float) -> np.ndarray:
    values = np.asarray(values, dtype=float).flatten()
    if len(values) == 0:
        return np.array([], dtype=float)
    if len(values) == 1:
        return np.zeros(1, dtype=float)
    return np.gradient(values, float(dt), edge_order=1)


def moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=float).flatten()
    window = max(1, int(window))
    if window <= 1 or len(signal) == 0:
        return signal.copy()
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(signal, (pad_left, pad_right), mode='edge')
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode='valid')


def rate_limit_signal(signal: np.ndarray, dt: float, accel_limit: float) -> np.ndarray:
    signal = np.asarray(signal, dtype=float).flatten()
    if len(signal) <= 1:
        return signal.copy()
    accel_limit = max(float(accel_limit), 1e-9)
    out = signal.copy()
    max_delta = accel_limit * float(dt)
    for k in range(1, len(out)):
        lower = out[k - 1] - max_delta
        upper = out[k - 1] + max_delta
        out[k] = float(np.clip(out[k], lower, upper))
    return out


def derive_reference_velocity_profile(
    values: np.ndarray,
    *,
    dt: float,
    smoothing_window: int,
    smoothing_passes: int,
    accel_limit: float,
) -> np.ndarray:
    vel = finite_difference_profile(values, dt)
    smoothing_window = max(1, int(smoothing_window))
    smoothing_passes = max(0, int(smoothing_passes))
    for _ in range(smoothing_passes):
        vel = moving_average(vel, smoothing_window)
    vel = rate_limit_signal(vel, float(dt), float(accel_limit))
    return vel


def derive_reference_world_velocities(
    x_ref_world: np.ndarray,
    z_ref_world: np.ndarray,
    *,
    dt: float,
    profile: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    vx = derive_reference_velocity_profile(
        x_ref_world,
        dt=float(dt),
        smoothing_window=int(getattr(profile, 'ref_velocity_smoothing_window', 5)),
        smoothing_passes=int(getattr(profile, 'ref_velocity_smoothing_passes', 2)),
        accel_limit=float(getattr(profile, 'ref_vx_accel_limit', 2.5)),
    )
    vz = derive_reference_velocity_profile(
        z_ref_world,
        dt=float(dt),
        smoothing_window=int(getattr(profile, 'ref_velocity_smoothing_window', 5)),
        smoothing_passes=int(getattr(profile, 'ref_velocity_smoothing_passes', 2)),
        accel_limit=float(getattr(profile, 'ref_vz_accel_limit', 2.0)),
    )
    return vx, vz


def clip_reference_velocity_to_caps(
    *,
    vx_world: np.ndarray,
    vz_world: np.ndarray,
    deck_vx: np.ndarray,
    deck_vz: np.ndarray,
    deck_z: np.ndarray,
    z_ref_world: np.ndarray,
    profile: Any,
    clip_vx: bool = False,
    clip_vz: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare default-guess world velocities from a reference trajectory.

    Design intent
    -------------
    - Vz clipping stays on by default because touchdown softness is non-negotiable.
    - Vx clipping is OFF by default because a hard horizontal-speed corridor proved too
      restrictive under the harsh-wind scenarios and was killing feasibility very early.
    """
    vx_world = np.asarray(vx_world, dtype=float).flatten()
    vz_world = np.asarray(vz_world, dtype=float).flatten()
    deck_vx = np.asarray(deck_vx, dtype=float).flatten()
    deck_vz = np.asarray(deck_vz, dtype=float).flatten()
    deck_z = np.asarray(deck_z, dtype=float).flatten()
    z_ref_world = np.asarray(z_ref_world, dtype=float).flatten()

    z_rel_ref = z_ref_world - deck_z
    vx_cap, vz_cap = relative_speed_caps_np(z_rel_ref, profile)

    if bool(clip_vx):
        vx_rel = np.clip(vx_world - deck_vx, -vx_cap, vx_cap)
    else:
        vx_rel = vx_world - deck_vx

    if bool(clip_vz):
        vz_rel = np.clip(vz_world - deck_vz, -vz_cap, 0.25)
    else:
        vz_rel = vz_world - deck_vz

    return deck_vx + vx_rel, deck_vz + vz_rel

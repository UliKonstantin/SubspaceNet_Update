"""
Single-step trajectory physics functions.

These are the canonical implementations of each state evolution model.
Both batch (TrajectoryDataHandler) and online (OnlineLearningTrajectoryGenerator)
generators call these instead of inlining the math.
"""
import numpy as np
from typing import Any, List, Optional, Sequence, Union


def resolve_sine_accel_dc_offsets(
    trajectory_config: Any,
    num_sources: int,
    rng: Optional[np.random.Generator] = None,
) -> Optional[np.ndarray]:
    """
    Per-source DC attractor offsets (degrees) for sine_accel_nonlinear.

    Returns None when disabled (legacy: damped convergence to 0).
    """
    fixed = getattr(trajectory_config, "sine_accel_dc_offsets", None)
    if fixed is not None:
        arr = np.atleast_1d(np.asarray(fixed, dtype=float))
        if arr.size == 1:
            arr = np.broadcast_to(arr, num_sources)
        if arr.size != num_sources:
            raise ValueError(
                f"sine_accel_dc_offsets length ({arr.size}) must match num_sources ({num_sources})"
            )
        return arr.copy()

    offset_range = getattr(trajectory_config, "sine_accel_dc_offset_range", None)
    if offset_range is None:
        return None
    if len(offset_range) != 2:
        raise ValueError(
            f"sine_accel_dc_offset_range must be [min, max], got {offset_range!r}"
        )
    lo, hi = float(offset_range[0]), float(offset_range[1])
    if lo > hi:
        raise ValueError(f"sine_accel_dc_offset_range min ({lo}) must be <= max ({hi})")
    gen = rng if rng is not None else np.random.default_rng()
    return gen.uniform(lo, hi, size=num_sources)


def sine_accel_step(
    theta_prev: np.ndarray,
    t: int,
    omega0: np.ndarray,
    kappa: np.ndarray,
    noise_std: float,
    damping: float = 0.99,
    dc_offset: Optional[Union[np.ndarray, Sequence[float], float]] = None,
) -> np.ndarray:
    """
    Sine acceleration nonlinear model (single step).
    θ_{k+1} = damping * θ_k + (1 - damping) * b + κ * sin(ω0 * t) + η_k

    When ``dc_offset`` (b) is None, the attractor is 0 (legacy behaviour).

    Args:
        theta_prev: Current angles [num_sources] in degrees
        t: Current time index
        omega0: Frequency per source [num_sources]
        kappa: Amplitude per source [num_sources]
        noise_std: Process noise standard deviation (degrees)
        damping: Damping factor (default 0.99)
        dc_offset: Optional per-source DC attractor (degrees)

    Returns:
        Next angles [num_sources] in degrees (unclamped)
    """
    oscillation = kappa * np.sin(omega0 * t)
    noise = np.random.randn(len(theta_prev)) * noise_std
    if dc_offset is not None:
        dc = np.atleast_1d(np.asarray(dc_offset, dtype=float))
        if dc.size == 1:
            dc = np.broadcast_to(dc, len(theta_prev))
        leak = (1.0 - damping) * dc
    else:
        leak = 0.0
    return damping * theta_prev + leak + oscillation + noise


def mult_noise_step(theta_prev: np.ndarray, omega0: float,
                    amp: float, base_std: float) -> np.ndarray:
    """
    Multiplicative noise nonlinear model (single step).
    θ_{k+1} = θ_k + ω0 * T + σ(θ_k) * η_k
    where σ(θ) = base_std * (1 + amp * sin²(θ_rad))

    Args:
        theta_prev: Current angles [num_sources] in degrees
        omega0: Drift rate (degrees per time step)
        amp: Amplitude of state-dependent noise modulation
        base_std: Base noise standard deviation (degrees)

    Returns:
        Next angles [num_sources] in degrees (unclamped)
    """
    theta_rad = theta_prev * (np.pi / 180.0)
    deterministic = omega0 * 1.0  # T = 1s
    std = base_std * (1.0 + amp * np.sin(theta_rad) ** 2)
    noise = np.random.randn(len(theta_prev)) * std
    return theta_prev + deterministic + noise


def random_walk_step(theta_prev: np.ndarray, std_dev: float) -> np.ndarray:
    """
    Random walk model (single step).
    θ_{k+1} = θ_k + w_k,  w_k ~ N(0, std_dev²)

    Args:
        theta_prev: Current angles [num_sources] in degrees
        std_dev: Step standard deviation (degrees)

    Returns:
        Next angles [num_sources] in degrees (unclamped)
    """
    noise = np.random.randn(len(theta_prev)) * std_dev
    return theta_prev + noise

"""
Single-step trajectory physics functions.

These are the canonical implementations of each state evolution model.
Both batch (TrajectoryDataHandler) and online (OnlineLearningTrajectoryGenerator)
generators call these instead of inlining the math.
"""
import numpy as np


def sine_accel_step(theta_prev: np.ndarray, t: int, omega0: np.ndarray,
                    kappa: np.ndarray, noise_std: float, damping: float = 0.99) -> np.ndarray:
    """
    Sine acceleration nonlinear model (single step).
    θ_{k+1} = damping * θ_k + κ * sin(ω0 * t) + η_k

    Args:
        theta_prev: Current angles [num_sources] in degrees
        t: Current time index
        omega0: Frequency per source [num_sources]
        kappa: Amplitude per source [num_sources]
        noise_std: Process noise standard deviation (degrees)
        damping: Damping factor (default 0.99)

    Returns:
        Next angles [num_sources] in degrees (unclamped)
    """
    oscillation = kappa * np.sin(omega0 * t)
    noise = np.random.randn(len(theta_prev)) * noise_std
    return damping * theta_prev + oscillation + noise


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

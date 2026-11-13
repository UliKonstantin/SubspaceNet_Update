"""Angle-unit helpers for Kalman filter Q/R.

Yaml ``process_noise_std_dev`` / ``measurement_noise_std_dev`` and trajectory
``sine_accel_noise_std`` are in **degrees**. Filter state is radians.
``initial_covariance`` (P0) is already in rad².
"""

from __future__ import annotations

import math

DEG2RAD = math.pi / 180.0


def angle_std_deg_to_variance_rad(std_deg: float) -> float:
    """Convert an angle std-dev in degrees to a variance in rad²."""
    std_rad = float(std_deg) * DEG2RAD
    return std_rad * std_rad

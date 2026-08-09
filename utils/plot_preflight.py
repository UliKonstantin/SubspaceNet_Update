"""Preflight checks before online-learning plot dispatch."""
from __future__ import annotations

import logging
from typing import List, Optional

from utils import drift_gates

logger = logging.getLogger("SubspaceNet.cli")


def estimate_ol_window_count(
    trajectory_length: int,
    window_size: int,
    stride: int,
) -> int:
    """Match OnlineLearningDataset window count (see create_online_learning_dataset)."""
    if trajectory_length < window_size or stride <= 0:
        return 0
    return (trajectory_length - window_size) // stride + 1


def minimum_trajectory_length_for_drift(
    window_size: int,
    stride: int,
    drift_warmup_windows: int,
    drift_guard_samples: int,
    adaptation_window_count: int = 5,
    post_eval_windows: int = 5,
) -> int:
    """Smallest trajectory_length that yields enough windows for drift + adaptation + eval."""
    min_windows = (
        drift_gates.first_z_window(drift_warmup_windows, drift_guard_samples)
        + 1
        + adaptation_window_count
        + post_eval_windows
    )
    # Invert (T - window_size) // stride + 1 >= min_windows
    required_span = (min_windows - 1) * stride + window_size
    return max(window_size, required_span)


def check_online_learning_plot_budget(config) -> List[str]:
    """
    Return human-readable warnings when OL trajectory is too short for drift/aggregate plots.
    """
    warnings: List[str] = []
    ol = getattr(config, "online_learning", None)
    if ol is None or not getattr(ol, "enabled", False):
        return warnings

    traj_len = int(getattr(ol, "trajectory_length", 0))
    window_size = int(getattr(ol, "window_size", 1))
    stride = int(getattr(ol, "stride", 1))
    warmup = int(getattr(ol, "drift_warmup_windows", 7))
    guard = int(getattr(ol, "drift_guard_samples", 3))
    adaptation = int(getattr(ol, "adaptation_window_count", 5))
    eta_interval = getattr(ol, "eta_update_interval_windows", None)

    num_windows = estimate_ol_window_count(traj_len, window_size, stride)
    first_g = drift_gates.first_g_window(warmup)
    first_z = drift_gates.first_z_window(warmup, guard)
    min_traj = minimum_trajectory_length_for_drift(
        window_size, stride, warmup, guard, adaptation
    )

    if num_windows <= first_g:
        warnings.append(
            f"Only ~{num_windows} OL windows (trajectory_length={traj_len}, "
            f"window_size={window_size}, stride={stride}) but GLRT Scope A starts at "
            f"window {first_g}. Drift/aggregate eta plots will be empty or N/A."
        )
    elif num_windows <= first_z:
        warnings.append(
            f"Only ~{num_windows} OL windows but first drift z-score window is {first_z}. "
            f"Increase trajectory_length (recommend >= {min_traj}, currently {traj_len})."
        )
    elif traj_len < min_traj:
        warnings.append(
            f"trajectory_length={traj_len} yields ~{num_windows} windows; recommend "
            f">= {min_traj} (~{estimate_ol_window_count(min_traj, window_size, stride)} windows) "
            f"for drift + {adaptation} adaptation windows + post-training eval."
        )

    if eta_interval and int(eta_interval) > 0 and int(eta_interval) < first_z:
        warnings.append(
            f"eta_update_interval_windows={eta_interval} is before first_z_window={first_z}; "
            f"distribution change occurs before drift can fire. Consider >= {first_z + 1}."
        )

    return warnings


def warn_online_learning_plot_budget(config, context: Optional[str] = None) -> None:
    """Log warnings once before OL plot generation."""
    for message in check_online_learning_plot_budget(config):
        prefix = f"{context}: " if context else ""
        logger.warning("%s%s", prefix, message)

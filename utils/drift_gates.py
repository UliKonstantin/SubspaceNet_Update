"""
Drift-detection gates (two stages).

1. **Scope A (changepoint GLRT)** — per-window max log-GLR over loss prefix (post-warmup).
2. **Scope B (z-score)** — compare current g to baseline g-history (minus tail guard).
"""
from __future__ import annotations

from typing import Sequence, Tuple
import numpy as np

GLRT_MIN_SEGMENT_SIZE = 5
# Scope B: minimum g-history baseline count before z-score (not configurable).
SCOPE_B_BASELINE_MIN_SAMPLES = 7


def scope_b_baseline_min_samples() -> int:
    return SCOPE_B_BASELINE_MIN_SAMPLES


def glrt_min_segment_size() -> int:
    return GLRT_MIN_SEGMENT_SIZE


def has_enough_losses_for_changepoint_glr(
    num_losses: int, min_segment_size: int | None = None
) -> bool:
    m = min_segment_size if min_segment_size is not None else GLRT_MIN_SEGMENT_SIZE
    return num_losses >= 2 * m + 1


def first_g_window(scope_a_warmup_windows: int, min_segment_size: int | None = None) -> int:
    """First window index where Scope A produces a log-GLR sample."""
    m = min_segment_size if min_segment_size is not None else GLRT_MIN_SEGMENT_SIZE
    return scope_a_warmup_windows + 2 * m


def first_z_window(
    scope_a_warmup_windows: int,
    guard_samples: int,
    min_segment_size: int | None = None,
) -> int:
    """First window index where Scope B z-score can be computed."""
    m = min_segment_size if min_segment_size is not None else GLRT_MIN_SEGMENT_SIZE
    return (
        scope_a_warmup_windows
        + 2 * m
        + guard_samples
        + SCOPE_B_BASELINE_MIN_SAMPLES
        - 1
    )


def drift_detection_milestones(
    scope_a_warmup_windows: int,
    guard_samples: int,
    min_segment_size: int | None = None,
) -> dict[str, int]:
    """
    Absolute window indices for drift-detection phase boundaries.

    - scope_a_loss_start: first window included in post-warmup GLRT loss prefix
    - first_g: first window where Scope A appends a g-scalar to history
    - first_z: first window where Scope B z-score (live drift trigger) is armed
    """
    m = min_segment_size if min_segment_size is not None else GLRT_MIN_SEGMENT_SIZE
    return {
        "scope_a_loss_start": scope_a_warmup_windows,
        "first_g": first_g_window(scope_a_warmup_windows, m),
        "first_z": first_z_window(scope_a_warmup_windows, guard_samples, m),
    }

def baseline_g_values(
    g_history: Sequence[float],
    guard_samples: int,
) -> np.ndarray:
    """Baseline g-scalars: all history except the last ``guard_samples`` tail."""
    n = len(g_history)
    if guard_samples > 0:
        if n <= guard_samples:
            return np.array([], dtype=float)
        return np.asarray(g_history[:-guard_samples], dtype=float)
    return np.asarray(g_history, dtype=float)


def can_compute_drift_z_score(history_length: int, guard_samples: int) -> bool:
    baseline_len = history_length - guard_samples if guard_samples > 0 else history_length
    return baseline_len >= SCOPE_B_BASELINE_MIN_SAMPLES


def compute_drift_z_score(
    current_g: float,
    g_history: Sequence[float],
    guard_samples: int,
) -> Tuple[float | None, float | None, float | None]:
    baseline = baseline_g_values(g_history, guard_samples)
    if len(baseline) < SCOPE_B_BASELINE_MIN_SAMPLES:
        return None, None, None
    baseline_mean = float(np.mean(baseline))
    baseline_std = float(np.std(baseline))
    if baseline_std <= 1e-10:
        return 0.0, baseline_mean, baseline_std
    z_score = (current_g - baseline_mean) / baseline_std
    return z_score, baseline_mean, baseline_std

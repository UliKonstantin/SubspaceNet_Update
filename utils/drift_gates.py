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


def recent_tau_passes(
    changepoint_post_warmup: int,
    prefix_len: int,
    recent_tau_k: int | None,
) -> bool:
    """
    Recent-τ gate: require argmax τ in the last ``recent_tau_k`` MSIE samples of the prefix.

    ``changepoint_post_warmup`` and ``prefix_len`` are post-warmup indices (0-based).
    When ``recent_tau_k`` is None, the gate is disabled (always passes).
    """
    if recent_tau_k is None:
        return True
    if prefix_len <= 0:
        return False
    min_tau = max(0, prefix_len - recent_tau_k)
    return changepoint_post_warmup >= min_tau


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


def average_z_score_trajectories(results_list: list) -> dict | None:
    """Average per-window GLRT z-scores across trajectories (align by window index)."""
    from collections import defaultdict

    by_window: dict[int, list[float]] = defaultdict(list)
    g_by_window: dict[int, list[float]] = defaultdict(list)
    baseline_by_window: dict[int, list[float]] = defaultdict(list)
    drift_detection_windows: list[int] = []
    training_start_windows: list[int] = []
    changepoint_at_detection_windows: list[int] = []

    for result in results_list:
        if result.get("status") != "success":
            continue
        online_results = result.get("online_learning_results", {})
        windows = online_results.get("glrt_z_score_windows") or []
        scores = online_results.get("glrt_z_scores") or []
        g_values = online_results.get("glrt_g_values") or []
        baseline_means = online_results.get("glrt_baseline_means") or []
        for window_idx, z_score in zip(windows, scores):
            by_window[int(window_idx)].append(float(z_score))
        for window_idx, g_val in zip(windows, g_values):
            g_by_window[int(window_idx)].append(float(g_val))
        for window_idx, b_mean in zip(windows, baseline_means):
            baseline_by_window[int(window_idx)].append(float(b_mean))
        if online_results.get("drift_detection_window") is not None:
            drift_detection_windows.append(int(online_results["drift_detection_window"]))
        if online_results.get("training_start_window") is not None:
            training_start_windows.append(int(online_results["training_start_window"]))
        tau_at_det = online_results.get("glrt_changepoint_window_at_detection")
        if tau_at_det is None:
            snap = online_results.get("glrt_at_detection") or {}
            tau_at_det = snap.get("changepoint_window")
        if tau_at_det is not None:
            changepoint_at_detection_windows.append(int(tau_at_det))

    if not by_window:
        return None

    ordered_windows = sorted(by_window.keys())
    avg_z_scores = [float(np.mean(by_window[w])) for w in ordered_windows]
    std_z_scores = [
        float(np.std(by_window[w])) if len(by_window[w]) > 1 else 0.0 for w in ordered_windows
    ]
    avg_g_values = [float(np.mean(g_by_window[w])) for w in ordered_windows]
    avg_baseline_means = [float(np.mean(baseline_by_window[w])) for w in ordered_windows]

    return {
        "windows": ordered_windows,
        "avg_z_scores": avg_z_scores,
        "std_z_scores": std_z_scores,
        "avg_g_values": avg_g_values,
        "avg_baseline_means": avg_baseline_means,
        "trajectory_count": len(results_list),
        "avg_drift_detection_window": float(np.mean(drift_detection_windows)) if drift_detection_windows else None,
        "std_drift_detection_window": float(np.std(drift_detection_windows)) if drift_detection_windows else None,
        "avg_changepoint_window_at_detection": (
            float(np.mean(changepoint_at_detection_windows)) if changepoint_at_detection_windows else None
        ),
        "std_changepoint_window_at_detection": (
            float(np.std(changepoint_at_detection_windows)) if changepoint_at_detection_windows else None
        ),
        "avg_training_start_window": float(np.mean(training_start_windows)) if training_start_windows else None,
        "std_training_start_window": float(np.std(training_start_windows)) if training_start_windows else None,
    }


def average_glrt_at_detection(results_list: list, window_index_offset: int = 0) -> dict | None:
    """
    Build an averaged GLRT snapshot at the drift-trigger prefix (MSIE losses up to detection).

    Averages per-window MSIE across trajectories, truncates to mean detection window, then
    runs offline GLRT on that prefix (same object the live trigger saw at fire time).
    """
    from simulation.drift.glrt import glrt_changepoint_detection

    snapshots = []
    detection_windows: list[int] = []
    for result in results_list:
        if result.get("status") != "success":
            continue
        snap = result.get("online_learning_results", {}).get("glrt_at_detection")
        if snap and snap.get("losses"):
            snapshots.append(snap)
            detection_windows.append(int(snap["window_idx"]))

    if not snapshots:
        return None

    avg_detection_window = float(np.mean(detection_windows))
    prefix_len = int(round(avg_detection_window)) - window_index_offset + 1
    prefix_len = max(prefix_len, GLRT_MIN_SEGMENT_SIZE * 2 + 1)

    loss_sums = np.zeros(prefix_len, dtype=float)
    loss_counts = np.zeros(prefix_len, dtype=int)
    for snap in snapshots:
        losses = snap["losses"]
        n = min(len(losses), prefix_len)
        for i in range(n):
            loss_sums[i] += losses[i]
            loss_counts[i] += 1

    if not np.all(loss_counts[:prefix_len] > 0):
        valid_len = int(np.max(np.where(loss_counts > 0)[0])) + 1 if np.any(loss_counts > 0) else 0
        prefix_len = valid_len

    if prefix_len < 2 * GLRT_MIN_SEGMENT_SIZE + 1:
        return None

    avg_losses = (loss_sums[:prefix_len] / np.maximum(loss_counts[:prefix_len], 1)).tolist()
    changepoint, log_glr, all_log_glr, candidate_points = glrt_changepoint_detection(
        avg_losses, min_segment_size=GLRT_MIN_SEGMENT_SIZE
    )

    return {
        "avg_losses": avg_losses,
        "changepoint_post_warmup": int(changepoint),
        "changepoint_window": int(changepoint + window_index_offset),
        "log_glr": float(log_glr),
        "all_log_glr": [float(v) for v in all_log_glr],
        "candidate_points_post_warmup": [int(p) for p in candidate_points],
        "window_index_offset": window_index_offset,
        "avg_detection_window": avg_detection_window,
        "std_detection_window": float(np.std(detection_windows)) if detection_windows else 0.0,
        "individual_detection_windows": detection_windows,
        "trajectory_count": len(snapshots),
        "avg_z_at_detection": float(np.mean([s["z_score"] for s in snapshots])),
    }

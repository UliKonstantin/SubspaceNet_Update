"""Tests for GLRT z-score trajectory export and plotting."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from utils.drift_gates import average_glrt_at_detection, average_z_score_trajectories
from utils.plotting.online_learning import (
    plot_glrt_adaptation_at_detection,
    plot_glrt_adaptation_g_vs_baseline,
    plot_glrt_adaptation_z_score_averaged,
)


def test_average_z_score_trajectories_aligns_by_window():
    results = [
        {
            "status": "success",
            "online_learning_results": {
                "glrt_z_score_windows": [29, 30, 31],
                "glrt_z_scores": [2.6, 3.1, 4.0],
                "glrt_g_values": [10.0, 12.0, 14.0],
                "glrt_baseline_means": [8.0, 9.0, 10.0],
                "drift_detection_window": 29,
                "training_start_window": 31,
                "glrt_at_detection": {
                    "window_idx": 29,
                    "losses": [0.1, 0.2, 0.15, 0.3, 0.25, 0.4, 0.35, 0.5, 0.45, 0.6, 0.55],
                    "changepoint_post_warmup": 5,
                    "changepoint_window": 15,
                    "log_glr": 3.0,
                    "all_log_glr": [0.5, 1.0, 3.0, 2.0, 1.5],
                    "candidate_points_post_warmup": [5, 6, 7, 8, 9],
                    "z_score": 2.6,
                    "baseline_mean": 8.0,
                    "baseline_std": 0.5,
                },
            },
        },
        {
            "status": "success",
            "online_learning_results": {
                "glrt_z_score_windows": [29, 30],
                "glrt_z_scores": [2.8, 3.5],
                "glrt_g_values": [10.5, 13.0],
                "glrt_baseline_means": [8.5, 9.5],
                "drift_detection_window": 29,
                "training_start_window": 31,
                "glrt_at_detection": {
                    "window_idx": 29,
                    "losses": [0.1, 0.2, 0.15, 0.3, 0.25, 0.4, 0.35, 0.5, 0.45, 0.6, 0.55],
                    "changepoint_post_warmup": 5,
                    "changepoint_window": 15,
                    "log_glr": 3.2,
                    "all_log_glr": [0.6, 1.1, 3.2, 2.1, 1.6],
                    "candidate_points_post_warmup": [5, 6, 7, 8, 9],
                    "z_score": 2.8,
                    "baseline_mean": 8.5,
                    "baseline_std": 0.6,
                },
            },
        },
    ]
    traj = average_z_score_trajectories(results)
    assert traj is not None
    assert traj["windows"] == [29, 30, 31]
    assert traj["avg_z_scores"][0] == 2.7
    assert traj["avg_z_scores"][1] == 3.3
    assert traj["avg_z_scores"][2] == 4.0
    assert traj["avg_drift_detection_window"] == 29.0
    assert traj["avg_changepoint_window_at_detection"] == 15.0
    assert traj["avg_training_start_window"] == 31.0
    assert traj["avg_g_values"][0] == 10.25
    assert traj["avg_baseline_means"][0] == 8.25


def _make_detection_losses(n: int = 20):
    return [0.1 + 0.01 * i for i in range(n)]


def test_average_glrt_at_detection_builds_prefix():
    losses = _make_detection_losses(20)
    results = [
        {
            "status": "success",
            "online_learning_results": {
                "glrt_at_detection": {
                    "window_idx": 29,
                    "losses": losses,
                    "z_score": 2.6,
                },
            },
        },
    ]
    snap = average_glrt_at_detection(results, window_index_offset=10)
    assert snap is not None
    assert len(snap["avg_losses"]) == 20  # windows 10..29 inclusive
    assert snap["avg_detection_window"] == 29.0


def test_plot_glrt_companion_pngs(tmp_path):
    glrt_results = {
        "adaptation_loss": {
            "trajectory_count": 1,
            "min_segment_size": 5,
            "z_score_trajectory": {
                "windows": [29, 30, 31],
                "avg_z_scores": [2.6, 3.1, 4.0],
                "std_z_scores": [0.0, 0.0, 0.0],
                "avg_g_values": [10.0, 12.0, 11.0],
                "avg_baseline_means": [8.0, 9.5, 10.0],
                "trajectory_count": 1,
                "avg_drift_detection_window": 29.0,
            },
            "at_detection": {
                "avg_losses": [0.1, 0.2, 0.15, 0.3, 0.25, 0.4, 0.35, 0.5, 0.45, 0.6, 0.55],
                "changepoint_post_warmup": 5,
                "changepoint_window": 15,
                "all_log_glr": [0.5, 1.0, 3.0, 2.0, 1.5],
                "candidate_points_post_warmup": [5, 6, 7, 8, 9],
                "window_index_offset": 10,
                "avg_detection_window": 29.0,
                "std_detection_window": 0.0,
                "avg_z_at_detection": 2.6,
                "trajectory_count": 1,
            },
        }
    }
    plot_glrt_adaptation_z_score_averaged(
        glrt_results,
        tmp_path,
        drift_z_threshold=2.5,
        eta_change_windows=[45],
        drift_detection_window=29,
    )
    plot_glrt_adaptation_g_vs_baseline(
        glrt_results,
        tmp_path,
        drift_z_threshold=2.5,
        eta_change_windows=[45],
        drift_detection_window=29,
    )
    plot_glrt_adaptation_at_detection(
        glrt_results,
        tmp_path,
        drift_warmup_windows=10,
        drift_guard_samples=3,
        eta_change_windows=[45],
    )
    assert (tmp_path / "glrt_adaptation_z_score_averaged.png").exists()
    assert (tmp_path / "glrt_adaptation_g_vs_baseline.png").exists()
    assert (tmp_path / "glrt_adaptation_at_detection_loss.png").exists()
    assert (tmp_path / "glrt_adaptation_at_detection_glrt.png").exists()


def test_plot_glrt_adaptation_z_score_averaged_writes_png(tmp_path):
    glrt_results = {
        "adaptation_loss": {
            "trajectory_count": 1,
            "z_score_trajectory": {
                "windows": [29, 30, 31],
                "avg_z_scores": [2.6, 3.1, 4.0],
                "std_z_scores": [0.0, 0.0, 0.0],
                "trajectory_count": 1,
                "avg_drift_detection_window": 29.0,
                "avg_training_start_window": 31.0,
            },
        }
    }
    plot_glrt_adaptation_z_score_averaged(
        glrt_results,
        tmp_path,
        drift_z_threshold=2.5,
        eta_change_windows=[32],
        drift_detection_window=29,
        training_start_window=31,
        time_to_learn=2,
    )
    assert (tmp_path / "glrt_adaptation_z_score_averaged.png").exists()

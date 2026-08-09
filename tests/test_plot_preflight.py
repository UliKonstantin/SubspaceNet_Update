"""Tests for online-learning plot preflight checks."""
from types import SimpleNamespace

from utils.plot_preflight import (
    check_online_learning_plot_budget,
    estimate_ol_window_count,
    minimum_trajectory_length_for_drift,
)
from utils import drift_gates


def test_estimate_ol_window_count_matches_dataset_formula():
    assert estimate_ol_window_count(20, 5, 3) == 6
    assert estimate_ol_window_count(125, 5, 3) == 41


def test_short_paper_like_config_warns():
    config = SimpleNamespace(
        online_learning=SimpleNamespace(
            enabled=True,
            trajectory_length=20,
            window_size=5,
            stride=3,
            drift_warmup_windows=10,
            drift_guard_samples=3,
            adaptation_window_count=5,
            eta_update_interval_windows=3,
        )
    )
    warnings = check_online_learning_plot_budget(config)
    assert any("GLRT Scope A" in w or "first drift z-score" in w for w in warnings)
    assert any("eta_update_interval_windows=3" in w for w in warnings)


def test_production_like_config_is_clean():
    config = SimpleNamespace(
        online_learning=SimpleNamespace(
            enabled=True,
            trajectory_length=125,
            window_size=5,
            stride=3,
            drift_warmup_windows=10,
            drift_guard_samples=3,
            adaptation_window_count=5,
            eta_update_interval_windows=32,
        )
    )
    assert check_online_learning_plot_budget(config) == []


def test_minimum_trajectory_length_covers_first_z():
    warmup, guard = 10, 3
    min_traj = minimum_trajectory_length_for_drift(5, 3, warmup, guard, adaptation_window_count=5)
    windows = estimate_ol_window_count(min_traj, 5, 3)
    assert windows >= drift_gates.first_z_window(warmup, guard) + 1 + 5

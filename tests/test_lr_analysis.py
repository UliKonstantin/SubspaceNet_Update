"""Tests for LR sweep optimality analysis utilities."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from utils.lr_analysis import (
    best_lr_per_eta,
    build_glrt_lr_mapping,
    fit_eta_to_lr_sigmoid,
    group_loss_curves_by_lr,
    postprocess_lr_sweep_analysis,
)


def _synthetic_heatmap() -> dict:
    return {
        "eta_values": [0.5, 0.5, 1.0, 1.0, 1.5, 1.5, 0.5, 1.0, 1.5],
        "lr_values": [0.001, 0.01, 0.001, 0.01, 0.001, 0.01, 0.02, 0.02, 0.02],
        "lr_types": [
            "static", "static", "static", "static", "static", "static",
            "adaptive", "adaptive", "adaptive",
        ],
        "lr_row_ids": [0, 1, 0, 1, 0, 1, "ADAPTIVE", "ADAPTIVE", "ADAPTIVE"],
        "avg_losses": [0.20, 0.12, 0.18, 0.10, 0.16, 0.09, 0.14, 0.13, 0.11],
    }


def _synthetic_drift() -> list:
    return [
        {"eta": 0.5, "main_log_glr": 70.0, "baseline_mean": 65.0},
        {"eta": 1.0, "main_log_glr": 80.0, "baseline_mean": 65.0},
        {"eta": 1.5, "main_log_glr": 85.0, "baseline_mean": 65.0},
    ]


def test_best_lr_per_eta_picks_minimum_loss():
    best = best_lr_per_eta(_synthetic_heatmap())
    assert [row["eta"] for row in best] == [0.5, 1.0, 1.5]
    assert best[0]["lr"] == 0.01
    assert best[1]["lr"] == 0.01
    assert best[2]["lr"] == 0.01


def test_group_loss_curves_by_lr_returns_sorted_curves():
    curves = group_loss_curves_by_lr(_synthetic_heatmap())
    assert len(curves) == 3
    assert all(np.all(np.diff(curve["etas"]) >= 0) for curve in curves)


def test_fit_eta_to_lr_sigmoid_runs():
    best = best_lr_per_eta(_synthetic_heatmap())
    etas = np.array([row["eta"] for row in best], dtype=float)
    lrs = np.array([row["lr"] for row in best], dtype=float)
    fit = fit_eta_to_lr_sigmoid(etas, lrs)
    assert len(fit.params) == 4
    assert all(np.isfinite(p) for p in fit.params)


def test_build_glrt_lr_mapping_joins_drift_and_heatmap():
    mapping = build_glrt_lr_mapping(_synthetic_heatmap(), _synthetic_drift())
    assert len(mapping) == 3
    assert mapping[0]["optimal_lr"] == 0.01


def test_postprocess_lr_sweep_analysis_writes_pngs(tmp_path: Path):
    heatmap = _synthetic_heatmap()
    drift_path = tmp_path / "drift_detection_dicts.json"
    drift_path.write_text(json.dumps(_synthetic_drift()), encoding="utf-8")

    created = postprocess_lr_sweep_analysis(tmp_path, heatmap)
    assert "optimal_lr_vs_eta" in created
    assert "loss_vs_eta_per_lr" in created
    assert "glrt_observable_to_optimal_lr" in created
    assert created["optimal_lr_vs_eta"].exists()
    assert created["loss_vs_eta_per_lr"].exists()
    assert created["glrt_observable_to_optimal_lr"].exists()

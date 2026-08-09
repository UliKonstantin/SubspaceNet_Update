"""Aggregate drift-detection metrics vs sweep eta."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from utils.plotting.style import apply_paper_plot_style, save_figure

logger = logging.getLogger(__name__)

DRIFT_METRICS_PLOT_FILENAME = "drift_detection_metrics_vs_eta.png"
DRIFT_DETECTION_JSON_FILENAME = "drift_detection_dicts.json"


def _drift_group_eta(entry: Dict[str, Any]) -> float:
    if "scenario_eta" in entry:
        return float(entry["scenario_eta"])
    return float(entry["eta"])


def _average_drift_dicts_by_eta(drift_dicts: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    eta_groups: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for entry in drift_dicts:
        eta_groups[_drift_group_eta(entry)].append(entry)

    etas = sorted(eta_groups.keys())
    averaged = {
        "eta": etas,
        "window_idx": [],
        "baseline_mean": [],
        "main_log_glr": [],
        "main_log_glr_std": [],
        "baseline_std": [],
        "current_glrt_z_score": [],
        "learning_rate_at_detection": [],
    }

    for eta in etas:
        group = eta_groups[eta]
        averaged["window_idx"].append(np.mean([d["window_idx"] for d in group]))
        averaged["baseline_mean"].append(np.mean([d["baseline_mean"] for d in group]))
        main_log_glr_values = [d["main_log_glr"] for d in group]
        averaged["main_log_glr"].append(np.mean(main_log_glr_values))
        averaged["main_log_glr_std"].append(np.std(main_log_glr_values))
        averaged["baseline_std"].append(np.mean([d["baseline_std"] for d in group]))
        averaged["current_glrt_z_score"].append(
            np.mean([d["current_glrt_z_score"] for d in group])
        )
        averaged["learning_rate_at_detection"].append(
            np.mean([d["learning_rate_at_detection"] for d in group])
        )

    return averaged


def plot_drift_detection_metrics_from_dicts(
    drift_dicts: List[Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
):
    """
    Plot drift detection metrics averaged per sweep eta.

    Returns the figure, or None when ``drift_dicts`` is empty.
    """
    if not drift_dicts:
        logger.warning("No drift detection data to plot")
        return None

    averaged_data = _average_drift_dicts_by_eta(drift_dicts)
    etas = averaged_data["eta"]

    apply_paper_plot_style()
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    metrics = [
        ("window_idx", "Window Index"),
        ("baseline_mean", "Baseline Mean"),
        ("baseline_std", "Baseline Std"),
        ("current_glrt_z_score", "Current GLRT Z-Score"),
        ("learning_rate_at_detection", "Learning Rate at Detection"),
    ]

    plot_idx = 0
    for metric_key, metric_label in metrics:
        ax = axes[plot_idx]
        ax.plot(etas, averaged_data[metric_key], "o-", linewidth=2, markersize=6)
        ax.set_xlabel("Eta", fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(f"{metric_label} vs Eta", fontsize=12)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    ax = axes[plot_idx]
    ax.errorbar(
        etas,
        averaged_data["main_log_glr"],
        yerr=averaged_data["main_log_glr_std"],
        fmt="o-",
        linewidth=2,
        markersize=6,
        capsize=5,
        capthick=2,
        label="Mean ± Std",
        elinewidth=1.5,
    )
    ax.set_xlabel("Eta", fontsize=11)
    ax.set_ylabel("Main Log-GLR", fontsize=11)
    ax.set_title("Main Log-GLR vs Eta (Mean ± Std)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plot_idx += 1

    axes[plot_idx].axis("off")
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        save_figure(fig, output_path)
        logger.info("Saved drift detection metrics plot to %s", output_path)
    else:
        plt.show()

    return fig


def plot_drift_detection_metrics(
    json_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
):
    """Load drift detection dicts from JSON and plot metrics vs eta."""
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as handle:
        drift_dicts = json.load(handle)

    return plot_drift_detection_metrics_from_dicts(drift_dicts, output_path)


def plot_drift_detection_metrics_in_output_dir(
    output_dir: Union[str, Path],
    *,
    json_name: str = DRIFT_DETECTION_JSON_FILENAME,
    output_name: str = DRIFT_METRICS_PLOT_FILENAME,
) -> Optional[Path]:
    """Load ``drift_detection_dicts.json`` from *output_dir* and save aggregate plot."""
    output_dir = Path(output_dir)
    json_path = output_dir / json_name
    if not json_path.exists():
        logger.debug("No drift detection JSON at %s; skipping metrics plot", json_path)
        return None

    with open(json_path, "r", encoding="utf-8") as handle:
        drift_dicts = json.load(handle)

    output_path = output_dir / output_name
    plot_drift_detection_metrics_from_dicts(drift_dicts, output_path)
    return output_path if output_path.exists() else None

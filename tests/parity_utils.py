"""Helpers for legacy vs CLI v2 numeric parity checks."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

WORKSPACE_ROOT = Path(__file__).parent.parent
PAPER_ETA_SWEEP_CONFIG = (
    WORKSPACE_ROOT / "configs/Used_for_paper/SineAccel_base_model_Online_learning_eta_sweep_config.yaml"
)

# Trimmed paper run: same physics, fast enough for CI
PAPER_ETA_TRIM_OVERRIDES = [
    "scenario_config.values=0.4,0.6",
    "online_learning.enable_lr_sweep=false",
    "online_learning.trajectory_length=15",
    "online_learning.max_iterations=2",
    "online_learning.dataset_size=1",
    "online_learning.time_to_learn=1",
    "logging.level=ERROR",
    "simulation.plot_results=false",
    "simulation.save_plots=false",
]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_ol_numeric_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract comparable scalar/list metrics from an online-learning result dict."""
    metrics: Dict[str, Any] = {}
    ol = result.get("online_learning_results", {})
    metrics["training_start_window"] = ol.get("training_start_window")
    metrics["training_end_window"] = ol.get("training_end_window")

    ar = result.get("averaged_results", {})
    for traj_key in (
        "averaged_pretrained_trajectory",
        "averaged_online_trajectory",
        "averaged_supervised_trajectory",
    ):
        sub = ar.get(traj_key, {})
        for loss_key in ("main_losses", "training_losses", "pre_ekf_losses"):
            losses = sub.get(loss_key, [])
            if losses:
                metrics[f"{traj_key}_{loss_key}"] = [float(x) for x in losses]

    for scalar_key in (
        "avg_glrt_z_score",
        "std_glrt_z_score",
        "avg_learning_rate_at_detection",
        "avg_actual_learning_rate",
    ):
        if result.get(scalar_key) is not None:
            metrics[scalar_key] = float(result[scalar_key])

    return metrics


def compare_numeric_metrics(
    legacy: Dict[str, Any],
    v2: Dict[str, Any],
    rtol: float = 1e-9,
    atol: float = 1e-9,
) -> Tuple[bool, List[str]]:
    """Return (all_match, list of human-readable diffs)."""
    diffs: List[str] = []
    all_keys = set(legacy.keys()) | set(v2.keys())

    for key in sorted(all_keys):
        lv, vv = legacy.get(key), v2.get(key)
        if lv is None and vv is None:
            continue
        if isinstance(lv, list) and isinstance(vv, list):
            if len(lv) != len(vv):
                diffs.append(f"{key}: length {len(lv)} vs {len(vv)}")
                continue
            for i, (a, b) in enumerate(zip(lv, vv)):
                if not np.isclose(a, b, rtol=rtol, atol=atol):
                    diffs.append(f"{key}[{i}]: {a} vs {b} (delta={abs(a-b):.2e})")
        elif isinstance(lv, (int, float)) and isinstance(vv, (int, float)):
            if not np.isclose(float(lv), float(vv), rtol=rtol, atol=atol):
                diffs.append(f"{key}: {lv} vs {vv}")
        elif lv != vv:
            diffs.append(f"{key}: {lv!r} vs {vv!r}")

    return len(diffs) == 0, diffs

"""LR sweep optimality analysis: best-LR selection and sigmoid fits."""

from __future__ import annotations

import json
import logging
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SigmoidFitResult:
    """Log-space sigmoid fit y = L_low + (L_high - L_low) / (1 + exp(-k*(x - x0)))."""

    params: Tuple[float, float, float, float]
    param_stderr: Tuple[float, float, float, float]
    r_squared: float
    input_name: str


def sigmoid(x: np.ndarray, L_low: float, L_high: float, k: float, x0: float) -> np.ndarray:
    return L_low + (L_high - L_low) / (1.0 + np.exp(-k * (x - x0)))


def load_heatmap_data(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_drift_detection_dicts(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in drift detection JSON, got {type(data).__name__}")
    return data


def best_lr_per_eta(heatmap_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return best (lowest-loss) LR entry per eta, sorted by eta."""
    eta_entries: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for eta, lr, lr_type, loss in zip(
        heatmap_data["eta_values"],
        heatmap_data["lr_values"],
        heatmap_data["lr_types"],
        heatmap_data["avg_losses"],
    ):
        eta_entries[float(eta)].append(
            {"eta": float(eta), "lr": float(lr), "lr_type": lr_type, "loss": float(loss)}
        )

    best_rows: List[Dict[str, Any]] = []
    for eta in sorted(eta_entries):
        best = min(eta_entries[eta], key=lambda entry: entry["loss"])
        best_rows.append(best)
    return best_rows


def group_loss_curves_by_lr(heatmap_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Group heatmap rows into one loss-vs-eta curve per LR run."""
    lr_row_ids = heatmap_data.get("lr_row_ids")
    if not lr_row_ids:
        lr_row_ids = [
            "ADAPTIVE" if lr_type == "adaptive" else lr
            for lr_type, lr in zip(heatmap_data["lr_types"], heatmap_data["lr_values"])
        ]

    curves: "OrderedDict[Any, Dict[str, Any]]" = OrderedDict()
    for eta, lr, lr_type, lr_row_id, loss in zip(
        heatmap_data["eta_values"],
        heatmap_data["lr_values"],
        heatmap_data["lr_types"],
        lr_row_ids,
        heatmap_data["avg_losses"],
    ):
        if lr_row_id not in curves:
            curves[lr_row_id] = {
                "lr_type": lr_type,
                "lr_value": float(lr),
                "lr_row_id": lr_row_id,
                "etas": [],
                "losses": [],
                "lr_values": [],
            }
        curves[lr_row_id]["etas"].append(float(eta))
        curves[lr_row_id]["losses"].append(float(loss))
        curves[lr_row_id]["lr_values"].append(float(lr))

    grouped: List[Dict[str, Any]] = []
    for curve in curves.values():
        order = np.argsort(curve["etas"])
        curve["etas"] = np.asarray(curve["etas"], dtype=float)[order]
        curve["losses"] = np.asarray(curve["losses"], dtype=float)[order]
        curve["lr_values"] = np.asarray(curve["lr_values"], dtype=float)[order]
        grouped.append(curve)
    return grouped


def _fit_sigmoid(
    x: np.ndarray,
    y: np.ndarray,
    p0: Tuple[float, float, float, float],
    bounds: Tuple[List[float], List[float]],
    input_name: str,
) -> SigmoidFitResult:
    popt, pcov = curve_fit(sigmoid, x, y, p0=p0, bounds=bounds, maxfev=10000)
    perr = np.sqrt(np.diag(pcov))
    predicted = sigmoid(x, *popt)
    ss_res = np.sum((y - predicted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return SigmoidFitResult(
        params=tuple(float(v) for v in popt),
        param_stderr=tuple(float(v) for v in perr),
        r_squared=float(r_squared),
        input_name=input_name,
    )


def fit_eta_to_lr_sigmoid(etas: np.ndarray, optimal_lrs: np.ndarray) -> SigmoidFitResult:
    log_lrs = np.log10(np.asarray(optimal_lrs, dtype=float))
    etas = np.asarray(etas, dtype=float)
    return _fit_sigmoid(
        etas,
        log_lrs,
        p0=(np.log10(0.001), np.log10(0.035), 15.0, 0.75),
        bounds=([-5, -3, 0.1, 0.3], [-1, 0, 100, 1.5]),
        input_name="eta",
    )


def fit_observable_to_lr_sigmoid(
    observables: np.ndarray,
    optimal_lrs: np.ndarray,
    input_name: str,
    x0_guess: Optional[float] = None,
) -> SigmoidFitResult:
    log_lrs = np.log10(np.asarray(optimal_lrs, dtype=float))
    observables = np.asarray(observables, dtype=float)
    x_min, x_max = float(observables.min()), float(observables.max())
    x_pad = max((x_max - x_min) * 0.1, 1e-3)
    if x0_guess is None:
        x0_guess = float(np.median(observables))
    return _fit_sigmoid(
        observables,
        log_lrs,
        p0=(float(log_lrs.min()), float(log_lrs.max()), 0.5, x0_guess),
        bounds=([-5, -3, 0.001, x_min - x_pad], [-1, 0, 10, x_max + x_pad]),
        input_name=input_name,
    )


def build_glrt_lr_mapping(
    heatmap_data: Dict[str, Any],
    drift_dicts: List[Dict[str, Any]],
    eta_control_threshold: float = 0.01,
) -> List[Dict[str, Any]]:
    """Join drift observables with optimal LR per eta (excludes eta≈0 control)."""
    eta_best = OrderedDict()
    for row in best_lr_per_eta(heatmap_data):
        eta_best[row["eta"]] = row

    eta_glrt: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for entry in drift_dicts:
        eta_glrt[float(entry["eta"])].append(entry)

    mapping: List[Dict[str, Any]] = []
    for eta in sorted(eta_best):
        if eta < eta_control_threshold or eta not in eta_glrt:
            continue
        group = eta_glrt[eta]
        avg_log_glr = float(np.mean([d["main_log_glr"] for d in group]))
        avg_glr_diff = float(
            np.mean([d["main_log_glr"] - d["baseline_mean"] for d in group])
        )
        mapping.append(
            {
                "eta": eta,
                "avg_log_glr": avg_log_glr,
                "avg_glr_diff": avg_glr_diff,
                "optimal_lr": eta_best[eta]["lr"],
                "lr_type": eta_best[eta]["lr_type"],
                "loss": eta_best[eta]["loss"],
            }
        )
    return mapping


def postprocess_lr_sweep_analysis(
    output_dir: Path,
    heatmap_data: Dict[str, Any],
    drift_dicts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Path]:
    """
    Generate LR optimality plots after an eta+LR sweep.

    Returns dict of plot stem -> saved path for plots that were created.
    """
    from utils.plotting import (
        plot_glrt_observable_to_optimal_lr,
        plot_loss_vs_eta_per_lr,
        plot_optimal_lr_vs_eta,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    created: Dict[str, Path] = {}

    optimal_path = plot_optimal_lr_vs_eta(heatmap_data, output_dir)
    if optimal_path is not None:
        created["optimal_lr_vs_eta"] = optimal_path

    per_lr_path = plot_loss_vs_eta_per_lr(heatmap_data, output_dir)
    if per_lr_path is not None:
        created["loss_vs_eta_per_lr"] = per_lr_path

    if drift_dicts is None:
        drift_path = output_dir / "drift_detection_dicts.json"
        if drift_path.exists():
            drift_dicts = load_drift_detection_dicts(drift_path)

    if drift_dicts:
        glrt_path = plot_glrt_observable_to_optimal_lr(heatmap_data, drift_dicts, output_dir)
        if glrt_path is not None:
            created["glrt_observable_to_optimal_lr"] = glrt_path
    else:
        logger.info("Skipping GLRT→LR plot: no drift_detection_dicts available")

    return created

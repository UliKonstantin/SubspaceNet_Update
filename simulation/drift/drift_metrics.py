"""Aggregate drift-detection metrics vs sweep eta."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from utils.plotting.style import (
    ETA_XLABEL,
    FIG_DOUBLE,
    PLOT_COLORS,
    WINDOW_XLABEL,
    apply_paper_plot_style,
    save_figure,
    style_axes,
)

logger = logging.getLogger(__name__)

DRIFT_METRICS_PLOT_FILENAME = "drift_detection_metrics_vs_eta.png"
DRIFT_DETECTION_JSON_FILENAME = "drift_detection_dicts.json"
SCENARIO_STUB_FILENAME = "scenario_results_stub.json"

# Paper yaml defaults (adaptive LR sigmoid); override via plot_drift_detection_metrics_in_output_dir(sigmoid_params=...)
DEFAULT_ADAPTIVE_LR_SIGMOID = {
    "lr_min": 0.0005,
    "lr_max": 0.02,
    "k_sigmoid": 0.7336,
    "dG0": 14.5,
}

# One formula per panel — together they cover detector + adaptive-LR policy.
DRIFT_PANEL_FORMULAS = {
    "trigger_window": (
        r"$w^* = \min\{w : z_w > z_{\mathrm{thr}}\}$"
    ),
    "log_glr": (
        r"$G_w = \max_{\tau}\left[\log L_1(\tau) - \log L_0\right]$"
    ),
    "z_score": (
        r"$z_w = \frac{G_w - \bar{G}_{\mathrm{base}}}{\sigma_{\mathrm{base}}},"
        r"\quad \bar{G}_{\mathrm{base}} = \mathrm{mean}\left(G_{1:w-G_{\mathrm{guard}}}\right)$"
    ),
    "adaptive_lr": (
        r"$\mathrm{LR} = 10^{\log_{10}\mathrm{LR}_{\min}"
        r"+ \frac{\log_{10}\mathrm{LR}_{\max}-\log_{10}\mathrm{LR}_{\min}}"
        r"{1+\exp[-k(\Delta G-\Delta G_0)]}}$"
    ),
    "delta_g": (
        r"$\Delta G = G_{w^*} - \bar{G}_{\mathrm{base}}$"
    ),
    "sigmoid_map": (
        r"$\log_{10}\mathrm{LR}(\Delta G) = \log_{10}\mathrm{LR}_{\min}"
        r"+ \frac{\log_{10}\mathrm{LR}_{\max}-\log_{10}\mathrm{LR}_{\min}}"
        r"{1+\exp[-k(\Delta G-\Delta G_0)]}$"
    ),
}


def _style_drift_panel(
    ax,
    *,
    title: str,
    formula_key: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """Title plus the panel's equation (centered under the title)."""
    formula = DRIFT_PANEL_FORMULAS[formula_key]
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", pad=16, fontsize=11)
    ax.text(
        0.5,
        1.02,
        formula,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#333333",
    )


def _compute_sigmoid_lr(dG: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    log_lr_min = np.log10(params["lr_min"])
    log_lr_max = np.log10(params["lr_max"])
    log_lr = log_lr_min + (log_lr_max - log_lr_min) / (
        1.0 + np.exp(-params["k_sigmoid"] * (dG - params["dG0"]))
    )
    return 10 ** log_lr


def _dedupe_drift_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []
    for entry in entries:
        key = (
            round(float(entry.get("learning_rate_at_detection", 0)), 8),
            round(float(entry.get("window_idx", 0)), 3),
            round(float(entry.get("main_log_glr", 0)), 4),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


def extract_adaptive_dg_rows_from_drift_dicts(
    drift_dicts: List[Dict[str, Any]],
    scenario_results: Dict[Any, Any],
) -> List[Dict[str, Any]]:
    """
    One adaptive-run ΔG row per scenario η.

    Prefer tagged ``use_adaptive_learning_rate`` entries; fall back to matching the
    adaptive sub-run changepoint window and trajectory η.
    """
    rows: List[Dict[str, Any]] = []
    for key, entry in scenario_results.items():
        if entry is None:
            continue
        eta = float(key)
        block = _glrt_adaptation_block(_unwrap_lr_sweep_result(entry))
        if block is None:
            continue

        trigger_win = block.get("avg_changepoint_window")
        target_lr = block.get("avg_learning_rate")

        tagged = [
            d
            for d in drift_dicts
            if _drift_group_eta(d) == eta and d.get("use_adaptive_learning_rate") is True
        ]
        if tagged:
            chosen = _dedupe_drift_entries(tagged)[0]
        else:
            pool = _dedupe_drift_entries(
                [
                    d
                    for d in drift_dicts
                    if _drift_group_eta(d) == eta and abs(float(d.get("eta", -1)) - eta) < 1e-9
                ]
            )
            if target_lr is not None:
                lr_pool = [
                    d
                    for d in pool
                    if abs(float(d["learning_rate_at_detection"]) - float(target_lr)) < 1e-6
                ]
                if lr_pool:
                    pool = lr_pool
            if trigger_win is not None and pool:
                pool = sorted(pool, key=lambda d: abs(float(d["window_idx"]) - float(trigger_win)))
            if not pool:
                continue
            chosen = pool[0]

        if chosen.get("dG_at_detection") is not None:
            dG = float(chosen["dG_at_detection"])
        else:
            dG = float(chosen["main_log_glr"]) - float(chosen["baseline_mean"])

        rows.append(
            {
                "eta": eta,
                "dG": dG,
                "G": float(chosen["main_log_glr"]),
                "baseline_mean": float(chosen["baseline_mean"]),
                "mapped_lr": float(chosen["learning_rate_at_detection"]),
                "dG0": chosen.get("adaptive_lr_dG0"),
            }
        )

    return sorted(rows, key=lambda row: row["eta"])


def _drift_group_eta(entry: Dict[str, Any]) -> float:
    if "scenario_eta" in entry:
        return float(entry["scenario_eta"])
    return float(entry["eta"])


def _unwrap_lr_sweep_result(entry: dict) -> dict:
    if not isinstance(entry, dict) or "lr_sweep_results" not in entry:
        return entry
    lr_results = entry["lr_sweep_results"]
    if isinstance(lr_results.get("adaptive"), dict):
        adaptive = lr_results["adaptive"]
        if isinstance(adaptive.get("result"), dict):
            return adaptive["result"]
    for value in lr_results.values():
        if isinstance(value, dict) and isinstance(value.get("result"), dict):
            return value["result"]
    return entry


def _glrt_adaptation_block(result: dict) -> Optional[Dict[str, Any]]:
    if not isinstance(result, dict):
        return None
    glrt = result.get("glrt_results")
    if glrt is None and isinstance(result.get("averaged_results"), dict):
        glrt = result["averaged_results"].get("glrt_results")
    if not isinstance(glrt, dict):
        return None
    block = glrt.get("adaptation_loss")
    return block if isinstance(block, dict) else None


def extract_adaptive_glrt_metrics_from_scenario_results(
    scenario_results: Dict[Any, Any],
) -> List[Dict[str, Any]]:
    """
    One row per sweep η from the adaptive-LR sub-run GLRT summary.

    Uses ``glrt_results.adaptation_loss`` (MSIE / adaptation-loss GLRT stream).
    """
    rows: List[Dict[str, Any]] = []
    for key, entry in scenario_results.items():
        if entry is None:
            continue
        eta = float(key)
        result = _unwrap_lr_sweep_result(entry)
        block = _glrt_adaptation_block(result)
        if block is None:
            continue
        rows.append(
            {
                "eta": eta,
                "trigger_window": block.get("avg_changepoint_window"),
                "trigger_window_std": block.get("std_changepoint_window", 0.0),
                "log_glr": block.get("avg_likelihood"),
                "log_glr_std": block.get("std_likelihood", 0.0),
                "z_score": block.get("avg_z_score"),
                "z_score_std": block.get("std_z_score", 0.0),
                "adaptive_lr": block.get("avg_learning_rate"),
                "adaptive_lr_std": block.get("std_learning_rate", 0.0),
            }
        )
    return sorted(rows, key=lambda row: row["eta"])


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


def plot_drift_detection_metrics_from_scenario_results(
    scenario_results: Dict[Any, Any],
    output_path: Optional[Union[str, Path]] = None,
    *,
    drift_dicts: Optional[List[Dict[str, Any]]] = None,
    sigmoid_params: Optional[Dict[str, float]] = None,
):
    """
    Plot GLRT drift-detection observables vs calibration-error sweep η.

    Data source: adaptive-LR sub-run only (``glrt_results.adaptation_loss``).
    Optional bottom row: ΔG diagnostic + sigmoid saturation check.
    """
    rows = extract_adaptive_glrt_metrics_from_scenario_results(scenario_results)
    if not rows:
        logger.warning("No adaptive GLRT metrics found in scenario results")
        return None

    dg_rows = extract_adaptive_dg_rows_from_drift_dicts(drift_dicts or [], scenario_results)
    sigmoid = {**DEFAULT_ADAPTIVE_LR_SIGMOID, **(sigmoid_params or {})}
    dG0 = float(dg_rows[0]["dG0"]) if dg_rows and dg_rows[0].get("dG0") is not None else sigmoid["dG0"]

    etas = np.array([row["eta"] for row in rows], dtype=float)
    eta_ticks = [f"{e:g}" for e in etas]

    apply_paper_plot_style()
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 12.5))
    panels = [
        (
            axes[0, 0],
            "trigger_window",
            "trigger_window_std",
            WINDOW_XLABEL,
            "GLRT trigger window",
            "trigger_window",
            PLOT_COLORS["online"],
        ),
        (
            axes[0, 1],
            "log_glr",
            "log_glr_std",
            "Log-GLR at changepoint",
            "GLRT statistic magnitude",
            "log_glr",
            PLOT_COLORS["glrt"],
        ),
        (
            axes[1, 0],
            "z_score",
            "z_score_std",
            "GLRT z-score at trigger",
            "Standardized exceedance over pre-drift baseline",
            "z_score",
            PLOT_COLORS["changepoint"],
        ),
        (
            axes[1, 1],
            "adaptive_lr",
            "adaptive_lr_std",
            "Adaptive LR at trigger",
            "Sigmoid-mapped learning rate",
            "adaptive_lr",
            PLOT_COLORS["adaptive"],
        ),
    ]

    for ax, value_key, std_key, ylabel, title, formula_key, color in panels:
        values = np.array([row[value_key] for row in rows], dtype=float)
        stds = np.array([row.get(std_key) or 0.0 for row in rows], dtype=float)
        if np.any(stds > 0):
            ax.errorbar(
                etas,
                values,
                yerr=stds,
                fmt="o-",
                color=color,
                linewidth=2,
                markersize=7,
                capsize=4,
                capthick=1.5,
                label="Mean ± std across trajectories",
            )
        else:
            ax.plot(etas, values, "o-", color=color, linewidth=2, markersize=7)
        _style_drift_panel(
            ax,
            title=title,
            formula_key=formula_key,
            xlabel=ETA_XLABEL,
            ylabel=ylabel,
        )
        ax.set_xticks(etas)
        ax.set_xticklabels(eta_ticks)

    lr_values = np.array([row["adaptive_lr"] for row in rows], dtype=float)
    lr_ax = axes[1, 1]
    if np.all(lr_values > 0) and lr_values.max() / lr_values.min() > 3:
        lr_ax.set_yscale("log")
    else:
        lr_mid = float(np.mean(lr_values))
        lr_span = max(float(np.max(lr_values) - np.min(lr_values)), lr_mid * 0.05, 1e-6)
        lr_ax.set_ylim(lr_mid - 1.6 * lr_span, lr_mid + 1.6 * lr_span)
        lr_ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)

    # Row 3 — adaptive LR sigmoid diagnostic
    ax_dg = axes[2, 0]
    ax_sig = axes[2, 1]
    if dg_rows:
        dg_etas = np.array([r["eta"] for r in dg_rows], dtype=float)
        dG_vals = np.array([r["dG"] for r in dg_rows], dtype=float)
        mapped_lrs = np.array([r["mapped_lr"] for r in dg_rows], dtype=float)

        ax_dg.plot(dg_etas, dG_vals, "o-", color=PLOT_COLORS["event"], linewidth=2, markersize=7, label=r"$\Delta G$ at trigger")
        ax_dg.axhline(dG0, color="#888888", linestyle="--", linewidth=1.5, label=rf"Sigmoid inflection $G_0$={dG0:g}")
        _style_drift_panel(
            ax_dg,
            title="Sigmoid input at trigger (adaptive run)",
            formula_key="delta_g",
            xlabel=ETA_XLABEL,
            ylabel=r"$\Delta G = G - \bar{G}_{\mathrm{base}}$",
        )
        ax_dg.legend(loc="best", fontsize=9)

        dG_dense = np.linspace(max(0, dG_vals.min() - 5), max(dG_vals.max(), dG0) + 5, 300)
        lr_curve = _compute_sigmoid_lr(dG_dense, sigmoid)
        ax_sig.plot(dG_dense, lr_curve, color="#888888", linewidth=1.8, label="Adaptive LR sigmoid")
        ax_sig.scatter(
            dG_vals, mapped_lrs, s=90, c=PLOT_COLORS["adaptive"], edgecolors="black",
            linewidths=0.7, zorder=5, label="Trigger events",
        )
        for eta_val, dG_val, lr_val in zip(dg_etas, dG_vals, mapped_lrs):
            ax_sig.annotate(
                f"η={eta_val:g}",
                (dG_val, lr_val),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )
        _style_drift_panel(
            ax_sig,
            title="Trigger ΔG → adaptive LR (sigmoid map)",
            formula_key="sigmoid_map",
            xlabel=r"$\Delta G$ at trigger",
            ylabel="Mapped LR at trigger",
        )
        ax_sig.set_yscale("log")
        ax_sig.axvline(dG0, color="#CCCCCC", linestyle=":", linewidth=1.2)
        ax_sig.legend(loc="lower right", fontsize=9)
        ax_dg.set_xticks(dg_etas)
        ax_dg.set_xticklabels([f"{e:g}" for e in dg_etas])
    else:
        for ax in (ax_dg, ax_sig):
            ax.axis("off")
            ax.text(0.5, 0.5, "No adaptive ΔG data\n(re-run sweep for tagged drift dicts)", ha="center", va="center")

    fig.suptitle(
        "GLRT drift-detection observables vs steering-error sweep (adaptive LR run)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        "Each point: one scenario η, adaptive learning rate, adaptation-loss GLRT stream "
        "(same trigger used to start online training)",
        ha="center",
        fontsize=10,
        style="italic",
        color="#555555",
    )
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.08, top=0.90, hspace=0.72, wspace=0.38)

    if output_path:
        output_path = Path(output_path)
        save_figure(fig, output_path, pad_inches=0.2)
        logger.info("Saved drift detection metrics plot to %s", output_path)
    else:
        plt.show()

    return fig


def plot_drift_detection_metrics_from_dicts(
    drift_dicts: List[Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
):
    """
    Fallback plot from raw ``drift_detection_dicts`` (debug / legacy).

    Prefer ``plot_drift_detection_metrics_from_scenario_results`` when a scenario
    stub is available — raw dicts mix all LR sweep runs and are easy to misread.
    """
    if not drift_dicts:
        logger.warning("No drift detection data to plot")
        return None

    averaged_data = _average_drift_dicts_by_eta(drift_dicts)
    etas = averaged_data["eta"]

    apply_paper_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=FIG_DOUBLE, sharex=True)

    panels = [
        (axes[0, 0], averaged_data["window_idx"], WINDOW_XLABEL, "GLRT trigger window", "Mean window index at trigger"),
        (axes[0, 1], averaged_data["main_log_glr"], "Log-GLR at trigger", "Adaptation-loss log-GLR", "Mean log-GLR when trigger fired"),
        (axes[1, 0], averaged_data["current_glrt_z_score"], "GLRT z-score at trigger", "Standardized exceedance", "Mean z-score when trigger fired"),
        (axes[1, 1], averaged_data["learning_rate_at_detection"], "LR at trigger", "Learning rate used", "Mean LR (all sweep runs — use scenario stub instead)"),
    ]
    for ax, values, ylabel, title, subtitle in panels:
        ax.plot(etas, values, "o-", linewidth=2, markersize=6, color=PLOT_COLORS["online"])
        style_axes(ax, xlabel=ETA_XLABEL, ylabel=ylabel, title=title)
        ax.text(0.02, 0.02, subtitle, transform=ax.transAxes, fontsize=9, va="bottom", color="#666666")

    fig.suptitle(
        "GLRT drift metrics vs η (raw dict average — includes all LR runs)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

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
    stub_name: str = SCENARIO_STUB_FILENAME,
    output_name: str = DRIFT_METRICS_PLOT_FILENAME,
    sigmoid_params: Optional[Dict[str, float]] = None,
) -> Optional[Path]:
    """Save aggregate drift plot; prefers ``scenario_results_stub.json`` when present."""
    output_dir = Path(output_dir)
    output_path = output_dir / output_name

    drift_dicts: List[Dict[str, Any]] = []
    json_path = output_dir / json_name
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as handle:
            drift_dicts = json.load(handle)

    stub_path = output_dir / stub_name
    if stub_path.exists():
        with open(stub_path, "r", encoding="utf-8") as handle:
            scenario_results = json.load(handle)
        fig = plot_drift_detection_metrics_from_scenario_results(
            scenario_results,
            output_path,
            drift_dicts=drift_dicts,
            sigmoid_params=sigmoid_params,
        )
        return output_path if fig is not None and output_path.exists() else None

    if not drift_dicts:
        logger.debug("No drift stub or JSON at %s; skipping metrics plot", output_dir)
        return None

    plot_drift_detection_metrics_from_dicts(drift_dicts, output_path)
    return output_path if output_path.exists() else None

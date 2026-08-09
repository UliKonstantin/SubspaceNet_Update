"""Plotting helpers — lr plots."""
from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from utils.plotting.style import apply_lr_analysis_plot_style


def plot_optimal_lr_vs_eta(heatmap_data: Dict, output_dir: Path) -> Path | None:
    """Plot optimal LR vs eta with log-space sigmoid fit."""
    import matplotlib.pyplot as plt

    from utils.lr_analysis import best_lr_per_eta, fit_eta_to_lr_sigmoid, sigmoid

    logger = logging.getLogger(__name__)
    best_rows = best_lr_per_eta(heatmap_data)
    if len(best_rows) < 2:
        logger.warning("Need at least 2 eta values for optimal LR plot; skipping")
        return None

    etas = np.array([row["eta"] for row in best_rows], dtype=float)
    best_lrs = np.array([row["lr"] for row in best_rows], dtype=float)
    best_losses = np.array([row["loss"] for row in best_rows], dtype=float)
    best_types = [row["lr_type"] for row in best_rows]

    fit = fit_eta_to_lr_sigmoid(etas, best_lrs)
    eta_dense = np.linspace(etas.min() * 0.8, etas.max() * 1.05, 300)
    lr_fit = 10 ** sigmoid(eta_dense, *fit.params)

    apply_lr_analysis_plot_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        eta_dense,
        lr_fit,
        color="#888888",
        linewidth=1.8,
        linestyle="-",
        zorder=3,
        label=(
            r"$\mathrm{LR}^*(G) = 10^{\,\log_{10}\mathrm{LR}_{\min}\;+\;"
            r"\frac{\log_{10}\mathrm{LR}_{\max}\,-\,\log_{10}\mathrm{LR}_{\min}}"
            r"{1\,+\,e^{-k(G\,-\,G_0)}}}$"
        ),
    )

    static_labeled = False
    adaptive_labeled = False
    for eta, lr, loss, lr_type in zip(etas, best_lrs, best_losses, best_types):
        color = "#E07B39" if lr_type == "adaptive" else "#3A76AF"
        marker = "D" if lr_type == "adaptive" else "o"
        label = None
        if lr_type == "adaptive" and not adaptive_labeled:
            label = "Adaptive"
            adaptive_labeled = True
        elif lr_type == "static" and not static_labeled:
            label = "Static"
            static_labeled = True
        ax.scatter(
            eta,
            lr,
            c=color,
            s=100,
            marker=marker,
            zorder=6,
            edgecolors="black",
            linewidths=0.7,
            label=label,
        )
        ax.annotate(
            f"{loss:.3f}",
            (eta, lr),
            textcoords="offset points",
            xytext=(0, 13),
            ha="center",
            fontsize=10,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#CCCCCC", alpha=0.85, lw=0.5),
        )

    ax.set_xlabel(r"Calibration Error $\eta$")
    ax.set_ylabel(r"Optimal Learning Rate $\mathrm{LR}^*$")
    ax.set_title(r"Optimal LR vs $\eta$", fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_xlim(etas.min() - 0.08, etas.max() + 0.08)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="#CCCCCC")
    plt.tight_layout()

    plot_path = Path(output_dir) / "optimal_lr_vs_eta.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        "Saved optimal LR plot to %s (eta sigmoid R²=%.4f)",
        plot_path,
        fit.r_squared,
    )
    return plot_path


def plot_loss_vs_eta_per_lr(heatmap_data: Dict, output_dir: Path) -> Path | None:
    """Plot post-learning loss vs eta, one curve per LR run."""
    import matplotlib.pyplot as plt

    from utils.lr_analysis import group_loss_curves_by_lr

    logger = logging.getLogger(__name__)
    curves = group_loss_curves_by_lr(heatmap_data)
    if not curves:
        logger.warning("No LR curves found for loss-vs-eta plot; skipping")
        return None

    etas = np.array(heatmap_data["eta_values"], dtype=float)
    static_styles = [
        {"color": "#2166AC", "marker": "o", "ls": "-"},
        {"color": "#4393C3", "marker": "s", "ls": "--"},
        {"color": "#92C5DE", "marker": "^", "ls": "-."},
        {"color": "#B2ABD2", "marker": "v", "ls": ":"},
        {"color": "#762A83", "marker": "p", "ls": "-"},
        {"color": "#1B7837", "marker": "h", "ls": "--"},
    ]
    adaptive_style = {"color": "#D6604D", "marker": "D", "ls": "-"}

    apply_lr_analysis_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    static_idx = 0
    for curve in curves:
        if curve["lr_type"] == "adaptive":
            style = adaptive_style
            lr_min = float(np.min(curve["lr_values"]))
            lr_max = float(np.max(curve["lr_values"]))
            label = f"Adaptive (LR: {lr_min:.4f} - {lr_max:.4f})"
        else:
            style = static_styles[static_idx % len(static_styles)]
            static_idx += 1
            label = f"Static LR = {curve['lr_value']}"

        ax.plot(
            curve["etas"],
            curve["losses"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["ls"],
            linewidth=2.2,
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=0.6,
            label=label,
            zorder=4,
        )

    ax.set_xlabel(r"Calibration Error $\eta$")
    ax.set_ylabel("Post-Learning RMSPE")
    ax.set_title(r"Post-Learning Loss vs $\eta$ for Each Learning Rate", fontweight="bold")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_xlim(etas.min() - 0.05, etas.max() + 0.05)
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="#CCCCCC", fontsize=11)
    plt.tight_layout()

    plot_path = Path(output_dir) / "loss_vs_eta_per_lr.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved per-LR loss plot to %s", plot_path)
    return plot_path


def plot_glrt_observable_to_optimal_lr(
    heatmap_data: Dict,
    drift_dicts: list,
    output_dir: Path,
) -> Path | None:
    """Plot sigmoid fits from GLRT observables to optimal LR."""
    import matplotlib.pyplot as plt

    from utils.lr_analysis import (
        build_glrt_lr_mapping,
        fit_observable_to_lr_sigmoid,
        sigmoid,
    )

    logger = logging.getLogger(__name__)
    mapping = build_glrt_lr_mapping(heatmap_data, drift_dicts)
    if len(mapping) < 2:
        logger.warning("Need at least 2 eta values with drift data for GLRT→LR plot; skipping")
        return None

    etas = np.array([row["eta"] for row in mapping], dtype=float)
    map_log_glrs = np.array([row["avg_log_glr"] for row in mapping], dtype=float)
    map_glr_diffs = np.array([row["avg_glr_diff"] for row in mapping], dtype=float)
    map_opt_lrs = np.array([row["optimal_lr"] for row in mapping], dtype=float)

    fit_glr = fit_observable_to_lr_sigmoid(map_log_glrs, map_opt_lrs, "main_log_glr")
    fit_diff = fit_observable_to_lr_sigmoid(map_glr_diffs, map_opt_lrs, "glr_diff")

    if fit_diff.r_squared > fit_glr.r_squared:
        logger.info(
            "GLR diff observable fits better than main_log_glr (R² %.4f vs %.4f)",
            fit_diff.r_squared,
            fit_glr.r_squared,
        )

    apply_lr_analysis_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    x_dense = np.linspace(map_log_glrs.min() * 0.9, map_log_glrs.max() * 1.05, 300)
    ax1.plot(
        x_dense,
        10 ** sigmoid(x_dense, *fit_glr.params),
        color="#888888",
        linewidth=1.8,
        zorder=3,
        label=(
            r"$\mathrm{LR}^*(G) = 10^{\,\log_{10}\mathrm{LR}_{\min}\;+\;"
            r"\frac{\Delta\log}{1+e^{-k(G-G_0)}}}$"
            f"\n$R^2 = {fit_glr.r_squared:.3f}$"
        ),
    )
    for eta, glr, lr in zip(etas, map_log_glrs, map_opt_lrs):
        ax1.scatter(glr, lr, s=100, c="#3A76AF", edgecolors="black", linewidths=0.7, zorder=6)
        ax1.annotate(
            f"$\\eta$={eta:g}",
            (glr, lr),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#CCC", alpha=0.85, lw=0.5),
        )
    ax1.set_xlabel("Main Log-GLR at Detection ($G$)")
    ax1.set_ylabel(r"Optimal Learning Rate $\mathrm{LR}^*$")
    ax1.set_title("LR* vs Log-GLR", fontweight="bold")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.25, linewidth=0.6)
    ax1.legend(loc="lower right", fontsize=10, framealpha=0.9, edgecolor="#CCC")

    x_dense2 = np.linspace(map_glr_diffs.min() * 0.9, map_glr_diffs.max() * 1.05, 300)
    ax2.plot(
        x_dense2,
        10 ** sigmoid(x_dense2, *fit_diff.params),
        color="#888888",
        linewidth=1.8,
        zorder=3,
        label=(
            r"$\mathrm{LR}^*(\Delta G) = 10^{\,\log_{10}\mathrm{LR}_{\min}\;+\;"
            r"\frac{\Delta\log}{1+e^{-k(\Delta G-\Delta G_0)}}}$"
            f"\n$R^2 = {fit_diff.r_squared:.3f}$"
        ),
    )
    for eta, diff, lr in zip(etas, map_glr_diffs, map_opt_lrs):
        ax2.scatter(diff, lr, s=100, c="#E07B39", marker="D", edgecolors="black", linewidths=0.7, zorder=6)
        ax2.annotate(
            f"$\\eta$={eta:g}",
            (diff, lr),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#CCC", alpha=0.85, lw=0.5),
        )
    ax2.set_xlabel(r"GLR Diff at Detection ($\Delta G = G - \bar{G}_{\mathrm{base}}$)")
    ax2.set_ylabel(r"Optimal Learning Rate $\mathrm{LR}^*$")
    ax2.set_title(r"LR* vs GLR Diff", fontweight="bold")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.25, linewidth=0.6)
    ax2.legend(loc="lower right", fontsize=10, framealpha=0.9, edgecolor="#CCC")

    plt.tight_layout()
    plot_path = Path(output_dir) / "glrt_observable_to_optimal_lr.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved GLRT→LR plot to %s", plot_path)
    return plot_path


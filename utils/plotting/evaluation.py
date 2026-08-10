"""Plotting helpers — evaluation."""
from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from utils.plotting.style import (
    PLOT_COLORS,
    apply_paper_plot_style,
    save_current_figure,
    save_figure,
    style_axes,
)
from utils.plotting.sweeps import SCENARIO_AXIS_LABELS, SCENARIO_PLOT_TITLES


def plot_loss_vs_scenario(scenario_results, scenario, output_dir):
    """Plot SubspaceNet snapshot and EKF posterior RMSPE vs sweep values."""
    apply_paper_plot_style()
    logger = logging.getLogger("SubspaceNet.plotting")
    x_vals = list(scenario_results.keys())
    esprit_losses, dnn_losses, ekf_losses = [], [], []
    for v in x_vals:
        res = scenario_results[v]
        if isinstance(res, (float, int)):
            esprit_loss, dnn_loss, ekf_loss = res, None, None
        elif isinstance(res, dict):
            esprit_loss = None
            if (
                "evaluation_results" in res
                and "classic_methods_test_losses" in res["evaluation_results"]
                and "ESPRIT" in res["evaluation_results"]["classic_methods_test_losses"]
            ):
                esprit_loss = res["evaluation_results"]["classic_methods_test_losses"]["ESPRIT"]
            eval_results = res.get("evaluation_results", res)
            dnn_loss = eval_results.get("dnn_test_loss")
            ekf_loss = eval_results.get("ekf_test_loss")
        else:
            esprit_loss = dnn_loss = ekf_loss = None
        esprit_losses.append(esprit_loss)
        dnn_losses.append(dnn_loss)
        ekf_losses.append(ekf_loss)
    if all(l is None for l in esprit_losses + dnn_losses + ekf_losses):
        logger.warning("All losses are None for scenario %s. Plot will be empty.", scenario)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if any(l is not None for l in esprit_losses):
        ax.plot(x_vals, esprit_losses, "-o", label="ESPRIT", color=PLOT_COLORS["esprit"])
    if any(l is not None for l in dnn_losses):
        ax.plot(x_vals, dnn_losses, "-s", label="SubspaceNet snapshot", color=PLOT_COLORS["dnn"])
    if any(l is not None for l in ekf_losses):
        ax.plot(x_vals, ekf_losses, "-^", label="EKF posterior", color=PLOT_COLORS["ekf"])
    x_label = SCENARIO_AXIS_LABELS.get(scenario, scenario)
    title = SCENARIO_PLOT_TITLES.get(scenario, f"DOA tracking error vs {scenario}")
    style_axes(ax, xlabel=x_label, ylabel="Mean RMSPE (rad)", title=title)
    ax.legend(loc="best")
    fig.tight_layout()
    plot_path = Path(output_dir) / f"loss_vs_{scenario}.png"
    save_current_figure(plot_path)
    return plot_path


def plot_2d_kalman_noise_sweep(scenario_results, output_dir):
    """Plot 2D heatmaps: DNN / EKF / ESPRIT loss vs KF noise parameters."""
    apply_paper_plot_style()
    logger = logging.getLogger("SubspaceNet.plotting")
    meas_noise_values = sorted(scenario_results.keys())
    proc_noise_values = sorted(list(scenario_results.values())[0].keys())
    dnn_loss_matrix = np.full((len(proc_noise_values), len(meas_noise_values)), np.nan)
    ekf_loss_matrix = np.full((len(proc_noise_values), len(meas_noise_values)), np.nan)
    esprit_loss_matrix = np.full((len(proc_noise_values), len(meas_noise_values)), np.nan)

    for i, meas_noise in enumerate(meas_noise_values):
        for j, proc_noise in enumerate(proc_noise_values):
            result = scenario_results[meas_noise][proc_noise]
            if isinstance(result, dict) and "evaluation_results" in result:
                ev = result["evaluation_results"]
                if ev.get("dnn_test_loss") is not None:
                    dnn_loss_matrix[j, i] = ev["dnn_test_loss"]
                if ev.get("ekf_test_loss") is not None:
                    ekf_loss_matrix[j, i] = ev["ekf_test_loss"]
                classic = ev.get("classic_methods_test_losses", {})
                if "ESPRIT" in classic:
                    esprit_loss_matrix[j, i] = classic["ESPRIT"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, matrix, title, cmap in (
        (axes[0], dnn_loss_matrix, "SubspaceNet snapshot", "viridis"),
        (axes[1], ekf_loss_matrix, "EKF posterior", "inferno"),
        (axes[2], esprit_loss_matrix, "ESPRIT", "plasma"),
    ):
        im = ax.imshow(matrix, cmap=cmap, aspect="auto", origin="lower")
        style_axes(
            ax,
            xlabel="Measurement noise σ",
            ylabel="Process noise σ",
            title=f"{title} RMSPE vs KF noise",
        )
        ax.set_xticks(range(len(meas_noise_values)))
        ax.set_xticklabels([f"{v:.3f}" for v in meas_noise_values], rotation=45, ha="right")
        ax.set_yticks(range(len(proc_noise_values)))
        ax.set_yticklabels([f"{v:.3f}" for v in proc_noise_values])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("RMSPE (rad)")
        for ii in range(len(meas_noise_values)):
            for jj in range(len(proc_noise_values)):
                if not np.isnan(matrix[jj, ii]):
                    ax.text(
                        ii, jj, f"{matrix[jj, ii]:.3f}",
                        ha="center", va="center", color="white", fontsize=8,
                    )
    fig.tight_layout()
    plot_path = Path(output_dir) / "kalman_noise_2d_heatmap.png"
    save_figure(fig, plot_path)
    _plot_kalman_noise_optimum_analysis(
        scenario_results, output_dir, dnn_loss_matrix, ekf_loss_matrix,
        esprit_loss_matrix, meas_noise_values, proc_noise_values,
    )
    return plot_path


def _plot_kalman_noise_optimum_analysis(
    scenario_results, output_dir, dnn_loss_matrix, ekf_loss_matrix,
    esprit_loss_matrix, meas_noise_values, proc_noise_values,
):
    apply_paper_plot_style()
    logger = logging.getLogger("SubspaceNet.plotting")
    fig, axes = plt.subplots(3, 2, figsize=(10, 11))
    panels = [
        (dnn_loss_matrix, meas_noise_values, "SubspaceNet", "Measurement noise σ", 0, 0),
        (dnn_loss_matrix, proc_noise_values, "SubspaceNet", "Process noise σ", 0, 1, True),
        (ekf_loss_matrix, meas_noise_values, "EKF", "Measurement noise σ", 1, 0),
        (ekf_loss_matrix, proc_noise_values, "EKF", "Process noise σ", 1, 1, True),
        (esprit_loss_matrix, meas_noise_values, "ESPRIT", "Measurement noise σ", 2, 0),
        (esprit_loss_matrix, proc_noise_values, "ESPRIT", "Process noise σ", 2, 1, True),
    ]
    for item in panels:
        matrix, xvals, name, xlabel, row, col = item[:6]
        axis = 0 if len(item) < 7 else 1
        valid = ~np.isnan(matrix)
        if not np.any(valid):
            continue
        means = np.nanmean(matrix, axis=axis)
        ax = axes[row, col]
        ax.plot(xvals, means, "o-")
        style_axes(ax, xlabel=xlabel, ylabel="Mean RMSPE (rad)", title=f"{name}: averaged slice")
    fig.tight_layout()
    save_figure(fig, Path(output_dir) / "kalman_noise_analysis.png")
    logger.info("Saved Kalman noise analysis plot")


def plot_eval_dnn_ekf_loss_vs_time(dnn_trajectory_results, output_dir):
    """Plot per-step SubspaceNet-only vs EKF posterior RMSPE vs GT."""
    import os
    import torch
    from DCD_MUSIC.src.metrics.rmspe_loss import RMSPELoss

    apply_paper_plot_style()
    logger = logging.getLogger(__name__)
    if not dnn_trajectory_results:
        logger.warning("Skipping eval KF plot: no trajectory results")
        return None

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rmspe_criterion = RMSPELoss().to(device)
    per_traj_dnn, per_traj_ekf = [], []

    for traj in dnn_trajectory_results:
        dnn_steps, ekf_steps = [], []
        for pred, kf, truth in zip(
            traj.get("model_predictions", []),
            traj.get("kf_predictions", []),
            traj.get("ground_truth", []),
        ):
            if pred is None or kf is None or truth is None or len(pred) == 0:
                continue
            with torch.no_grad():
                p = torch.tensor(np.asarray(pred), device=device, dtype=torch.float64).unsqueeze(0)
                k = torch.tensor(np.asarray(kf), device=device, dtype=torch.float64).unsqueeze(0)
                tr = torch.tensor(np.asarray(truth), device=device, dtype=torch.float64).unsqueeze(0)
                dnn_steps.append(rmspe_criterion(p, tr).item())
                ekf_steps.append(rmspe_criterion(k, tr).item())
        if dnn_steps:
            per_traj_dnn.append(dnn_steps)
            per_traj_ekf.append(ekf_steps)

    if not per_traj_dnn:
        logger.warning("Skipping eval KF plot: no valid step losses")
        return None

    max_len = max(len(s) for s in per_traj_dnn)
    dnn_avg = [float(np.mean([s[i] for s in per_traj_dnn if i < len(s)])) for i in range(max_len)]
    ekf_avg = [float(np.mean([s[i] for s in per_traj_ekf if i < len(s)])) for i in range(max_len)]
    steps = np.arange(max_len)
    gain = np.array(dnn_avg) - np.array(ekf_avg)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot(steps, dnn_avg, "-s", linewidth=2, markersize=4, label="SubspaceNet-only", color=PLOT_COLORS["dnn"])
    ax1.plot(steps, ekf_avg, "-o", linewidth=2, markersize=4, label="EKF posterior", color=PLOT_COLORS["ekf"])
    style_axes(ax1, xlabel="", ylabel="RMSPE vs GT (rad)", title="Batch eval: SubspaceNet vs EKF")
    ax1.legend(loc="best")

    ax2.plot(steps, gain, "-d", linewidth=2, markersize=4, label="KF gain (pre-EKF − EKF)", color=PLOT_COLORS["gain"])
    ax2.axhline(0.0, color="black", linestyle="-", alpha=0.35, linewidth=1)
    style_axes(ax2, xlabel="Trajectory step", ylabel="RMSPE reduction (rad)", title="EKF improvement over SubspaceNet-only")
    ax2.legend(loc="best")
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "eval_kf_gain_comparison.png")
    save_figure(fig, plot_path)
    logger.info("Eval KF comparison plot saved to: %s", plot_path)
    return plot_path

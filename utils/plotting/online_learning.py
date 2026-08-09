"""Plotting helpers — online learning."""
from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

def plot_online_learning_results_structured(output_dir, pretrained_trajectory_results, online_trajectory_results, reference_metric_config, adaptation_loss_config, training_start_window=None, training_end_window=None, eta_change_windows=None):
    """
    Plot online learning results using structured data approach.
    
    Args:
        output_dir: Output directory for saving plots
        pretrained_trajectory_results: List of TrajectoryResults for pretrained model
        online_trajectory_results: List of TrajectoryResults for online model
        reference_metric_config: Configuration string for main loss (e.g., "supervised_rmspe")
        adaptation_loss_config: Configuration string for training reference loss (e.g., "multimoment")
        training_start_window: Window index where training started (optional)
        training_end_window: Window index where training ended (optional)
        eta_change_windows: List of window indices where eta changed (optional)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data from structured results
    def extract_loss_data(trajectory_results_list, loss_type):
        """Extract loss data from trajectory results."""
        all_losses = []
        all_eta_values = []
        all_window_indices = []
        
        for trajectory_results in trajectory_results_list:
            for i, window_result in enumerate(trajectory_results.window_results):
                if loss_type == "main":
                    loss_value = window_result.loss_metrics.reference_metric_loss
                elif loss_type == "training_reference":
                    loss_value = window_result.loss_metrics.adaptation_loss
                else:
                    continue
                    
                all_losses.append(loss_value)
                all_eta_values.append(trajectory_results.window_eta_values[i])
                all_window_indices.append(trajectory_results.window_indices[i])
        
        return all_losses, all_eta_values, all_window_indices
    
    # Extract data for both models
    pretrained_main_losses, pretrained_eta_values, pretrained_window_indices = extract_loss_data(pretrained_trajectory_results, "main")
    online_main_losses, online_eta_values, online_window_indices = extract_loss_data(online_trajectory_results, "main")
    
    pretrained_training_losses, _, _ = extract_loss_data(pretrained_trajectory_results, "training_reference")
    online_training_losses, _, _ = extract_loss_data(online_trajectory_results, "training_reference")
    
    # Plot 1: Main Loss Comparison
    plt.figure(figsize=(12, 8))
    
    # Plot pretrained model main losses
    plt.subplot(2, 1, 1)
    plt.plot(pretrained_window_indices, pretrained_main_losses, 'b-', linewidth=2, label='Pretrained Model', marker='o', markersize=4)
    plt.plot(online_window_indices, online_main_losses, 'r-', linewidth=2, label='Algorithm 1', marker='s', markersize=4)
    
    # Add eta change markers to first subplot
    if eta_change_windows:
        for eta_window in eta_change_windows:
            if eta_window >= 1:
                plt.axvline(x=eta_window, color='red', linestyle='--', alpha=0.3, linewidth=1)
                plt.text(eta_window, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.1, 'Distribution Change', rotation=90, verticalalignment='bottom', horizontalalignment='center',
                        color='red', fontsize=14)
    
    # Add training markers to first subplot
    if training_start_window is not None and training_start_window >= 1:
        plt.axvline(x=training_start_window, color='orange', linestyle='-', alpha=0.7, linewidth=2)
        plt.text(training_start_window, plt.ylim()[1] * 0.9, 'Training Start', rotation=90, verticalalignment='top', 
                color='orange', fontweight='bold', fontsize=24)
    
    if training_end_window is not None and training_end_window >= 1:
        plt.axvline(x=training_end_window, color='purple', linestyle='-', alpha=0.7, linewidth=2)
        plt.text(training_end_window, plt.ylim()[1] * 0.9, 'Training End', rotation=90, verticalalignment='top', 
                color='purple', fontweight='bold', fontsize=24)
    
    plt.xlabel('Window Index', fontsize=28)
    plt.ylabel('RMSPE (Supervised)', fontsize=28)
    plt.title('RMSPE (Supervised) Comparison', fontsize=30, fontweight='bold')
    plt.legend(fontsize=26)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=24)
    
    # Plot 2: Training Reference Loss Comparison
    plt.subplot(2, 1, 2)
    plt.plot(pretrained_window_indices, pretrained_training_losses, 'b-', linewidth=2, label='Pretrained Model', marker='o', markersize=4)
    plt.plot(online_window_indices, online_training_losses, 'r-', linewidth=2, label='Algorithm 1', marker='s', markersize=4)
    
    # Add eta change markers to second subplot
    if eta_change_windows:
        for eta_window in eta_change_windows:
            if eta_window >= 1:
                plt.axvline(x=eta_window, color='red', linestyle='--', alpha=0.3, linewidth=1)
                plt.text(eta_window, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.1, 'Distribution Change', rotation=90, verticalalignment='bottom', horizontalalignment='center',
                        color='red', fontsize=14)
    
    # Add training markers to second subplot
    if training_start_window is not None and training_start_window >= 1:
        plt.axvline(x=training_start_window, color='orange', linestyle='-', alpha=0.7, linewidth=2)
        plt.text(training_start_window, plt.ylim()[1] * 0.9, 'Training Start', rotation=90, verticalalignment='top', 
                color='orange', fontweight='bold', fontsize=24)
    
    if training_end_window is not None and training_end_window >= 1:
        plt.axvline(x=training_end_window, color='purple', linestyle='-', alpha=0.7, linewidth=2)
        plt.text(training_end_window, plt.ylim()[1] * 0.9, 'Training End', rotation=90, verticalalignment='top', 
                color='purple', fontweight='bold', fontsize=24)
    
    plt.xlabel('Window Index', fontsize=28)
    plt.ylabel('MSIE (Unsupervised)', fontsize=28)
    plt.title('MSIE (Unsupervised) Comparison', fontsize=30, fontweight='bold')
    plt.legend(fontsize=26)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=24)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'online_learning_structured_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Structured online learning comparison plot saved to: {plot_path}")


def _derive_eta_change_windows(window_indices, eta_values, tol=1e-6):
    """Return actual window indices where eta transitions in a plotted series."""
    if not window_indices or not eta_values or len(window_indices) != len(eta_values):
        return []
    return [
        window_indices[i]
        for i in range(1, len(eta_values))
        if abs(eta_values[i] - eta_values[i - 1]) > tol
    ]


def _filter_series_after_training(window_indices, values, training_end_window):
    """Keep only post-training evaluation windows (strictly after training_end_window)."""
    if training_end_window is None:
        return [], []
    filtered_indices, filtered_values = [], []
    for w, v in zip(window_indices, values):
        if w > training_end_window:
            filtered_indices.append(w)
            filtered_values.append(v)
    return filtered_indices, filtered_values


def plot_averaged_online_learning_results(
    output_dir,
    averaged_pretrained_metrics,
    averaged_online_metrics,
    reference_metric_config,
    adaptation_loss_config,
    training_start_window=None,
    training_end_window=None,
    drift_detection_window=None,
    eta_change_windows=None,
    averaged_supervised_metrics=None,
):
    """
    Plot online learning results using directly averaged metrics (no TrajectoryResults conversion).

    Pretrained model is shown for all windows. Algorithm 1 and supervised oracle appear only
    after training_end_window (post-adaptation evaluation phase).

    Distribution Change = η increment (eta_change_windows). One marker per logged change.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    logger = logging.getLogger(__name__)
    logger.info("Creating averaged online learning results plot...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data directly from averaged metrics
    pretrained_window_indices = averaged_pretrained_metrics.get("window_indices", [])
    pretrained_main_losses = averaged_pretrained_metrics.get("reference_metric_losses", [])
    pretrained_training_losses = averaged_pretrained_metrics.get("adaptation_losses", [])
    pretrained_eta_values = averaged_pretrained_metrics.get("window_eta_values", [])
    
    online_window_indices = averaged_online_metrics.get("window_indices", [])
    online_main_losses = averaged_online_metrics.get("reference_metric_losses", [])
    online_training_losses = averaged_online_metrics.get("adaptation_losses", [])
    online_eta_values = averaged_online_metrics.get("window_eta_values", [])
    
    # Extract supervised model data if available
    supervised_window_indices = []
    supervised_main_losses = []
    supervised_training_losses = []
    supervised_eta_values = []
    
    if averaged_supervised_metrics is not None:
        supervised_window_indices = averaged_supervised_metrics.get("window_indices", [])
        supervised_main_losses = averaged_supervised_metrics.get("reference_metric_losses", [])
        supervised_training_losses = averaged_supervised_metrics.get("adaptation_losses", [])
        supervised_eta_values = averaged_supervised_metrics.get("window_eta_values", [])

    # Online / supervised: post-training eval only
    online_window_indices, online_main_losses = _filter_series_after_training(
        online_window_indices, online_main_losses, training_end_window
    )
    online_window_indices_msie, online_training_losses = _filter_series_after_training(
        averaged_online_metrics.get("window_indices", []),
        averaged_online_metrics.get("adaptation_losses", []),
        training_end_window,
    )
    supervised_window_indices, supervised_main_losses = _filter_series_after_training(
        supervised_window_indices, supervised_main_losses, training_end_window
    )

    distribution_change_windows = _derive_eta_change_windows(
        pretrained_window_indices, pretrained_eta_values
    )
    if not distribution_change_windows and eta_change_windows:
        distribution_change_windows = list(eta_change_windows)

    def _add_phase_markers(ax):
        distribution_labeled = False
        for eta_window in distribution_change_windows:
            if eta_window >= 0:
                ax.axvline(
                    x=eta_window, color='red', linestyle='--', alpha=0.5, linewidth=1.5,
                    label='Distribution Change' if not distribution_labeled else None,
                )
                distribution_labeled = True
        if drift_detection_window is not None and drift_detection_window >= 0:
            ax.axvline(
                x=drift_detection_window, color='crimson', linestyle=':', alpha=0.7, linewidth=1.5,
                label='Drift Detected (GLRT)',
            )
        if training_start_window is not None and training_start_window >= 0:
            ax.axvline(
                x=training_start_window, color='orange', linestyle='-', alpha=0.7, linewidth=2,
                label='Training Start',
            )
        if training_end_window is not None and training_end_window >= 0:
            ax.axvline(
                x=training_end_window, color='purple', linestyle='-', alpha=0.7, linewidth=2,
                label='Training End',
            )
    
    # Create separate figure for Main Loss Comparison
    fig1 = plt.figure(figsize=(14, 5))
    ax1 = fig1.add_subplot(111)
    if pretrained_main_losses and pretrained_window_indices:
        ax1.plot(pretrained_window_indices, pretrained_main_losses, 'b-', linewidth=3, 
                label='Pretrained Model', marker='o', markersize=6)
    if online_main_losses and online_window_indices:
        ax1.plot(online_window_indices, online_main_losses, 'r-', linewidth=3, 
                label='Algorithm 1', marker='s', markersize=6)
    if supervised_main_losses and supervised_window_indices:
        ax1.plot(supervised_window_indices, supervised_main_losses, 'g-', linewidth=3, 
                label='Supervised Trained Model', marker='^', markersize=6)
    
    _add_phase_markers(ax1)
    
    ax1.set_xlabel('Window Index', fontsize=20)
    ax1.set_ylabel('RMSPE (Supervised)', fontsize=20)
    ax1.set_title('RMSPE (Supervised)', fontsize=22, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    # Set x-axis range dynamically based on available data
    all_window_indices = []
    if pretrained_window_indices:
        all_window_indices.extend(pretrained_window_indices)
    if online_window_indices:
        all_window_indices.extend(online_window_indices)
    if supervised_window_indices:
        all_window_indices.extend(supervised_window_indices)
    if all_window_indices:
        max_window = max(all_window_indices)
        ax1.set_xlim(0, max_window + 1)  # Add small padding
    else:
        ax1.set_xlim(0, 60)  # Fallback if no data
    ax1.legend(fontsize=14, loc='best')
    plt.tight_layout()
    
    # Create separate figure for Training Reference Loss Comparison
    fig2 = plt.figure(figsize=(14, 5))
    ax2 = fig2.add_subplot(111)
    if pretrained_training_losses and pretrained_window_indices:
        ax2.plot(pretrained_window_indices, pretrained_training_losses, 'b-', linewidth=3, 
                label='Pretrained Model', marker='o', markersize=6)
    if online_training_losses and online_window_indices_msie:
        ax2.plot(online_window_indices_msie, online_training_losses, 'r-', linewidth=3, 
                label='Algorithm 1', marker='s', markersize=6)
    
    _add_phase_markers(ax2)
    
    ax2.set_xlabel('Window Index', fontsize=20)
    ax2.set_ylabel('MSIE (Unsupervised)', fontsize=20)
    ax2.set_title('MSIE (Unsupervised)', fontsize=22, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    # Set x-axis range dynamically based on available data (same as ax1)
    if all_window_indices:
        max_window = max(all_window_indices)
        ax2.set_xlim(0, max_window + 1)  # Add small padding
    else:
        ax2.set_xlim(0, 60)  # Fallback if no data
    ax2.legend(fontsize=14, loc='best')
    plt.tight_layout()
    
    # Save the plots separately
    plot_path_main = os.path.join(output_dir, 'averaged_online_learning_comparison_main_loss.png')
    plot_path_training = os.path.join(output_dir, 'averaged_online_learning_comparison_training_loss.png')
    fig1.savefig(plot_path_main, dpi=150, bbox_inches='tight')
    fig2.savefig(plot_path_training, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    plt.close(fig2)

    kf_plot_path = plot_averaged_kf_gain_comparison(
        output_dir,
        averaged_pretrained_metrics,
        averaged_online_metrics=averaged_online_metrics,
        training_end_window=training_end_window,
        drift_detection_window=drift_detection_window,
        eta_change_windows=distribution_change_windows,
        training_start_window=training_start_window,
    )
    
    logger.info(f"Averaged online learning comparison plots saved to: {plot_path_main} and {plot_path_training}")
    if kf_plot_path:
        logger.info(f"KF gain comparison plot saved to: {kf_plot_path}")
    return plot_path_main, plot_path_training


def plot_averaged_kf_gain_comparison(
    output_dir,
    averaged_pretrained_metrics,
    averaged_online_metrics=None,
    training_start_window=None,
    training_end_window=None,
    drift_detection_window=None,
    eta_change_windows=None,
):
    """
    Plot SubspaceNet-only vs EKF posterior (both vs GT) and the KF improvement gap.

    Pretrained curves span all windows. Online curves (if provided) appear only after training_end_window.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    logger = logging.getLogger(__name__)

    pretrained_indices = averaged_pretrained_metrics.get("window_indices", [])
    pretrained_pre_ekf = averaged_pretrained_metrics.get("pre_ekf_losses", [])
    pretrained_ekf = averaged_pretrained_metrics.get("reference_metric_losses", [])
    pretrained_eta = averaged_pretrained_metrics.get("window_eta_values", [])

    if not pretrained_indices or not pretrained_pre_ekf or not pretrained_ekf:
        logger.warning("Skipping KF gain plot: missing pretrained pre_ekf or EKF loss series")
        return None

    os.makedirs(output_dir, exist_ok=True)

    distribution_change_windows = _derive_eta_change_windows(pretrained_indices, pretrained_eta)
    if not distribution_change_windows and eta_change_windows:
        distribution_change_windows = list(eta_change_windows)

    online_indices, online_pre_ekf, online_ekf = [], [], []
    if averaged_online_metrics is not None:
        online_indices, online_pre_ekf = _filter_series_after_training(
            averaged_online_metrics.get("window_indices", []),
            averaged_online_metrics.get("pre_ekf_losses", []),
            training_end_window,
        )
        _, online_ekf = _filter_series_after_training(
            averaged_online_metrics.get("window_indices", []),
            averaged_online_metrics.get("reference_metric_losses", []),
            training_end_window,
        )

    def _add_phase_markers(ax):
        distribution_labeled = False
        for eta_window in distribution_change_windows:
            if eta_window >= 0:
                ax.axvline(
                    x=eta_window, color='red', linestyle='--', alpha=0.5, linewidth=1.5,
                    label='Distribution Change' if not distribution_labeled else None,
                )
                distribution_labeled = True
        if drift_detection_window is not None and drift_detection_window >= 0:
            ax.axvline(
                x=drift_detection_window, color='crimson', linestyle=':', alpha=0.7, linewidth=1.5,
                label='Drift Detected (GLRT)',
            )
        if training_start_window is not None and training_start_window >= 0:
            ax.axvline(
                x=training_start_window, color='orange', linestyle='-', alpha=0.7, linewidth=2,
                label='Training Start',
            )
        if training_end_window is not None and training_end_window >= 0:
            ax.axvline(
                x=training_end_window, color='purple', linestyle='-', alpha=0.7, linewidth=2,
                label='Training End',
            )

    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(
        pretrained_indices, pretrained_pre_ekf, 'b-', linewidth=3,
        label='SubspaceNet-only (pre-EKF)', marker='s', markersize=6,
    )
    ax1.plot(
        pretrained_indices, pretrained_ekf, 'r-', linewidth=3,
        label='EKF posterior', marker='o', markersize=6,
    )
    if online_indices and online_pre_ekf and online_ekf:
        ax1.plot(
            online_indices, online_pre_ekf, color='cornflowerblue', linewidth=2, linestyle='--',
            label='Algorithm 1 SubspaceNet-only', marker='^', markersize=5,
        )
        ax1.plot(
            online_indices, online_ekf, color='salmon', linewidth=2, linestyle='--',
            label='Algorithm 1 EKF posterior', marker='v', markersize=5,
        )
    _add_phase_markers(ax1)
    ax1.set_xlabel('Window Index', fontsize=20)
    ax1.set_ylabel('RMSPE vs GT (rad)', fontsize=20)
    ax1.set_title('SubspaceNet-only vs EKF Posterior (Supervised)', fontsize=22, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.legend(fontsize=14, loc='best')

    ax2 = fig.add_subplot(2, 1, 2)
    pretrained_gain = np.array(pretrained_pre_ekf) - np.array(pretrained_ekf)
    ax2.plot(
        pretrained_indices, pretrained_gain, 'g-', linewidth=3,
        label='Pretrained KF gain (pre-EKF − EKF)', marker='d', markersize=6,
    )
    ax2.axhline(y=0.0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    if online_indices and online_pre_ekf and online_ekf:
        online_gain = np.array(online_pre_ekf) - np.array(online_ekf)
        ax2.plot(
            online_indices, online_gain, color='darkgreen', linewidth=2, linestyle='--',
            label='Algorithm 1 KF gain', marker='x', markersize=6,
        )
    _add_phase_markers(ax2)
    ax2.set_xlabel('Window Index', fontsize=20)
    ax2.set_ylabel('RMSPE reduction (rad)', fontsize=20)
    ax2.set_title('EKF Improvement vs SubspaceNet-only (positive = KF helped)', fontsize=22, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.legend(fontsize=14, loc='best')

    all_indices = list(pretrained_indices)
    if online_indices:
        all_indices.extend(online_indices)
    if all_indices:
        ax1.set_xlim(0, max(all_indices) + 1)
        ax2.set_xlim(0, max(all_indices) + 1)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'averaged_kf_gain_comparison.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return plot_path


def plot_training_curves(metrics: dict, output_dir) -> None:
    """Plot and save training loss/accuracy curves from stored trainer metrics."""
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime
    from pathlib import Path

    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)
    plots_dir = output_dir / metrics.get("plots_subdir", "plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    train_losses = metrics.get("train_losses", [])
    valid_losses = metrics.get("valid_losses", [])
    train_accuracies = metrics.get("train_accuracies", [])
    valid_accuracies = metrics.get("valid_accuracies", [])
    train_angles_losses = metrics.get("train_angles_losses", [])
    valid_angles_losses = metrics.get("valid_angles_losses", [])
    train_ranges_losses = metrics.get("train_ranges_losses", [])
    valid_ranges_losses = metrics.get("valid_ranges_losses", [])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_epochs = max(len(train_losses), len(valid_losses), 1)
    epochs = np.arange(1, n_epochs + 1)
    plot_kwargs = {"marker": "o", "linewidth": 2, "markersize": 6}

    def _finite_series(values):
        arr = np.asarray(values, dtype=float)
        if arr.size == 0 or not np.any(np.isfinite(arr)):
            return None
        return arr

    def _save_epoch_plot(y_series, ylabel, title, filename):
        plt.figure(figsize=(10, 6))
        plotted = False
        for values, label in y_series:
            arr = _finite_series(values)
            if arr is None:
                continue
            x = epochs[: len(arr)]
            plt.plot(x, arr, label=label, **plot_kwargs)
            plotted = True
        if not plotted:
            plt.close()
            logger.warning("Skipping %s: no finite training metrics to plot", filename)
            return
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        if n_epochs == 1:
            plt.xlim(0.5, 1.5)
            plt.xticks([1])
        else:
            plt.xlim(0.5, n_epochs + 0.5)
            plt.xticks(epochs)
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / filename)
        plt.close()

    _save_epoch_plot(
        [(train_losses, "Training Loss"), (valid_losses, "Validation Loss")],
        ylabel="Loss",
        title="Training and Validation Loss",
        filename=f"loss_curve_{timestamp}.png",
    )

    train_acc_pct = np.asarray(train_accuracies, dtype=float) * 100
    valid_acc_pct = np.asarray(valid_accuracies, dtype=float) * 100
    _save_epoch_plot(
        [(train_acc_pct, "Training Accuracy"), (valid_acc_pct, "Validation Accuracy")],
        ylabel="Accuracy (%)",
        title="Training and Validation Accuracy",
        filename=f"accuracy_curve_{timestamp}.png",
    )

    if train_angles_losses and valid_angles_losses:
        plt.figure(figsize=(10, 6))
        plt.plot(train_angles_losses, label="Training Angle Loss")
        plt.plot(valid_angles_losses, label="Validation Angle Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Angle Loss")
        plt.title("Training and Validation Angle Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / f"angle_loss_curve_{timestamp}.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(train_ranges_losses, label="Training Range Loss")
        plt.plot(valid_ranges_losses, label="Validation Range Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Range Loss")
        plt.title("Training and Validation Range Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / f"range_loss_curve_{timestamp}.png")
        plt.close()

    logger.info("Training curves saved under %s", plots_dir)


def plot_glrt_averaged_drift_results(
    glrt_results: dict,
    output_dir,
    drift_warmup_windows: int,
    drift_guard_samples: int,
    eta_change_windows=None,
) -> None:
    """Plot averaged GLRT drift-detection figures for adaptation and reference losses."""
    import matplotlib.pyplot as plt
    from pathlib import Path

    from simulation.runners.sandbox import glrt_changepoint_detection, plot_results
    from utils import drift_gates

    logger = logging.getLogger(__name__)
    glrt_window_offset = drift_warmup_windows
    eta_markers = eta_change_windows if eta_change_windows else None
    gate_milestones = drift_gates.drift_detection_milestones(
        drift_warmup_windows, drift_guard_samples
    )

    for key, label, loss_path, glrt_path in (
        (
            "adaptation_loss",
            "Adaptation Loss",
            "glrt_adaptation_loss_averaged_loss.png",
            "glrt_adaptation_loss_averaged_glrt.png",
        ),
        (
            "reference_metric",
            "Reference Metric",
            "glrt_reference_metric_averaged_loss.png",
            "glrt_reference_metric_averaged_glrt.png",
        ),
    ):
        if key not in glrt_results:
            continue
        data = glrt_results[key]
        avg_losses = data["avg_losses"]
        min_segment_size = data["min_segment_size"]
        plot_offset = data.get("window_index_offset", glrt_window_offset)
        if len(avg_losses) < 2 * min_segment_size + 1:
            continue
        try:
            changepoint, _, all_log_glr, candidate_points = glrt_changepoint_detection(
                avg_losses, min_segment_size=min_segment_size
            )
            fig_loss, fig_glrt = plot_results(
                avg_losses,
                changepoint,
                all_log_glr,
                candidate_points,
                window_index_offset=plot_offset,
                event_windows=eta_markers,
                gate_milestones=gate_milestones,
            )
            title = (
                f"GLRT Drift Detection - {label} "
                f'(Averaged across {data["trajectory_count"]} trajectories)'
            )
            if data.get("avg_changepoint_window") is not None:
                title += (
                    f'\nAvg Changepoint Window: {data["avg_changepoint_window"]:.2f} '
                    f'± {data["std_changepoint_window"]:.2f}, '
                    f'Avg Log-GLR: {data["avg_likelihood"]:.4f} '
                    f'± {data["std_likelihood"]:.4f}'
                )
            fig_loss.suptitle(title + " - Loss", fontsize=14)
            fig_glrt.suptitle(title + " - GLRT Statistics", fontsize=14)
            fig_loss.subplots_adjust(top=0.88)
            fig_glrt.subplots_adjust(top=0.88)
            loss_plot_path = Path(output_dir) / loss_path
            glrt_plot_path = Path(output_dir) / glrt_path
            fig_loss.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
            fig_glrt.savefig(glrt_plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig_loss)
            plt.close(fig_glrt)
            logger.info("Saved averaged GLRT %s plots to %s and %s", label, loss_plot_path, glrt_plot_path)
        except Exception as exc:
            logger.warning("Failed to plot averaged GLRT %s results: %s", label, exc)


def plot_single_online_learning_run(result: dict, output_dir, config) -> None:
    """Dispatch all single-run online learning plots from structured run results."""
    import datetime
    from pathlib import Path

    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)
    ol_results = result.get("online_learning_results", {})
    averaged_data = result.get("averaged_results")
    if not averaged_data:
        logger.warning("Skipping OL plots: no averaged_results in run output")
        return

    training_start_window = ol_results.get("training_start_window")
    training_end_window = ol_results.get("training_end_window")
    drift_detection_window = ol_results.get("drift_detection_window")
    eta_change_windows = ol_results.get("eta_change_windows", [])
    reference_metric_config = ol_results.get("reference_metric_config", "unknown")
    adaptation_loss_config = ol_results.get("adaptation_loss_config", "unknown")

    plot_averaged_online_learning_results(
        output_dir,
        averaged_data["averaged_pretrained_trajectory"],
        averaged_data["averaged_online_trajectory"],
        reference_metric_config,
        adaptation_loss_config,
        training_start_window=training_start_window,
        training_end_window=training_end_window,
        drift_detection_window=drift_detection_window,
        eta_change_windows=eta_change_windows,
        averaged_supervised_metrics=averaged_data.get("averaged_supervised_trajectory"),
    )

    glrt_results = averaged_data.get("glrt_results") or result.get("glrt_results")
    if glrt_results:
        online_config = getattr(config, "online_learning", None)
        drift_warmup = getattr(online_config, "drift_warmup_windows", 7)
        drift_guard = getattr(online_config, "drift_guard_samples", 3)
        plot_glrt_averaged_drift_results(
            glrt_results,
            output_dir,
            drift_warmup,
            drift_guard,
            eta_change_windows=eta_change_windows,
        )

    if getattr(config.online_learning, "plot_trajectory", False):
        pretrained_trajectory_results = ol_results.get("pretrained_trajectory_results", [])
        plot_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        stride = config.online_learning.stride
        for traj_idx, traj_result in enumerate(pretrained_trajectory_results):
            suffix = f"_traj{traj_idx}" if len(pretrained_trajectory_results) > 1 else ""
            plot_online_learning_trajectory(
                traj_result.window_labels,
                output_dir,
                f"{plot_ts}{suffix}",
                window_indices=traj_result.window_indices,
                stride=stride,
            )

    logger.info("Single-run online learning plots written to %s", output_dir)


def plot_online_learning_results(output_dir, window_losses, window_covariances, window_eta_values, window_updates, window_pre_ekf_losses, window_labels, ekf_covariances, ekf_kalman_gains=None, ekf_kalman_gain_times_innovation=None, ekf_y_s_inv_y=None, online_window_losses=None, online_window_covariances=None, online_pre_ekf_losses=None, online_ekf_innovations=None, online_ekf_kalman_gains=None, online_ekf_kalman_gain_times_innovation=None, online_ekf_y_s_inv_y=None, online_window_indices=None, training_window_losses=None, training_window_covariances=None, training_pre_ekf_losses=None, training_ekf_innovations=None, training_ekf_kalman_gains=None, training_ekf_kalman_gain_times_innovation=None, training_ekf_y_s_inv_y=None, training_window_indices=None, learning_start_window=None, window_delta_rmspe_losses=None, window_delta_rmape_losses=None, online_delta_rmspe_losses=None, online_delta_rmape_losses=None, training_delta_rmspe_losses=None, training_delta_rmape_losses=None, window_pre_ekf_angles_pred=None, online_pre_ekf_angles_pred=None, training_pre_ekf_angles_pred=None, window_ekf_predictions=None, online_ekf_predictions=None, training_ekf_predictions=None, window_avg_ekf_angle_pred=None, window_avg_pre_ekf_angle_pred=None, online_avg_ekf_angle_pred=None, online_avg_pre_ekf_angle_pred=None, training_avg_ekf_angle_pred=None, training_avg_pre_ekf_angle_pred=None):
    """
    Plot online learning results including plots as a function of eta.
    Also includes online model results and training data when available.
    
    Args:
        output_dir: Output directory for saving plots
        window_losses: Static model window losses
        window_covariances: Static model window covariances
        window_eta_values: Eta values for each window
        window_updates: Model update flags for each window
        window_pre_ekf_losses: Static model pre-EKF losses
        window_labels: Window labels for trajectory plotting
        ekf_covariances: EKF covariances
        ekf_kalman_gains: EKF Kalman gains (optional)
        ekf_kalman_gain_times_innovation: EKF Kalman gain times innovation (optional)
        ekf_y_s_inv_y: EKF y*S^-1*y values (optional)
        online_window_losses: Online model window losses (optional)
        online_window_covariances: Online model window covariances (optional)
        online_pre_ekf_losses: Online model pre-EKF losses (optional)
        online_ekf_innovations: Online model EKF innovations (optional)
        online_ekf_kalman_gains: Online model EKF Kalman gains (optional)
        online_ekf_kalman_gain_times_innovation: Online model EKF Kalman gain times innovation (optional)
        online_ekf_y_s_inv_y: Online model EKF y*S^-1*y values (optional)
        online_window_indices: Online model window indices (optional)
        training_window_losses: Training model window losses (optional)
        training_window_covariances: Training model window covariances (optional)
        training_pre_ekf_losses: Training model pre-EKF losses (optional)
        training_ekf_innovations: Training model EKF innovations (optional)
        training_ekf_kalman_gains: Training model EKF Kalman gains (optional)
        training_ekf_kalman_gain_times_innovation: Training model EKF Kalman gain times innovation (optional)
        training_ekf_y_s_inv_y: Training model EKF y*S^-1*y values (optional)
        training_window_indices: Training model window indices (optional)
        learning_start_window: Learning start window index (optional)
        window_delta_rmspe_losses: Static model delta losses (optional)
        window_delta_rmape_losses: Static model delta RMAPE losses (optional)
        online_delta_rmspe_losses: Online model delta losses (optional)
        online_delta_rmape_losses: Online model delta RMAPE losses (optional)
        training_delta_rmspe_losses: Training model delta losses (optional)
        training_delta_rmape_losses: Training model delta RMAPE losses (optional)
        window_pre_ekf_angles_pred: Static model pre-EKF angle predictions (optional)
        online_pre_ekf_angles_pred: Online model pre-EKF angle predictions (optional)
        training_pre_ekf_angles_pred: Training model pre-EKF angle predictions (optional)
        window_ekf_predictions: Static model EKF predictions (optional)
        online_ekf_predictions: Online model EKF predictions (optional)
        training_ekf_predictions: Training model EKF predictions (optional)
        window_avg_ekf_angle_pred: Static model averaged EKF angle predictions (optional)
        window_avg_pre_ekf_angle_pred: Static model averaged pre-EKF angle predictions (optional)
        online_avg_ekf_angle_pred: Online model averaged EKF angle predictions (optional)
        online_avg_pre_ekf_angle_pred: Online model averaged pre-EKF angle predictions (optional)
        training_avg_ekf_angle_pred: Training model averaged EKF angle predictions (optional)
        training_avg_pre_ekf_angle_pred: Training model averaged pre-EKF angle predictions (optional)
        
    Returns:
        Tuple of plot paths (main_plot_path, trajectory_plot_path)
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import datetime
        from pathlib import Path
        import torch
        
        logger = logging.getLogger(__name__)
        
        def safe_convert_to_list(data):
            """Safely convert tensor or list to list for plotting"""
            if data is None:
                return None
            if isinstance(data, torch.Tensor):
                return data.cpu().numpy().tolist()
            elif isinstance(data, list):
                return data
            else:
                return list(data)
        
        # Convert tensor inputs to lists to avoid boolean context issues
        window_delta_rmspe_losses = safe_convert_to_list(window_delta_rmspe_losses)
        window_delta_rmape_losses = safe_convert_to_list(window_delta_rmape_losses)
        online_delta_rmspe_losses = safe_convert_to_list(online_delta_rmspe_losses)
        online_delta_rmape_losses = safe_convert_to_list(online_delta_rmape_losses)
        training_delta_rmspe_losses = safe_convert_to_list(training_delta_rmspe_losses)
        training_delta_rmape_losses = safe_convert_to_list(training_delta_rmape_losses)
        
        # Convert nested tensor structures to avoid boolean context issues
        def convert_nested_tensors(data):
            """Convert nested tensor structures to lists"""
            if data is None:
                return None
            if isinstance(data, torch.Tensor):
                return data.cpu().numpy().tolist()
            elif isinstance(data, list):
                return [convert_nested_tensors(item) for item in data]
            else:
                return data
        
        # Convert all EKF-related data that might contain tensors
        ekf_kalman_gains = convert_nested_tensors(ekf_kalman_gains)
        ekf_kalman_gain_times_innovation = convert_nested_tensors(ekf_kalman_gain_times_innovation)
        ekf_y_s_inv_y = convert_nested_tensors(ekf_y_s_inv_y)
        online_ekf_kalman_gains = convert_nested_tensors(online_ekf_kalman_gains)
        online_ekf_kalman_gain_times_innovation = convert_nested_tensors(online_ekf_kalman_gain_times_innovation)
        online_ekf_y_s_inv_y = convert_nested_tensors(online_ekf_y_s_inv_y)
        training_ekf_kalman_gains = convert_nested_tensors(training_ekf_kalman_gains)
        training_ekf_kalman_gain_times_innovation = convert_nested_tensors(training_ekf_kalman_gain_times_innovation)
        training_ekf_y_s_inv_y = convert_nested_tensors(training_ekf_y_s_inv_y)
        
        # Create timestamp and plot directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_dir = Path(output_dir) / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        def set_adjusted_ylim(ax, data, padding=0.1):
            """Helper function to set y limits excluding first point"""
            if len(data) > 1:
                data_without_first = data[1:]
                ymin = min(data_without_first)
                ymax = max(data_without_first)
                range_y = ymax - ymin
                ax.set_ylim([ymin - range_y * padding, ymax + range_y * padding])
        
        # Check if online learning data is available
        has_online_data = (online_window_losses is not None and 
                          len(online_window_losses) > 0 and 
                          online_pre_ekf_losses is not None and 
                          len(online_pre_ekf_losses) > 0)
        
        # Check if training data is available
        has_training_data = (training_window_losses is not None and 
                            len(training_window_losses) > 0 and 
                            training_pre_ekf_losses is not None and 
                            len(training_pre_ekf_losses) > 0 and
                            training_window_indices is not None and
                            len(training_window_indices) > 0)
        
        # Calculate differences between static and online models if online data is available
        static_vs_online_ekf_diff = None
        static_vs_online_pre_ekf_diff = None
        if has_online_data:
            # Get static model data for the online windows
            if online_window_indices is not None and len(online_window_indices) > 0:
                # Use actual online window indices to get corresponding static data
                static_ekf_for_comparison = [window_losses[i] for i in online_window_indices]
                static_pre_ekf_for_comparison = [window_pre_ekf_losses[i] for i in online_window_indices]
            else:
                # Fallback: use the last N windows where N is the length of online data
                start_idx = max(0, len(window_losses) - len(online_window_losses))
                static_ekf_for_comparison = window_losses[start_idx:]
                static_pre_ekf_for_comparison = window_pre_ekf_losses[start_idx:]
            
            # Calculate differences (positive = online better)
            static_vs_online_ekf_diff = np.array(static_ekf_for_comparison) - np.array(online_window_losses)
            static_vs_online_pre_ekf_diff = np.array(static_pre_ekf_for_comparison) - np.array(online_pre_ekf_losses)
        
        # Find indices where eta changes
        eta_changes = []
        eta_values = []
        for i in range(1, len(window_eta_values)):
            if abs(window_eta_values[i] - window_eta_values[i-1]) > 1e-6:
                eta_changes.append(i)
                eta_values.append(window_eta_values[i])
        
        # Calculate training end window (training ends after 13 training windows)
        training_end_window = None
        if has_training_data and training_window_indices is not None and len(training_window_indices) > 0:
            # Training ends after 13 training windows, so the last training window is window 12 (0-indexed)
            # The training end window is the window where training completed and online inference started
            if len(training_window_indices) >= 13:
                training_end_window = training_window_indices[12]  # 13th training window (0-indexed)
            else:
                # If less than 13 training windows, use the last training window
                training_end_window = training_window_indices[-1]
        
        # Calculate training start window (when time_to_learn is reached)
        training_start_window = None
        if learning_start_window is not None:
            training_start_window = learning_start_window
        elif has_training_data and training_window_indices is not None and len(training_window_indices) > 0:
            # If learning_start_window is not provided, use the first training window index
            training_start_window = training_window_indices[0]
        
        # Create figure with multiple subplots (4x2 layout for 8 plots)
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Plot loss vs window index
        ax1 = fig.add_subplot(4, 2, 1)
        x = np.arange(len(window_losses))[1:]  # Start from second sample
        
        # Color scheme:
        # 1. SubspaceNet without EKF (pre-EKF): Blue
        # 2. SubspaceNet with EKF (EKF): Red  
        # 3. Training/Online with EKF: Light red (related to red)
        # 4. Training/Online without EKF: Light blue (related to blue)
        
        # Static model plots
        ax1.plot(x, np.array(window_losses)[1:], 'red', marker='o', linewidth=2, label='Static Model EKF Loss')
        ax1.plot(x, np.array(window_pre_ekf_losses)[1:], 'blue', marker='s', linewidth=2, label='Static Model SubspaceNet Loss')
        
        # Add online model data if available
        if has_online_data:
            # Use actual online window indices if available, otherwise fall back to learning start window
            if online_window_indices is not None and len(online_window_indices) > 0:
                online_x = np.array(online_window_indices)  # Use all window indices
            elif learning_start_window is not None:
                online_x = np.arange(learning_start_window, learning_start_window + len(online_window_losses))[1:]  # Start from second sample
            else:
                online_x = np.arange(len(online_window_losses))[1:]  # Start from second sample
            ax1.plot(online_x, np.array(online_window_losses), 'lightcoral', marker='d', linewidth=2, label='Online Model EKF Loss')
            ax1.plot(online_x, np.array(online_pre_ekf_losses), 'lightblue', marker='^', linewidth=2, label='Online Model SubspaceNet Loss')
        
        # Add training data if available
        if has_training_data:
            # Use actual training window indices
            training_x = np.array(training_window_indices)  # Use all window indices
            ax1.plot(training_x, np.array(training_window_losses), 'lightcoral', marker='*', linewidth=2, label='Training Model EKF Loss', linestyle='--')
            ax1.plot(training_x, np.array(training_pre_ekf_losses), 'lightblue', marker='s', linewidth=2, label='Training Model SubspaceNet Loss', linestyle='--')
            
            # Connect training to online if both are available
            if has_online_data and online_window_indices is not None and len(online_window_indices) > 0:
                # Connect last training point to first online point
                last_training_x = training_x[-1]  # Window 12
                first_online_x = online_x[0]      # Window 13
                last_training_ekf = np.array(training_window_losses)[-1]
                first_online_ekf = np.array(online_window_losses)[0]
                last_training_subspace = np.array(training_pre_ekf_losses)[-1]
                first_online_subspace = np.array(online_pre_ekf_losses)[0]
                
                # Draw connecting lines
                ax1.plot([last_training_x, first_online_x], [last_training_ekf, first_online_ekf], 'lightcoral', linestyle='-', linewidth=2, alpha=0.7)
                ax1.plot([last_training_x, first_online_x], [last_training_subspace, first_online_subspace], 'lightblue', linestyle='-', linewidth=2, alpha=0.7)
        
        # Set y-axis limits to include all data points with some padding
        all_loss_values = []
        all_loss_values.extend(np.array(window_losses)[1:])
        all_loss_values.extend(np.array(window_pre_ekf_losses)[1:])
        
        if has_online_data:
            all_loss_values.extend(np.array(online_window_losses))
            all_loss_values.extend(np.array(online_pre_ekf_losses))
        
        if has_training_data:
            all_loss_values.extend(np.array(training_window_losses))
            all_loss_values.extend(np.array(training_pre_ekf_losses))
        
        if all_loss_values:
            min_loss = min(all_loss_values)
            max_loss = max(all_loss_values)
            padding = (max_loss - min_loss) * 0.05  # 5% padding
            ax1.set_ylim([min_loss - padding, max_loss + padding])
        else:
            # Fallback to original limit if no data
            ax1.set_ylim([None, 0.14])
        
        # Add eta change markers (adjusted for starting from second sample)
        for idx, eta in zip(eta_changes, eta_values):
            if idx >= 1:  # Only show markers from second sample onwards
                ax1.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                ax1.text(idx, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.1, 'Distribution Change', rotation=90, verticalalignment='bottom', horizontalalignment='center', color='red', fontsize=14)
        
        # Add training end marker if available
        if training_end_window is not None and training_end_window >= 1:
            ax1.axvline(x=training_end_window, color='purple', linestyle='-', alpha=0.7, linewidth=2)
            ax1.text(training_end_window, 0.13, 'Training End', rotation=90, verticalalignment='top', 
                    color='purple', fontweight='bold', fontsize=24)
        
        # Add training start marker if available
        if training_start_window is not None and training_start_window >= 1:
            ax1.axvline(x=training_start_window, color='orange', linestyle='-', alpha=0.7, linewidth=2)
            ax1.text(training_start_window, 0.13, 'Training Start', rotation=90, verticalalignment='top', 
                    color='orange', fontweight='bold', fontsize=24)
        
        # Add labels and title
        ax1.set_xlabel('Window Index')
        ax1.set_ylabel('Loss')
        title = 'Loss vs Window Index (Starting from Window 1)\nRMSPE = √(1/N * Σ(θ_pred - θ_true)²)'
        if has_online_data:
            title += '\n(Static + Online Models)'
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Plot EKF improvement vs window index (reversed: SubspaceNet - EKF)
        ax2 = fig.add_subplot(4, 2, 2)
        x = np.arange(len(window_losses))[1:]  # Start from second sample
        static_improvement = np.array(window_pre_ekf_losses)[1:] - np.array(window_losses)[1:]  # Reversed calculation, starting from second sample
        ax2.plot(x, static_improvement, 'green', marker='o', linewidth=2, label='Static Model Improvement')
        
        # Add online model improvement if available
        if has_online_data:
            # Use actual online window indices if available, otherwise fall back to learning start window
            if online_window_indices is not None and len(online_window_indices) > 0:
                online_x = np.array(online_window_indices)  # Use all window indices
            elif learning_start_window is not None:
                online_x = np.arange(learning_start_window, learning_start_window + len(online_window_losses))[1:]  # Start from second sample
            else:
                online_x = np.arange(len(online_window_losses))[1:]  # Start from second sample
            online_improvement = np.array(online_pre_ekf_losses) - np.array(online_window_losses)
            ax2.plot(online_x, online_improvement, 'lightgreen', marker='d', linewidth=2, label='Online Model Improvement')
        
        # Add training model improvement if available
        if has_training_data:
            # Use actual training window indices
            training_x = np.array(training_window_indices)  # Use all window indices
            training_improvement = np.array(training_pre_ekf_losses) - np.array(training_window_losses)
            ax2.plot(training_x, training_improvement, 'lightgreen', marker='*', linewidth=2, label='Training Model Improvement', linestyle='--')
            
            # Connect training to online if both are available
            if has_online_data and online_window_indices is not None and len(online_window_indices) > 0:
                # Connect last training point to first online point
                last_training_x = training_x[-1]  # Window 12
                first_online_x = online_x[0]      # Window 13
                last_training_improvement = training_improvement[-1]
                first_online_improvement = online_improvement[0]
                
                # Draw connecting line
                ax2.plot([last_training_x, first_online_x], [last_training_improvement, first_online_improvement], 'lightgreen', linestyle='-', linewidth=2, alpha=0.7)
        
        # Add eta change markers (adjusted for starting from second sample)
        for idx, eta in zip(eta_changes, eta_values):
            if idx >= 1:  # Only show markers from second sample onwards
                ax2.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                ax2.text(idx, ax2.get_ylim()[0] + (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.1, 'Distribution Change', rotation=90, verticalalignment='bottom', horizontalalignment='center', color='red', fontsize=14)
        
        # Add training end marker if available
        if training_end_window is not None and training_end_window >= 1:
            ax2.axvline(x=training_end_window, color='purple', linestyle='-', alpha=0.7, linewidth=2)
            ax2.text(training_end_window, ax2.get_ylim()[1], 'Training End', rotation=90, verticalalignment='top', 
                    color='purple', fontweight='bold', fontsize=24)
        
        # Add training start marker if available
        if training_start_window is not None and training_start_window >= 1:
            ax2.axvline(x=training_start_window, color='orange', linestyle='-', alpha=0.7, linewidth=2)
            ax2.text(training_start_window, ax2.get_ylim()[1], 'Training Start', rotation=90, verticalalignment='top', 
                    color='orange', fontweight='bold', fontsize=24)
        
        # Add labels and title
        ax2.set_xlabel('Window Index')
        ax2.set_ylabel('Loss Difference')
        title = 'SubspaceNet Loss - EKF Loss vs Window Index (Starting from Window 1)\nImprovement = L_SubspaceNet - L_EKF'
        if has_online_data:
            title += '\n(Static + Online Models)'
        ax2.set_title(title)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Plot averaged angle predictions vs window index
        ax3 = fig.add_subplot(4, 2, 3)
        
        # Define colors for each source (3 sources)
        source_colors = ['blue', 'red', 'green']
        
        # Plot static model averaged angles if available
        if window_avg_ekf_angle_pred is not None and len(window_avg_ekf_angle_pred) > 0:
            x = np.arange(len(window_avg_ekf_angle_pred))[1:]  # Start from second sample
            
            # Convert to numpy array - should be (num_windows, num_sources) shape
            try:
                window_avg_ekf_angles = np.array(window_avg_ekf_angle_pred)[1:]  # Skip first window
                window_avg_pre_ekf_angles = np.array(window_avg_pre_ekf_angle_pred)[1:] if window_avg_pre_ekf_angle_pred else None
                
                # Check if we have a proper 2D array (windows x sources)
                if len(window_avg_ekf_angles.shape) == 2:
                    num_sources = window_avg_ekf_angles.shape[1]
                    
                    # Plot each source separately with different colors
                    for src_idx in range(num_sources):
                        color = source_colors[src_idx % len(source_colors)]
                        # EKF angles for this source
                        ekf_angles_src = window_avg_ekf_angles[:, src_idx]
                        ax3.plot(x, ekf_angles_src, color=color, marker='o', linewidth=2, 
                                label=f'Static EKF Source {src_idx+1}')
                        
                        # Pre-EKF angles for this source
                        if window_avg_pre_ekf_angles is not None:
                            pre_ekf_angles_src = window_avg_pre_ekf_angles[:, src_idx]
                            ax3.plot(x, pre_ekf_angles_src, color=color, marker='s', 
                                    linewidth=2, linestyle='--', label=f'Static Pre-EKF Source {src_idx+1}')
                else:
                    # Fallback for 1D array
                    ax3.plot(x, window_avg_ekf_angles, 'blue', marker='o', linewidth=2, label='Static Model Avg EKF Angles')
                    if window_avg_pre_ekf_angles is not None:
                        ax3.plot(x, window_avg_pre_ekf_angles, 'lightblue', marker='s', linewidth=2, linestyle='--', 
                                label='Static Model Avg Pre-EKF Angles')
            except Exception as e:
                # Fallback: plot as single values
                ax3.plot(x, window_avg_ekf_angle_pred[1:], 'blue', marker='o', linewidth=2, label='Static Model Avg EKF Angles')
                if window_avg_pre_ekf_angle_pred:
                    ax3.plot(x, window_avg_pre_ekf_angle_pred[1:], 'lightblue', marker='s', linewidth=2, linestyle='--', 
                            label='Static Model Avg Pre-EKF Angles')
        
        # Add online model averaged angles if available
        if has_online_data and online_avg_ekf_angle_pred is not None and len(online_avg_ekf_angle_pred) > 0:
            # Use actual online window indices if available
            if online_window_indices is not None and len(online_window_indices) > 0:
                online_x = np.array(online_window_indices)
            else:
                online_x = np.arange(len(online_avg_ekf_angle_pred))
            
            online_ekf_angles = np.array(online_avg_ekf_angle_pred)
            online_pre_ekf_angles = np.array(online_avg_pre_ekf_angle_pred) if online_avg_pre_ekf_angle_pred else None
            
            # Plot online averaged angles
            if len(online_ekf_angles.shape) == 2:
                num_sources = online_ekf_angles.shape[1]
                for src_idx in range(num_sources):
                    color = source_colors[src_idx % len(source_colors)]
                    # EKF angles (solid line with diamond marker)
                    ax3.plot(online_x, online_ekf_angles[:, src_idx], color=color, marker='d', linewidth=2, 
                            label=f'Online EKF Source {src_idx+1}', alpha=0.8)
                    # Pre-EKF angles (dashed line with triangle marker)
                    if online_pre_ekf_angles is not None:
                        ax3.plot(online_x, online_pre_ekf_angles[:, src_idx], color=color, marker='^', linewidth=2, 
                                linestyle='--', label=f'Online Pre-EKF Source {src_idx+1}', alpha=0.8)
            else:
                # Fallback
                ax3.plot(online_x, online_ekf_angles, 'red', marker='d', linewidth=2, label='Online Model Avg EKF Angles')
                if online_pre_ekf_angles is not None:
                    ax3.plot(online_x, online_pre_ekf_angles, 'lightcoral', marker='^', linewidth=2, linestyle='--', 
                            label='Online Model Avg Pre-EKF Angles')
        
        # Add training model averaged angles if available
        if has_training_data and training_avg_ekf_angle_pred is not None and len(training_avg_ekf_angle_pred) > 0:
            training_x = np.array(training_window_indices)
            training_ekf_angles = np.array(training_avg_ekf_angle_pred)
            training_pre_ekf_angles = np.array(training_avg_pre_ekf_angle_pred) if training_avg_pre_ekf_angle_pred else None
            
            # Plot training averaged angles with dotted style
            if len(training_ekf_angles.shape) == 2:
                num_sources = training_ekf_angles.shape[1]
                for src_idx in range(num_sources):
                    color = source_colors[src_idx % len(source_colors)]
                    # EKF angles (dotted line with star marker)
                    ax3.plot(training_x, training_ekf_angles[:, src_idx], color=color, marker='*', linewidth=2, 
                            linestyle=':', label=f'Training EKF Source {src_idx+1}', alpha=0.7)
                    # Pre-EKF angles (dotted line with square marker)
                    if training_pre_ekf_angles is not None:
                        ax3.plot(training_x, training_pre_ekf_angles[:, src_idx], color=color, marker='s', linewidth=2, 
                                linestyle=':', label=f'Training Pre-EKF Source {src_idx+1}', alpha=0.7)
            else:
                # Fallback
                ax3.plot(training_x, training_ekf_angles, 'green', marker='*', linewidth=2, linestyle=':', 
                        label='Training Model Avg EKF Angles', alpha=0.7)
                if training_pre_ekf_angles is not None:
                    ax3.plot(training_x, training_pre_ekf_angles, 'lightgreen', marker='s', linewidth=2, linestyle=':', 
                            label='Training Model Avg Pre-EKF Angles', alpha=0.7)
        
        # Add eta change markers
        for idx, eta in zip(eta_changes, eta_values):
            if idx >= 1:  # Only show markers from second sample onwards
                ax3.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                ax3.text(idx, ax3.get_ylim()[0] + (ax3.get_ylim()[1] - ax3.get_ylim()[0]) * 0.1, 'Distribution Change', rotation=90, verticalalignment='bottom', horizontalalignment='center', color='red', fontsize=14)
        
        # Add training end marker if available
        if training_end_window is not None and training_end_window >= 1:
            ax3.axvline(x=training_end_window, color='purple', linestyle='-', alpha=0.7, linewidth=2)
            ax3.text(training_end_window, ax3.get_ylim()[1] * 0.9, 'Training End', rotation=90, verticalalignment='top', 
                    color='purple', fontweight='bold', fontsize=24)
        
        # Add training start marker if available
        if training_start_window is not None and training_start_window >= 1:
            ax3.axvline(x=training_start_window, color='orange', linestyle='-', alpha=0.7, linewidth=2)
            ax3.text(training_start_window, ax3.get_ylim()[1] * 0.9, 'Training Start', rotation=90, verticalalignment='top', 
                    color='orange', fontweight='bold', fontsize=24)
        
        # Add labels and title
        ax3.set_xlabel('Window Index')
        ax3.set_ylabel('Average Angle Predictions (radians)')
        title = 'Averaged Angle Predictions vs Window Index (Starting from Window 1)\nTime-Averaged EKF and Pre-EKF Predictions'
        if has_online_data:
            title += '\n(Static + Online Models)'
        ax3.set_title(title)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=24)  # Move legend outside plot
        ax3.grid(True, alpha=0.3)
        
        # 4. Plot Subspace-Kalman Delta Loss vs Window Index (RMSPE - L2)
        ax4 = fig.add_subplot(4, 2, 4)
        x = np.arange(len(window_delta_rmspe_losses))[1:]  # Start from second sample
        
        # Static model prediction delta (baseline model)
        ax4.plot(x, np.array(window_delta_rmspe_losses)[1:], 'navy', marker='o', linewidth=2.5, markersize=7, alpha=0.9, label='Static Model (RMSPE-L2)')
        
        # Static model RMAPE delta if available
        if window_delta_rmape_losses is not None and len(window_delta_rmape_losses) > 0:
            ax4.plot(x, np.array(window_delta_rmape_losses)[1:], 'royalblue', marker='s', linewidth=2.5, markersize=7, alpha=0.9, label='Static Model (RMAPE-L1)')
        
        # Add online training phase delta if available
        if has_training_data and training_delta_rmspe_losses is not None and len(training_delta_rmspe_losses) > 0:
            # Use actual training window indices
            training_x = np.array(training_window_indices)
            ax4.plot(training_x, np.array(training_delta_rmspe_losses), 'darkgreen', marker='o', linewidth=2.5, markersize=7, linestyle='--', alpha=0.9, label='Online Training (RMSPE-L2)')
            
            # Training RMAPE delta if available
            if training_delta_rmape_losses is not None and len(training_delta_rmape_losses) > 0:
                ax4.plot(training_x, np.array(training_delta_rmape_losses), 'forestgreen', marker='s', linewidth=2.5, markersize=7, linestyle='--', alpha=0.9, label='Online Training (RMAPE-L1)')
        
        # Add online inference phase delta if available
        if has_online_data and online_delta_rmspe_losses is not None and len(online_delta_rmspe_losses) > 0:
            # Use actual online window indices if available
            if online_window_indices is not None and len(online_window_indices) > 0:
                online_x = np.array(online_window_indices)
            else:
                online_x = np.arange(len(online_delta_rmspe_losses))
            ax4.plot(online_x, np.array(online_delta_rmspe_losses), 'crimson', marker='o', linewidth=2.5, markersize=7, alpha=0.9, label='Online Inference (RMSPE-L2)')
            
            # Online inference RMAPE delta if available
            if online_delta_rmape_losses is not None and len(online_delta_rmape_losses) > 0:
                ax4.plot(online_x, np.array(online_delta_rmape_losses), 'red', marker='s', linewidth=2.5, markersize=7, alpha=0.9, label='Online Inference (RMAPE-L1)')
            
            # Connect training to online inference if both are available
            if has_training_data and training_delta_rmspe_losses is not None and len(training_delta_rmspe_losses) > 0:
                # Connect last training point to first online point (RMSPE only for cleaner visualization)
                last_training_x = training_x[-1]
                first_online_x = online_x[0]
                last_training_delta = np.array(training_delta_rmspe_losses)[-1]
                first_online_delta = np.array(online_delta_rmspe_losses)[0]
                
                # Draw connecting line for phase transition
                ax4.plot([last_training_x, first_online_x], [last_training_delta, first_online_delta], 'darkgreen', linestyle='-', linewidth=2, alpha=0.5)
        
        # Add eta change markers
        for idx, eta in zip(eta_changes, eta_values):
            if idx >= 1:  # Only show markers from second sample onwards
                ax4.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                ax4.text(idx, ax4.get_ylim()[0] + (ax4.get_ylim()[1] - ax4.get_ylim()[0]) * 0.1, 'Distribution Change', rotation=90, verticalalignment='bottom', horizontalalignment='center', color='red', fontsize=14)
        
        # Add training end marker if available
        if training_end_window is not None and training_end_window >= 1:
            ax4.axvline(x=training_end_window, color='purple', linestyle='-', alpha=0.7, linewidth=2)
            ax4.text(training_end_window, ax4.get_ylim()[1], 'Training End', rotation=90, verticalalignment='top', 
                    color='purple', fontweight='bold', fontsize=24)
        
        # Add training start marker if available
        if training_start_window is not None and training_start_window >= 1:
            ax4.axvline(x=training_start_window, color='orange', linestyle='-', alpha=0.7, linewidth=2)
            ax4.text(training_start_window, ax4.get_ylim()[1], 'Training Start', rotation=90, verticalalignment='top', 
                    color='orange', fontweight='bold', fontsize=24)
        
        ax4.set_xlabel('Window Index')
        ax4.set_ylabel('Prediction Delta')
        ax4.set_title('Subspace-Kalman Prediction Delta vs Window Index\nStatic → Training → Inference Pipeline')
        ax4.legend(loc='upper right', framealpha=0.9, fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # 5. Plot absolute value of average Kalman gain times innovation vs window index
        if ekf_kalman_gain_times_innovation is not None:
            ax5 = fig.add_subplot(4, 2, 5)
            # Calculate average K*y per window
            avg_k_times_y = []
            for window_k_times_y in ekf_kalman_gain_times_innovation:
                window_avg = []
                for step_k_times_y in window_k_times_y:
                    if step_k_times_y:  # Check if there are any values in this step
                        window_avg.extend(step_k_times_y)
                if window_avg:
                    avg_k_times_y.append(np.mean(window_avg))
                else:
                    avg_k_times_y.append(0)
            
            x = np.arange(len(avg_k_times_y))[1:]  # Start from second sample
            ax5.plot(x, np.abs(np.array(avg_k_times_y)[1:]), 'orange', marker='v', linewidth=2, label='Static Model |Average K*Innovation|')
            
            # Add online model K*y if available
            if has_online_data and online_ekf_kalman_gain_times_innovation is not None and len(online_ekf_kalman_gain_times_innovation) > 0:
                # Calculate average online K*y per window
                online_avg_k_times_y = []
                for window_k_times_y in online_ekf_kalman_gain_times_innovation:
                    window_avg = []
                    for step_k_times_y in window_k_times_y:
                        if step_k_times_y:  # Check if there are any values in this step
                            window_avg.extend(step_k_times_y)
                    if window_avg:
                        online_avg_k_times_y.append(np.mean(window_avg))
                    else:
                        online_avg_k_times_y.append(0)
                
                # Use actual online window indices if available, otherwise fall back to learning start window
                if online_window_indices is not None and len(online_window_indices) > 0:
                    online_x = np.array(online_window_indices)  # Use all window indices
                elif learning_start_window is not None:
                    online_x = np.arange(learning_start_window, learning_start_window + len(online_avg_k_times_y))[1:]  # Start from second sample
                else:
                    online_x = np.arange(len(online_avg_k_times_y))[1:]  # Start from second sample
                ax5.plot(online_x, np.abs(np.array(online_avg_k_times_y)), 'red', marker='s', linewidth=2, label='Online Model |Average K*Innovation|')
            
            # Add training model K*y if available
            if has_training_data and training_ekf_kalman_gain_times_innovation is not None and len(training_ekf_kalman_gain_times_innovation) > 0:
                # Calculate average training K*y per window
                training_avg_k_times_y = []
                for window_k_times_y in training_ekf_kalman_gain_times_innovation:
                    window_avg = []
                    for step_k_times_y in window_k_times_y:
                        if step_k_times_y:  # Check if there are any values in this step
                            window_avg.extend(step_k_times_y)
                    if window_avg:
                        training_avg_k_times_y.append(np.mean(window_avg))
                    else:
                        training_avg_k_times_y.append(0)
                
                # Use actual training window indices
                training_x = np.array(training_window_indices)  # Use all window indices
                ax5.plot(training_x, np.abs(np.array(training_avg_k_times_y)), 'brown', marker='*', linewidth=2, label='Training Model |Average K*Innovation|', linestyle='--')
                
                # Connect training to online if both are available
                if has_online_data and online_avg_k_times_y is not None and len(online_avg_k_times_y) > 0:
                    # Connect last training point to first online point
                    last_training_x = training_x[-1]  # Window 12
                    first_online_x = online_x[0]      # Window 13
                    last_training_k_times_y = np.abs(np.array(training_avg_k_times_y)[-1])
                    first_online_k_times_y = np.abs(np.array(online_avg_k_times_y)[0])
                    
                    # Draw connecting line
                    ax5.plot([last_training_x, first_online_x], [last_training_k_times_y, first_online_k_times_y], 'brown', linestyle='-', linewidth=2, alpha=0.7)
            
            # Add eta change markers (adjusted for starting from second sample)
            for idx, eta in zip(eta_changes, eta_values):
                if idx >= 1:  # Only show markers from second sample onwards
                    ax5.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                    ax5.text(idx, ax5.get_ylim()[0] + (ax5.get_ylim()[1] - ax5.get_ylim()[0]) * 0.1, 'Distribution Change', rotation=90, verticalalignment='bottom', horizontalalignment='center', color='red', fontsize=14)
            
            # Add training end marker if available
            if training_end_window is not None and training_end_window >= 1:
                ax5.axvline(x=training_end_window, color='purple', linestyle='-', alpha=0.7, linewidth=2)
                ax5.text(training_end_window, ax5.get_ylim()[1], 'Training End', rotation=90, verticalalignment='top', 
                        color='purple', fontweight='bold', fontsize=24)
            
            # Add training start marker if available
            if training_start_window is not None and training_start_window >= 1:
                ax5.axvline(x=training_start_window, color='orange', linestyle='-', alpha=0.7, linewidth=2)
                ax5.text(training_start_window, ax5.get_ylim()[1], 'Training Start', rotation=90, verticalalignment='top', 
                        color='orange', fontweight='bold', fontsize=24)
            
            # Add labels and title
            ax5.set_xlabel('Window Index')
            ax5.set_ylabel('|Average K*Innovation|')
            title = 'Absolute Average Kalman Gain × Innovation vs Window Index\n|K_k × ν_k| = |K_k × (z_k - H x̂_k|k-1)|'
            if has_online_data:
                title += '\n(Static + Online Models)'
            ax5.set_title(title)
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Plot average Kalman gain vs window index
        if ekf_kalman_gains is not None:
            ax6 = fig.add_subplot(4, 2, 6)
            # Calculate average Kalman gain per window
            avg_kalman_gains = []
            for window_gains in ekf_kalman_gains:
                window_avg = []
                for step_gains in window_gains:
                    if step_gains:  # Check if there are any gains in this step
                        window_avg.extend(step_gains)
                if window_avg:
                    avg_kalman_gains.append(np.mean(window_avg))
                else:
                    avg_kalman_gains.append(0)
            
            x = np.arange(len(avg_kalman_gains))[1:]  # Start from second sample
            ax6.plot(x, np.array(avg_kalman_gains)[1:], 'purple', marker='d', label='Static Model Average Kalman Gain')
            
            # Add online model Kalman gains if available
            if has_online_data and online_ekf_kalman_gains is not None and len(online_ekf_kalman_gains) > 0:
                # Calculate average online Kalman gain per window
                online_avg_kalman_gains = []
                for window_gains in online_ekf_kalman_gains:
                    window_avg = []
                    for step_gains in window_gains:
                        if step_gains is not None and len(step_gains) > 0:  # Check if there are any gains in this step
                            if isinstance(step_gains, torch.Tensor):
                                window_avg.extend(step_gains.cpu().numpy().tolist())
                            else:
                                window_avg.extend(step_gains)
                    if window_avg:
                        online_avg_kalman_gains.append(np.mean(window_avg))
                    else:
                        online_avg_kalman_gains.append(0)
                
                # Use actual online window indices if available, otherwise fall back to learning start window
                if online_window_indices is not None and len(online_window_indices) > 0:
                    online_x = np.array(online_window_indices)  # Use all window indices
                elif learning_start_window is not None:
                    online_x = np.arange(learning_start_window, learning_start_window + len(online_avg_kalman_gains))[1:]  # Start from second sample
                else:
                    online_x = np.arange(len(online_avg_kalman_gains))[1:]  # Start from second sample
                ax6.plot(online_x, np.array(online_avg_kalman_gains), 'orange', marker='^', label='Online Model Average Kalman Gain')
            
            # Add training model Kalman gains if available
            if has_training_data and training_ekf_kalman_gains is not None and len(training_ekf_kalman_gains) > 0:
                # Calculate average training Kalman gain per window
                training_avg_kalman_gains = []
                for window_gains in training_ekf_kalman_gains:
                    window_avg = []
                    for step_gains in window_gains:
                        if step_gains is not None and len(step_gains) > 0:  # Check if there are any gains in this step
                            if isinstance(step_gains, torch.Tensor):
                                window_avg.extend(step_gains.cpu().numpy().tolist())
                            else:
                                window_avg.extend(step_gains)
                    if window_avg:
                        training_avg_kalman_gains.append(np.mean(window_avg))
                    else:
                        training_avg_kalman_gains.append(0)
                
                # Use actual training window indices
                training_x = np.array(training_window_indices)  # Use all window indices
                ax6.plot(training_x, np.array(training_avg_kalman_gains), 'brown', marker='*', label='Training Model Average Kalman Gain', linestyle='--')
                
                # Connect training to online if both are available
                if has_online_data and online_avg_kalman_gains is not None and len(online_avg_kalman_gains) > 0:
                    # Connect last training point to first online point
                    last_training_x = training_x[-1]  # Window 12
                    first_online_x = online_x[0]      # Window 13
                    last_training_kalman = np.array(training_avg_kalman_gains)[-1]
                    first_online_kalman = np.array(online_avg_kalman_gains)[0]
                    
                    # Draw connecting line
                    ax6.plot([last_training_x, first_online_x], [last_training_kalman, first_online_kalman], 'brown', linestyle='-', linewidth=2, alpha=0.7)
            
            # Add eta change markers (adjusted for starting from second sample)
            for idx, eta in zip(eta_changes, eta_values):
                if idx >= 1:  # Only show markers from second sample onwards
                    ax6.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                    ax6.text(idx, ax6.get_ylim()[0] + (ax6.get_ylim()[1] - ax6.get_ylim()[0]) * 0.1, 'Distribution Change', rotation=90, verticalalignment='bottom', horizontalalignment='center', color='red', fontsize=14)
            
            # Add training end marker if available
            if training_end_window is not None and training_end_window >= 1:
                ax6.axvline(x=training_end_window, color='purple', linestyle='-', alpha=0.7, linewidth=2)
                ax6.text(training_end_window, ax6.get_ylim()[1], 'Training End', rotation=90, verticalalignment='top', 
                        color='purple', fontweight='bold', fontsize=24)
        
            # Add training start marker if available
            if training_start_window is not None and training_start_window >= 1:
                ax6.axvline(x=training_start_window, color='orange', linestyle='-', alpha=0.7, linewidth=2)
                ax6.text(training_start_window, ax6.get_ylim()[1], 'Training Start', rotation=90, verticalalignment='top', 
                        color='orange', fontweight='bold', fontsize=24)
        
            # Add labels and title
            ax6.set_xlabel('Window Index')
            ax6.set_ylabel('Average Kalman Gain')
            title = 'Average Kalman Gain vs Window Index (Starting from Window 1)\nK_k = P_k|k-1 H^T (H P_k|k-1 H^T + R)^-1'
            if has_online_data:
                title += '\n(Static + Online Models)'
            ax6.set_title(title)
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7. Plot average K*y vs window index
        if ekf_kalman_gain_times_innovation is not None:
            ax7 = fig.add_subplot(4, 2, 7)
            # Calculate average K*y per window
            avg_k_times_y = []
            for window_k_times_y in ekf_kalman_gain_times_innovation:
                window_avg = []
                for step_k_times_y in window_k_times_y:
                    if step_k_times_y:  # Check if there are any values in this step
                        window_avg.extend(step_k_times_y)
                if window_avg:
                    avg_k_times_y.append(np.mean(window_avg))
                else:
                    avg_k_times_y.append(0)
            
            x = np.arange(len(avg_k_times_y))[1:]  # Start from second sample
            ax7.plot(x, np.array(avg_k_times_y)[1:], 'orange', marker='v', label='Static Model Average K*Innovation')
            
            # Add online model K*y if available
            if has_online_data and online_ekf_kalman_gain_times_innovation is not None and len(online_ekf_kalman_gain_times_innovation) > 0:
                # Calculate average online K*y per window
                online_avg_k_times_y = []
                for window_k_times_y in online_ekf_kalman_gain_times_innovation:
                    window_avg = []
                    for step_k_times_y in window_k_times_y:
                        if step_k_times_y:  # Check if there are any values in this step
                            window_avg.extend(step_k_times_y)
                    if window_avg:
                        online_avg_k_times_y.append(np.mean(window_avg))
                    else:
                        online_avg_k_times_y.append(0)
                
                # Use actual online window indices if available, otherwise fall back to learning start window
                if online_window_indices is not None and len(online_window_indices) > 0:
                    online_x = np.array(online_window_indices)  # Use all window indices
                elif learning_start_window is not None:
                    online_x = np.arange(learning_start_window, learning_start_window + len(online_avg_k_times_y))[1:]  # Start from second sample
                else:
                    online_x = np.arange(len(online_avg_k_times_y))[1:]  # Start from second sample
                ax7.plot(online_x, np.array(online_avg_k_times_y), 'red', marker='s', label='Online Model Average K*Innovation')
            
            # Add training model K*y if available
            if has_training_data and training_ekf_kalman_gain_times_innovation is not None and len(training_ekf_kalman_gain_times_innovation) > 0:
                # Calculate average training K*y per window
                training_avg_k_times_y = []
                for window_k_times_y in training_ekf_kalman_gain_times_innovation:
                    window_avg = []
                    for step_k_times_y in window_k_times_y:
                        if step_k_times_y:  # Check if there are any values in this step
                            window_avg.extend(step_k_times_y)
                    if window_avg:
                        training_avg_k_times_y.append(np.mean(window_avg))
                    else:
                        training_avg_k_times_y.append(0)
                
                # Use actual training window indices
                training_x = np.array(training_window_indices)  # Use all window indices
                ax7.plot(training_x, np.array(training_avg_k_times_y), 'brown', marker='*', label='Training Model Average K*Innovation', linestyle='--')
                
                # Connect training to online if both are available
                if has_online_data and online_avg_k_times_y is not None and len(online_avg_k_times_y) > 0:
                    # Connect last training point to first online point
                    last_training_x = training_x[-1]  # Window 12
                    first_online_x = online_x[0]      # Window 13
                    last_training_k_times_y = np.array(training_avg_k_times_y)[-1]
                    first_online_k_times_y = np.array(online_avg_k_times_y)[0]
                    
                    # Draw connecting line
                    ax7.plot([last_training_x, first_online_x], [last_training_k_times_y, first_online_k_times_y], 'brown', linestyle='-', linewidth=2, alpha=0.7)
            
            # Add eta change markers (adjusted for starting from second sample)
            for idx, eta in zip(eta_changes, eta_values):
                if idx >= 1:  # Only show markers from second sample onwards
                    ax7.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                    ax7.text(idx, ax7.get_ylim()[0] + (ax7.get_ylim()[1] - ax7.get_ylim()[0]) * 0.1, 'Distribution Change', rotation=90, verticalalignment='bottom', horizontalalignment='center', color='red', fontsize=14)
            
            # Add training end marker if available
            if training_end_window is not None and training_end_window >= 1:
                ax7.axvline(x=training_end_window, color='purple', linestyle='-', alpha=0.7, linewidth=2)
                ax7.text(training_end_window, ax7.get_ylim()[1], 'Training End', rotation=90, verticalalignment='top', 
                        color='purple', fontweight='bold', fontsize=24)
        
            # Add training start marker if available
            if training_start_window is not None and training_start_window >= 1:
                ax7.axvline(x=training_start_window, color='orange', linestyle='-', alpha=0.7, linewidth=2)
                ax7.text(training_start_window, ax7.get_ylim()[1], 'Training Start', rotation=90, verticalalignment='top', 
                        color='orange', fontweight='bold', fontsize=24)
        
            # Add labels and title
            ax7.set_xlabel('Window Index')
            ax7.set_ylabel('Average K*Innovation')
            title = 'Average Kalman Gain × Innovation vs Window Index (Starting from Window 1)\nK_k × ν_k = K_k × (z_k - H x̂_k|k-1)'
            if has_online_data:
                title += '\n(Static + Online Models)'
            ax7.set_title(title)
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # 8. Plot average y*(S^-1)*y vs window index
        if ekf_y_s_inv_y is not None:
            ax8 = fig.add_subplot(4, 2, 8)
            # Calculate average y*(S^-1)*y per window
            avg_y_s_inv_y = []
            for window_y_s_inv_y in ekf_y_s_inv_y:
                window_avg = []
                for step_y_s_inv_y in window_y_s_inv_y:
                    if step_y_s_inv_y:  # Check if there are any values in this step
                        window_avg.extend(step_y_s_inv_y)
                if window_avg:
                    avg_y_s_inv_y.append(np.mean(window_avg))
                else:
                    avg_y_s_inv_y.append(0)
            
            x = np.arange(len(avg_y_s_inv_y))[1:]  # Start from second sample
            ax8.plot(x, np.array(avg_y_s_inv_y)[1:], 'red', marker='^', label='Static Model Average y*(S^-1)*y')
            
            # Add online model y*(S^-1)*y if available
            if has_online_data and online_ekf_y_s_inv_y is not None and len(online_ekf_y_s_inv_y) > 0:
                # Calculate average online y*(S^-1)*y per window
                online_avg_y_s_inv_y = []
                for window_y_s_inv_y in online_ekf_y_s_inv_y:
                    window_avg = []
                    for step_y_s_inv_y in window_y_s_inv_y:
                        if step_y_s_inv_y:  # Check if there are any values in this step
                            window_avg.extend(step_y_s_inv_y)
                    if window_avg:
                        online_avg_y_s_inv_y.append(np.mean(window_avg))
                    else:
                        online_avg_y_s_inv_y.append(0)
                
                # Use actual online window indices if available, otherwise fall back to learning start window
                if online_window_indices is not None and len(online_window_indices) > 0:
                    online_x = np.array(online_window_indices)  # Use all window indices
                elif learning_start_window is not None:
                    online_x = np.arange(learning_start_window, learning_start_window + len(online_avg_y_s_inv_y))[1:]  # Start from second sample
                else:
                    online_x = np.arange(len(online_avg_y_s_inv_y))[1:]  # Start from second sample
                ax8.plot(online_x, np.array(online_avg_y_s_inv_y), 'brown', marker='*', label='Online Model Average y*(S^-1)*y')
            
            # Add training model y*(S^-1)*y if available
            if has_training_data and training_ekf_y_s_inv_y is not None and len(training_ekf_y_s_inv_y) > 0:
                # Calculate average training y*(S^-1)*y per window
                training_avg_y_s_inv_y = []
                for window_y_s_inv_y in training_ekf_y_s_inv_y:
                    window_avg = []
                    for step_y_s_inv_y in window_y_s_inv_y:
                        if step_y_s_inv_y:  # Check if there are any values in this step
                            window_avg.extend(step_y_s_inv_y)
                    if window_avg:
                        training_avg_y_s_inv_y.append(np.mean(window_avg))
                    else:
                        training_avg_y_s_inv_y.append(0)
                
                # Use actual training window indices
                training_x = np.array(training_window_indices)  # Use all window indices
                ax8.plot(training_x, np.array(training_avg_y_s_inv_y), 'gray', marker='o', label='Training Model Average y*(S^-1)*y', linestyle='--')
                
                # Connect training to online if both are available
                if has_online_data and online_avg_y_s_inv_y is not None and len(online_avg_y_s_inv_y) > 0:
                    # Connect last training point to first online point
                    last_training_x = training_x[-1]  # Window 12
                    first_online_x = online_x[0]      # Window 13
                    last_training_y_s_inv_y = np.array(training_avg_y_s_inv_y)[-1]
                    first_online_y_s_inv_y = np.array(online_avg_y_s_inv_y)[0]
                    
                    # Draw connecting line
                    ax8.plot([last_training_x, first_online_x], [last_training_y_s_inv_y, first_online_y_s_inv_y], 'gray', linestyle='-', linewidth=2, alpha=0.7)
            
            # Add eta change markers (adjusted for starting from second sample)
            for idx, eta in zip(eta_changes, eta_values):
                if idx >= 1:  # Only show markers from second sample onwards
                    ax8.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                    ax8.text(idx, ax8.get_ylim()[0] + (ax8.get_ylim()[1] - ax8.get_ylim()[0]) * 0.1, 'Distribution Change', rotation=90, verticalalignment='bottom', horizontalalignment='center', color='red', fontsize=14)
            
            # Add training end marker if available
            if training_end_window is not None and training_end_window >= 1:
                ax8.axvline(x=training_end_window, color='purple', linestyle='-', alpha=0.7, linewidth=2)
                ax8.text(training_end_window, ax8.get_ylim()[1], 'Training End', rotation=90, verticalalignment='top', 
                        color='purple', fontweight='bold', fontsize=24)
            
            # Add training start marker if available
            if training_start_window is not None and training_start_window >= 1:
                ax8.axvline(x=training_start_window, color='orange', linestyle='-', alpha=0.7, linewidth=2)
                ax8.text(training_start_window, ax8.get_ylim()[1], 'Training Start', rotation=90, verticalalignment='top', 
                        color='orange', fontweight='bold', fontsize=24)
            
            # Add labels and title
            ax8.set_xlabel('Window Index')
            ax8.set_ylabel('Average y*(S^-1)*y')
            title = 'Average Innovation Covariance Metric vs Window Index (Starting from Window 1)\ny*(S^-1)*y = ν^T S^-1 ν'
            if has_online_data:
                title += '\n(Static + Online Models)'
            ax8.set_title(title)
            ax8.legend()
            ax8.grid(True, alpha=0.3)

        
        # Adjust layout and save
        plt.tight_layout()
        plot_path = plot_dir / f"online_learning_results_{timestamp}.png"
        plt.savefig(plot_path)
        plt.close()
        
        # Plot online learning trajectory
        trajectory_plot_path = plot_online_learning_trajectory(window_labels, plot_dir, timestamp)
        
        logger.info(f"Online learning plots saved: {plot_path.name}")
        return plot_path, trajectory_plot_path
        
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.warning("matplotlib not available for plotting")
        return None, None
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error plotting online learning results: {e}")
        return None, None


def _collect_deduplicated_trajectory_steps(window_labels, window_indices=None, stride=1):
    """
    Build one GT label per trajectory step, deduplicating overlapping sliding windows.

    Returns:
        sorted_steps: ascending trajectory step indices
        step_labels: list of label arrays (radians), one per sorted step
        step_window_ids: window index that last wrote each step
    """
    import numpy as np

    if window_indices is None:
        window_indices = list(range(len(window_labels)))

    step_map = {}
    step_window_ids = {}
    for window_idx, window_label_list in zip(window_indices, window_labels):
        start_step = int(window_idx) * int(stride)
        for step_offset, step_labels in enumerate(window_label_list):
            global_step = start_step + step_offset
            step_map[global_step] = np.asarray(step_labels)
            step_window_ids[global_step] = int(window_idx)

    sorted_steps = sorted(step_map.keys())
    return sorted_steps, [step_map[s] for s in sorted_steps], [step_window_ids[s] for s in sorted_steps]


def plot_online_learning_trajectory(
    window_labels,
    plot_dir,
    timestamp,
    window_indices=None,
    stride=1,
):
    """
    Plot the full trajectory across all windows for online learning.
    
    Args:
        window_labels: List of labels for each window, where each window contains a list of numpy arrays
        plot_dir: Directory to save the plots
        timestamp: Timestamp for the plot filename
        window_indices: Actual window indices (required when stride < window_size)
        stride: Sliding-window stride used during online learning
        
    Returns:
        Path to saved trajectory plot, or None
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from pathlib import Path
        
        logger = logging.getLogger(__name__)

        sorted_steps, step_labels_list, step_window_ids = _collect_deduplicated_trajectory_steps(
            window_labels, window_indices=window_indices, stride=stride
        )
        if not step_labels_list:
            logger.warning("No trajectory data available for plotting")
            return None

        all_angles = []
        all_distances = []
        for global_step, step_labels in zip(sorted_steps, step_labels_list):
            angles_deg = step_labels * (180.0 / np.pi)
            all_angles.append(angles_deg)
            base_distance = 20.0 + global_step * 1.0
            all_distances.append(np.full(len(step_labels), base_distance))

        max_sources = max(len(angles) for angles in all_angles)
        total_steps = len(all_angles)
        
        # Pad arrays to have consistent dimensions
        padded_angles = np.full((total_steps, max_sources), np.nan)
        padded_distances = np.full((total_steps, max_sources), np.nan)
        
        for step_idx, (angles, distances) in enumerate(zip(all_angles, all_distances)):
            num_sources = len(angles)
            padded_angles[step_idx, :num_sources] = angles
            padded_distances[step_idx, :num_sources] = distances
        
        # Create the trajectory plot
        plt.figure(figsize=(12, 10))
        
        # Plot each source trajectory
        for s in range(max_sources):
            # Get valid data for this source (some steps might have fewer sources)
            valid_mask = ~np.isnan(padded_angles[:, s])
            if np.any(valid_mask):
                angles_rad = padded_angles[valid_mask, s] * (np.pi / 180.0)  # Convert back to radians for plotting
                distances = padded_distances[valid_mask, s]
                
                # Convert from polar to Cartesian coordinates
                x = distances * np.cos(angles_rad)
                y = distances * np.sin(angles_rad)
                
                # Plot trajectory
                plt.plot(x, y, '-o', markersize=4, label=f'Source {s+1}')
                
                # Mark start and end points
                if len(x) > 0:
                    plt.plot(x[0], y[0], 'go', markersize=8)  # Green for start
                    plt.plot(x[-1], y[-1], 'ro', markersize=8)  # Red for end
        
        # Plot radar location
        plt.plot(0, 0, 'bD', markersize=12, label='Radar')
        
        # Add distance circles
        for d in [20, 30, 40, 50]:
            circle = plt.Circle((0, 0), d, fill=False, linestyle='--', alpha=0.3)
            plt.gca().add_patch(circle)
            plt.text(0, d, f'{d}m', va='bottom', ha='center')
        
        # Add angle lines
        for a in range(-90, 91, 30):
            a_rad = a * (np.pi/180)
            plt.plot([0, 60*np.cos(a_rad)], [0, 60*np.sin(a_rad)], 'k:', alpha=0.2)
            plt.text(55*np.cos(a_rad), 55*np.sin(a_rad), f'{a}°', 
                    va='center', ha='center', bbox=dict(facecolor='white', alpha=0.5))
        
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.title(
            f'Online Learning Full Trajectory (T={total_steps} steps, Sources={max_sources}, '
            f'Windows={len(window_labels)}, stride={stride})'
        )
        plt.legend()

        plot_path = Path(plot_dir) / f"online_learning_trajectory_{timestamp}.png"
        plt.savefig(plot_path)
        plt.close()

        # Angle vs trajectory step (far-field; more useful than fake-range xy for sine_accel)
        fig, ax = plt.subplots(figsize=(12, 5))
        padded = np.full((total_steps, max_sources), np.nan)
        for i, angles in enumerate(all_angles):
            padded[i, : len(angles)] = angles
        for s in range(max_sources):
            ax.plot(sorted_steps, padded[:, s], "-o", markersize=3, label=f"Source {s + 1}")
        ax.set_xlabel("Trajectory step")
        ax.set_ylabel("DOA (degrees)")
        ax.set_title("Ground-truth DOA vs trajectory step")
        ax.grid(True, alpha=0.3)
        ax.legend()
        angles_plot_path = Path(plot_dir) / f"online_learning_trajectory_angles_{timestamp}.png"
        fig.savefig(angles_plot_path, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Online learning trajectory plots saved to {plot_dir}:")
        logger.info(f"  - XY trajectory: {plot_path.name}")
        logger.info(f"  - Angles vs step: {angles_plot_path.name}")

        return plot_path
        
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.warning("matplotlib not available for plotting online learning trajectory")
        return None
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error plotting online learning trajectory: {e}")
        return None



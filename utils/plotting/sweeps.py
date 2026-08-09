"""Plotting helpers — sweeps."""
from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

SCENARIO_AXIS_LABELS = {
    "snr": "SNR (dB)",
    "m": "Number of sources (M)",
    "t": "Snapshots (T)",
    "eta": "Steering error η",
    "trajectory_length": "Trajectory length (steps)",
}

SCENARIO_PLOT_TITLES = {
    "snr": "DOA tracking error vs SNR",
    "m": "DOA tracking error vs number of sources",
    "t": "DOA tracking error vs snapshots",
    "eta": "DOA tracking error vs steering error η",
    "trajectory_length": "DOA tracking error vs trajectory length",
}

def _first_trajectory_from_ol_results(ol_results: dict):
    """Return the first TrajectoryResults object from OL result payloads."""
    for key in (
        "pretrained_trajectory_results",
        "pretrained_model_trajectory_results",
        "online_trajectory_results",
        "online_model_trajectory_results",
    ):
        val = ol_results.get(key)
        if isinstance(val, list) and val:
            return val[0]
        if val is not None and hasattr(val, "window_results"):
            return val
    return None


def _nested_ekf_series_from_trajectory(traj) -> tuple:
    """Build legacy nested EKF series expected by the 4D grid plotter."""
    innovations, k_times_y, y_s_inv_y = [], [], []
    if traj is None or not getattr(traj, "window_results", None):
        return innovations, k_times_y, y_s_inv_y

    for wr in traj.window_results:
        if not getattr(wr, "is_valid", True):
            continue
        wm = wr.window_metrics
        innovations.append([wm.avg_ekf_innovations or []])
        k_times_y.append([wm.avg_ekf_kalman_gain_times_innovation or []])
        y_s_inv_y.append([wm.avg_ekf_y_s_inv_y or []])
    return innovations, k_times_y, y_s_inv_y


def _loss_series_from_trajectory(traj) -> tuple:
    """Extract parallel loss / eta / index series from a trajectory object."""
    if traj is None or not getattr(traj, "window_results", None):
        return [], [], [], []

    window_losses, pre_ekf_losses, window_eta_values, window_indices = [], [], [], []
    for w_idx, wr in zip(getattr(traj, "window_indices", []), traj.window_results):
        if not getattr(wr, "is_valid", True):
            continue
        window_indices.append(w_idx)
        window_losses.append(wr.loss_metrics.reference_metric_loss)
        pre_ekf_losses.append(wr.loss_metrics.pre_ekf_loss)
        window_eta_values.append(wr.window_metrics.eta_value)
    return window_losses, pre_ekf_losses, window_eta_values, window_indices


def _scalar_ekf_window_metrics(wr) -> tuple:
    """Scalar EKF diagnostic averages for one window (aligned with 4D grid plots)."""
    wm = wr.window_metrics
    inns = wm.avg_ekf_innovations or []
    kty = wm.avg_ekf_kalman_gain_times_innovation or []
    ysy = wm.avg_ekf_y_s_inv_y or []
    avg_inn = float(np.mean([abs(x) for x in inns])) if inns else 0.0
    avg_kty = float(np.mean(kty)) if kty else 0.0
    avg_ysy = float(np.mean(ysy)) if ysy else 0.0
    return avg_inn, avg_kty, avg_ysy


def _normalize_online_learning_results_for_4d_grid(ol_results: dict) -> dict:
    """Adapt structured trajectory OL results to the flat lists used by 4D grid plots."""
    if ol_results.get("window_losses"):
        return ol_results

    pretrained = ol_results.get("pretrained_trajectory_results")
    if isinstance(pretrained, list) and pretrained:
        pretrained = pretrained[0]
    elif pretrained is None:
        pretrained = ol_results.get("pretrained_model_trajectory_results")

    online = ol_results.get("online_trajectory_results")
    if isinstance(online, list) and online:
        online = online[0]
    elif online is None:
        online = ol_results.get("online_model_trajectory_results")

    window_losses, pre_ekf_losses, window_eta_values, static_window_indices = _loss_series_from_trajectory(
        pretrained
    )
    ekf_innovations, ekf_kalman_gain_times_innovation, ekf_y_s_inv_y = _nested_ekf_series_from_trajectory(
        pretrained
    )

    training_start = ol_results.get("training_start_window")
    training_end = ol_results.get("training_end_window")
    learning_start_window = ol_results.get("learning_start_window", training_start)

    online_window_losses, online_pre_ekf_losses, online_window_indices = [], [], []
    training_window_losses, training_pre_ekf_losses, training_window_indices = [], [], []
    online_avg_innovations, online_avg_k_times_y, online_avg_y_s_inv_y = [], [], []
    training_avg_innovations, training_avg_k_times_y, training_avg_y_s_inv_y = [], [], []
    if online is not None and getattr(online, "window_results", None):
        for w_idx, wr in zip(getattr(online, "window_indices", []), online.window_results):
            if not getattr(wr, "is_valid", True):
                continue
            loss = wr.loss_metrics.reference_metric_loss
            pre = wr.loss_metrics.pre_ekf_loss
            avg_inn, avg_kty, avg_ysy = _scalar_ekf_window_metrics(wr)
            if training_end is not None and w_idx > training_end:
                online_window_losses.append(loss)
                online_pre_ekf_losses.append(pre)
                online_window_indices.append(w_idx)
                online_avg_innovations.append(avg_inn)
                online_avg_k_times_y.append(avg_kty)
                online_avg_y_s_inv_y.append(avg_ysy)
            elif training_start is not None and w_idx >= training_start:
                training_window_losses.append(loss)
                training_pre_ekf_losses.append(pre)
                training_window_indices.append(w_idx)
                training_avg_innovations.append(avg_inn)
                training_avg_k_times_y.append(avg_kty)
                training_avg_y_s_inv_y.append(avg_ysy)

    normalized = dict(ol_results)
    normalized.update(
        {
            "window_losses": window_losses,
            "pre_ekf_losses": pre_ekf_losses,
            "window_eta_values": window_eta_values,
            "static_window_indices": static_window_indices,
            "ekf_innovations": ekf_innovations,
            "ekf_kalman_gain_times_innovation": ekf_kalman_gain_times_innovation,
            "ekf_y_s_inv_y": ekf_y_s_inv_y,
            "online_window_losses": online_window_losses,
            "online_pre_ekf_losses": online_pre_ekf_losses,
            "online_window_indices": online_window_indices,
            "training_window_losses": training_window_losses,
            "training_pre_ekf_losses": training_pre_ekf_losses,
            "training_window_indices": training_window_indices,
            "online_avg_innovations": online_avg_innovations,
            "online_avg_k_times_y": online_avg_k_times_y,
            "online_avg_y_s_inv_y": online_avg_y_s_inv_y,
            "training_avg_innovations": training_avg_innovations,
            "training_avg_k_times_y": training_avg_k_times_y,
            "training_avg_y_s_inv_y": training_avg_y_s_inv_y,
            "learning_start_window": learning_start_window,
        }
    )
    return normalized


def plot_eta_comparison_4d_grid(scenario_results, output_dir):
    """
    Plot comparison between different eta scenarios with identical other settings.
    
    For each combination of process_noise, kf_process_noise, and kf_measurement_noise,
    compare metrics across different eta values.
    
    Args:
        scenario_results: 4D dict with structure {proc_noise: {kf_proc_noise: {kf_meas_noise: {eta: result}}}}
        output_dir: Output directory for saving the plots
        
    Returns:
        List of paths to the saved plots
    """
    logger = logging.getLogger("SubspaceNet.plotting")
    
    # Create timestamp and plot directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = Path(output_dir) / "eta_comparison_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    saved_plots = []
    
    # Extract all parameter combinations (excluding eta)
    proc_noise_values = list(scenario_results.keys())
    
    logger.info(f"Creating eta comparison plots for {len(proc_noise_values)} process noise values")
    
    combination_count = 0
    total_combinations = 0
    
    # Count total combinations first
    for proc_noise in proc_noise_values:
        for kf_proc_noise in scenario_results[proc_noise].keys():
            for kf_meas_noise in scenario_results[proc_noise][kf_proc_noise].keys():
                total_combinations += 1
    
    for proc_noise in proc_noise_values:
        for kf_proc_noise in scenario_results[proc_noise].keys():
            for kf_meas_noise in scenario_results[proc_noise][kf_proc_noise].keys():
                combination_count += 1
                
                # Get all eta results for this parameter combination
                eta_results = scenario_results[proc_noise][kf_proc_noise][kf_meas_noise]
                eta_values = sorted(eta_results.keys())
                
                if len(eta_values) < 2:
                    logger.warning(f"Skipping combination {combination_count}/{total_combinations}: only {len(eta_values)} eta values")
                    continue
                
                logger.info(f"Processing combination {combination_count}/{total_combinations}: proc_noise={proc_noise}, kf_proc_noise={kf_proc_noise}, kf_meas_noise={kf_meas_noise}")
                
                # Extract metrics for each eta value
                metrics_by_eta = {}
                valid_etas = []
                
                for eta in eta_values:
                    result = eta_results[eta]
                    
                    # Check if result is valid and has online learning results
                    if (result.get("status") == "success" and 
                        "online_learning_results" in result):
                        
                        ol_results = _normalize_online_learning_results_for_4d_grid(
                            result["online_learning_results"]
                        )
                        
                        # Extract required metrics
                        window_losses = ol_results.get("window_losses", [])
                        pre_ekf_losses = ol_results.get("pre_ekf_losses", [])
                        window_eta_values = ol_results.get("window_eta_values", [])
                        ekf_innovations = ol_results.get("ekf_innovations", [])
                        ekf_kalman_gain_times_innovation = ol_results.get("ekf_kalman_gain_times_innovation", [])
                        ekf_y_s_inv_y = ol_results.get("ekf_y_s_inv_y", [])
                        
                        # Extract online learning data if available
                        online_window_losses = ol_results.get("online_window_losses", [])
                        online_pre_ekf_losses = ol_results.get("online_pre_ekf_losses", [])
                        online_window_indices = ol_results.get("online_window_indices", [])
                        training_window_losses = ol_results.get("training_window_losses", [])
                        training_pre_ekf_losses = ol_results.get("training_pre_ekf_losses", [])
                        training_window_indices = ol_results.get("training_window_indices", [])
                        learning_start_window = ol_results.get("learning_start_window", None)
                        static_window_indices = ol_results.get("static_window_indices", [])
                        
                        # Extract online learning EKF data
                        online_ekf_innovations = ol_results.get("online_ekf_innovations", [])
                        online_ekf_kalman_gain_times_innovation = ol_results.get("online_ekf_kalman_gain_times_innovation", [])
                        online_ekf_y_s_inv_y = ol_results.get("online_ekf_y_s_inv_y", [])
                        training_ekf_innovations = ol_results.get("training_ekf_innovations", [])
                        training_ekf_kalman_gain_times_innovation = ol_results.get("training_ekf_kalman_gain_times_innovation", [])
                        training_ekf_y_s_inv_y = ol_results.get("training_ekf_y_s_inv_y", [])
                        
                        # Calculate derived metrics
                        if len(window_losses) == len(pre_ekf_losses) and len(window_losses) > 1:
                            # EKF improvement = pre_ekf_loss - ekf_loss
                            ekf_improvement = [pre - post for pre, post in zip(pre_ekf_losses[1:], window_losses[1:])]
                            
                            # Calculate average innovation magnitude per window
                            avg_innovations = []
                            for window_innovations in ekf_innovations:
                                window_avg = []
                                for step_innovations in window_innovations:
                                    if step_innovations:
                                        window_avg.extend([abs(inn) for inn in step_innovations])
                                if window_avg:
                                    avg_innovations.append(np.mean(window_avg))
                                else:
                                    avg_innovations.append(0)
                            
                            # Calculate average K*y per window
                            avg_k_times_y = []
                            for window_k_times_y in ekf_kalman_gain_times_innovation:
                                window_avg = []
                                for step_k_times_y in window_k_times_y:
                                    if step_k_times_y:
                                        window_avg.extend(step_k_times_y)
                                if window_avg:
                                    avg_k_times_y.append(np.mean(window_avg))
                                else:
                                    avg_k_times_y.append(0)
                            
                            # Calculate average y*S^-1*y per window
                            avg_y_s_inv_y = []
                            for window_y_s_inv_y in ekf_y_s_inv_y:
                                window_avg = []
                                for step_y_s_inv_y in window_y_s_inv_y:
                                    if step_y_s_inv_y:
                                        window_avg.extend(step_y_s_inv_y)
                                if window_avg:
                                    avg_y_s_inv_y.append(np.mean(window_avg))
                                else:
                                    avg_y_s_inv_y.append(0)
                            
                            # Calculate online learning metrics if available
                            has_online_data = (len(online_window_losses) > 0 and len(online_window_indices) > 0)
                            has_training_data = (len(training_window_losses) > 0 and len(training_window_indices) > 0)
                            
                            online_avg_innovations = list(ol_results.get("online_avg_innovations", []))
                            online_avg_k_times_y = list(ol_results.get("online_avg_k_times_y", []))
                            online_avg_y_s_inv_y = list(ol_results.get("online_avg_y_s_inv_y", []))
                            training_avg_innovations = list(ol_results.get("training_avg_innovations", []))
                            training_avg_k_times_y = list(ol_results.get("training_avg_k_times_y", []))
                            training_avg_y_s_inv_y = list(ol_results.get("training_avg_y_s_inv_y", []))

                            if has_online_data and not online_avg_innovations and online_ekf_innovations:
                                for window_innovations in online_ekf_innovations:
                                    window_avg = []
                                    for step_innovations in window_innovations:
                                        if step_innovations:
                                            window_avg.extend([abs(inn) for inn in step_innovations])
                                    if window_avg:
                                        online_avg_innovations.append(np.mean(window_avg))
                                    else:
                                        online_avg_innovations.append(0)
                                        
                                for window_k_times_y in online_ekf_kalman_gain_times_innovation:
                                    window_avg = []
                                    for step_k_times_y in window_k_times_y:
                                        if step_k_times_y:
                                            window_avg.extend(step_k_times_y)
                                    if window_avg:
                                        online_avg_k_times_y.append(np.mean(window_avg))
                                    else:
                                        online_avg_k_times_y.append(0)
                                        
                                for window_y_s_inv_y in online_ekf_y_s_inv_y:
                                    window_avg = []
                                    for step_y_s_inv_y in window_y_s_inv_y:
                                        if step_y_s_inv_y:
                                            window_avg.extend(step_y_s_inv_y)
                                    if window_avg:
                                        online_avg_y_s_inv_y.append(np.mean(window_avg))
                                    else:
                                        online_avg_y_s_inv_y.append(0)
                            
                            if has_training_data and not training_avg_innovations and training_ekf_innovations:
                                for window_innovations in training_ekf_innovations:
                                    window_avg = []
                                    for step_innovations in window_innovations:
                                        if step_innovations:
                                            window_avg.extend([abs(inn) for inn in step_innovations])
                                    if window_avg:
                                        training_avg_innovations.append(np.mean(window_avg))
                                    else:
                                        training_avg_innovations.append(0)
                                        
                                for window_k_times_y in training_ekf_kalman_gain_times_innovation:
                                    window_avg = []
                                    for step_k_times_y in window_k_times_y:
                                        if step_k_times_y:
                                            window_avg.extend(step_k_times_y)
                                    if window_avg:
                                        training_avg_k_times_y.append(np.mean(window_avg))
                                    else:
                                        training_avg_k_times_y.append(0)
                                        
                                for window_y_s_inv_y in training_ekf_y_s_inv_y:
                                    window_avg = []
                                    for step_y_s_inv_y in window_y_s_inv_y:
                                        if step_y_s_inv_y:
                                            window_avg.extend(step_y_s_inv_y)
                                    if window_avg:
                                        training_avg_y_s_inv_y.append(np.mean(window_avg))
                                    else:
                                        training_avg_y_s_inv_y.append(0)
                            
                            metrics_by_eta[eta] = {
                                "window_losses": window_losses[1:],  # Exclude first window
                                "static_window_indices": static_window_indices[1:] if len(static_window_indices) > 1 else [],
                                "ekf_improvement": ekf_improvement,
                                "window_eta_values": window_eta_values[1:] if len(window_eta_values) > 1 else [],  # Exclude first window
                                "avg_innovations": avg_innovations[1:] if len(avg_innovations) > 1 else [],
                                "avg_k_times_y": avg_k_times_y[1:] if len(avg_k_times_y) > 1 else [],
                                "avg_y_s_inv_y": avg_y_s_inv_y[1:] if len(avg_y_s_inv_y) > 1 else [],
                                # Online learning data
                                "has_online_data": has_online_data,
                                "has_training_data": has_training_data,
                                "online_window_losses": online_window_losses,
                                "online_pre_ekf_losses": online_pre_ekf_losses,
                                "online_window_indices": online_window_indices,
                                "online_avg_innovations": online_avg_innovations,
                                "online_avg_k_times_y": online_avg_k_times_y,
                                "online_avg_y_s_inv_y": online_avg_y_s_inv_y,
                                "training_window_losses": training_window_losses,
                                "training_pre_ekf_losses": training_pre_ekf_losses,
                                "training_window_indices": training_window_indices,
                                "training_avg_innovations": training_avg_innovations,
                                "training_avg_k_times_y": training_avg_k_times_y,
                                "training_avg_y_s_inv_y": training_avg_y_s_inv_y,
                                "learning_start_window": learning_start_window
                            }
                            valid_etas.append(eta)
                        else:
                            logger.warning(f"Invalid data for eta={eta}: mismatched lengths or insufficient data")
                    else:
                        logger.warning(f"Invalid result for eta={eta}: {result.get('status', 'unknown status')}")
                
                if len(valid_etas) < 2:
                    logger.warning(f"Skipping combination {combination_count}/{total_combinations}: only {len(valid_etas)} valid eta results")
                    continue
                
                # Create comparison plot for this parameter combination
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))

                def _static_x(metrics):
                    x = metrics.get("static_window_indices")
                    losses = metrics.get("window_losses", [])
                    if x and len(x) == len(losses):
                        return np.array(x)
                    return np.arange(1, len(losses) + 1)

                first_eta = valid_etas[0]
                training_start_window = metrics_by_eta[first_eta].get("learning_start_window")
                training_end_window = None
                for eta in valid_etas:
                    ol_raw = eta_results[eta].get("online_learning_results", {})
                    if ol_raw.get("training_end_window") is not None:
                        training_end_window = ol_raw["training_end_window"]
                        break
                has_any_online = any(metrics_by_eta[e]["has_online_data"] for e in valid_etas)
                
                # For each eta scenario, find where eta changes occur within that scenario
                eta_change_markers = {}
                for eta in valid_etas:
                    eta_change_markers[eta] = {"positions": [], "values": []}
                    
                    # Get the window eta values for this scenario
                    scenario_eta_values = metrics_by_eta[eta]["window_eta_values"]
                    scenario_x = _static_x(metrics_by_eta[eta])
                    
                    if len(scenario_eta_values) > 1 and len(scenario_x) == len(scenario_eta_values):
                        for i in range(1, len(scenario_eta_values)):
                            if abs(scenario_eta_values[i] - scenario_eta_values[i - 1]) > 1e-6:
                                eta_change_markers[eta]["positions"].append(scenario_x[i])
                                eta_change_markers[eta]["values"].append(scenario_eta_values[i])
                
                # Plot 1: Window Losses (Both EKF and SubspaceNet)
                ax1 = axes[0, 0]
                
                # Define colors for different eta values
                colors = plt.cm.tab10(np.linspace(0, 1, len(valid_etas)))
                
                for i, eta in enumerate(valid_etas):
                    color = colors[i]
                    static_x = _static_x(metrics_by_eta[eta])
                    
                    # Plot EKF Loss (solid line, circle markers)
                    ax1.plot(static_x, metrics_by_eta[eta]["window_losses"], 
                            color=color, linestyle='-', marker='o', 
                            label=f'EKF η={eta:.3f}', linewidth=2, markersize=4)
                    
                    # Plot SubspaceNet Loss (dashed line, square markers)
                    # Calculate SubspaceNet loss from EKF loss + improvement
                    subspacenet_losses = [ekf_loss + improvement for ekf_loss, improvement in 
                                        zip(metrics_by_eta[eta]["window_losses"], metrics_by_eta[eta]["ekf_improvement"])]
                    ax1.plot(static_x, subspacenet_losses, 
                            color=color, linestyle='--', marker='s', 
                            label=f'SubspaceNet η={eta:.3f}', linewidth=2, markersize=4)
                    
                    # Add online learning data if available
                    if metrics_by_eta[eta]["has_online_data"]:
                        online_x = np.array(metrics_by_eta[eta]["online_window_indices"])
                        ax1.plot(online_x, metrics_by_eta[eta]["online_window_losses"], 
                                color=color, linestyle='-', marker='d', 
                                label=f'Online EKF η={eta:.3f}', linewidth=2, markersize=6)
                        ax1.plot(online_x, metrics_by_eta[eta]["online_pre_ekf_losses"], 
                                color=color, linestyle='--', marker='^', 
                                label=f'Online SubspaceNet η={eta:.3f}', linewidth=2, markersize=6)
                    
                    # Add training data if available
                    if metrics_by_eta[eta]["has_training_data"]:
                        training_x = np.array(metrics_by_eta[eta]["training_window_indices"])
                        ax1.plot(training_x, metrics_by_eta[eta]["training_window_losses"], 
                                color=color, linestyle='-', marker='*', 
                                label=f'Training EKF η={eta:.3f}', linewidth=2, markersize=6, alpha=0.7)
                        ax1.plot(training_x, metrics_by_eta[eta]["training_pre_ekf_losses"], 
                                color=color, linestyle='--', marker='s', 
                                label=f'Training SubspaceNet η={eta:.3f}', linewidth=2, markersize=6, alpha=0.7)
                        
                        # Connect training to online if both are available
                        if metrics_by_eta[eta]["has_online_data"]:
                            last_training_x = training_x[-1]
                            first_online_x = online_x[0]
                            last_training_ekf = metrics_by_eta[eta]["training_window_losses"][-1]
                            first_online_ekf = metrics_by_eta[eta]["online_window_losses"][0]
                            last_training_subspace = metrics_by_eta[eta]["training_pre_ekf_losses"][-1]
                            first_online_subspace = metrics_by_eta[eta]["online_pre_ekf_losses"][0]
                            
                            # Draw connecting lines
                            ax1.plot([last_training_x, first_online_x], [last_training_ekf, first_online_ekf], 
                                    color=color, linestyle='-', linewidth=2, alpha=0.7)
                            ax1.plot([last_training_x, first_online_x], [last_training_subspace, first_online_subspace], 
                                    color=color, linestyle='-', linewidth=2, alpha=0.7)
                    
                # Add eta change markers for all scenarios (combine unique positions to avoid duplication)
                all_eta_positions = set()
                all_eta_markers = {}
                for eta in valid_etas:
                    for pos, eta_val in zip(eta_change_markers[eta]["positions"], eta_change_markers[eta]["values"]):
                        all_eta_positions.add(pos)
                        all_eta_markers[pos] = eta_val
                
                for pos in sorted(all_eta_positions):
                    ax1.axvline(x=pos, color='red', linestyle='--', alpha=0.3)
                    ax1.text(pos, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.1, 'Distribution Change', rotation=90, verticalalignment='bottom', horizontalalignment='center', color='red', fontsize=14)
                
                ax1.set_xlabel('Window Index')
                ax1.set_ylabel('Loss')
                ax1.set_title('EKF vs SubspaceNet Loss vs Window Index\nRMSPE = √(1/N * Σ(θ_pred - θ_true)²)\nSolid=EKF, Dashed=SubspaceNet')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim([None, 0.14])  # Match the original plot limit
                
                # Plot 2: EKF Improvement
                ax2 = axes[0, 1]
                for i, eta in enumerate(valid_etas):
                    color = colors[i]
                    static_x = _static_x(metrics_by_eta[eta])
                    ax2.plot(static_x, metrics_by_eta[eta]["ekf_improvement"], 
                            color=color, linestyle='-', marker='s', 
                            label=f'η={eta:.3f}', linewidth=2, markersize=4)
                    
                    # Add online learning improvement if available
                    if metrics_by_eta[eta]["has_online_data"]:
                        online_x = np.array(metrics_by_eta[eta]["online_window_indices"])
                        online_improvement = [pre - post for pre, post in zip(metrics_by_eta[eta]["online_pre_ekf_losses"], metrics_by_eta[eta]["online_window_losses"])]
                        ax2.plot(online_x, online_improvement, 
                                color=color, linestyle='-', marker='d', 
                                label=f'Online η={eta:.3f}', linewidth=2, markersize=6)
                    
                    # Add training improvement if available
                    if metrics_by_eta[eta]["has_training_data"]:
                        training_x = np.array(metrics_by_eta[eta]["training_window_indices"])
                        training_improvement = [pre - post for pre, post in zip(metrics_by_eta[eta]["training_pre_ekf_losses"], metrics_by_eta[eta]["training_window_losses"])]
                        ax2.plot(training_x, training_improvement, 
                                color=color, linestyle='-', marker='*', 
                                label=f'Training η={eta:.3f}', linewidth=2, markersize=6, alpha=0.7)
                        
                        # Connect training to online if both are available
                        if metrics_by_eta[eta]["has_online_data"]:
                            last_training_x = training_x[-1]
                            first_online_x = online_x[0]
                            last_training_improvement = training_improvement[-1]
                            first_online_improvement = online_improvement[0]
                            
                            # Draw connecting line
                            ax2.plot([last_training_x, first_online_x], [last_training_improvement, first_online_improvement], 
                                    color=color, linestyle='-', linewidth=2, alpha=0.7)
                    
                # Add eta change markers (same positions as ax1)
                for pos in sorted(all_eta_positions):
                    if pos >= 1:
                        ax2.axvline(x=pos, color='red', linestyle='--', alpha=0.3)
                        ax2.text(pos, ax2.get_ylim()[0] + (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.1, 'Distribution Change', rotation=90, verticalalignment='bottom', horizontalalignment='center', color='red', fontsize=14)
                
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
                if has_any_online:
                    title += '\n(Static + Online Models)'
                ax2.set_title(title)
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Plot 3: Average Innovation Magnitude
                ax3 = axes[0, 2]
                for i, eta in enumerate(valid_etas):
                    if metrics_by_eta[eta]["avg_innovations"]:
                        color = colors[i]
                        static_x = _static_x(metrics_by_eta[eta])
                        innov = metrics_by_eta[eta]["avg_innovations"]
                        plot_x = static_x[: len(innov)] if len(static_x) != len(innov) else static_x
                        ax3.plot(plot_x, innov, 
                                color=color, linestyle='-', marker='d', 
                                label=f'η={eta:.3f}', linewidth=2, markersize=4)
                    
                    # Add online learning innovations if available
                    if metrics_by_eta[eta]["has_online_data"] and metrics_by_eta[eta]["online_avg_innovations"]:
                        online_x = np.array(metrics_by_eta[eta]["online_window_indices"])
                        ax3.plot(online_x, metrics_by_eta[eta]["online_avg_innovations"], 
                                color=color, linestyle='-', marker='d', 
                                label=f'Online η={eta:.3f}', linewidth=2, markersize=6)
                    
                    # Add training innovations if available
                    if metrics_by_eta[eta]["has_training_data"] and metrics_by_eta[eta]["training_avg_innovations"]:
                        training_x = np.array(metrics_by_eta[eta]["training_window_indices"])
                        ax3.plot(training_x, metrics_by_eta[eta]["training_avg_innovations"], 
                                color=color, linestyle='-', marker='*', 
                                label=f'Training η={eta:.3f}', linewidth=2, markersize=6, alpha=0.7)
                        
                        # Connect training to online if both are available
                        if metrics_by_eta[eta]["has_online_data"] and metrics_by_eta[eta]["online_avg_innovations"]:
                            last_training_x = training_x[-1]
                            first_online_x = online_x[0]
                            last_training_innovation = metrics_by_eta[eta]["training_avg_innovations"][-1]
                            first_online_innovation = metrics_by_eta[eta]["online_avg_innovations"][0]
                            
                            # Draw connecting line
                            ax3.plot([last_training_x, first_online_x], [last_training_innovation, first_online_innovation], 
                                    color=color, linestyle='-', linewidth=2, alpha=0.7)
                
                # Add eta change markers for all scenarios (combine unique positions to avoid duplication)
                all_eta_positions = set()
                all_eta_markers = {}
                for eta in valid_etas:
                    for pos, eta_val in zip(eta_change_markers[eta]["positions"], eta_change_markers[eta]["values"]):
                        all_eta_positions.add(pos)
                        all_eta_markers[pos] = eta_val
                
                for pos in sorted(all_eta_positions):
                    ax3.axvline(x=pos, color='red', linestyle='--', alpha=0.3)
                    ax3.text(pos, ax3.get_ylim()[0] + (ax3.get_ylim()[1] - ax3.get_ylim()[0]) * 0.1, 'Distribution Change', rotation=90, verticalalignment='bottom', horizontalalignment='center', color='red', fontsize=14)
                
                ax3.set_xlabel('Window Index')
                ax3.set_ylabel('Average Innovation')
                ax3.set_title('|EKF Innovation| vs Window Index\nInnovation = z_k - H x̂_k|k-1')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
                # Plot 4: Average |K*y|
                ax4 = axes[1, 0]
                for i, eta in enumerate(valid_etas):
                    if metrics_by_eta[eta]["avg_k_times_y"]:
                        color = colors[i]
                        static_x = _static_x(metrics_by_eta[eta])
                        kty = metrics_by_eta[eta]["avg_k_times_y"]
                        plot_x = static_x[: len(kty)] if len(static_x) != len(kty) else static_x
                        ax4.plot(plot_x, np.abs(kty), 
                                color=color, linestyle='-', marker='v', 
                                label=f'η={eta:.3f}', linewidth=2, markersize=4)
                    
                    # Add online learning K*y if available
                    if metrics_by_eta[eta]["has_online_data"] and metrics_by_eta[eta]["online_avg_k_times_y"]:
                        online_x = np.array(metrics_by_eta[eta]["online_window_indices"])
                        ax4.plot(online_x, np.abs(metrics_by_eta[eta]["online_avg_k_times_y"]), 
                                color=color, linestyle='-', marker='v', 
                                label=f'Online η={eta:.3f}', linewidth=2, markersize=6)
                    
                    # Add training K*y if available
                    if metrics_by_eta[eta]["has_training_data"] and metrics_by_eta[eta]["training_avg_k_times_y"]:
                        training_x = np.array(metrics_by_eta[eta]["training_window_indices"])
                        ax4.plot(training_x, np.abs(metrics_by_eta[eta]["training_avg_k_times_y"]), 
                                color=color, linestyle='-', marker='*', 
                                label=f'Training η={eta:.3f}', linewidth=2, markersize=6, alpha=0.7)
                        
                        # Connect training to online if both are available
                        if metrics_by_eta[eta]["has_online_data"] and metrics_by_eta[eta]["online_avg_k_times_y"]:
                            last_training_x = training_x[-1]
                            first_online_x = online_x[0]
                            last_training_k_times_y = metrics_by_eta[eta]["training_avg_k_times_y"][-1]
                            first_online_k_times_y = metrics_by_eta[eta]["online_avg_k_times_y"][0]
                            
                            # Draw connecting line
                            ax4.plot([last_training_x, first_online_x], [abs(last_training_k_times_y), abs(first_online_k_times_y)], 
                                    color=color, linestyle='-', linewidth=2, alpha=0.7)
                
                # Add eta change markers for all scenarios (combine unique positions to avoid duplication)
                all_eta_positions = set()
                all_eta_markers = {}
                for eta in valid_etas:
                    for pos, eta_val in zip(eta_change_markers[eta]["positions"], eta_change_markers[eta]["values"]):
                        all_eta_positions.add(pos)
                        all_eta_markers[pos] = eta_val
                
                for pos in sorted(all_eta_positions):
                    ax4.axvline(x=pos, color='red', linestyle='--', alpha=0.3)
                    ax4.text(pos, ax4.get_ylim()[0] + (ax4.get_ylim()[1] - ax4.get_ylim()[0]) * 0.1, 'Distribution Change', rotation=90, verticalalignment='bottom', horizontalalignment='center', color='red', fontsize=14)
                
                ax4.set_xlabel('Window Index')
                ax4.set_ylabel('|Average K*Innovation|')
                ax4.set_title('Average |Kalman Gain × Innovation| vs Window Index\n|K_k × ν_k| = |K_k × (z_k - H x̂_k|k-1)|')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                
                # Plot 5: Average y*S^-1*y
                ax5 = axes[1, 1]
                for i, eta in enumerate(valid_etas):
                    if metrics_by_eta[eta]["avg_y_s_inv_y"]:
                        color = colors[i]
                        static_x = _static_x(metrics_by_eta[eta])
                        ysy = metrics_by_eta[eta]["avg_y_s_inv_y"]
                        plot_x = static_x[: len(ysy)] if len(static_x) != len(ysy) else static_x
                        ax5.plot(plot_x, ysy, 
                                color=color, linestyle='-', marker='^', 
                                label=f'η={eta:.3f}', linewidth=2, markersize=4)
                    
                    # Add online learning y*S^-1*y if available
                    if metrics_by_eta[eta]["has_online_data"] and metrics_by_eta[eta]["online_avg_y_s_inv_y"]:
                        online_x = np.array(metrics_by_eta[eta]["online_window_indices"])
                        ax5.plot(online_x, metrics_by_eta[eta]["online_avg_y_s_inv_y"], 
                                color=color, linestyle='-', marker='^', 
                                label=f'Online η={eta:.3f}', linewidth=2, markersize=6)
                    
                    # Add training y*S^-1*y if available
                    if metrics_by_eta[eta]["has_training_data"] and metrics_by_eta[eta]["training_avg_y_s_inv_y"]:
                        training_x = np.array(metrics_by_eta[eta]["training_window_indices"])
                        ax5.plot(training_x, metrics_by_eta[eta]["training_avg_y_s_inv_y"], 
                                color=color, linestyle='-', marker='*', 
                                label=f'Training η={eta:.3f}', linewidth=2, markersize=6, alpha=0.7)
                        
                        # Connect training to online if both are available
                        if metrics_by_eta[eta]["has_online_data"] and metrics_by_eta[eta]["online_avg_y_s_inv_y"]:
                            last_training_x = training_x[-1]
                            first_online_x = online_x[0]
                            last_training_y_s_inv_y = metrics_by_eta[eta]["training_avg_y_s_inv_y"][-1]
                            first_online_y_s_inv_y = metrics_by_eta[eta]["online_avg_y_s_inv_y"][0]
                            
                            # Draw connecting line
                            ax5.plot([last_training_x, first_online_x], [last_training_y_s_inv_y, first_online_y_s_inv_y], 
                                    color=color, linestyle='-', linewidth=2, alpha=0.7)
                
                # Add eta change markers for all scenarios (combine unique positions to avoid duplication)
                all_eta_positions = set()
                all_eta_markers = {}
                for eta in valid_etas:
                    for pos, eta_val in zip(eta_change_markers[eta]["positions"], eta_change_markers[eta]["values"]):
                        all_eta_positions.add(pos)
                        all_eta_markers[pos] = eta_val
                
                for pos in sorted(all_eta_positions):
                    ax5.axvline(x=pos, color='red', linestyle='--', alpha=0.3)
                    ax5.text(pos, ax5.get_ylim()[0] + (ax5.get_ylim()[1] - ax5.get_ylim()[0]) * 0.1, 'Distribution Change', rotation=90, verticalalignment='bottom', horizontalalignment='center', color='red', fontsize=14)
                
                ax5.set_xlabel('Window Index')
                ax5.set_ylabel('Average y*(S^-1)*y')
                ax5.set_title('Average Innovation Covariance Metric vs Window Index\ny*(S^-1)*y = ν^T S^-1 ν')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
                
                # Plot 6: Summary statistics
                ax6 = axes[1, 2]
                # Calculate mean values for summary
                summary_metrics = []
                summary_labels = []
                summary_colors = []
                for i, eta in enumerate(valid_etas):
                    mean_ekf_loss = np.mean(metrics_by_eta[eta]["window_losses"])
                    mean_improvement = np.mean(metrics_by_eta[eta]["ekf_improvement"])
                    mean_subspacenet_loss = mean_ekf_loss + mean_improvement
                    summary_metrics.append([mean_ekf_loss, mean_subspacenet_loss, mean_improvement])
                    summary_labels.append(f'η={eta:.3f}')
                    summary_colors.append(colors[i])
                
                summary_metrics = np.array(summary_metrics)
                x_pos = np.arange(len(summary_labels))
                
                # Create bar plot with color-coded eta values
                width = 0.25
                for i, (eta, color) in enumerate(zip(valid_etas, summary_colors)):
                    ax6.bar(x_pos[i] - width, summary_metrics[i, 0], width, 
                           label=f'EKF η={eta:.3f}' if i == 0 else '', 
                           color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
                    ax6.bar(x_pos[i], summary_metrics[i, 1], width, 
                           label=f'SubspaceNet η={eta:.3f}' if i == 0 else '', 
                           color=color, alpha=0.4, edgecolor='black', linewidth=0.5)
                    
                ax6_twin = ax6.twinx()
                ax6_twin.bar(x_pos + width, summary_metrics[:, 2], width, 
                           label='Mean EKF Improvement', alpha=0.9, color='orange', 
                           edgecolor='black', linewidth=0.5)
                
                ax6.set_xlabel('Eta Values')
                ax6.set_ylabel('Mean Loss')
                ax6_twin.set_ylabel('Mean EKF Improvement', color='orange')
                ax6.set_title('Summary: EKF vs SubspaceNet Loss by Eta\nDark=EKF, Light=SubspaceNet')
                ax6.set_xticks(x_pos)
                ax6.set_xticklabels(summary_labels)
                ax6.grid(True, alpha=0.3)
                
                # Add combined legend for the summary plot
                lines1, labels1 = ax6.get_legend_handles_labels()
                lines2, labels2 = ax6_twin.get_legend_handles_labels()
                ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                
                # Add overall title
                fig.suptitle(f'Eta Comparison: proc_noise={proc_noise:.3f}, kf_proc_noise={kf_proc_noise:.3f}, kf_meas_noise={kf_meas_noise:.3f}', 
                            fontsize=16, y=0.98)
                
                plt.tight_layout()
                
                # Save the plot
                plot_filename = f"eta_comparison_pn{proc_noise:.3f}_kfpn{kf_proc_noise:.3f}_kfmn{kf_meas_noise:.3f}_{timestamp}.png"
                plot_path = plot_dir / plot_filename
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                saved_plots.append(plot_path)
                logger.info(f"Saved eta comparison plot {combination_count}/{total_combinations}: {plot_path.name}")
    
    logger.info(f"Completed eta comparison plotting: {len(saved_plots)} plots saved to {plot_dir}")
    return saved_plots

def _unwrap_result_for_lr_sweep(result: dict) -> dict:
    """When LR sweep is enabled, result has lr_sweep_results; extract the nested online-learning result."""
    if not isinstance(result, dict) or 'lr_sweep_results' not in result:
        return result
    lr_results = result['lr_sweep_results']
    # Prefer adaptive result; otherwise use first available
    if 'adaptive' in lr_results and isinstance(lr_results['adaptive'], dict) and lr_results['adaptive'].get('result'):
        return lr_results['adaptive']['result']
    for k, v in lr_results.items():
        if isinstance(v, dict) and v.get('result'):
            return v['result']
    return result


def plot_scenario_results(scenario_results: dict, output_dir: Path, scenario_type: str = 'snr') -> None:
    """
    Plot scenario results comparing average dB loss of online learning vs pretrained models.
    
    Args:
        scenario_results: Dictionary mapping scenario values (SNR or eta) to their results
        output_dir: Output directory for saving the plot
        scenario_type: Type of scenario ('snr' or 'eta'), defaults to 'snr'
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    
    logger = logging.getLogger(__name__)
    logger.info(f"Creating scenario results plot for {scenario_type.upper()} scenario...")
    
    # Extract scenario values and sort them (preserve original key type for lookup)
    # Keys can be int, float, or string depending on how they were stored
    scenario_keys = [key for key in scenario_results.keys() if scenario_results[key] is not None]
    scenario_values = sorted([float(key) for key in scenario_keys])
    
    # Use appropriate labels based on scenario type
    value_label = scenario_type.upper() if scenario_type.lower() == 'snr' else 'Eta'
    value_unit = ' (dB)' if scenario_type.lower() == 'snr' else ''
    
    # Initialize lists to store average dB losses
    online_avg_db_losses = []
    pretrained_avg_db_losses = []
    supervised_avg_db_losses = []
    
    for val in scenario_values:
        # Try to find the matching key (could be float, int, or string)
        result = None
        for key in scenario_keys:
            if abs(float(key) - val) < 1e-10:  # Handle floating point comparison
                result = scenario_results[key]
                break
        
        if result is None:
            logger.warning(f"{value_label} {val} not found in results, skipping...")
            continue
        result = _unwrap_result_for_lr_sweep(result)
        # Extract averaged results (use the already calculated averages)
        online_avg_db = None
        pretrained_avg_db = None
        supervised_avg_db = None
        
        # First, try to get the averaged results if available
        if 'averaged_results' in result:
            averaged_data = result['averaged_results']
            
            # Get averaged online model dB losses (use last 10 windows for consistency)
            if 'averaged_online_trajectory' in averaged_data:
                online_metrics = averaged_data['averaged_online_trajectory']
                online_db_losses = online_metrics.get('reference_metric_losses_db', [])
                if online_db_losses:
                    # Use last 10 windows or all if fewer than 10
                    num_windows_to_use = min(10, len(online_db_losses))
                    online_avg_db = np.mean(online_db_losses[-num_windows_to_use:])
                    logger.info(f"{value_label} {val}: Online model - averaged from last {num_windows_to_use} windows, avg dB loss = {online_avg_db:.2f}")
            
            # Get averaged pretrained model dB losses (use last 10 windows for consistency)
            if 'averaged_pretrained_trajectory' in averaged_data:
                pretrained_metrics = averaged_data['averaged_pretrained_trajectory']
                pretrained_db_losses = pretrained_metrics.get('reference_metric_losses_db', [])
                if pretrained_db_losses:
                    # Use last 10 windows or all if fewer than 10
                    num_windows_to_use = min(10, len(pretrained_db_losses))
                    pretrained_avg_db = np.mean(pretrained_db_losses[-num_windows_to_use:])
                    logger.info(f"{value_label} {val}: Pretrained model - averaged from last {num_windows_to_use} windows, avg dB loss = {pretrained_avg_db:.2f}")
            
            # Get averaged supervised model dB losses (use last 10 windows for consistency)
            if 'averaged_supervised_trajectory' in averaged_data:
                supervised_metrics = averaged_data['averaged_supervised_trajectory']
                supervised_db_losses = supervised_metrics.get('reference_metric_losses_db', [])
                if supervised_db_losses:
                    # Use last 10 windows or all if fewer than 10
                    num_windows_to_use = min(10, len(supervised_db_losses))
                    supervised_avg_db = np.mean(supervised_db_losses[-num_windows_to_use:])
                    logger.info(f"{value_label} {val}: Supervised trained model - averaged from last {num_windows_to_use} windows, avg dB loss = {supervised_avg_db:.2f}")
        
        # Fallback to individual trajectory results if averaged results not available
        if online_avg_db is None or pretrained_avg_db is None or supervised_avg_db is None:
            logger.warning(f"{value_label} {val}: Some averaged results not available, falling back to individual trajectory extraction")
            
            if 'online_learning_results' in result:
                online_learning_data = result['online_learning_results']
                
                # Get online model trajectory results
                if online_avg_db is None and 'online_trajectory_results' in online_learning_data:
                    online_results = online_learning_data['online_trajectory_results']
                    if online_results and isinstance(online_results, list) and len(online_results) > 0:
                        trajectory_results = online_results[0]  # Get the first trajectory result
                        total_windows = len(trajectory_results.window_results)
                        num_windows_to_use = min(10, total_windows)
                        start_window = max(0, total_windows - num_windows_to_use)
                        post_learning_db_losses = []
                        for window_result in trajectory_results.window_results[start_window:]:
                            if hasattr(window_result, 'loss_metrics') and hasattr(window_result.loss_metrics, 'reference_metric_loss_db'):
                                post_learning_db_losses.append(window_result.loss_metrics.reference_metric_loss_db)
                        if post_learning_db_losses:
                            online_avg_db = np.mean(post_learning_db_losses)
                            logger.info(f"{value_label} {val}: Online model (fallback) - last {len(post_learning_db_losses)} windows, avg dB loss = {online_avg_db:.2f}")
                
                # Get pretrained model trajectory results  
                if pretrained_avg_db is None and 'pretrained_trajectory_results' in online_learning_data:
                    pretrained_results = online_learning_data['pretrained_trajectory_results']
                    if pretrained_results and isinstance(pretrained_results, list) and len(pretrained_results) > 0:
                        pretrained_trajectory_results = pretrained_results[0]  # Get the first trajectory result
                        total_windows = len(pretrained_trajectory_results.window_results)
                        num_windows_to_use = min(10, total_windows)
                        start_window = max(0, total_windows - num_windows_to_use)
                        post_learning_db_losses = []
                        for window_result in pretrained_trajectory_results.window_results[start_window:]:
                            if hasattr(window_result, 'loss_metrics') and hasattr(window_result.loss_metrics, 'reference_metric_loss_db'):
                                post_learning_db_losses.append(window_result.loss_metrics.reference_metric_loss_db)
                        if post_learning_db_losses:
                            pretrained_avg_db = np.mean(post_learning_db_losses)
                            logger.info(f"{value_label} {val}: Pretrained model (fallback) - last {len(post_learning_db_losses)} windows, avg dB loss = {pretrained_avg_db:.2f}")
                
                # Get supervised model trajectory results (fallback)
                if supervised_avg_db is None and 'supervised_model_trajectory_results' in online_learning_data:
                    supervised_results = online_learning_data['supervised_model_trajectory_results']
                    if supervised_results and isinstance(supervised_results, list) and len(supervised_results) > 0:
                        supervised_trajectory_results = supervised_results[0]  # Get the first trajectory result
                        total_windows = len(supervised_trajectory_results.window_results)
                        num_windows_to_use = min(10, total_windows)
                        start_window = max(0, total_windows - num_windows_to_use)
                        post_learning_db_losses = []
                        for window_result in supervised_trajectory_results.window_results[start_window:]:
                            if hasattr(window_result, 'loss_metrics') and hasattr(window_result.loss_metrics, 'reference_metric_loss_db'):
                                post_learning_db_losses.append(window_result.loss_metrics.reference_metric_loss_db)
                        if post_learning_db_losses:
                            supervised_avg_db = np.mean(post_learning_db_losses)
                            logger.info(f"{value_label} {val}: Supervised trained model (fallback) - last {len(post_learning_db_losses)} windows, avg dB loss = {supervised_avg_db:.2f}")
        
        # Store results
        online_avg_db_losses.append(online_avg_db if online_avg_db is not None else np.nan)
        pretrained_avg_db_losses.append(pretrained_avg_db if pretrained_avg_db is not None else np.nan)
        supervised_avg_db_losses.append(supervised_avg_db if supervised_avg_db is not None else np.nan)
        
        # Log with proper None handling
        parts = []
        if online_avg_db is not None:
            parts.append(f"Online avg dB loss = {online_avg_db:.2f}")
        if pretrained_avg_db is not None:
            parts.append(f"Pretrained avg dB loss = {pretrained_avg_db:.2f}")
        if supervised_avg_db is not None:
            parts.append(f"Supervised avg dB loss = {supervised_avg_db:.2f}")
        
        if parts:
            logger.info(f"{value_label} {val}: {', '.join(parts)}")
        else:
            logger.info(f"{value_label} {val}: No loss data available")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot all three models
    plt.plot(scenario_values, online_avg_db_losses, 'o-', label='Algorithm 1', linewidth=2, markersize=8)
    plt.plot(scenario_values, pretrained_avg_db_losses, 's-', label='Pretrained Model', linewidth=2, markersize=8)
    
    # Add supervised model if data is available
    if any(not np.isnan(loss) for loss in supervised_avg_db_losses):
        plt.plot(scenario_values, supervised_avg_db_losses, '^-', label='Supervised Trained Model', linewidth=2, markersize=8)
    
    # Customize the plot with scenario-appropriate labels
    plt.xlabel(f'{value_label}{value_unit}', fontsize=20)
    plt.ylabel('Average RMSPE (Supervised) (dB)', fontsize=20)
    title_text = f'Averaged RMSPE (Supervised) vs {value_label}'
    if scenario_type.lower() == 'snr':
        title_text += ' (SNR)'
    plt.title(title_text, fontsize=24, fontweight='bold')
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3)
    
    # Set custom x-axis ticks based on scenario type
    min_val = min(scenario_values)
    max_val = max(scenario_values)
    
    if scenario_type.lower() == 'snr':
        # For SNR: use 5dB spacing
        start_tick = int(min_val // 5) * 5  # Round down to nearest 5
        end_tick = int(max_val // 5) * 5 + 5  # Round up to nearest 5
        x_ticks = list(range(start_tick, end_tick + 1, 5))
        plt.xticks(x_ticks, fontsize=18)
        # Set axis limits with exact SNR range (0-10) if applicable
        if min_val >= 0 and max_val <= 10:
            plt.xlim(0, 10)
    else:
        # For eta: use automatic ticks
        plt.xticks(fontsize=18)
    
    # Set custom y-axis ticks with 5dB spacing
    all_losses = online_avg_db_losses + pretrained_avg_db_losses
    if any(not np.isnan(loss) for loss in supervised_avg_db_losses):
        all_losses += supervised_avg_db_losses
    
    # Filter out NaN values for min/max calculation
    valid_losses = [loss for loss in all_losses if not np.isnan(loss)]
    min_loss = min(valid_losses) if valid_losses else -30
    max_loss = max(valid_losses) if valid_losses else 0
    
    # Create ticks with 5dB spacing for y-axis, starting from the nearest 5dB value below min_loss
    start_y_tick = int(min_loss // 5) * 5  # Round down to nearest 5
    end_y_tick = int(max_loss // 5) * 5 + 5  # Round up to nearest 5
    
    # Generate y-axis ticks with 5dB spacing
    y_ticks = list(range(start_y_tick, end_y_tick + 1, 5))
    plt.yticks(y_ticks, fontsize=18)
    
    # Set y-axis limits
    if scenario_type.lower() == 'snr' and min_val >= 0 and max_val <= 10:
        plt.ylim(min_loss - 2.5, max_loss + 2.5)
    else:
        plt.ylim(min_loss - 2.5, max_loss + 2.5)
    
    # Add some styling
    plt.tight_layout()
    
    # Save the plot
    plot_path = output_dir / 'scenario_results_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved scenario results plot to {plot_path}")


def plot_eta_scenario_comparison(scenario_results: dict, output_dir: Path) -> None:
    """
    Plot eta scenario comparison showing drift detection metrics.
    
    Creates 3 subplots:
    1. Change detection window vs eta
    2. GLRT z-score at detection vs eta  
    3. Learning rate at detection vs eta
    
    Args:
        scenario_results: Dictionary mapping eta values (as strings) to their results
        output_dir: Output directory for saving the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    
    logger = logging.getLogger(__name__)
    logger.info("Creating eta scenario comparison plot...")
    
    # Extract eta values and sort them (preserve original key type for lookup)
    # Keys can be int, float, or string depending on how they were stored
    eta_keys = [key for key in scenario_results.keys() if scenario_results[key] is not None]
    eta_values = sorted([float(key) for key in eta_keys])
    
    if not eta_values:
        logger.warning("No valid eta values found in scenario results")
        return
    
    # Initialize lists to store metrics
    detection_windows = []
    detection_window_stds = []
    z_scores = []
    z_score_stds = []
    learning_rates = []
    learning_rate_stds = []
    
    for eta in eta_values:
        # Try to find the matching key (could be float, int, or string)
        result = None
        for key in eta_keys:
            if abs(float(key) - eta) < 1e-10:  # Handle floating point comparison
                result = scenario_results[key]
                break
        
        if result is None:
            logger.warning(f"Eta {eta} not found in results, skipping...")
            detection_windows.append(np.nan)
            detection_window_stds.append(np.nan)
            z_scores.append(np.nan)
            z_score_stds.append(np.nan)
            learning_rates.append(np.nan)
            learning_rate_stds.append(np.nan)
            continue
        result = _unwrap_result_for_lr_sweep(result)
        # Extract GLRT results
        detection_window = None
        detection_window_std = None
        z_score = None
        z_score_std = None
        learning_rate = None
        learning_rate_std = None
        
        # Try to get from glrt_results (top level)
        if 'glrt_results' in result and 'adaptation_loss' in result['glrt_results']:
            main_glrt = result['glrt_results']['adaptation_loss']
            detection_window = main_glrt.get('avg_changepoint_window')
            detection_window_std = main_glrt.get('std_changepoint_window')
            z_score = main_glrt.get('avg_z_score')
            z_score_std = main_glrt.get('std_z_score')
            learning_rate = main_glrt.get('avg_learning_rate')
            learning_rate_std = main_glrt.get('std_learning_rate')
        
        # Fallback to averaged_results -> glrt_results
        if detection_window is None and 'averaged_results' in result:
            averaged_data = result['averaged_results']
            if 'glrt_results' in averaged_data and 'adaptation_loss' in averaged_data['glrt_results']:
                main_glrt = averaged_data['glrt_results']['adaptation_loss']
                detection_window = main_glrt.get('avg_changepoint_window')
                detection_window_std = main_glrt.get('std_changepoint_window')
                z_score = main_glrt.get('avg_z_score')
                z_score_std = main_glrt.get('std_z_score')
                learning_rate = main_glrt.get('avg_learning_rate')
                learning_rate_std = main_glrt.get('std_learning_rate')
        
        # Store results (use None as np.nan for plotting)
        detection_windows.append(detection_window if detection_window is not None else np.nan)
        detection_window_stds.append(detection_window_std if detection_window_std is not None else 0.0)
        z_scores.append(z_score if z_score is not None else np.nan)
        z_score_stds.append(z_score_std if z_score_std is not None else 0.0)
        learning_rates.append(learning_rate if learning_rate is not None else np.nan)
        learning_rate_stds.append(learning_rate_std if learning_rate_std is not None else 0.0)
        
        logger.info(f"Eta {eta}: Detection window = {detection_window:.2f} ± {detection_window_std:.2f}, "
                   f"Z-score = {z_score:.4f} ± {z_score_std:.4f}, "
                   f"LR = {learning_rate:.6f} ± {learning_rate_std:.6f}" 
                   if detection_window is not None and z_score is not None and learning_rate is not None 
                   else f"Eta {eta}: Incomplete GLRT data")
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Subplot 1: Change detection window vs eta
    ax1 = axes[0]
    valid_mask = ~np.isnan(detection_windows)
    if np.any(valid_mask):
        ax1.errorbar(np.array(eta_values)[valid_mask], np.array(detection_windows)[valid_mask],
                    yerr=np.array(detection_window_stds)[valid_mask], 
                    fmt='o-', linewidth=2, markersize=8, capsize=5, capthick=2, label='Detection Window')
    ax1.set_xlabel('Eta', fontsize=14)
    ax1.set_ylabel('Change Detection Window', fontsize=14)
    ax1.set_title('Drift Detection Window vs Eta', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
    # Subplot 2: GLRT z-score vs eta
    ax2 = axes[1]
    valid_mask = ~np.isnan(z_scores)
    if np.any(valid_mask):
        ax2.errorbar(np.array(eta_values)[valid_mask], np.array(z_scores)[valid_mask],
                    yerr=np.array(z_score_stds)[valid_mask],
                    fmt='s-', linewidth=2, markersize=8, capsize=5, capthick=2, 
                    color='green', label='GLRT Z-Score')
    ax2.set_xlabel('Eta', fontsize=14)
    ax2.set_ylabel('GLRT Z-Score at Detection', fontsize=14)
    ax2.set_title('GLRT Z-Score vs Eta', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    
    # Subplot 3: Learning rate vs eta
    ax3 = axes[2]
    valid_mask = ~np.isnan(learning_rates)
    if np.any(valid_mask):
        ax3.errorbar(np.array(eta_values)[valid_mask], np.array(learning_rates)[valid_mask],
                    yerr=np.array(learning_rate_stds)[valid_mask],
                    fmt='^-', linewidth=2, markersize=8, capsize=5, capthick=2,
                    color='orange', label='Learning Rate')
    ax3.set_xlabel('Eta', fontsize=14)
    ax3.set_ylabel('Learning Rate at Detection', fontsize=14)
    ax3.set_title('Learning Rate vs Eta', fontsize=16, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=12)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = output_dir / "eta_scenario_drift_detection_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved eta scenario comparison plot to {plot_path}")
    
    logger.info(f"Scenario results plot saved to: {plot_path}")
    
    # GLRT drift detection violin plots
    _plot_glrt_scenario_results(scenario_results, output_dir)


def _plot_glrt_scenario_results(scenario_results: dict, output_dir: Path) -> None:
    """
    Plot GLRT drift detection results across scenarios using violin plots.
    
    Creates two plots:
    1. Changepoint window detection (avg ± std) as a function of scenario
    2. Likelihood (avg ± std) at changepoint window as a function of scenario
    
    Args:
        scenario_results: Dictionary mapping scenario values (e.g., SNR) to their results
        output_dir: Output directory for saving the plots
    """
    try:
        import pandas as pd
        import seaborn as sns
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.warning("pandas or seaborn not available, skipping GLRT violin plots")
        return
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    logger = logging.getLogger(__name__)
    
    # Extract scenario values and sort them
    scenario_values = sorted([float(val) for val in scenario_results.keys() if scenario_results[val] is not None])
    
    if not scenario_values:
        logger.warning("No valid scenario results found for GLRT plotting")
        return
    
    # Collect GLRT data for each scenario
    changepoint_data_by_scenario = {}  # {scenario_value: [list of changepoint windows]}
    likelihood_data_by_scenario = {}   # {scenario_value: [list of likelihoods]}
    
    for scenario_val in scenario_values:
        result = None
        for key in scenario_results:
            if scenario_results[key] is not None and abs(float(key) - scenario_val) < 1e-10:
                result = scenario_results[key]
                break
        if not result:
            continue
        result = _unwrap_result_for_lr_sweep(result)
        # Try to get GLRT results from different locations
        glrt_results = None
        if "glrt_results" in result:
            glrt_results = result["glrt_results"]
        elif "averaged_results" in result and "glrt_results" in result["averaged_results"]:
            glrt_results = result["averaged_results"]["glrt_results"]
        
        if glrt_results and "ref_loss" in glrt_results:
            ref_data = glrt_results["ref_loss"]
            
            # Get individual trajectory values for violin plot
            if "individual_changepoint_windows" in ref_data:
                changepoint_data_by_scenario[scenario_val] = ref_data["individual_changepoint_windows"]
            
            if "individual_likelihoods" in ref_data:
                likelihood_data_by_scenario[scenario_val] = ref_data["individual_likelihoods"]
    
    if not changepoint_data_by_scenario and not likelihood_data_by_scenario:
        logger.warning("No GLRT data found in scenario results")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Changepoint window detection
    if changepoint_data_by_scenario:
        # Prepare data for violin plot
        scenario_labels = []
        changepoint_values = []
        
        for scenario_val in scenario_values:
            if scenario_val in changepoint_data_by_scenario:
                values = changepoint_data_by_scenario[scenario_val]
                if values:
                    scenario_labels.extend([str(scenario_val)] * len(values))
                    changepoint_values.extend(values)
        
        if changepoint_values:
            # Create violin plot
            data_for_violin = pd.DataFrame({
                'Scenario': scenario_labels,
                'Changepoint Window': changepoint_values
            })
            
            sns.violinplot(data=data_for_violin, x='Scenario', y='Changepoint Window', ax=ax1)
            ax1.set_xlabel('Scenario (SNR)', fontsize=14)
            ax1.set_ylabel('Changepoint Window', fontsize=14)
            ax1.set_title('GLRT Changepoint Window Detection Across Scenarios', fontsize=16, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Add mean and std markers
            handles_added = False
            for i, scenario_val in enumerate(scenario_values):
                if scenario_val in changepoint_data_by_scenario:
                    values = changepoint_data_by_scenario[scenario_val]
                    if values:
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        ax1.scatter(i, mean_val, color='red', marker='o', s=100, zorder=5, label='Mean ± Std' if not handles_added else '')
                        ax1.errorbar(i, mean_val, yerr=std_val, color='red', capsize=5, capthick=2, zorder=5, label='' if handles_added else '')
                        handles_added = True
            
            if handles_added:
                ax1.legend(fontsize=10, loc='best')
    
    # Plot 2: Likelihood at changepoint window
    if likelihood_data_by_scenario:
        # Prepare data for violin plot
        scenario_labels = []
        likelihood_values = []
        
        for scenario_val in scenario_values:
            if scenario_val in likelihood_data_by_scenario:
                values = likelihood_data_by_scenario[scenario_val]
                if values:
                    scenario_labels.extend([str(scenario_val)] * len(values))
                    likelihood_values.extend(values)
        
        if likelihood_values:
            # Create violin plot
            data_for_violin = pd.DataFrame({
                'Scenario': scenario_labels,
                'Likelihood (Log-GLR)': likelihood_values
            })
            
            sns.violinplot(data=data_for_violin, x='Scenario', y='Likelihood (Log-GLR)', ax=ax2)
            ax2.set_xlabel('Scenario (SNR)', fontsize=14)
            ax2.set_ylabel('Likelihood (Log-GLR)', fontsize=14)
            ax2.set_title('GLRT Likelihood at Changepoint Window (Reference Loss)', fontsize=16, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Add mean and std markers
            handles_added = False
            for i, scenario_val in enumerate(scenario_values):
                if scenario_val in likelihood_data_by_scenario:
                    values = likelihood_data_by_scenario[scenario_val]
                    if values:
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        ax2.scatter(i, mean_val, color='red', marker='o', s=100, zorder=5, label='Mean ± Std' if not handles_added else '')
                        ax2.errorbar(i, mean_val, yerr=std_val, color='red', capsize=5, capthick=2, zorder=5, label='' if handles_added else '')
                        handles_added = True
            
            if handles_added:
                ax2.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = output_dir / 'glrt_scenario_results.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"GLRT scenario results violin plots saved to: {plot_path}")


def plot_performance_improvement_table(scenario_results: dict, output_dir: Path) -> Path:
    """
    Plot a table showing performance improvement of online model vs pretrained model across SNR values.
    
    Args:
        scenario_results: Dictionary mapping SNR values to their results (same as plot_scenario_results)
        output_dir: Output directory for saving the plot
        
    Returns:
        Path to saved plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    logger = logging.getLogger(__name__)
    logger.info("Creating performance improvement table...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract and sort SNR values
    snr_values = sorted([float(snr) for snr in scenario_results.keys()])
    
    # Initialize data structures for the table
    rmspe_improvements = []
    msie_improvements = []
    supervised_rmspe_improvements = []
    supervised_msie_improvements = []
    
    for snr in snr_values:
        if snr not in scenario_results:
            logger.warning(f"SNR {snr} not found in results, using NaN...")
            rmspe_improvements.append(np.nan)
            msie_improvements.append(np.nan)
            continue
            
        result = scenario_results[snr]
        
        # Initialize improvements as NaN
        rmspe_improvement = np.nan
        msie_improvement = np.nan
        supervised_rmspe_improvement = np.nan
        supervised_msie_improvement = np.nan
        
        # Try to get averaged results first
        if 'averaged_results' in result:
            averaged_data = result['averaged_results']
            
            # Get averaged metrics for all models
            if ('averaged_pretrained_trajectory' in averaged_data and 
                'averaged_online_trajectory' in averaged_data):
                
                pretrained_metrics = averaged_data['averaged_pretrained_trajectory']
                online_metrics = averaged_data['averaged_online_trajectory']
                supervised_metrics = averaged_data.get('averaged_supervised_trajectory')
                
                # Get last 15 windows for RMSPE (main losses)
                pretrained_rmspe = pretrained_metrics.get('reference_metric_losses', [])
                online_rmspe = online_metrics.get('reference_metric_losses', [])
                
                if pretrained_rmspe and online_rmspe:
                    # Use last 15 windows or all if fewer than 15
                    num_windows = min(15, len(pretrained_rmspe), len(online_rmspe))
                    pretrained_last15 = pretrained_rmspe[-num_windows:]
                    online_last15 = online_rmspe[-num_windows:]
                    
                    # Calculate standalone values
                    pretrained_rmspe_avg = np.mean(pretrained_last15)
                    online_rmspe_avg = np.mean(online_last15)
                    
                    # Calculate L2 distance (average improvement)
                    # Positive means online model is better (lower loss)
                    improvements = [pre - onl for pre, onl in zip(pretrained_last15, online_last15)]
                    rmspe_improvement = np.mean(improvements)
                    
                    logger.info(f"SNR {snr}: RMSPE - Pretrained: {pretrained_rmspe_avg:.6f} rad ({(pretrained_rmspe_avg / np.pi) * 180:.3f}°), Algorithm 1: {online_rmspe_avg:.6f} rad ({(online_rmspe_avg / np.pi) * 180:.3f}°), Improvement: {rmspe_improvement:.6f} rad ({(rmspe_improvement / np.pi) * 180:.3f}°)")
                
                # Calculate supervised model RMSPE improvement if available
                if supervised_metrics is not None:
                    supervised_rmspe = supervised_metrics.get('reference_metric_losses', [])
                    if supervised_rmspe:
                        num_windows = min(15, len(pretrained_rmspe), len(supervised_rmspe))
                        supervised_last15 = supervised_rmspe[-num_windows:]
                        
                        # Calculate supervised model improvement vs pretrained
                        supervised_improvements = [pre - sup for pre, sup in zip(pretrained_last15, supervised_last15)]
                        supervised_rmspe_improvement = np.mean(supervised_improvements)
                        
                        supervised_rmspe_avg = np.mean(supervised_last15)
                        logger.info(f"SNR {snr}: RMSPE Supervised - Supervised Trained: {supervised_rmspe_avg:.6f} rad ({(supervised_rmspe_avg / np.pi) * 180:.3f}°), vs Pretrained Improvement: {supervised_rmspe_improvement:.6f} rad ({(supervised_rmspe_improvement / np.pi) * 180:.3f}°)")
                
                # Get last 15 windows for MSIE (training reference losses)
                pretrained_msie = pretrained_metrics.get('adaptation_losses', [])
                online_msie = online_metrics.get('adaptation_losses', [])
                
                if pretrained_msie and online_msie:
                    # Use last 15 windows or all if fewer than 15
                    num_windows = min(15, len(pretrained_msie), len(online_msie))
                    pretrained_last15 = pretrained_msie[-num_windows:]
                    online_last15 = online_msie[-num_windows:]
                    
                    # Calculate standalone values
                    pretrained_msie_avg = np.mean(pretrained_last15)
                    online_msie_avg = np.mean(online_last15)
                    
                    # Calculate L2 distance (average improvement)
                    improvements = [pre - onl for pre, onl in zip(pretrained_last15, online_last15)]
                    msie_improvement = np.mean(improvements)
                    
                    logger.info(f"SNR {snr}: MSIE - Pretrained: {pretrained_msie_avg:.6f} rad ({(pretrained_msie_avg / np.pi) * 180:.3f}°), Algorithm 1: {online_msie_avg:.6f} rad ({(online_msie_avg / np.pi) * 180:.3f}°), Improvement: {msie_improvement:.6f} rad ({(msie_improvement / np.pi) * 180:.3f}°)")
                
                # Calculate supervised model MSIE improvement if available
                if supervised_metrics is not None:
                    supervised_msie = supervised_metrics.get('adaptation_losses', [])
                    if supervised_msie:
                        num_windows = min(15, len(pretrained_msie), len(supervised_msie))
                        supervised_last15 = supervised_msie[-num_windows:]
                        
                        # Calculate supervised model improvement vs pretrained
                        supervised_improvements = [pre - sup for pre, sup in zip(pretrained_last15, supervised_last15)]
                        supervised_msie_improvement = np.mean(supervised_improvements)
                        
                        supervised_msie_avg = np.mean(supervised_last15)
                        logger.info(f"SNR {snr}: MSIE Supervised - Supervised Trained: {supervised_msie_avg:.6f} rad ({(supervised_msie_avg / np.pi) * 180:.3f}°), vs Pretrained Improvement: {supervised_msie_improvement:.6f} rad ({(supervised_msie_improvement / np.pi) * 180:.3f}°)")
        
        # Store the improvements
        rmspe_improvements.append(rmspe_improvement)
        msie_improvements.append(msie_improvement)
        supervised_rmspe_improvements.append(supervised_rmspe_improvement)
        supervised_msie_improvements.append(supervised_msie_improvement)
    
    # Create the table plot with minimal margins
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.axis('off')  # Turn off axes for table
    
    # Remove all margins and set tight spacing
    fig.subplots_adjust(left=0, right=1, top=0.85, bottom=0.02)
    
    # Prepare table data (SNR as rows, loss types as columns)
    snr_labels = [f'SNR {int(snr)}' for snr in snr_values]
    table_headers = ['SNR (dB)', 'RMSPE (Alg 1)', 'MSIE (Alg 1)', 'RMSPE (Supervised)']
    
    # Format the improvement values with degree conversion
    table_data = []
    for i, snr in enumerate(snr_values):
        # Convert to degrees and add degree symbol
        rmspe_val = f'{(rmspe_improvements[i] / np.pi) * 180:.3f}°' if not np.isnan(rmspe_improvements[i]) else 'N/A'
        msie_val = f'{(msie_improvements[i] / np.pi) * 180:.3f}°' if not np.isnan(msie_improvements[i]) else 'N/A'
        supervised_rmspe_val = f'{(supervised_rmspe_improvements[i] / np.pi) * 180:.3f}°' if not np.isnan(supervised_rmspe_improvements[i]) else 'N/A'
        table_data.append([snr_labels[i], rmspe_val, msie_val, supervised_rmspe_val])
    
    # Create the table positioned to avoid overlap with titles
    table = ax.table(cellText=table_data, colLabels=table_headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.2, 0.267, 0.267, 0.267])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(16)  # Keep reasonable base font size
    table.scale(1.2, 2.5)  # Make table bigger
    
    # Style header row
    for i in range(len(table_headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_fontsize(16)  # Keep original header font size
    
    # Style data rows with color coding based on improvement
    for i in range(len(table_data)):
        for j in range(len(table_headers)):
            if j == 0:  # First column (SNR labels)
                table[(i+1, j)].set_facecolor('#E7E6E6')  # Light gray for SNR labels
                table[(i+1, j)].set_text_props(weight='bold')
                table[(i+1, j)].set_fontsize(16)  # Keep original SNR label font size
            else:  # Data columns (RMSPE and MSIE for both algorithms)
                # Color code based on improvement value
                if j == 1:  # Algorithm 1 RMSPE
                    improvement = rmspe_improvements[i]
                elif j == 2:  # Algorithm 1 MSIE
                    improvement = msie_improvements[i]
                elif j == 3:  # Supervised RMSPE
                    improvement = supervised_rmspe_improvements[i]
                else:
                    improvement = np.nan
                
                if not np.isnan(improvement):
                    if improvement > 0:  # Algorithm is better than pretrained
                        table[(i+1, j)].set_facecolor('#C6EFCE')  # Light green
                    elif improvement < 0:  # Pretrained model is better
                        table[(i+1, j)].set_facecolor('#FFC7CE')  # Light red
                    else:  # No difference
                        table[(i+1, j)].set_facecolor('#FFFFFF')  # White
                else:
                    table[(i+1, j)].set_facecolor('#F2F2F2')  # Light gray for N/A
                
                table[(i+1, j)].set_fontsize(16)  # Larger but reasonable data cell font size for degree values
    
    # Add title and subtitle with proper spacing to avoid overlap
    plt.suptitle('Performance Improvement of Algorithms vs Pretrained Model', fontsize=20, fontweight='bold', y=0.95)
    ax.text(0.5, 0.82, 'Average L2 Distance (Pretrained - Algorithm) over Last 15 Windows', 
            ha='center', va='center', transform=ax.transAxes, fontsize=18, style='italic')
    
    # Save the plot with absolute minimal whitespace
    plot_path = output_dir / 'performance_improvement_table.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    logger.info(f"Saved performance improvement table to {plot_path}")
    return plot_path


def plot_lr_sweep_heatmap(heatmap_data: Dict, output_dir: Path) -> Path:
    """
    Create heatmap of average RMSPE loss vs eta and LR values.
    
    Args:
        heatmap_data: Dictionary with keys:
        - eta_values: List of eta values
        - lr_values: List of LR values
        - lr_types: List of "static" or "adaptive"
        - lr_row_ids: List of unique row IDs (int for static run index, "ADAPTIVE" for adaptive)
        - avg_losses: List of average RMSPE losses
        output_dir: Output directory for saving plot
    
    Returns:
        Path to saved plot file
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    logger = logging.getLogger(__name__)
    logger.info("Creating LR sweep heatmap...")
    
    # Build DataFrame with lr_row_id for unique rows (allows duplicate LRs as separate rows)
    df_dict = {
        'eta': heatmap_data['eta_values'],
        'lr': heatmap_data['lr_values'],
        'lr_type': heatmap_data['lr_types'],
        'loss': heatmap_data['avg_losses']
    }
    if 'lr_row_ids' in heatmap_data and heatmap_data['lr_row_ids']:
        df_dict['lr_row_id'] = heatmap_data['lr_row_ids']
    else:
        # Fallback: use lr for static, ADAPTIVE for adaptive (old behavior, collapses duplicates)
        df_dict['lr_row_id'] = [
            'ADAPTIVE' if lt == 'adaptive' else lr
            for lt, lr in zip(heatmap_data['lr_types'], heatmap_data['lr_values'])
        ]
    
    df = pd.DataFrame(df_dict)
    
    # Get unique eta values
    eta_unique = sorted(df['eta'].unique())
    
    # Separate static and adaptive
    static_df = df[df['lr_type'] == 'static'].copy()
    adaptive_df = df[df['lr_type'] == 'adaptive'].copy()
    
    # Build row order: static runs by lr_row_id (0, 1, 2, ...), then ADAPTIVE
    static_row_ids = sorted([r for r in df['lr_row_id'].unique() if r != 'ADAPTIVE'], key=lambda x: x if isinstance(x, (int, np.integer)) else 999)
    row_order = static_row_ids.copy()
    if not adaptive_df.empty and 'ADAPTIVE' in df['lr_row_id'].values:
        row_order.append('ADAPTIVE')
    
    # Create pivot table using lr_row_id (unique per run, so duplicate LRs get separate rows)
    pivot_all = df.pivot_table(
        values='loss',
        index='lr_row_id',
        columns='eta',
        aggfunc='mean'
    )
    
    # Reindex to match desired row order
    pivot_all = pivot_all.reindex(row_order)
    
    # Ensure all eta columns are present
    pivot_all = pivot_all.reindex(columns=eta_unique)
    
    # Create figure with appropriate height based on number of rows
    fig, ax = plt.subplots(figsize=(14, max(8, len(row_order) * 0.8)))
    
    # Create proper discrete heatmap using pcolormesh
    # Get the data matrix
    data_matrix = pivot_all.values
    
    # Create meshgrid for proper cell positioning
    y_positions = np.arange(len(row_order) + 1)
    x_positions = np.arange(len(eta_unique) + 1)
    
    # Create heatmap using pcolormesh for proper discrete cells
    # Use reversed viridis: yellow = low error, dark = high error
    im = ax.pcolormesh(x_positions, y_positions, data_matrix, 
                       cmap='viridis_r', shading='flat', edgecolors='white', linewidths=1.5)
    
    # Set ticks at cell centers
    ax.set_xticks(x_positions[:-1] + 0.5)
    ax.set_yticks(y_positions[:-1] + 0.5)
    ax.set_xticklabels([f"{eta:.2f}" for eta in eta_unique])
    
    # Create row labels: lr_value for each row (from first occurrence), "Adaptive" for adaptive
    y_labels = []
    for row_id in row_order:
        if row_id == 'ADAPTIVE':
            y_labels.append("Adaptive")
        else:
            row_data = df[df['lr_row_id'] == row_id]
            lr_val = row_data['lr'].iloc[0] if not row_data.empty else row_id
            y_labels.append(f"{lr_val:.6f}" if isinstance(lr_val, (int, float, np.floating)) else str(row_id))
    
    ax.set_yticklabels(y_labels)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average RMSPE Loss', rotation=270, labelpad=20)
    
    # Add text annotations in each cell
    for i, lr_val in enumerate(row_order):
        for j, eta_val in enumerate(eta_unique):
            cell_value = pivot_all.iloc[i, j]
            if pd.notna(cell_value):
                # For adaptive row, also show the LR value
                if lr_val == 'ADAPTIVE':
                    # This is adaptive row - get the actual LR value for this eta
                    adaptive_row_data = adaptive_df[adaptive_df['eta'] == eta_val]
                    if not adaptive_row_data.empty:
                        adaptive_lr_val = adaptive_row_data.iloc[0]['lr']
                        cell_text = f"{cell_value:.4f}\nLR: {adaptive_lr_val:.6f}"
                    else:
                        cell_text = f"{cell_value:.4f}"
                else:
                    cell_text = f"{cell_value:.4f}"
                
                # Determine text color based on cell value (white for dark, black for light)
                # Use median as threshold
                median_loss = np.nanmedian(data_matrix)
                text_color = 'white' if cell_value > median_loss else 'black'
                
                ax.text(j + 0.5, i + 0.5, cell_text,
                       ha="center", va="center", color=text_color, fontsize=9, weight='bold')
    
    # Labels and title
    ax.set_xlabel('Eta Value', fontsize=12)
    ax.set_ylabel('Learning Rate Value', fontsize=12)
    ax.set_title('Average RMSPE Loss Heatmap: Eta vs Learning Rate', fontsize=14)
    
    # Invert y-axis so first row is at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "lr_sweep_heatmap.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"LR sweep heatmap saved to {plot_path}")
    return plot_path


def plot_performance_improvement_table_eta(scenario_results: dict, output_dir: Path) -> Path:
    """
    Create a performance improvement table comparing online learning vs pretrained models for eta scenario.
    
    Similar to plot_performance_improvement_table but for eta values instead of SNR.
    
    Args:
        scenario_results: Dictionary mapping eta values to their results (same as plot_scenario_results)
        output_dir: Output directory for saving the plot
        
    Returns:
        Path to saved plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    logger = logging.getLogger(__name__)
    logger.info("Creating performance improvement table for eta scenario...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract and sort eta values (preserve original key type for lookup)
    # Keys can be int, float, or string depending on how they were stored
    eta_keys = [key for key in scenario_results.keys() if scenario_results[key] is not None]
    eta_values = sorted([float(key) for key in eta_keys])
    
    if not eta_values:
        logger.warning("No valid eta values found in scenario results")
        return None
    
    # Initialize data structures for the table
    rmspe_improvements = []
    msie_improvements = []
    supervised_rmspe_improvements = []
    supervised_msie_improvements = []
    
    for eta in eta_values:
        # Try to find the matching key (could be float, int, or string)
        result = None
        for key in eta_keys:
            if abs(float(key) - eta) < 1e-10:  # Handle floating point comparison
                result = scenario_results[key]
                break
        
        if result is None:
            logger.warning(f"Eta {eta} not found in results, using NaN...")
            rmspe_improvements.append(np.nan)
            msie_improvements.append(np.nan)
            supervised_rmspe_improvements.append(np.nan)
            supervised_msie_improvements.append(np.nan)
            continue
        result = _unwrap_result_for_lr_sweep(result)
        # Initialize improvements as NaN
        rmspe_improvement = np.nan
        msie_improvement = np.nan
        supervised_rmspe_improvement = np.nan
        supervised_msie_improvement = np.nan
        
        # Try to get averaged results first
        if 'averaged_results' in result:
            averaged_data = result['averaged_results']
            
            # Get averaged metrics for all models
            if ('averaged_pretrained_trajectory' in averaged_data and 
                'averaged_online_trajectory' in averaged_data):
                
                pretrained_metrics = averaged_data['averaged_pretrained_trajectory']
                online_metrics = averaged_data['averaged_online_trajectory']
                supervised_metrics = averaged_data.get('averaged_supervised_trajectory')
                
                # Get last 15 windows for RMSPE (main losses)
                pretrained_rmspe = pretrained_metrics.get('reference_metric_losses', [])
                online_rmspe = online_metrics.get('reference_metric_losses', [])
                
                if pretrained_rmspe and online_rmspe:
                    # Use last 15 windows or all if fewer than 15
                    num_windows = min(15, len(pretrained_rmspe), len(online_rmspe))
                    pretrained_last15 = pretrained_rmspe[-num_windows:]
                    online_last15 = online_rmspe[-num_windows:]
                    
                    # Calculate standalone values
                    pretrained_rmspe_avg = np.mean(pretrained_last15)
                    online_rmspe_avg = np.mean(online_last15)
                    
                    # Calculate L2 distance (average improvement)
                    # Positive means online model is better (lower loss)
                    improvements = [pre - onl for pre, onl in zip(pretrained_last15, online_last15)]
                    rmspe_improvement = np.mean(improvements)
                    
                    logger.info(f"Eta {eta}: RMSPE - Pretrained: {pretrained_rmspe_avg:.6f} rad ({(pretrained_rmspe_avg / np.pi) * 180:.3f}°), Algorithm 1: {online_rmspe_avg:.6f} rad ({(online_rmspe_avg / np.pi) * 180:.3f}°), Improvement: {rmspe_improvement:.6f} rad ({(rmspe_improvement / np.pi) * 180:.3f}°)")
                
                # Calculate supervised model RMSPE improvement if available
                if supervised_metrics is not None:
                    supervised_rmspe = supervised_metrics.get('reference_metric_losses', [])
                    if supervised_rmspe:
                        num_windows = min(15, len(pretrained_rmspe), len(supervised_rmspe))
                        supervised_last15 = supervised_rmspe[-num_windows:]
                        
                        # Calculate supervised model improvement vs pretrained
                        supervised_improvements = [pre - sup for pre, sup in zip(pretrained_last15, supervised_last15)]
                        supervised_rmspe_improvement = np.mean(supervised_improvements)
                        
                        supervised_rmspe_avg = np.mean(supervised_last15)
                        logger.info(f"Eta {eta}: RMSPE Supervised - Supervised Trained: {supervised_rmspe_avg:.6f} rad ({(supervised_rmspe_avg / np.pi) * 180:.3f}°), vs Pretrained Improvement: {supervised_rmspe_improvement:.6f} rad ({(supervised_rmspe_improvement / np.pi) * 180:.3f}°)")
                
                # Get last 15 windows for MSIE (training reference losses)
                pretrained_msie = pretrained_metrics.get('adaptation_losses', [])
                online_msie = online_metrics.get('adaptation_losses', [])
                
                if pretrained_msie and online_msie:
                    # Use last 15 windows or all if fewer than 15
                    num_windows = min(15, len(pretrained_msie), len(online_msie))
                    pretrained_last15 = pretrained_msie[-num_windows:]
                    online_last15 = online_msie[-num_windows:]
                    
                    # Calculate standalone values
                    pretrained_msie_avg = np.mean(pretrained_last15)
                    online_msie_avg = np.mean(online_last15)
                    
                    # Calculate L2 distance (average improvement)
                    improvements = [pre - onl for pre, onl in zip(pretrained_last15, online_last15)]
                    msie_improvement = np.mean(improvements)
                    
                    logger.info(f"Eta {eta}: MSIE - Pretrained: {pretrained_msie_avg:.6f} rad ({(pretrained_msie_avg / np.pi) * 180:.3f}°), Algorithm 1: {online_msie_avg:.6f} rad ({(online_msie_avg / np.pi) * 180:.3f}°), Improvement: {msie_improvement:.6f} rad ({(msie_improvement / np.pi) * 180:.3f}°)")
                
                # Calculate supervised model MSIE improvement if available
                if supervised_metrics is not None:
                    supervised_msie = supervised_metrics.get('adaptation_losses', [])
                    if supervised_msie:
                        num_windows = min(15, len(pretrained_msie), len(supervised_msie))
                        supervised_last15 = supervised_msie[-num_windows:]
                        
                        # Calculate supervised model improvement vs pretrained
                        supervised_improvements = [pre - sup for pre, sup in zip(pretrained_last15, supervised_last15)]
                        supervised_msie_improvement = np.mean(supervised_improvements)
                        
                        supervised_msie_avg = np.mean(supervised_last15)
                        logger.info(f"Eta {eta}: MSIE Supervised - Supervised Trained: {supervised_msie_avg:.6f} rad ({(supervised_msie_avg / np.pi) * 180:.3f}°), vs Pretrained Improvement: {supervised_msie_improvement:.6f} rad ({(supervised_msie_improvement / np.pi) * 180:.3f}°)")
        
        # Store the improvements
        rmspe_improvements.append(rmspe_improvement)
        msie_improvements.append(msie_improvement)
        supervised_rmspe_improvements.append(supervised_rmspe_improvement)
        supervised_msie_improvements.append(supervised_msie_improvement)
    
    # Create the table plot with minimal margins
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.axis('off')  # Turn off axes for table
    
    # Remove all margins and set tight spacing
    fig.subplots_adjust(left=0, right=1, top=0.85, bottom=0.02)
    
    # Prepare table data (Eta as rows, loss types as columns)
    eta_labels = [f'Eta {eta:.2f}' for eta in eta_values]
    table_headers = ['Eta', 'RMSPE (Alg 1)', 'MSIE (Alg 1)', 'RMSPE (Supervised)']
    
    # Format the improvement values with degree conversion
    table_data = []
    for i, eta in enumerate(eta_values):
        # Convert to degrees and add degree symbol
        rmspe_val = f'{(rmspe_improvements[i] / np.pi) * 180:.3f}°' if not np.isnan(rmspe_improvements[i]) else 'N/A'
        msie_val = f'{(msie_improvements[i] / np.pi) * 180:.3f}°' if not np.isnan(msie_improvements[i]) else 'N/A'
        supervised_rmspe_val = f'{(supervised_rmspe_improvements[i] / np.pi) * 180:.3f}°' if not np.isnan(supervised_rmspe_improvements[i]) else 'N/A'
        table_data.append([eta_labels[i], rmspe_val, msie_val, supervised_rmspe_val])
    
    # Create the table positioned to avoid overlap with titles
    table = ax.table(cellText=table_data, colLabels=table_headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.2, 0.267, 0.267, 0.267])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.scale(1.2, 2.5)
    
    # Style header row
    for i in range(len(table_headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_fontsize(16)
    
    # Style data rows with color coding based on improvement
    for i in range(len(table_data)):
        for j in range(len(table_headers)):
            if j == 0:  # First column (Eta labels)
                table[(i+1, j)].set_facecolor('#E7E6E6')
                table[(i+1, j)].set_text_props(weight='bold')
                table[(i+1, j)].set_fontsize(16)
            else:  # Data columns (RMSPE and MSIE for both algorithms)
                # Color code based on improvement value
                if j == 1:  # Algorithm 1 RMSPE
                    improvement = rmspe_improvements[i]
                elif j == 2:  # Algorithm 1 MSIE
                    improvement = msie_improvements[i]
                elif j == 3:  # Supervised RMSPE
                    improvement = supervised_rmspe_improvements[i]
                else:
                    improvement = np.nan
                
                if not np.isnan(improvement):
                    if improvement > 0:  # Algorithm is better than pretrained
                        table[(i+1, j)].set_facecolor('#C6EFCE')
                    elif improvement < 0:  # Pretrained model is better
                        table[(i+1, j)].set_facecolor('#FFC7CE')
                    else:  # No difference
                        table[(i+1, j)].set_facecolor('#FFFFFF')
                else:
                    table[(i+1, j)].set_facecolor('#F2F2F2')
                
                table[(i+1, j)].set_fontsize(16)
    
    # Add title and subtitle with proper spacing
    plt.suptitle('Performance Improvement of Algorithms vs Pretrained Model (Eta Scenario)', 
                 fontsize=20, fontweight='bold', y=0.95)
    ax.text(0.5, 0.82, 'Average L2 Distance (Pretrained - Algorithm) over Last 15 Windows', 
            ha='center', va='center', transform=ax.transAxes, fontsize=18, style='italic')
    
    # Save the plot
    plot_path = output_dir / 'performance_improvement_table_eta.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    logger.info(f"Saved performance improvement table for eta scenario to {plot_path}")
    return plot_path
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    
    logger.info(f"Performance improvement table saved to: {plot_path}")
    return plot_path



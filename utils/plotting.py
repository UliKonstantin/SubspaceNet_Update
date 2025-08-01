import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
import datetime

def plot_loss_vs_scenario(scenario_results, scenario, output_dir):
    """
    Plot ESPRIT and DNN loss vs. scenario values and save the plot.
    """
    logger = logging.getLogger("SubspaceNet.plotting")
    x_vals = list(scenario_results.keys())
    esprit_losses = []
    dnn_losses = []
    ekf_losses = []
    for v in x_vals:
        res = scenario_results[v]
        # If result is a float, treat as ESPRIT loss
        if isinstance(res, float) or isinstance(res, int):
            esprit_loss = res
            dnn_loss = None
            ekf_loss = None
        elif isinstance(res, dict):
            esprit_loss = None
            if 'evaluation_results' in res and 'classic_methods_test_losses' in res['evaluation_results'] and 'ESPRIT' in res['evaluation_results']['classic_methods_test_losses']:
                esprit_loss = res['evaluation_results']['classic_methods_test_losses']['ESPRIT']
            dnn_loss = res['evaluation_results'].get('dnn_test_loss')
            ekf_loss = res['evaluation_results'].get('ekf_test_loss')
        else:
            esprit_loss = None
            dnn_loss = None
            ekf_loss = None
        esprit_losses.append(esprit_loss)
        dnn_losses.append(dnn_loss)
        ekf_losses.append(ekf_loss)
        logger.debug(f"eta={v}: ESPRIT loss={esprit_loss}, DNN loss={dnn_loss}, EKF loss={ekf_loss}")
    if all(l is None for l in esprit_losses) and all(l is None for l in dnn_losses) and all(l is None for l in ekf_losses):
        logger.warning(f"All losses are None for scenario {scenario}. Plot will be empty.")
    plt.figure(figsize=(10, 6))
    if any(l is not None for l in esprit_losses):
        plt.plot(x_vals, esprit_losses, '-o', label='ESPRIT loss', color='green')
    if any(l is not None for l in dnn_losses):
        plt.plot(x_vals, dnn_losses, '-s', label='DNN loss', color='blue')
    if any(l is not None for l in ekf_losses):
        plt.plot(x_vals, ekf_losses, '-^', label='EKF loss', color='red')
    plt.xlabel(scenario)
    plt.ylabel('Loss')
    plt.title(f'Loss vs. {scenario}')
    plt.legend()
    plt.grid(True)
    plot_path = Path(output_dir) / f"loss_vs_{scenario}.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def plot_2d_kalman_noise_sweep(scenario_results, output_dir):
    """
    Plot a 2D heatmap showing DNN loss vs. measurement and process noise standard deviations.
    
    Args:
        scenario_results: 2D dict with structure {meas_noise: {proc_noise: result}}
        output_dir: Output directory for saving the plot
        
    Returns:
        Path to the saved plot
    """
    logger = logging.getLogger("SubspaceNet.plotting")
    
    # Extract measurement and process noise values
    meas_noise_values = sorted(scenario_results.keys())
    proc_noise_values = sorted(list(scenario_results.values())[0].keys())
    
    logger.info(f"Creating 2D heatmap for {len(meas_noise_values)} x {len(proc_noise_values)} combinations")
    
    # Create 2D arrays for the heatmap
    dnn_loss_matrix = np.full((len(proc_noise_values), len(meas_noise_values)), np.nan)
    ekf_loss_matrix = np.full((len(proc_noise_values), len(meas_noise_values)), np.nan)
    esprit_loss_matrix = np.full((len(proc_noise_values), len(meas_noise_values)), np.nan)
    
    # Fill the matrices
    for i, meas_noise in enumerate(meas_noise_values):
        for j, proc_noise in enumerate(proc_noise_values):
            result = scenario_results[meas_noise][proc_noise]
            
            # Extract DNN loss
            dnn_loss = None
            if isinstance(result, dict) and 'evaluation_results' in result:
                dnn_loss = result['evaluation_results'].get('dnn_test_loss')
            
            if dnn_loss is not None:
                dnn_loss_matrix[j, i] = dnn_loss
            
            # Extract EKF loss
            ekf_loss = None
            if isinstance(result, dict) and 'evaluation_results' in result:
                ekf_loss = result['evaluation_results'].get('ekf_test_loss')
            
            if ekf_loss is not None:
                ekf_loss_matrix[j, i] = ekf_loss
            
            # Extract ESPRIT loss for comparison
            esprit_loss = None
            if isinstance(result, dict) and 'evaluation_results' in result:
                classic_losses = result['evaluation_results'].get('classic_methods_test_losses', {})
                if 'ESPRIT' in classic_losses:
                    esprit_loss = classic_losses['ESPRIT']
            
            if esprit_loss is not None:
                esprit_loss_matrix[j, i] = esprit_loss
    
    # Create the figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    
    # Plot DNN loss heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(dnn_loss_matrix, cmap='viridis', aspect='auto', origin='lower')
    ax1.set_xlabel('Measurement Noise Std Dev')
    ax1.set_ylabel('Process Noise Std Dev')
    ax1.set_title('DNN Loss vs. Kalman Filter Noise Parameters')
    ax1.set_xticks(range(len(meas_noise_values)))
    ax1.set_xticklabels([f'{v:.3f}' for v in meas_noise_values], rotation=45)
    ax1.set_yticks(range(len(proc_noise_values)))
    ax1.set_yticklabels([f'{v:.3f}' for v in proc_noise_values])
    
    # Add colorbar for DNN loss
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('DNN Loss')
    
    # Add text annotations for DNN loss values
    for i in range(len(meas_noise_values)):
        for j in range(len(proc_noise_values)):
            if not np.isnan(dnn_loss_matrix[j, i]):
                text = ax1.text(i, j, f'{dnn_loss_matrix[j, i]:.3f}',
                               ha="center", va="center", color="white", fontsize=8)
    
    # Plot EKF loss heatmap
    ax2 = axes[1]
    im2 = ax2.imshow(ekf_loss_matrix, cmap='inferno', aspect='auto', origin='lower')
    ax2.set_xlabel('Measurement Noise Std Dev')
    ax2.set_ylabel('Process Noise Std Dev')
    ax2.set_title('EKF Loss vs. Kalman Filter Noise Parameters')
    ax2.set_xticks(range(len(meas_noise_values)))
    ax2.set_xticklabels([f'{v:.3f}' for v in meas_noise_values], rotation=45)
    ax2.set_yticks(range(len(proc_noise_values)))
    ax2.set_yticklabels([f'{v:.3f}' for v in proc_noise_values])
    
    # Add colorbar for EKF loss
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('EKF Loss')
    
    # Add text annotations for EKF loss values
    for i in range(len(meas_noise_values)):
        for j in range(len(proc_noise_values)):
            if not np.isnan(ekf_loss_matrix[j, i]):
                text = ax2.text(i, j, f'{ekf_loss_matrix[j, i]:.3f}',
                               ha="center", va="center", color="white", fontsize=8)
    
    # Plot ESPRIT loss heatmap (if available)
    ax3 = axes[2]
    im3 = ax3.imshow(esprit_loss_matrix, cmap='plasma', aspect='auto', origin='lower')
    ax3.set_xlabel('Measurement Noise Std Dev')
    ax3.set_ylabel('Process Noise Std Dev')
    ax3.set_title('ESPRIT Loss vs. Kalman Filter Noise Parameters')
    ax3.set_xticks(range(len(meas_noise_values)))
    ax3.set_xticklabels([f'{v:.3f}' for v in meas_noise_values], rotation=45)
    ax3.set_yticks(range(len(proc_noise_values)))
    ax3.set_yticklabels([f'{v:.3f}' for v in proc_noise_values])
    
    # Add colorbar for ESPRIT loss
    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.set_label('ESPRIT Loss')
    
    # Add text annotations for ESPRIT loss values
    for i in range(len(meas_noise_values)):
        for j in range(len(proc_noise_values)):
            if not np.isnan(esprit_loss_matrix[j, i]):
                text = ax3.text(i, j, f'{esprit_loss_matrix[j, i]:.3f}',
                               ha="center", va="center", color="white", fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = Path(output_dir) / "kalman_noise_2d_heatmap.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create individual optimum analysis plot
    _plot_kalman_noise_optimum_analysis(scenario_results, output_dir, dnn_loss_matrix, 
                                       ekf_loss_matrix, esprit_loss_matrix, meas_noise_values, proc_noise_values)
    
    return plot_path


def _plot_kalman_noise_optimum_analysis(scenario_results, output_dir, dnn_loss_matrix, 
                                       ekf_loss_matrix, esprit_loss_matrix, meas_noise_values, proc_noise_values):
    """
    Create additional analysis plots for Kalman noise optimization.
    """
    logger = logging.getLogger("SubspaceNet.plotting")
    
    # Find optimal points
    valid_dnn_mask = ~np.isnan(dnn_loss_matrix)
    if np.any(valid_dnn_mask):
        dnn_min_idx = np.unravel_index(np.nanargmin(dnn_loss_matrix), dnn_loss_matrix.shape)
        optimal_proc_noise_dnn = proc_noise_values[dnn_min_idx[0]]
        optimal_meas_noise_dnn = meas_noise_values[dnn_min_idx[1]]
        optimal_dnn_loss = dnn_loss_matrix[dnn_min_idx]
        
        logger.info(f"Optimal DNN performance: loss={optimal_dnn_loss:.6f} at meas_noise={optimal_meas_noise_dnn:.3f}, proc_noise={optimal_proc_noise_dnn:.3f}")
    
    # Find optimal EKF point
    valid_ekf_mask = ~np.isnan(ekf_loss_matrix)
    if np.any(valid_ekf_mask):
        ekf_min_idx = np.unravel_index(np.nanargmin(ekf_loss_matrix), ekf_loss_matrix.shape)
        optimal_proc_noise_ekf = proc_noise_values[ekf_min_idx[0]]
        optimal_meas_noise_ekf = meas_noise_values[ekf_min_idx[1]]
        optimal_ekf_loss = ekf_loss_matrix[ekf_min_idx]
        
        logger.info(f"Optimal EKF performance: loss={optimal_ekf_loss:.6f} at meas_noise={optimal_meas_noise_ekf:.3f}, proc_noise={optimal_proc_noise_ekf:.3f}")
    
    valid_esprit_mask = ~np.isnan(esprit_loss_matrix)
    if np.any(valid_esprit_mask):
        esprit_min_idx = np.unravel_index(np.nanargmin(esprit_loss_matrix), esprit_loss_matrix.shape)
        optimal_proc_noise_esprit = proc_noise_values[esprit_min_idx[0]]
        optimal_meas_noise_esprit = meas_noise_values[esprit_min_idx[1]]
        optimal_esprit_loss = esprit_loss_matrix[esprit_min_idx]
        
        logger.info(f"Optimal ESPRIT performance: loss={optimal_esprit_loss:.6f} at meas_noise={optimal_meas_noise_esprit:.3f}, proc_noise={optimal_proc_noise_esprit:.3f}")
    
    # Create summary plot showing loss vs individual parameters
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    
    # DNN loss vs measurement noise (averaged over process noise)
    if np.any(valid_dnn_mask):
        mean_dnn_vs_meas = np.nanmean(dnn_loss_matrix, axis=0)
        axes[0, 0].plot(meas_noise_values, mean_dnn_vs_meas, 'bo-')
        axes[0, 0].set_xlabel('Measurement Noise Std Dev')
        axes[0, 0].set_ylabel('Mean DNN Loss')
        axes[0, 0].set_title('DNN Loss vs Measurement Noise (averaged)')
        axes[0, 0].grid(True)
    
    # DNN loss vs process noise (averaged over measurement noise)
    if np.any(valid_dnn_mask):
        mean_dnn_vs_proc = np.nanmean(dnn_loss_matrix, axis=1)
        axes[0, 1].plot(proc_noise_values, mean_dnn_vs_proc, 'ro-')
        axes[0, 1].set_xlabel('Process Noise Std Dev')
        axes[0, 1].set_ylabel('Mean DNN Loss')
        axes[0, 1].set_title('DNN Loss vs Process Noise (averaged)')
        axes[0, 1].grid(True)
    
    # EKF loss vs measurement noise (averaged over process noise)
    if np.any(valid_ekf_mask):
        mean_ekf_vs_meas = np.nanmean(ekf_loss_matrix, axis=0)
        axes[1, 0].plot(meas_noise_values, mean_ekf_vs_meas, 'co-')
        axes[1, 0].set_xlabel('Measurement Noise Std Dev')
        axes[1, 0].set_ylabel('Mean EKF Loss')
        axes[1, 0].set_title('EKF Loss vs Measurement Noise (averaged)')
        axes[1, 0].grid(True)
    
    # EKF loss vs process noise (averaged over measurement noise)
    if np.any(valid_ekf_mask):
        mean_ekf_vs_proc = np.nanmean(ekf_loss_matrix, axis=1)
        axes[1, 1].plot(proc_noise_values, mean_ekf_vs_proc, 'ko-')
        axes[1, 1].set_xlabel('Process Noise Std Dev')
        axes[1, 1].set_ylabel('Mean EKF Loss')
        axes[1, 1].set_title('EKF Loss vs Process Noise (averaged)')
        axes[1, 1].grid(True)
    
    # ESPRIT loss vs measurement noise (averaged over process noise)
    if np.any(valid_esprit_mask):
        mean_esprit_vs_meas = np.nanmean(esprit_loss_matrix, axis=0)
        axes[2, 0].plot(meas_noise_values, mean_esprit_vs_meas, 'go-')
        axes[2, 0].set_xlabel('Measurement Noise Std Dev')
        axes[2, 0].set_ylabel('Mean ESPRIT Loss')
        axes[2, 0].set_title('ESPRIT Loss vs Measurement Noise (averaged)')
        axes[2, 0].grid(True)
    
    # ESPRIT loss vs process noise (averaged over measurement noise)
    if np.any(valid_esprit_mask):
        mean_esprit_vs_proc = np.nanmean(esprit_loss_matrix, axis=1)
        axes[2, 1].plot(proc_noise_values, mean_esprit_vs_proc, 'mo-')
        axes[2, 1].set_xlabel('Process Noise Std Dev')
        axes[2, 1].set_ylabel('Mean ESPRIT Loss')
        axes[2, 1].set_title('ESPRIT Loss vs Process Noise (averaged)')
        axes[2, 1].grid(True)
    
    plt.tight_layout()
    
    # Save the analysis plot
    analysis_plot_path = Path(output_dir) / "kalman_noise_analysis.png"
    plt.savefig(analysis_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved Kalman noise analysis plot to {analysis_plot_path}") 


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
                        
                        ol_results = result["online_learning_results"]
                        
                        # Extract required metrics
                        window_losses = ol_results.get("window_losses", [])
                        pre_ekf_losses = ol_results.get("pre_ekf_losses", [])
                        window_eta_values = ol_results.get("window_eta_values", [])
                        ekf_innovations = ol_results.get("ekf_innovations", [])
                        ekf_kalman_gain_times_innovation = ol_results.get("ekf_kalman_gain_times_innovation", [])
                        ekf_y_s_inv_y = ol_results.get("ekf_y_s_inv_y", [])
                        
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
                            
                            metrics_by_eta[eta] = {
                                "window_losses": window_losses[1:],  # Exclude first window
                                "ekf_improvement": ekf_improvement,
                                "window_eta_values": window_eta_values[1:] if len(window_eta_values) > 1 else [],  # Exclude first window
                                "avg_innovations": avg_innovations[1:] if len(avg_innovations) > 1 else [],
                                "avg_k_times_y": avg_k_times_y[1:] if len(avg_k_times_y) > 1 else [],
                                "avg_y_s_inv_y": avg_y_s_inv_y[1:] if len(avg_y_s_inv_y) > 1 else []
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
                
                # Determine the number of windows (assuming all eta values have same length)
                first_eta = valid_etas[0]
                n_windows = len(metrics_by_eta[first_eta]["window_losses"])
                window_indices = np.arange(1, n_windows + 1)  # Start from window 1
                
                # For each eta scenario, find where eta changes occur within that scenario
                eta_change_markers = {}
                for eta in valid_etas:
                    eta_change_markers[eta] = {"positions": [], "values": []}
                    
                    # Get the window eta values for this scenario
                    scenario_eta_values = metrics_by_eta[eta]["window_eta_values"]
                    
                    if len(scenario_eta_values) > 1:
                        # Find indices where eta changes (similar to original implementation)
                        for i in range(1, len(scenario_eta_values)):
                            if abs(scenario_eta_values[i] - scenario_eta_values[i-1]) > 1e-6:
                                # Window index adjustment: +1 because we start from window 1 (excluded first window)
                                eta_change_markers[eta]["positions"].append(i + 1)
                                eta_change_markers[eta]["values"].append(scenario_eta_values[i])
                
                # Plot 1: Window Losses (Both EKF and SubspaceNet)
                ax1 = axes[0, 0]
                
                # Define colors for different eta values
                colors = plt.cm.tab10(np.linspace(0, 1, len(valid_etas)))
                
                for i, eta in enumerate(valid_etas):
                    color = colors[i]
                    
                    # Plot EKF Loss (solid line, circle markers)
                    ax1.plot(window_indices, metrics_by_eta[eta]["window_losses"], 
                            color=color, linestyle='-', marker='o', 
                            label=f'EKF η={eta:.3f}', linewidth=2, markersize=4)
                    
                    # Plot SubspaceNet Loss (dashed line, square markers)
                    # Calculate SubspaceNet loss from EKF loss + improvement
                    subspacenet_losses = [ekf_loss + improvement for ekf_loss, improvement in 
                                        zip(metrics_by_eta[eta]["window_losses"], metrics_by_eta[eta]["ekf_improvement"])]
                    ax1.plot(window_indices, subspacenet_losses, 
                            color=color, linestyle='--', marker='s', 
                            label=f'SubspaceNet η={eta:.3f}', linewidth=2, markersize=4)
                    
                # Add eta change markers for all scenarios (combine unique positions to avoid duplication)
                all_eta_positions = set()
                all_eta_markers = {}
                for eta in valid_etas:
                    for pos, eta_val in zip(eta_change_markers[eta]["positions"], eta_change_markers[eta]["values"]):
                        all_eta_positions.add(pos)
                        all_eta_markers[pos] = eta_val
                
                for pos in sorted(all_eta_positions):
                    ax1.axvline(x=pos, color='red', linestyle='--', alpha=0.3)
                    ax1.text(pos, 0.13, f'η={all_eta_markers[pos]:.3f}', rotation=90, verticalalignment='top', fontsize=8)
                
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
                    ax2.plot(window_indices, metrics_by_eta[eta]["ekf_improvement"], 
                            color=color, linestyle='-', marker='s', 
                            label=f'η={eta:.3f}', linewidth=2, markersize=4)
                    
                # Add eta change markers for all scenarios (combine unique positions to avoid duplication)
                all_eta_positions = set()
                all_eta_markers = {}
                for eta in valid_etas:
                    for pos, eta_val in zip(eta_change_markers[eta]["positions"], eta_change_markers[eta]["values"]):
                        all_eta_positions.add(pos)
                        all_eta_markers[pos] = eta_val
                
                for pos in sorted(all_eta_positions):
                    ax2.axvline(x=pos, color='red', linestyle='--', alpha=0.3)
                    ax2.text(pos, ax2.get_ylim()[1], f'η={all_eta_markers[pos]:.3f}', rotation=90, verticalalignment='top', fontsize=8)
                
                ax2.set_xlabel('Window Index')
                ax2.set_ylabel('Loss Difference')
                ax2.set_title('SubspaceNet Loss - EKF Loss vs Window Index\nImprovement = L_SubspaceNet - L_EKF')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Plot 3: Average Innovation Magnitude
                ax3 = axes[0, 2]
                for i, eta in enumerate(valid_etas):
                    if metrics_by_eta[eta]["avg_innovations"]:
                        color = colors[i]
                        ax3.plot(window_indices, metrics_by_eta[eta]["avg_innovations"], 
                                color=color, linestyle='-', marker='d', 
                                label=f'η={eta:.3f}', linewidth=2, markersize=4)
                
                # Add eta change markers for all scenarios (combine unique positions to avoid duplication)
                all_eta_positions = set()
                all_eta_markers = {}
                for eta in valid_etas:
                    for pos, eta_val in zip(eta_change_markers[eta]["positions"], eta_change_markers[eta]["values"]):
                        all_eta_positions.add(pos)
                        all_eta_markers[pos] = eta_val
                
                for pos in sorted(all_eta_positions):
                    ax3.axvline(x=pos, color='red', linestyle='--', alpha=0.3)
                    ax3.text(pos, ax3.get_ylim()[1], f'η={all_eta_markers[pos]:.3f}', rotation=90, verticalalignment='top', fontsize=8)
                
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
                        ax4.plot(window_indices, np.abs(metrics_by_eta[eta]["avg_k_times_y"]), 
                                color=color, linestyle='-', marker='v', 
                                label=f'η={eta:.3f}', linewidth=2, markersize=4)
                
                # Add eta change markers for all scenarios (combine unique positions to avoid duplication)
                all_eta_positions = set()
                all_eta_markers = {}
                for eta in valid_etas:
                    for pos, eta_val in zip(eta_change_markers[eta]["positions"], eta_change_markers[eta]["values"]):
                        all_eta_positions.add(pos)
                        all_eta_markers[pos] = eta_val
                
                for pos in sorted(all_eta_positions):
                    ax4.axvline(x=pos, color='red', linestyle='--', alpha=0.3)
                    ax4.text(pos, ax4.get_ylim()[1], f'η={all_eta_markers[pos]:.3f}', rotation=90, verticalalignment='top', fontsize=8)
                
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
                        ax5.plot(window_indices, metrics_by_eta[eta]["avg_y_s_inv_y"], 
                                color=color, linestyle='-', marker='^', 
                                label=f'η={eta:.3f}', linewidth=2, markersize=4)
                
                # Add eta change markers for all scenarios (combine unique positions to avoid duplication)
                all_eta_positions = set()
                all_eta_markers = {}
                for eta in valid_etas:
                    for pos, eta_val in zip(eta_change_markers[eta]["positions"], eta_change_markers[eta]["values"]):
                        all_eta_positions.add(pos)
                        all_eta_markers[pos] = eta_val
                
                for pos in sorted(all_eta_positions):
                    ax5.axvline(x=pos, color='red', linestyle='--', alpha=0.3)
                    ax5.text(pos, ax5.get_ylim()[1], f'η={all_eta_markers[pos]:.3f}', rotation=90, verticalalignment='top', fontsize=8)
                
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
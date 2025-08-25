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
                        
                        # Extract online learning data if available
                        online_window_losses = ol_results.get("online_window_losses", [])
                        online_pre_ekf_losses = ol_results.get("online_pre_ekf_losses", [])
                        online_window_indices = ol_results.get("online_window_indices", [])
                        training_window_losses = ol_results.get("training_window_losses", [])
                        training_pre_ekf_losses = ol_results.get("training_pre_ekf_losses", [])
                        training_window_indices = ol_results.get("training_window_indices", [])
                        learning_start_window = ol_results.get("learning_start_window", None)
                        
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
                            
                            # Calculate online learning derived metrics
                            online_avg_innovations = []
                            online_avg_k_times_y = []
                            online_avg_y_s_inv_y = []
                            training_avg_innovations = []
                            training_avg_k_times_y = []
                            training_avg_y_s_inv_y = []
                            
                            if has_online_data and online_ekf_innovations:
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
                            
                            if has_training_data and training_ekf_innovations:
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


def plot_single_trajectory_results(trajectory_results, trajectory_idx, output_dir):
    """
    Plot results for a single trajectory including loss difference and innovation.
    Also includes online learning results when available.
    
    Args:
        trajectory_results: Dictionary containing single trajectory results
        trajectory_idx: Index of the trajectory being plotted
        output_dir: Output directory for saving plots
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import datetime
        from pathlib import Path
        
        # Create timestamp and plot directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_dir = Path(output_dir) / "plots" / "single_trajectories"
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract data from trajectory results
        pre_ekf_losses = np.array(trajectory_results['pre_ekf_losses'])  # Shape: (30,)
        window_losses = np.array(trajectory_results['window_losses'])    # Shape: (30,) - post EKF losses
        ekf_innovations = np.array(trajectory_results['ekf_innovations']) # Shape: (30, 5, 3)
        window_eta_values = trajectory_results['window_eta_values']
        
        # Check if online learning results are available
        has_online_results = ('online_window_losses' in trajectory_results and 
                            len(trajectory_results['online_window_losses']) > 0)
        
        # Calculate loss difference: pre_ekf_loss - post_ekf_loss (positive means EKF improved)
        loss_difference = pre_ekf_losses - window_losses
        
        # Calculate average innovation magnitude per window
        # Aggregate across steps (5) and sources (3) to get one value per window (30)
        avg_innovations_per_window = []
        for window_idx in range(ekf_innovations.shape[0]):  # 30 windows
            window_innovations = []
            for step_idx in range(ekf_innovations.shape[1]):  # 5 steps
                for source_idx in range(ekf_innovations.shape[2]):  # 3 sources
                    innovation_val = ekf_innovations[window_idx, step_idx, source_idx]
                    window_innovations.append(abs(innovation_val))  # Use absolute value
            
            if window_innovations:
                avg_innovations_per_window.append(np.mean(window_innovations))
            else:
                avg_innovations_per_window.append(0)
        
        avg_innovations_per_window = np.array(avg_innovations_per_window)
        
        # Find indices where eta changes
        eta_changes = []
        eta_values = []
        for i in range(1, len(window_eta_values)):
            if abs(window_eta_values[i] - window_eta_values[i-1]) > 1e-6:
                eta_changes.append(i)
                eta_values.append(window_eta_values[i])
        
        # Determine number of subplots based on available data
        if has_online_results:
            # Create figure with 6 subplots (2x3 layout) to accommodate difference plot
            fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
            
            # Extract online learning data
            online_window_losses = np.array(trajectory_results['online_window_losses'])
            online_pre_ekf_losses = np.array(trajectory_results['online_pre_ekf_losses'])
            online_ekf_innovations = np.array(trajectory_results['online_ekf_innovations'])
            
            # Calculate online loss difference
            online_loss_difference = online_pre_ekf_losses - online_window_losses
            
            # Calculate online average innovation magnitude per window
            online_avg_innovations_per_window = []
            for window_idx in range(online_ekf_innovations.shape[0]):
                window_innovations = []
                for step_idx in range(online_ekf_innovations.shape[1]):
                    for source_idx in range(online_ekf_innovations.shape[2]):
                        innovation_val = online_ekf_innovations[window_idx, step_idx, source_idx]
                        window_innovations.append(abs(innovation_val))
                
                if window_innovations:
                    online_avg_innovations_per_window.append(np.mean(window_innovations))
                else:
                    online_avg_innovations_per_window.append(0)
            
            online_avg_innovations_per_window = np.array(online_avg_innovations_per_window)
            
            # Window indices for online results
            online_window_indices = np.arange(len(online_window_losses))
            
            # Calculate difference between static and online models
            # We need to align the data properly - static model has full trajectory, online has subset
            # For now, we'll calculate difference for the online windows only
            static_losses_for_comparison = window_losses[online_window_indices]  # Get static losses for online windows
            static_pre_ekf_for_comparison = pre_ekf_losses[online_window_indices]  # Get static pre-ekf for online windows
            
            # Calculate differences
            loss_difference_static_vs_online = static_losses_for_comparison - online_window_losses  # Positive = online better
            pre_ekf_difference_static_vs_online = static_pre_ekf_for_comparison - online_pre_ekf_losses  # Positive = online better
            
        else:
            # Create figure with 2 subplots (original layout)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            ax3, ax4 = None, None
        
        # Window indices for static model results
        window_indices = np.arange(len(window_losses))
        
        # Plot 1: Static model loss difference (Pre-EKF Loss - Post-EKF Loss)
        ax1.plot(window_indices, loss_difference, 'g-', marker='o', linewidth=2, markersize=6, label='Static Model: Pre-EKF Loss - Post-EKF Loss')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='No Improvement Line')
        
        # Add eta change markers
        for idx, eta in zip(eta_changes, eta_values):
            ax1.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
            ax1.text(idx, ax1.get_ylim()[1], f'η={eta:.3f}', rotation=90, verticalalignment='top')
        
        ax1.set_xlabel('Window Index')
        ax1.set_ylabel('Loss Difference')
        ax1.set_title(f'Trajectory {trajectory_idx}: Static Model Loss Difference\n(Positive = EKF Improved, Negative = EKF Worse)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Static model innovation magnitude
        ax2.plot(window_indices, avg_innovations_per_window, 'b-', marker='s', linewidth=2, markersize=6, label='Static Model: Average Innovation Magnitude')
        
        # Add eta change markers
        for idx, eta in zip(eta_changes, eta_values):
            ax2.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
            ax2.text(idx, ax2.get_ylim()[1], f'η={eta:.3f}', rotation=90, verticalalignment='top')
        
        ax2.set_xlabel('Window Index')
        ax2.set_ylabel('Average Innovation Magnitude')
        ax2.set_title(f'Trajectory {trajectory_idx}: Static Model Innovation Magnitude\n|Innovation| = |z_k - H x̂_k|k-1| = |measurement - prediction|')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add online learning plots if available
        if has_online_results and ax3 is not None and ax4 is not None and ax5 is not None and ax6 is not None:
            # Plot 3: Online model loss difference
            ax3.plot(online_window_indices, online_loss_difference, 'purple', marker='d', linewidth=2, markersize=6, label='Online Model: Pre-EKF Loss - Post-EKF Loss')
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='No Improvement Line')
            
            # Add eta change markers for online results
            for idx, eta in zip(eta_changes, eta_values):
                if idx < len(online_window_indices):
                    ax3.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                    ax3.text(idx, ax3.get_ylim()[1], f'η={eta:.3f}', rotation=90, verticalalignment='top')
            
            ax3.set_xlabel('Window Index')
            ax3.set_ylabel('Loss Difference')
            ax3.set_title(f'Trajectory {trajectory_idx}: Online Model Loss Difference\n(Positive = EKF Improved, Negative = EKF Worse)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Online model innovation magnitude
            ax4.plot(online_window_indices, online_avg_innovations_per_window, 'orange', marker='^', linewidth=2, markersize=6, label='Online Model: Average Innovation Magnitude')
            
            # Add eta change markers for online results
            for idx, eta in zip(eta_changes, eta_values):
                if idx < len(online_window_indices):
                    ax4.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                    ax4.text(idx, ax4.get_ylim()[1], f'η={eta:.3f}', rotation=90, verticalalignment='top')
            
            ax4.set_xlabel('Window Index')
            ax4.set_ylabel('Average Innovation Magnitude')
            ax4.set_title(f'Trajectory {trajectory_idx}: Online Model Innovation Magnitude\n|Innovation| = |z_k - H x̂_k|k-1| = |measurement - prediction|')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Plot 5: Static vs Online EKF Loss Difference
            ax5.plot(online_window_indices, loss_difference_static_vs_online, 'red', marker='s', linewidth=2, markersize=6, label='Static - Online EKF Loss')
            ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='No Difference Line')
            
            # Add eta change markers
            for idx, eta in zip(eta_changes, eta_values):
                if idx < len(online_window_indices):
                    ax5.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                    ax5.text(idx, ax5.get_ylim()[1], f'η={eta:.3f}', rotation=90, verticalalignment='top')
            
            ax5.set_xlabel('Window Index')
            ax5.set_ylabel('Loss Difference')
            ax5.set_title(f'Trajectory {trajectory_idx}: Static vs Online EKF Loss Difference\n(Positive = Online Better, Negative = Static Better)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # Plot 6: Static vs Online Pre-EKF Loss Difference
            ax6.plot(online_window_indices, pre_ekf_difference_static_vs_online, 'brown', marker='*', linewidth=2, markersize=6, label='Static - Online Pre-EKF Loss')
            ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='No Difference Line')
            
            # Add eta change markers
            for idx, eta in zip(eta_changes, eta_values):
                if idx < len(online_window_indices):
                    ax6.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                    ax6.text(idx, ax6.get_ylim()[1], f'η={eta:.3f}', rotation=90, verticalalignment='top')
            
            ax6.set_xlabel('Window Index')
            ax6.set_ylabel('Loss Difference')
            ax6.set_title(f'Trajectory {trajectory_idx}: Static vs Online Pre-EKF Loss Difference\n(Positive = Online Better, Negative = Static Better)')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        plot_filename = f"single_trajectory_{trajectory_idx}_{timestamp}.png"
        if has_online_results:
            plot_filename = f"single_trajectory_{trajectory_idx}_with_online_{timestamp}.png"
        
        plot_path = plot_dir / plot_filename
        plt.savefig(plot_path)
        plt.close()
        
        logger = logging.getLogger(__name__)
        logger.info(f"Single trajectory plot saved: {plot_path}")
        if has_online_results:
            logger.info(f"Online learning results included in plot for trajectory {trajectory_idx}")
        
        return plot_path
        
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.warning("matplotlib not available for single trajectory plotting")
        return None
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error plotting single trajectory results for trajectory {trajectory_idx}: {e}")
        return None


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
        
        # Set y-axis limit to 0.14 for loss plot only
        ax1.set_ylim([None, 0.14])
        
        # Add eta change markers (adjusted for starting from second sample)
        for idx, eta in zip(eta_changes, eta_values):
            if idx >= 1:  # Only show markers from second sample onwards
                ax1.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                ax1.text(idx, 0.13, f'η={eta:.3f}', rotation=90, verticalalignment='top')
        
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
                ax2.text(idx, ax2.get_ylim()[1], f'η={eta:.3f}', rotation=90, verticalalignment='top')
        
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
                ax3.text(idx, ax3.get_ylim()[1] * 0.9, f'η={eta:.3f}', rotation=90, verticalalignment='top', fontsize=8)
        
        # Add labels and title
        ax3.set_xlabel('Window Index')
        ax3.set_ylabel('Average Angle Predictions (radians)')
        title = 'Averaged Angle Predictions vs Window Index (Starting from Window 1)\nTime-Averaged EKF and Pre-EKF Predictions'
        if has_online_data:
            title += '\n(Static + Online Models)'
        ax3.set_title(title)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)  # Move legend outside plot
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
                ax4.text(idx, ax4.get_ylim()[1], f'η={eta:.3f}', rotation=90, verticalalignment='top')
        
        ax4.set_xlabel('Window Index')
        ax4.set_ylabel('Prediction Delta')
        ax4.set_title('Subspace-Kalman Prediction Delta vs Window Index\nStatic → Training → Inference Pipeline')
        ax4.legend(loc='upper right', framealpha=0.9, fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # 5. Plot covariance vs window index (DUPLICATE of plot 3)
        ax5 = fig.add_subplot(4, 2, 5)
        x = np.arange(len(window_covariances))[1:]  # Start from second sample
        ax5.plot(x, np.array(window_covariances)[1:], 'orange', marker='o', linewidth=2, label='Static Model Average Covariance')
        
        # Add online model covariance if available
        if has_online_data and online_window_covariances is not None and len(online_window_covariances) > 0:
            # Use actual online window indices if available, otherwise fall back to learning start window
            if online_window_indices is not None and len(online_window_indices) > 0:
                online_x = np.array(online_window_indices)  # Use all window indices
            elif learning_start_window is not None:
                online_x = np.arange(learning_start_window, learning_start_window + len(online_window_covariances))[1:]  # Start from second sample
            else:
                online_x = np.arange(len(online_window_covariances))[1:]  # Start from second sample
            ax5.plot(online_x, np.array(online_window_covariances), 'gold', marker='d', linewidth=2, label='Online Model Average Covariance')
        
        # Add training model covariance if available
        if has_training_data and training_window_covariances is not None and len(training_window_covariances) > 0:
            # Use actual training window indices
            training_x = np.array(training_window_indices)  # Use all window indices
            ax5.plot(training_x, np.array(training_window_covariances), 'gold', marker='*', linewidth=2, label='Training Model Average Covariance', linestyle='--')
            
            # Connect training to online if both are available
            if has_online_data and online_window_covariances is not None and len(online_window_covariances) > 0:
                # Connect last training point to first online point
                last_training_x = training_x[-1]  # Window 12
                first_online_x = online_x[0]      # Window 13
                last_training_cov = np.array(training_window_covariances)[-1]
                first_online_cov = np.array(online_window_covariances)[0]
                
                # Draw connecting line
                ax5.plot([last_training_x, first_online_x], [last_training_cov, first_online_cov], 'gold', linestyle='-', linewidth=2, alpha=0.7)
        
        # Add eta change markers (adjusted for starting from second sample)
        for idx, eta in zip(eta_changes, eta_values):
            if idx >= 1:  # Only show markers from second sample onwards
                ax5.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                ax5.text(idx, max(np.array(window_covariances)[1:]), f'η={eta:.3f}', rotation=90, verticalalignment='top')
        
        # Highlight windows where model was updated (adjusted for starting from second sample)
        if window_updates:
            update_indices = [i for i, updated in enumerate(window_updates) if updated and i >= 1]
            update_covs = [window_covariances[i] for i in update_indices]
            ax5.scatter(update_indices, update_covs, color='r', s=80, marker='o', label='Model Updated')
        
        # Add labels and title
        ax5.set_xlabel('Window Index')
        ax5.set_ylabel('Average Covariance')
        title = 'Covariance vs Window Index (Starting from Window 1) - DUPLICATE\nP_k|k = (I - K_k H) P_k|k-1'
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
                    ax6.text(idx, ax6.get_ylim()[1], f'η={eta:.3f}', rotation=90, verticalalignment='top')
        
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
                    ax7.text(idx, ax7.get_ylim()[1], f'η={eta:.3f}', rotation=90, verticalalignment='top')
        
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
                    ax8.text(idx, ax8.get_ylim()[1], f'η={eta:.3f}', rotation=90, verticalalignment='top')
            
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


def plot_online_learning_trajectory(window_labels, plot_dir, timestamp):
    """
    Plot the full trajectory across all windows for online learning.
    
    Args:
        window_labels: List of labels for each window, where each window contains a list of numpy arrays
        plot_dir: Directory to save the plots
        timestamp: Timestamp for the plot filename
        
    Returns:
        Path to saved trajectory plot
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from pathlib import Path
        
        logger = logging.getLogger(__name__)
        
        # Flatten all labels across all windows to get the complete trajectory
        all_angles = []
        all_distances = []
        window_indices = []  # Track which window each step belongs to
        
        for window_idx, window_label_list in enumerate(window_labels):
            for step_idx, step_labels in enumerate(window_label_list):
                # step_labels is a numpy array containing [angle1, angle2, ..., angleM] (in radians)
                # For far-field, we assume distances are constant (e.g., 30m)
                # For near-field, distances would be included in the labels
                
                # Convert radians to degrees for plotting
                angles_deg = step_labels * (180.0 / np.pi)
                all_angles.append(angles_deg)
                
                # Use the same range increment mechanism as in generate_trajectories
                # Start at 20m and increment by 1m each step
                global_step = len(all_angles) - 1  # Current global step index
                base_distance = 20.0 + global_step * 1.0  # Increment by 1m each step
                
                num_sources = len(step_labels)
                distances = np.full(num_sources, base_distance)
                all_distances.append(distances)
                
                # Track window index
                window_indices.append(window_idx)
        
        if not all_angles:
            logger.warning("No trajectory data available for plotting")
            return None
        
        # Find the maximum number of sources across all steps
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
        plt.title(f'Online Learning Full Trajectory (T={total_steps}, Sources={max_sources}, Windows={len(window_labels)})')
        plt.legend()
        
        # Add window boundary markers if there are multiple windows
        if len(window_labels) > 1:
            # Find window boundaries
            window_boundaries = []
            current_window = window_indices[0]
            for i, window_idx in enumerate(window_indices):
                if window_idx != current_window:
                    window_boundaries.append(i)
                    current_window = window_idx
            
            # Add vertical lines for window boundaries
            for boundary_idx in window_boundaries:
                if boundary_idx < len(all_angles):
                    # Get the position at the boundary
                    angles_at_boundary = all_angles[boundary_idx]
                    distances_at_boundary = all_distances[boundary_idx]
                    
                    # Plot boundary markers for each source
                    for s in range(len(angles_at_boundary)):
                        if not np.isnan(angles_at_boundary[s]):
                            angle_rad = angles_at_boundary[s] * (np.pi / 180.0)
                            distance = distances_at_boundary[s]
                            x = distance * np.cos(angle_rad)
                            y = distance * np.sin(angle_rad)
                            plt.plot(x, y, 'ks', markersize=10, markerfacecolor='none', markeredgewidth=2)
        
        # Save the plot
        plot_path = Path(plot_dir) / f"online_learning_trajectory_{timestamp}.png"
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Online learning trajectory plot saved to {plot_dir}:")
        logger.info(f"  - Trajectory plot: {plot_path.name}")
        
        return plot_path
        
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.warning("matplotlib not available for plotting online learning trajectory")
        return None
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error plotting online learning trajectory: {e}")
        return None

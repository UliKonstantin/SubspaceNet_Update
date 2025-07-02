import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging

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
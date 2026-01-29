import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import json
from pathlib import Path
import argparse
from collections import defaultdict

def glrt_changepoint_detection(losses, min_segment_size=5):
    """
    Detect change point using Generalized Likelihood Ratio Test (GLRT).
    
    Assumes Gaussian distribution for log-likelihood computation.
    Tests hypothesis H0 (no change) vs H1 (change at point tau).
    
    Parameters:
    -----------
    losses : array-like
        Time series of loss values
    min_segment_size : int
        Minimum number of samples in each segment
    
    Returns:
    --------
    changepoint : int
        Detected change point index
    log_glr : float
        Log Generalized Likelihood Ratio at the change point
    all_log_glr : array
        Log GLR values for all candidate change points
    """
    losses = np.array(losses)
    n = len(losses)
    
    # Under H0: single Gaussian distribution
    mu_0 = np.mean(losses)
    sigma_0 = np.std(losses, ddof=1)
    
    # Compute log-likelihood under H0
    log_L0 = -n/2 * np.log(2 * np.pi) - n/2 * np.log(sigma_0**2) - \
             np.sum((losses - mu_0)**2) / (2 * sigma_0**2)
    
    # Test all possible change points
    all_log_glr = np.zeros(n - 2*min_segment_size)
    candidate_points = range(min_segment_size, n - min_segment_size)
    
    for i, tau in enumerate(candidate_points):
        # Split data at candidate change point
        segment1 = losses[:tau]
        segment2 = losses[tau:]
        
        # Compute statistics for each segment
        n1, n2 = len(segment1), len(segment2)
        mu1, mu2 = np.mean(segment1), np.mean(segment2)
        sigma1 = np.std(segment1, ddof=1)
        sigma2 = np.std(segment2, ddof=1)
        
        # Avoid numerical issues with very small variances
        sigma1 = max(sigma1, 1e-10)
        sigma2 = max(sigma2, 1e-10)
        
        # Compute log-likelihood under H1 (change at tau)
        log_L1_seg1 = -n1/2 * np.log(2 * np.pi) - n1/2 * np.log(sigma1**2) - \
                      np.sum((segment1 - mu1)**2) / (2 * sigma1**2)
        log_L1_seg2 = -n2/2 * np.log(2 * np.pi) - n2/2 * np.log(sigma2**2) - \
                      np.sum((segment2 - mu2)**2) / (2 * sigma2**2)
        log_L1 = log_L1_seg1 + log_L1_seg2
        
        # Compute log Generalized Likelihood Ratio
        all_log_glr[i] = log_L1 - log_L0
    
    # Find maximum log GLR
    max_idx = np.argmax(all_log_glr)
    changepoint = candidate_points[max_idx]
    max_log_glr = all_log_glr[max_idx]
    
    return changepoint, max_log_glr, all_log_glr, candidate_points


def plot_results(losses, changepoint, all_log_glr, candidate_points):
    """
    Visualize the loss time series and GLRT statistics.
    Returns two separate figures: one for loss, one for GLRT statistics.
    """
    # Create separate figure for loss plot
    fig_loss = plt.figure(figsize=(14, 5))
    ax1 = fig_loss.add_subplot(111)
    ax1.plot(losses, 'b-', linewidth=1.5, label='RMSPE Loss')
    ax1.axvline(x=changepoint, color='r', linestyle='--', linewidth=2, 
                label=f'Detected Change Point (t={changepoint})')
    ax1.set_xlabel('Time Window', fontsize=12)
    ax1.set_ylabel('RMSPE Loss', fontsize=12)
    ax1.set_title('Model Loss Over Time with Detected Change Point', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.88])  # Leave space at top for suptitle
    
    # Create separate figure for GLRT statistics plot
    fig_glrt = plt.figure(figsize=(14, 5))
    ax2 = fig_glrt.add_subplot(111)
    ax2.plot(candidate_points, all_log_glr, 'g-', linewidth=1.5)
    ax2.axvline(x=changepoint, color='r', linestyle='--', linewidth=2, 
                label=f'Maximum log-GLR (t={changepoint})')
    ax2.set_xlabel('Candidate Change Point', fontsize=12)
    ax2.set_ylabel('Log Generalized Likelihood Ratio', fontsize=12)
    ax2.set_title('GLRT Statistics Across All Candidate Change Points', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.88])  # Leave space at top for suptitle
    
    return fig_loss, fig_glrt


def plot_drift_detection_metrics(json_path, output_path=None):
    """
    Load drift detection dicts from JSON file and plot metrics as functions of eta.
    
    If multiple dicts exist for the same eta (multiple trajectories), values are averaged.
    
    Parameters:
    -----------
    json_path : str or Path
        Path to the JSON file containing drift detection dicts
    output_path : str or Path, optional
        Path to save the plot. If None, displays the plot.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    # Load JSON data
    with open(json_path, 'r') as f:
        drift_dicts = json.load(f)
    
    if not drift_dicts:
        print("Warning: No drift detection data found in JSON file")
        return
    
    # Group dicts by eta value
    eta_groups = defaultdict(list)
    for d in drift_dicts:
        eta_groups[d['eta']].append(d)
    
    # Average values for each eta
    etas = sorted(eta_groups.keys())
    averaged_data = {
        'eta': etas,
        'window_idx': [],
        'baseline_mean': [],
        'main_log_glr': [],
        'main_log_glr_std': [],
        'baseline_std': [],
        'current_glrt_z_score': [],
        'learning_rate_at_detection': []
    }
    
    for eta in etas:
        group = eta_groups[eta]
        averaged_data['window_idx'].append(np.mean([d['window_idx'] for d in group]))
        averaged_data['baseline_mean'].append(np.mean([d['baseline_mean'] for d in group]))
        main_log_glr_values = [d['main_log_glr'] for d in group]
        averaged_data['main_log_glr'].append(np.mean(main_log_glr_values))
        averaged_data['main_log_glr_std'].append(np.std(main_log_glr_values))
        averaged_data['baseline_std'].append(np.mean([d['baseline_std'] for d in group]))
        averaged_data['current_glrt_z_score'].append(np.mean([d['current_glrt_z_score'] for d in group]))
        averaged_data['learning_rate_at_detection'].append(np.mean([d['learning_rate_at_detection'] for d in group]))
    
    # Create subplots (4 rows, 2 columns to accommodate the new plot)
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()
    
    metrics = [
        ('window_idx', 'Window Index'),
        ('baseline_mean', 'Baseline Mean'),
        ('baseline_std', 'Baseline Std'),
        ('current_glrt_z_score', 'Current GLRT Z-Score'),
        ('learning_rate_at_detection', 'Learning Rate at Detection')
    ]
    
    plot_idx = 0
    for metric_key, metric_label in metrics:
        ax = axes[plot_idx]
        ax.plot(etas, averaged_data[metric_key], 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('Eta', fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(f'{metric_label} vs Eta', fontsize=12)
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Add mean+std plot for main_log_glr
    ax = axes[plot_idx]
    main_log_glr_mean = averaged_data['main_log_glr']
    main_log_glr_std = averaged_data['main_log_glr_std']
    ax.errorbar(etas, main_log_glr_mean, yerr=main_log_glr_std, fmt='o-', 
                linewidth=2, markersize=6, capsize=5, capthick=2, 
                label='Mean ± Std', elinewidth=1.5)
    ax.set_xlabel('Eta', fontsize=11)
    ax.set_ylabel('Main Log-GLR', fontsize=11)
    ax.set_title('Main Log-GLR vs Eta (Mean ± Std)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plot_idx += 1
    
    # Hide unused subplot
    axes[plot_idx].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    return fig


# # Your example data
# losses = [np.float64(0.004931669023353607), np.float64(0.007578440988436341), 
#           np.float64(0.012997683663852512), np.float64(0.00808196676429361), 
#           np.float64(0.0037428371026180683), np.float64(0.0036191873881034555), 
#           np.float64(0.0068395142117515205), np.float64(0.005118306514341384), 
#           np.float64(0.004571315133944154), np.float64(0.008349625742994249), 
#           np.float64(0.005849287793971598), np.float64(0.003415681341430172), 
#           np.float64(0.0038480300654191524), np.float64(0.0037694890226703135), 
#           np.float64(0.005921079907566309), np.float64(0.008513692754786462), 
#           np.float64(0.006546487710438669), np.float64(0.004433163041248918), 
#           np.float64(0.0064321842044591905), np.float64(0.0048534156614914534), 
#           np.float64(0.00431691390927881), np.float64(0.1368792290240526), 
#           np.float64(0.20899184323847295), np.float64(0.21095313012599945), 
#           np.float64(0.21011458970606328), np.float64(0.21278823256492616), 
#           np.float64(0.2093567244708538), np.float64(0.2244790482521057), 
#           np.float64(0.17682151876389982), np.float64(0.1454287474602461), 
#           np.float64(0.2027802936732769), np.float64(0.2186738930642605), 
#           np.float64(0.21005682542920112), np.float64(0.18476634681224824), 
#           np.float64(0.21587180890142918), np.float64(0.22898483499884606), 
#           np.float64(0.18468307211995125), np.float64(0.18271939642727375), 
#           np.float64(0.1825578884780407), np.float64(0.2150580244511366), 
#           np.float64(0.23237869411706924), np.float64(0.17873112492263318), 
#           np.float64(0.18701113909482955), np.float64(0.216712242141366), 
#           np.float64(0.2319300489127636), np.float64(0.1864546513557434), 
#           np.float64(0.15158024199306966), np.float64(0.22755918517708779), 
#           np.float64(0.212598287910223), np.float64(0.18568192034959793), 
#           np.float64(0.19146738044917583), np.float64(0.2041339661180973), 
#           np.float64(0.22155723571777344), np.float64(0.2082066160440445), 
#           np.float64(0.15259424425661564), np.float64(0.19504203505814074), 
#           np.float64(0.21823965817689894), np.float64(0.2262695948779583), 
#           np.float64(0.17920041956007482)]

# # Run change point detection
# print("Running GLRT Change Point Detection...")
# print("=" * 60)
# 
# changepoint, max_log_glr, all_log_glr, candidate_points = glrt_changepoint_detection(
#     losses, min_segment_size=5
# )
# 
# print(f"\nDetected Change Point: Window {changepoint}")
# print(f"Maximum Log-GLR: {max_log_glr:.4f}")
# 
# # Compute statistics before and after change point
# losses_array = np.array(losses)
# before_change = losses_array[:changepoint]
# after_change = losses_array[changepoint:]
# 
# print(f"\nStatistics Before Change (windows 0-{changepoint-1}):")
# print(f"  Mean: {np.mean(before_change):.6f}")
# print(f"  Std:  {np.std(before_change):.6f}")
# print(f"  Min:  {np.min(before_change):.6f}")
# print(f"  Max:  {np.max(before_change):.6f}")
# 
# print(f"\nStatistics After Change (windows {changepoint}-{len(losses)-1}):")
# print(f"  Mean: {np.mean(after_change):.6f}")
# print(f"  Std:  {np.std(after_change):.6f}")
# print(f"  Min:  {np.min(after_change):.6f}")
# print(f"  Max:  {np.max(after_change):.6f}")
# 
# print(f"\nMean increase factor: {np.mean(after_change) / np.mean(before_change):.2f}x")
# 
# # Create visualization
# fig = plot_results(losses, changepoint, all_log_glr, candidate_points)
# plt.savefig('simulation/runners/', dpi=300, bbox_inches='tight')
# print("\n✓ Visualization saved to 'simulation/runners/sandbox.py'")
# 
# plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot drift detection metrics from JSON file")
    parser.add_argument("json_path", type=str, help="Path to the JSON file containing drift detection dicts")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output path for the plot (optional, displays if not provided)")
    
    args = parser.parse_args()
    
    plot_drift_detection_metrics(args.json_path, args.output)

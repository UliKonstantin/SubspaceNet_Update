import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot loss time series with detected change point
    ax1.plot(losses, 'b-', linewidth=1.5, label='RMSPE Loss')
    ax1.axvline(x=changepoint, color='r', linestyle='--', linewidth=2, 
                label=f'Detected Change Point (t={changepoint})')
    ax1.set_xlabel('Time Window', fontsize=12)
    ax1.set_ylabel('RMSPE Loss', fontsize=12)
    ax1.set_title('Model Loss Over Time with Detected Change Point', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot log GLR statistics
    ax2.plot(candidate_points, all_log_glr, 'g-', linewidth=1.5)
    ax2.axvline(x=changepoint, color='r', linestyle='--', linewidth=2, 
                label=f'Maximum log-GLR (t={changepoint})')
    ax2.set_xlabel('Candidate Change Point', fontsize=12)
    ax2.set_ylabel('Log Generalized Likelihood Ratio', fontsize=12)
    ax2.set_title('GLRT Statistics Across All Candidate Change Points', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# Your example data
losses = [np.float64(0.004931669023353607), np.float64(0.007578440988436341), 
          np.float64(0.012997683663852512), np.float64(0.00808196676429361), 
          np.float64(0.0037428371026180683), np.float64(0.0036191873881034555), 
          np.float64(0.0068395142117515205), np.float64(0.005118306514341384), 
          np.float64(0.004571315133944154), np.float64(0.008349625742994249), 
          np.float64(0.005849287793971598), np.float64(0.003415681341430172), 
          np.float64(0.0038480300654191524), np.float64(0.0037694890226703135), 
          np.float64(0.005921079907566309), np.float64(0.008513692754786462), 
          np.float64(0.006546487710438669), np.float64(0.004433163041248918), 
          np.float64(0.0064321842044591905), np.float64(0.0048534156614914534), 
          np.float64(0.00431691390927881), np.float64(0.1368792290240526), 
          np.float64(0.20899184323847295), np.float64(0.21095313012599945), 
          np.float64(0.21011458970606328), np.float64(0.21278823256492616), 
          np.float64(0.2093567244708538), np.float64(0.2244790482521057), 
          np.float64(0.17682151876389982), np.float64(0.1454287474602461), 
          np.float64(0.2027802936732769), np.float64(0.2186738930642605), 
          np.float64(0.21005682542920112), np.float64(0.18476634681224824), 
          np.float64(0.21587180890142918), np.float64(0.22898483499884606), 
          np.float64(0.18468307211995125), np.float64(0.18271939642727375), 
          np.float64(0.1825578884780407), np.float64(0.2150580244511366), 
          np.float64(0.23237869411706924), np.float64(0.17873112492263318), 
          np.float64(0.18701113909482955), np.float64(0.216712242141366), 
          np.float64(0.2319300489127636), np.float64(0.1864546513557434), 
          np.float64(0.15158024199306966), np.float64(0.22755918517708779), 
          np.float64(0.212598287910223), np.float64(0.18568192034959793), 
          np.float64(0.19146738044917583), np.float64(0.2041339661180973), 
          np.float64(0.22155723571777344), np.float64(0.2082066160440445), 
          np.float64(0.15259424425661564), np.float64(0.19504203505814074), 
          np.float64(0.21823965817689894), np.float64(0.2262695948779583), 
          np.float64(0.17920041956007482)]

# Run change point detection
print("Running GLRT Change Point Detection...")
print("=" * 60)

changepoint, max_log_glr, all_log_glr, candidate_points = glrt_changepoint_detection(
    losses, min_segment_size=5
)

print(f"\nDetected Change Point: Window {changepoint}")
print(f"Maximum Log-GLR: {max_log_glr:.4f}")

# Compute statistics before and after change point
losses_array = np.array(losses)
before_change = losses_array[:changepoint]
after_change = losses_array[changepoint:]

print(f"\nStatistics Before Change (windows 0-{changepoint-1}):")
print(f"  Mean: {np.mean(before_change):.6f}")
print(f"  Std:  {np.std(before_change):.6f}")
print(f"  Min:  {np.min(before_change):.6f}")
print(f"  Max:  {np.max(before_change):.6f}")

print(f"\nStatistics After Change (windows {changepoint}-{len(losses)-1}):")
print(f"  Mean: {np.mean(after_change):.6f}")
print(f"  Std:  {np.std(after_change):.6f}")
print(f"  Min:  {np.min(after_change):.6f}")
print(f"  Max:  {np.max(after_change):.6f}")

print(f"\nMean increase factor: {np.mean(after_change) / np.mean(before_change):.2f}x")

# Create visualization
fig = plot_results(losses, changepoint, all_log_glr, candidate_points)
plt.savefig('simulation/runners/', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved to 'simulation/runners/sandbox.py'")

plt.show()
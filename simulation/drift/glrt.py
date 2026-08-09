import numpy as np


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

    mu_0 = np.mean(losses)
    sigma_0 = np.std(losses, ddof=1)

    log_L0 = -n / 2 * np.log(2 * np.pi) - n / 2 * np.log(sigma_0**2) - np.sum(
        (losses - mu_0) ** 2
    ) / (2 * sigma_0**2)

    all_log_glr = np.zeros(n - 2 * min_segment_size)
    candidate_points = range(min_segment_size, n - min_segment_size)

    for i, tau in enumerate(candidate_points):
        segment1 = losses[:tau]
        segment2 = losses[tau:]

        n1, n2 = len(segment1), len(segment2)
        mu1, mu2 = np.mean(segment1), np.mean(segment2)
        sigma1 = max(np.std(segment1, ddof=1), 1e-10)
        sigma2 = max(np.std(segment2, ddof=1), 1e-10)

        log_L1_seg1 = -n1 / 2 * np.log(2 * np.pi) - n1 / 2 * np.log(sigma1**2) - np.sum(
            (segment1 - mu1) ** 2
        ) / (2 * sigma1**2)
        log_L1_seg2 = -n2 / 2 * np.log(2 * np.pi) - n2 / 2 * np.log(sigma2**2) - np.sum(
            (segment2 - mu2) ** 2
        ) / (2 * sigma2**2)
        log_L1 = log_L1_seg1 + log_L1_seg2

        all_log_glr[i] = log_L1 - log_L0

    max_idx = np.argmax(all_log_glr)
    changepoint = candidate_points[max_idx]
    max_log_glr = all_log_glr[max_idx]

    return changepoint, max_log_glr, all_log_glr, candidate_points

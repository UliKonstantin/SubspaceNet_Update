# GLRT-Based Drift Detection and Adaptive Learning Rate System

## Overview

This document explains the mathematical framework for detecting distribution drift in online learning scenarios using Generalized Likelihood Ratio Test (GLRT) and how the detected drift magnitude is used to adaptively adjust the learning rate.

---

## 1. Generalized Likelihood Ratio Test (GLRT) for Changepoint Detection

### 1.1 Problem Formulation

Given a time series of loss values $\{L_1, L_2, \ldots, L_n\}$, we want to detect if there exists a changepoint $\tau$ where the underlying distribution of the losses changes.

**Hypothesis Testing Framework:**

- **H₀ (Null Hypothesis)**: No changepoint exists. All losses follow a single Gaussian distribution.
- **H₁ (Alternative Hypothesis)**: A changepoint exists at position $\tau$. Losses before and after $\tau$ follow different Gaussian distributions.

### 1.2 Statistical Model

Under both hypotheses, we assume losses follow Gaussian distributions:

- **H₀**: $L_i \sim \mathcal{N}(\mu_0, \sigma_0^2)$ for all $i = 1, \ldots, n$
- **H₁**: $L_i \sim \mathcal{N}(\mu_1, \sigma_1^2)$ for $i \leq \tau$ and $L_i \sim \mathcal{N}(\mu_2, \sigma_2^2)$ for $i > \tau$

### 1.3 Log-Likelihood Under H₀

Under the null hypothesis, we estimate parameters using all data:

$$\mu_0 = \frac{1}{n}\sum_{i=1}^{n} L_i$$

$$\sigma_0^2 = \frac{1}{n-1}\sum_{i=1}^{n}(L_i - \mu_0)^2 \quad \text{(sample variance with Bessel's correction)}$$

The log-likelihood under H₀ is:

$$\log L_0 = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma_0^2) - \frac{1}{2\sigma_0^2}\sum_{i=1}^{n}(L_i - \mu_0)^2$$

### 1.4 Log-Likelihood Under H₁

For each candidate changepoint $\tau \in [\tau_{\min}, \tau_{\max}]$ (where $\tau_{\min} = m$ and $\tau_{\max} = n-m$ for minimum segment size $m$), we estimate parameters for each segment:

**Segment 1 (before changepoint):**
$$\mu_1 = \frac{1}{\tau}\sum_{i=1}^{\tau} L_i, \quad \sigma_1^2 = \frac{1}{\tau-1}\sum_{i=1}^{\tau}(L_i - \mu_1)^2$$

**Segment 2 (after changepoint):**
$$\mu_2 = \frac{1}{n-\tau}\sum_{i=\tau+1}^{n} L_i, \quad \sigma_2^2 = \frac{1}{n-\tau-1}\sum_{i=\tau+1}^{n}(L_i - \mu_2)^2$$

The log-likelihood under H₁ is:

$$\log L_1(\tau) = \log L_1^{\text{seg1}} + \log L_1^{\text{seg2}}$$

where:
$$\log L_1^{\text{seg1}} = -\frac{\tau}{2}\log(2\pi) - \frac{\tau}{2}\log(\sigma_1^2) - \frac{1}{2\sigma_1^2}\sum_{i=1}^{\tau}(L_i - \mu_1)^2$$

$$\log L_1^{\text{seg2}} = -\frac{n-\tau}{2}\log(2\pi) - \frac{n-\tau}{2}\log(\sigma_2^2) - \frac{1}{2\sigma_2^2}\sum_{i=\tau+1}^{n}(L_i - \mu_2)^2$$

### 1.5 Log Generalized Likelihood Ratio

For each candidate changepoint $\tau$, we compute:

$$\log\text{GLR}(\tau) = \log L_1(\tau) - \log L_0$$

This represents the evidence in favor of a changepoint at position $\tau$ versus no changepoint.

### 1.6 Changepoint Detection

The detected changepoint is:

$$\tau^* = \arg\max_{\tau \in [\tau_{\min}, \tau_{\max}]} \log\text{GLR}(\tau)$$

The maximum log-GLR value is:

$$\log\text{GLR}^* = \max_{\tau} \log\text{GLR}(\tau)$$

**Note**: We require a minimum segment size $m$ (typically $m=3$ or $m=5$) to ensure sufficient samples for reliable parameter estimation in each segment.

---

## 2. Statistical Drift Detection Using Z-Score

### 2.1 Baseline Estimation

The GLRT changepoint detection is performed at each window $t$, producing a sequence of log-GLR values: $\{G_1, G_2, \ldots, G_t\}$.

To detect significant drift, we maintain a rolling baseline from recent history:

- **Baseline window size**: $W_b$ (default: 20 windows)
- **Minimum samples for statistics**: $N_{\min}$ (default: 10 windows)

The baseline statistics are computed from the most recent $W_b$ log-GLR values (excluding the current value):

$$\bar{G}_b = \frac{1}{W_b-1}\sum_{i=t-W_b}^{t-1} G_i$$

$$s_b = \sqrt{\frac{1}{W_b-2}\sum_{i=t-W_b}^{t-1}(G_i - \bar{G}_b)^2}$$

### 2.2 Z-Score Computation

Once we have accumulated at least $N_{\min}$ samples, we compute the z-score for the current window:

$$z_t = \frac{G_t - \bar{G}_b}{s_b}$$

The z-score measures how many standard deviations the current log-GLR value is from the baseline mean.

### 2.3 Drift Detection Criterion

Drift is detected when:

$$z_t > z_{\text{threshold}}$$

where $z_{\text{threshold}}$ is a configurable threshold (default: 2.5, corresponding to approximately the 99.4th percentile of a standard normal distribution).

**Intuition**: A high z-score indicates that the current log-GLR is unusually large compared to recent history, suggesting a significant distribution change.

### 2.4 Delayed Activation

After drift is detected at window $t_d$, learning activation is delayed by $T_d$ windows (configurable via `time_to_learn` parameter):

$$\text{Learning activated at window } t_a = t_d + T_d$$

This delay provides a buffer to confirm the drift is persistent rather than a transient spike.

---

## 3. Adaptive Learning Rate Based on Drift Magnitude

### 3.1 Motivation

The magnitude of drift, as quantified by the difference between current GLRT value and baseline mean, should influence the learning rate:
- **Large drift** (large GLRT difference) → Higher learning rate for faster adaptation
- **Small drift** (small GLRT difference) → Lower learning rate for stable, gradual adaptation

**Note**: Z-score is still used for drift detection (Section 2), but the adaptive learning rate uses the GLRT difference directly.

### 3.2 Learning Rate Formula

When adaptive learning rate is enabled, the learning rate is computed as:

$$\text{LR}_{\text{adaptive}} = \text{LR}_{\text{base}} \times (1 + k \times \max(0, G_t - G_{\text{baseline}}))$$

where:
- $G_t$ = current GLRT value (main_log_glr)
- $G_{\text{baseline}}$ = baseline mean of GLRT history ($\bar{G}_b$)
- $k = 0.03$ (scaling constant)

### 3.3 Formula Explanation

The formula directly uses the difference between the current GLRT value and the baseline mean:

**Step-by-step computation:**

1. **Compute GLRT difference**: $\Delta G = \max(0, G_t - G_{\text{baseline}})$
   - Only positive differences are used (negative differences indicate values below baseline, use base LR)
   
2. **Apply scaling**: $\text{LR}_{\text{adaptive}} = \text{LR}_{\text{base}} \times (1 + k \times \Delta G)$
   - Minimum learning rate: $\text{LR}_{\text{base}}$ (when $\Delta G = 0$ or negative)
   - Learning rate increases linearly with GLRT difference
   - No upper bound (unlike the previous z-score-based formula)

### 3.4 Learning Rate Examples

| GLRT ($G_t$) | Baseline ($G_{\text{baseline}}$) | Difference ($\Delta G$) | Learning Rate (if base=0.001, k=0.03) |
|--------------|----------------------------------|-------------------------|----------------------------------------|
| 2.0          | 2.0                              | 0.0                     | 0.001                                  |
| 3.0          | 2.0                              | 1.0                     | 0.00103                                |
| 5.0          | 2.0                              | 3.0                     | 0.00109                                |
| 10.0         | 2.0                              | 8.0                     | 0.00124                                |
| 1.0          | 2.0                              | 0.0 (clamped)           | 0.001                                  |

**Key Properties:**
- Monotonically increasing with GLRT difference
- Linear scaling (no saturation)
- Minimum learning rate equals base learning rate
- Simple, interpretable formula

---

## 4. Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Window t: Collect loss data L_t                             │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ Extract loss sequence: {L_1, ..., L_t}                      │
│ Perform GLRT changepoint detection                           │
│   → Compute log-GLR(τ) for all candidate τ                  │
│   → Find τ* = argmax log-GLR(τ)                             │
│   → Store G_t = max log-GLR(τ)                              │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ Update GLRT history: [G_1, ..., G_t]                        │
│ (Keep only last W_b values)                                  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ If history length ≥ N_min:                                   │
│   Compute baseline: μ_b, σ_b from history[-W_b:-1]          │
│   Compute z-score: z_t = (G_t - μ_b) / σ_b                  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
                   ┌────────┐
                   │ z_t >  │
                   │z_thresh│
                   └────┬───┘
                        │
            ┌───────────┴───────────┐
            │                       │
           Yes                     No
            │                       │
            ▼                       ▼
   ┌──────────────┐         ┌──────────────┐
   │ Store:       │         │ Continue     │
   │ t_d = t      │         │ monitoring   │
   │ z_d = z_t    │         └──────────────┘
   └──────┬───────┘
          │
          ▼
   ┌──────────────────────────────────────┐
   │ Wait T_d windows                     │
   │ (time_to_learn delay)                │
   └──────┬───────────────────────────────┘
          │
          ▼
   ┌──────────────────────────────────────┐
   │ Calculate adaptive learning rate:    │
   │   ΔG = max(0, G_t - G_baseline)      │
   │   LR = LR_base × (1 + k × ΔG)        │
   │   where k = 0.03                      │
   └──────┬───────────────────────────────┘
          │
          ▼
   ┌──────────────────────────────────────┐
   │ Activate online learning             │
   │ with adaptive learning rate          │
   └──────────────────────────────────────┘
```

---

## 5. Configuration Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `min_segment_size` | $m$ | 3 | Minimum samples per segment for GLRT |
| `glrt_baseline_window_size` | $W_b$ | 20 | Number of windows for baseline estimation |
| `glrt_min_samples_for_statistics` | $N_{\min}$ | 10 | Minimum samples before computing z-score |
| `glrt_detection_z_threshold` | $z_{\text{threshold}}$ | 2.5 | Z-score threshold for drift detection |
| `time_to_learn` | $T_d$ | Configurable | Windows to wait after detection before learning |
| `use_adaptive_learning_rate` | - | False | Enable/disable adaptive LR |
| `learning_rate` | $\text{LR}_{\text{base}}$ | 0.001 | Base learning rate |
| adaptive LR scaling constant | $k$ | 0.03 | Scaling factor for GLRT difference in adaptive LR formula |

---

## 6. Advantages of This Approach

1. **Statistical Rigor**: GLRT provides a principled statistical test for changepoint detection
2. **Relative Detection**: Z-score normalization makes detection relative to recent history, adapting to different loss scales
3. **Robust to Outliers**: Baseline smoothing and delayed activation reduce false positives
4. **Adaptive Response**: Learning rate scales with drift magnitude (GLRT difference) for optimal adaptation
5. **Simple Formula**: Linear scaling based on GLRT difference provides interpretable and predictable behavior
6. **Separation of Concerns**: Z-score used for detection, GLRT difference used for learning rate adaptation

---

## 7. Mathematical Properties

### 7.1 GLRT Properties

- **Likelihood Ratio Test**: GLRT is optimal for detecting changes in distribution parameters under certain regularity conditions
- **Computational Efficiency**: O(n²) complexity for n windows, acceptable for online scenarios
- **Minimum Segment Constraint**: Ensures statistical validity by requiring sufficient samples per segment

### 7.2 Z-Score Properties

- **Standard Normal Approximation**: Under stable conditions, z-scores approximately follow standard normal distribution
- **One-Sided Test**: Only positive z-scores trigger detection (loss increases indicate drift)
- **Adaptive Threshold**: Threshold of 2.5 corresponds to ~99.4% confidence level

### 7.3 Adaptive Learning Rate Properties

- **Monotonicity**: Learning rate is non-decreasing with GLRT difference $\Delta G$
- **Linearity**: Learning rate scales linearly with GLRT difference (no saturation)
- **Lower Bound**: Minimum learning rate equals base learning rate (when $\Delta G \leq 0$)
- **Interpretability**: Direct relationship between GLRT magnitude and learning rate scaling

---

## References

- **GLRT Theory**: Generalized Likelihood Ratio Test is a well-established method in change detection (see Basseville & Nikiforov, 1993)
- **Z-Score Normalization**: Standard statistical technique for relative anomaly detection
- **Adaptive Learning Rates**: Common in online learning and adaptive control systems

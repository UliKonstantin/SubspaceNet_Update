# Online Learning Algorithm — SubspaceNet with EKF and GLRT Drift Detection

This document describes the algorithm that executes when online learning is invoked. It focuses on the logic and mechanics of the pipeline — what runs, in what order, and why — independent of any specific sweep configuration or trajectory type.

---

## Overview

The system solves **online Direction-of-Arrival (DOA) adaptation**: a neural network (SubspaceNet) was trained offline under a fixed array calibration regime. At inference time, the array's calibration error (parameterized by η) can drift, degrading the model's predictions. The pipeline detects this drift and fine-tunes the model in a windowed, online fashion using a differentiable Extended Kalman Filter (EKF) as the training signal.

**Three actors** run in parallel throughout every trajectory:
1. **Pretrained model** — the original frozen model, always evaluated with EKF, serves as the degradation baseline.
2. **Online model** — an independent copy of the pretrained model that adapts via gradient descent after drift is detected, trained with an *unsupervised* loss (no ground truth labels required).
3. **Supervised model** — a second independent copy trained with a *supervised* loss (using ground truth angles). Serves as an oracle upper bound for comparison.

---

## Algorithm Structure

### Stage 0 — Initialization

Before any window is processed:
- The pretrained SubspaceNet is loaded from disk.
- Two independent copies are created: the *online model* (unsupervised) and the *supervised model*.
- For each new trajectory, η is reset to its initial value (0), and the three model copies are re-initialized to the same pretrained weights.
- A windowed dataloader is created that generates data on-demand. Each item in the dataloader is one window of `window_size` consecutive time steps, advanced by `stride` steps.
- State variables are cleared: `drift_detected = False`, `learning_done = False`, `online_training_count = 0`.
- GLRT history from previous trajectories is *retained* (not reset), providing a better statistical baseline.

---

### Stage 1 — Per-Window Evaluation Loop

The pipeline iterates over windows. Each window contains `window_size` time steps. At each window index `w`:

#### 1a. Dynamic η update (optional)
If `eta_update_interval_windows` is set and `w > 0` and `w % eta_update_interval_windows == 0`:
- η is incremented by `eta_increment`, clamped to `[min_eta, max_eta]`.
- The dataset generator immediately begins producing samples with the new η.
- The first time η changes, a flag `first_eta_change` is recorded.

#### 1b. Pretrained model — window evaluation

For each time step `t` within the window:

1. **SubspaceNet forward pass**
   The array snapshot `X_t ∈ ℂ^{N×T}` is passed through the pre-trained SubspaceNet. The model produces raw DOA angle estimates `θ̂_t^{pre}` (one per source). This step runs under `torch.no_grad()` for the pretrained model.

2. **Permutation alignment**
   Source ordering is ambiguous. An optimal permutation is computed over all `M!` orderings by minimizing RMSPE between predicted and true angles. All subsequent computations use this aligned ordering.

3. **EKF state initialization (first step of each window)**
   - First window: EKF state is seeded from the ground truth angles.
   - Subsequent windows: state is seeded from the last EKF prediction and covariance of the previous window (with its own permutation alignment applied).

4. **EKF predict and update (per source)**
   Each source has its own independent `ExtendedKalmanFilter1D`. For source `i`:

   **Predict step:**
   ```
   x̂_{t|t-1}^i = f(x̂_{t-1|t-1}^i)          # nonlinear state transition
   F_t^i = ∂f/∂x |_{x̂_{t-1}}                # linearized Jacobian
   P_{t|t-1}^i = F_t^i · P_{t-1}^i · (F_t^i)^T + Q_t^i
   ```

   **Update step** (measurement = SubspaceNet output `θ̂_t^{pre,i}`):
   ```
   y_t^i   = θ̂_t^{pre,i} - h(x̂_{t|t-1}^i)   # innovation  (h = identity)
   S_t^i   = P_{t|t-1}^i + R                  # innovation covariance
   K_t^i   = P_{t|t-1}^i / S_t^i              # Kalman gain
   x̂_t^i  = x̂_{t|t-1}^i + K_t^i · y_t^i     # updated state
   P_t^i   = (1 - K_t^i) · P_{t|t-1}^i        # updated covariance
   ```

   The EKF outputs `θ̂_t^{ekf,i}` (refined angle) and `P_t^i` (state covariance).

5. **Loss calculation (window level)**
   After all time steps in the window are processed, three losses are computed over the stacked window tensor `[window_size, M]`:

   | Loss name | Formula | Purpose |
   |---|---|---|
   | `pre_ekf_loss` | RMSPE(θ̂^pre, θ^true) | Raw model quality |
   | `reference_metric_loss` | RMSPE(θ̂^ekf, θ^true) when `reference_metric=supervised_rmspe` | Eval-only tracking (plot 1, tables) |
   | `adaptation_loss` | RMSPE(θ̂^ekf, θ̂^pre) when `adaptation_loss=unsupervised_rmspe` | MSIE: GLRT trigger, backprop, plot 2 |

   RMSPE uses angular wrapping: errors are mapped into `[-π/2, π/2]` before squaring, making the metric periodic-aware.

---

### Stage 2 — GLRT Drift Detection

After each new window result is appended, the pipeline runs a **Generalized Likelihood Ratio Test (GLRT)** over the history of `adaptation_loss` values (MSIE stream). A separate GLRT on `reference_metric_loss` is computed for analysis only — it does not trigger drift.

**GLRT changepoint detection:**
Given a loss sequence `{L_1, …, L_w}`, the test finds the split point `τ*` that maximizes the log-likelihood ratio between a two-segment model (before/after change) and a single-segment model:
```
τ* = argmax_τ  log GLR(τ)
```
The Log-GLR at the optimal split is the GLRT statistic `G_w`.

**Statistical thresholding:**
Rather than a fixed GLR threshold, a rolling z-score is used on the `adaptation_loss` = RMSPE(θ̂^ekf, θ̂^pre) GLRT series:
- First `drift_warmup_windows` trajectory windows are skipped entirely (no GLRT, no g-history).
- Scope A runs on post-warmup losses from window `W + 2m` (`m = GLRT_MIN_SEGMENT_SIZE = 5` internal).
- Baseline for z-score uses all g-history except the last `drift_guard_samples` entries. Z-score once `len(baseline) >= drift_warmup_windows`.
- Rolling history is unbounded by default (`drift_history_max_size: null`).
- Z-score: `z_w = (G_w - μ_baseline) / σ_baseline`
- Drift is declared at window `w*` when `z_w > drift_z_threshold` and drift has not been declared previously.

First g at `W + 2m`; first z at `2W + 2m + G - 1` (defaults W=7, m=5, G=3 → g@17, z@26).

This approach automatically adapts to the statistical regime of the specific signal and suppresses false alarms due to natural loss fluctuation.

**Time-to-learn delay:**
Actual online learning does not start at detection window `w*`. It starts at `w* + time_to_learn`. This delay allows additional windows of post-drift data to accumulate before adaptation begins, providing a more stable training signal.

---

### Stage 3 — Adaptive Learning Rate

When drift is first detected (at `w*`), the learning rate for the upcoming adaptation phase is computed immediately — before the baseline is contaminated by post-drift GLRT values:

If `use_adaptive_learning_rate = True`, the learning rate is mapped from the GLRT statistic via a sigmoid:
```
dG   = G_{w*} - μ_baseline
log_lr = log10(lr_min) + [log10(lr_max) - log10(lr_min)] / (1 + exp(-k · (dG - dG0)))
lr*  = 10^{log_lr}
```
Parameters `lr_min`, `lr_max`, `k`, `dG0` define the sigmoid shape.

If `use_adaptive_learning_rate = False`, the configured `learning_rate` is used directly.

This learning rate `lr*` is fixed for the entire adaptation phase of this trajectory — it does not change as the baseline shifts post-drift.

---

### Stage 4 — Online Adaptation (per window, while `learning_done = False`)

When `window_idx >= w* + time_to_learn` and `drift_detected = True` and `learning_done = False`:

The same time-series window is used for **both training and evaluation**:

#### 4a. Gradient descent phase

The online model is put in `train()` mode. For each of `num_gd_steps` gradient descent iterations:

1. Zero gradients.
2. For each time step `t` in the window:
   - Forward pass with `require_grad=True` → `θ̂_t^{pre}` (differentiable tensor).
   - Permutation alignment (detached for alignment, but gradient flows through prediction).
   - EKF predict and update — **the EKF is differentiable**: `updated_state` and `kalman_gain_times_innovation` are tensors that carry gradient information back to the SubspaceNet parameters.
3. Stack all time steps into a window tensor.
4. Compute the training loss (unsupervised RMSPE):
   ```
   L_train = RMSPE(θ̂^{ekf}, θ̂^{pre}) / window_size
   ```
   This loss encourages the SubspaceNet output to be consistent with the EKF's smoothed estimate — no ground truth labels are required.
5. `loss.backward()` — gradients flow from the EKF output, through the EKF update equations, back into the SubspaceNet.
6. `optimizer.step()` — Adam optimizer updates SubspaceNet parameters.

After `num_gd_steps` iterations, the model is set back to `eval()` mode.

#### 4b. Post-training evaluation

The same window data is re-evaluated in `no_grad()` mode using the just-updated online model, following the same pretrained-model evaluation path (forward → align → EKF → losses). This gives the online model's current performance on the window.

#### 4c. Supervised model (oracle track)

In parallel, the supervised model undergoes the same training procedure but with:
```
L_train_supervised = RMSPE(θ̂^{ekf}, θ^{true}) / window_size
```
This oracle track shows what adaptation would look like with perfect label access.

#### 4d. Learning termination

`learning_done` is set to `True` after `online_training_count >= adaptation_window_count` training windows (default 5). Once set, both online and supervised models stop updating and switch to pure evaluation mode.

---

### Stage 5 — Post-Learning Evaluation Phase

Once `learning_done = True`, every subsequent window evaluates all three models in `no_grad()` mode:
- Pretrained model (frozen, continuously degrading as η increases).
- Online model (frozen after 5 training windows).
- Supervised model (frozen after 5 training windows).

The gap between pretrained and online model performance quantifies the benefit of unsupervised adaptation.

---

## Data Flow Diagram

```
Array snapshot X_t
       │
       ▼
 ┌─────────────┐
 │  SubspaceNet │  ──→  θ̂^pre  (pre-EKF DOA estimates, M angles)
 └─────────────┘
       │
       ▼ permutation alignment
       │
 ┌─────────────────────────────────────────────┐
 │  ExtendedKalmanFilter1D  ×  M (one per src) │
 │                                             │
 │   predict:  x̂_{t|t-1} = f(x̂_{t-1})        │
 │   update:   x̂_t = x̂_{t|t-1} + K·(θ̂^pre - x̂_{t|t-1}) │
 └─────────────────────────────────────────────┘
       │
       ▼
    θ̂^ekf  (EKF-refined DOA estimates)
       │
       ├──→  reference_metric_loss = RMSPE(θ̂^ekf, θ^true)   [eval / plot 1]
       ├──→  pre_ekf_loss         = RMSPE(θ̂^pre, θ^true)       [eval]
       └──→  adaptation_loss     = RMSPE(θ̂^ekf, θ̂^pre)       [GLRT trigger + backprop]
                                    │
                                    ▼
                            GLRT(adaptation_loss series)
                                    │
                            z-score > threshold?
                                    │ yes
                                    ▼
                         wait time_to_learn windows
                                    │
                                    ▼
                   training: L_train = RMSPE(θ̂^ekf, θ̂^pre)
                   backprop through EKF → SubspaceNet
                   Adam step (lr = lr* from detection time)
```

---

## Multi-Trajectory Averaging

The full pipeline runs over `dataset_size` independent trajectories. For each trajectory, all window metrics are collected. After all trajectories complete, results are averaged window-by-window across trajectories to reduce variance. Plots and final metrics are produced from the averaged results.

---

## Key Design Choices

**Why unsupervised training loss?**
The loss `RMSPE(θ̂^ekf, θ̂^pre)` measures how much the EKF disagrees with the raw SubspaceNet prediction. When the model is miscalibrated (η is large), EKF — which has a better motion prior — diverges more from the noisy model output. Minimizing this disagreement pushes the model's output toward what the EKF expects, effectively using the Kalman smoother as a self-supervision signal.

**Why z-score rather than a fixed GLR threshold?**
The absolute GLRT statistic depends on the noise level, window size, and loss scale. A z-score normalizes by the recent historical distribution, making the threshold consistent across different η regimes and loss magnitudes.

**Why fix the learning rate at detection time?**
After drift is detected, the GLRT history gets contaminated by post-drift (high-loss) values, which inflates the baseline. Computing the adaptive LR before that contamination ensures the LR reflects the true magnitude of the drift signal, not the degraded post-drift statistics.

**Why a differentiable EKF?**
Running backpropagation through the EKF update equations means the gradient signal carries information about tracking quality (covariance, Kalman gain) back to the SubspaceNet. This is richer than training the model in isolation: the gradient rewards predictions that are both accurate and temporally consistent under the motion prior.

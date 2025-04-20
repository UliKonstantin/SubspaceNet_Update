# Kalman Filter Integration Plan

**Goal:** Integrate a simple Kalman Filter (KF) into the trajectory evaluation pipeline to track source angles, using model predictions as measurements. Initially, the KF will assume an identity observation model (H=I) and near-zero measurement noise (R≈0).

**Phases:**

## Phase 1: Data Generation Modification (Random Walk)

1.  **Target File:** `simulation/runners/data.py`
2.  **Target Function:** `TrajectoryDataHandler._generate_trajectories`
3.  **Specific Change:** Modify the generation logic **only** for `trajectory_type == TrajectoryType.RANDOM_WALK`.
    *   **State-Space Model:** Implement the process model θ\_k = θ\_{k-1} + w\_k, where θ\_k is the angle at step k.
    *   **Process Noise (w\_k):**
        *   Define `w_k` drawn from a zero-mean Gaussian distribution with standard deviation σ\_w.
        *   **Action:** Add a new configuration parameter `trajectory.random_walk_std_dev` (default: 1.0 degrees) to `config/schema.py`.
        *   **Action:** In `_generate_trajectories`, retrieve `random_walk_std_dev` from the config.
        *   **Action:** Replace existing random walk logic with `w_k = torch.randn(num_sources) * random_walk_std_dev`.
        *   **Action:** Update angle: `angle_trajectories[i, t, :num_sources] = angle_trajectories[i, t-1, :num_sources] + w_k`.
    *   **Angle Clamping:** Retain the logic to clamp generated angles within the valid DOA range (`angle_min`, `angle_max`).
    *   **Other Trajectory Types:** Ensure no changes to other `trajectory_type` values.

## Phase 2: Kalman Filter Class Implementation

1.  **Target File:** Create new file: `simulation/kalman_filter.py`
2.  **Class Definition:** Create `KalmanFilter1D` class.
    *   **State Representation:**
        *   `self.x`: State estimate (scalar angle θ).
        *   `self.P`: State covariance (scalar variance P).
    *   **`__init__(self, Q, R, P0=1.0)`:**
        *   Params: `Q` (process noise var), `R` (measurement noise var), `P0` (initial state var).
        *   **Action:** Store `Q`, `R`, `P0`. Initialize `self.x = None`, `self.P = P0`.
        *   **Action:** Set `R` to a small value (e.g., `1e-6`) if passed as 0.
    *   **`initialize_state(self, x0)`:**
        *   **Action:** Set `self.x = x0`, `self.P = self.P0`.
    *   **`predict(self)`:**
        *   Implements time update (F=1, B=0).
        *   **Action:** Predict state: `x_pred = self.x`.
        *   **Action:** Predict covariance: `P_pred = self.P + self.Q`.
        *   **Action:** Update internal state: `self.x = x_pred`, `self.P = P_pred`.
        *   **Action:** Return `x_pred`.
    *   **`update(self, z)`:**
        *   Implements measurement update (H=1).
        *   Params: `z` (scalar measurement).
        *   **Action:** Calculate Innovation Covariance: `S = self.P + self.R`.
        *   **Action:** Calculate Kalman Gain: `K = self.P / S`.
        *   **Action:** Calculate Innovation: `y = z - self.x`.
        *   **Action:** Update state estimate: `x_new = self.x + K * y`.
        *   **Action:** Update state covariance: `P_new = (1 - K) * self.P`.
        *   **Action:** Update internal state: `self.x = x_new`, `self.P = P_new`.

## Phase 3: Integration into Evaluation Pipeline

1.  **Target File:** `simulation/core.py`
2.  **Target Function:** `Simulation._run_evaluation_pipeline`
3.  **Specific Changes:**
    *   **Import:** Add `from simulation.kalman_filter import KalmanFilter1D`.
    *   **Configuration:**
        *   **Action:** Define a new `KalmanFilterConfig` Pydantic model in `config/schema.py` with fields `process_noise_std_dev` (Optional, defaults to link to `trajectory.random_walk_std_dev`), `measurement_noise_std_dev` (default: 1e-3), `initial_covariance` (default: 1.0).
        *   **Action:** In `_run_evaluation_pipeline`, retrieve these values from `self.config.kalman_filter`.
        *   **Action:** Calculate `kf_Q = config.kalman_filter.process_noise_std_dev**2` (or use `trajectory.random_walk_std_dev` if KF specific one isn't set).
        *   **Action:** Calculate `kf_R = config.kalman_filter.measurement_noise_std_dev**2`.
        *   **Action:** Set `kf_P0 = config.kalman_filter.initial_covariance`.
    *   **Initialization (Inside trajectory loop `for traj_idx...`):**
        *   **Action:** Before the `step` loop, determine `num_sources` for the trajectory.
        *   **Action:** Get initial true angles: `initial_true_angles = labels[traj_idx, 0, :num_sources].cpu().numpy()`.
        *   **Action:** Create and initialize KFs: `k_filters = [KalmanFilter1D(Q=kf_Q, R=kf_R, P0=kf_P0) for _ in range(num_sources)]`.
        *   **Action:** Initialize state for each filter: `k_filters[s_idx].initialize_state(initial_true_angles[s_idx])`.
    *   **Prediction & Update (Inside step loop `for step...`):**
        *   **Action:** **Before** `traj_preds.append(...)`: Call KF predict for each source: `kf_step_predictions[s_idx] = k_filters[s_idx].predict()`. Store the result in `traj_kf_preds.append(kf_step_predictions)`.
        *   **Action:** Remove the old placeholder line for `traj_kf_preds`.
        *   **Action:** **After** getting `angles_pred` from the model: Call KF update for each source: `k_filters[s_idx].update(measurement)`, where `measurement` is the corresponding angle from `angles_pred`.
    *   **Storage:** Ensure `traj_kf_preds` stores the array of predicted angles from KF for each step.
    *   **Logging:** Update log message to reflect KF predictions are being stored. 
# High-Level Pseudo-code for Evaluation Pipeline with Kalman Filter

## File: `simulation/core.py`

### FUNCTION `_run_evaluation_pipeline(self)`

**BEGIN**

  // Ensure necessary model and test dataloader are available (Error handling omitted)
  // Assume KF parameters (Q, R, P0) are determined

  INITIALIZE overall `results` (loss, metrics, trajectory details)

  // --- Process Test Data ---
  FOR EACH `batch` IN `test_dataloader`
    GET `trajectories`, `source_counts`, `ground_truth_labels` FROM `batch`

    // --- Process Each Trajectory in Batch ---
    FOR EACH `trajectory_index` FROM 0 TO `batch_size` - 1
      INITIALIZE trajectory-specific storage: `model_preds`, `kf_preds`, `true_vals`
      GET `current_trajectory_data`
      GET `current_source_counts`

      // --- Initialize KFs for this Trajectory ---
      DETERMINE `num_sources` for this trajectory
      GET `initial_true_angles` for step 0 ( \(x_0\) )
      CREATE & INITIALIZE `kalman_filters` (one `KalmanFilter1D` per source using \(x_0\), Q, R, P0)

      // --- Process Each Step in Trajectory (k=1...T)---
      FOR EACH `step` `k` FROM 0 TO `trajectory_length` - 1
        GET `step_input_data` for the model
        GET `step_true_angles` ( \(x_k\) ground truth)

        // --- Kalman Filter Predict Step ---
        INITIALIZE `kf_step_predictions` AS ARRAY size `num_sources`
        FOR EACH `filter` IN `kalman_filters` (one per source)
           ASSIGN predicted state \(\hat{x}_{k|k-1}\) = CALL `filter.predict()`
           STORE \(\hat{x}_{k|k-1}\) in `kf_step_predictions` for the corresponding source
        ENDFOR
        STORE `kf_step_predictions` array in `kf_preds` list for step `k`

        // --- Model Measurement Step ---
        GET `model_predicted_angles` ( \(z_k\) measurement) by calling `self.trained_model`
        STORE `model_predicted_angles` in `model_preds` list for step `k`
        STORE `step_true_angles` in `true_vals` list for step `k`

        // --- Kalman Filter Update Step ---
        FOR EACH `filter`, `measurement` IN ZIP(`kalman_filters`, `model_predicted_angles`)
          CALL `filter.update`(`measurement` \(z_k\)) // Updates filter's internal state to \(\hat{x}_{k|k}\)
        ENDFOR

        // --- Accumulate Loss ---
        CALCULATE `step_loss` based on error between `model_predicted_angles` and `step_true_angles` (e.g., using RMSPE)
        ADD `step_loss` to `total_loss`

      ENDFOR // Step loop (k)

      // --- Store Detailed Results for This Trajectory ---
      ADD {`model_preds`, `kf_preds`, `true_vals`, `source_counts`} to overall `trajectory_results` list

    ENDFOR // Trajectory loop

  ENDFOR // Batch loop

  // --- Calculate Final Metrics ---
  CALCULATE overall `test_loss` from `total_loss`
  CALCULATE other overall metrics (using `trajectory_results`)
  STORE overall metrics and detailed `trajectory_results` in `self.results`

**END FUNCTION**

---

## File: `simulation/kalman_filter.py`

### CLASS `KalmanFilter1D` (Mathematical Focus)

  // State: \(x\), Covariance: \(P\)
  // Process Noise Variance: \(Q\), Measurement Noise Variance: \(R\)
  // State Transition \(F=1\), Observation Model \(H=1\)

  #### FUNCTION `__init__(Q, R, P0)`
    STORE \(Q, R, P0\). Handle \(R=0 \implies R = \epsilon\).
    INITIALIZE state \(x \leftarrow \text{None}\), covariance \(P \leftarrow P0\).

  #### FUNCTION `initialize_state(x0)`
    SET \(x \leftarrow x0\), SET \(P \leftarrow P0\).

  #### FUNCTION `predict()` -> predicted state \(\hat{x}_{k|k-1}\)
    **BEGIN**
      // Predict State: \(\hat{x}_{k|k-1} = F \hat{x}_{k-1|k-1} = \hat{x}_{k-1|k-1}\)
      ASSIGN \(\hat{x}_{k|k-1} \leftarrow x\)

      // Predict Covariance: \(P_{k|k-1} = F P_{k-1|k-1} F^T + Q = P_{k-1|k-1} + Q\)
      ASSIGN \(P_{k|k-1} \leftarrow P + Q\)

      // Update internal state for next step
      SET \(x \leftarrow \hat{x}_{k|k-1}\)
      SET \(P \leftarrow P_{k|k-1}\)

      RETURN \(\hat{x}_{k|k-1}\)
    **END**

  #### FUNCTION `update(z)` -> updated state \(\hat{x}_{k|k}\)
    // Input: measurement \(z_k\)
    **BEGIN**
      // Innovation: \(y_k = z_k - H \hat{x}_{k|k-1} = z_k - \hat{x}_{k|k-1}\)
      ASSIGN \(y_k \leftarrow z - x\) // `x` currently holds \(\hat{x}_{k|k-1}\)

      // Innovation Covariance: \(S_k = H P_{k|k-1} H^T + R = P_{k|k-1} + R\)
      ASSIGN \(S_k \leftarrow P + R\) // `P` currently holds \(P_{k|k-1}\)

      // Kalman Gain: \(K_k = P_{k|k-1} H^T S_k^{-1} = P_{k|k-1} / S_k\)
      ASSIGN \(K_k \leftarrow P / S_k\)

      // Update State Estimate: \(\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k y_k\)
      ASSIGN \(\hat{x}_{k|k} \leftarrow x + K_k \times y_k\)

      // Update Estimate Covariance: \(P_{k|k} = (I - K_k H) P_{k|k-1} = (1 - K_k) P_{k|k-1}\)
      ASSIGN \(P_{k|k} \leftarrow (1 - K_k) \times P\)

      // Update internal state
      SET \(x \leftarrow \hat{x}_{k|k}\)
      SET \(P \leftarrow P_{k|k}\)

      RETURN \(\hat{x}_{k|k}\)
    **END**

</rewritten_file> 
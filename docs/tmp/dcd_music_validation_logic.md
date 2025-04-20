# Pseudo-code for Far-Field SubspaceNet Evaluation

## File: `DCD_MUSIC/src/evaluation.py` (Simplified for Far-Field SubspaceNet)

### FUNCTION `evaluate_dnn_model(model, dataset, mode)`

**BEGIN**

  // Initialize accumulators
  INITIALIZE `total_loss` = 0.0
  INITIALIZE `total_accuracy` = NULL
  INITIALIZE `num_samples` = 0

  SET `model` to evaluation mode (`model.eval()`)

  // Disable gradient calculation
  **BEGIN TORCH_NO_GRAD**

    // Iterate through evaluation dataset
    FOR EACH `data_batch` IN `dataset`
      // --- Model-Specific Evaluation Step (Assuming model is Far-Field SubspaceNet) ---
      IF `mode` IS "valid" THEN
        ASSIGN `step_loss`, `step_accuracy` = CALL `model.validation_step`(`data_batch`) // Calls __validation_step_far_field
      ELSE // mode is "test"
        ASSIGN `step_loss`, `step_accuracy` = CALL `model.test_step`(`data_batch`) // Assumed similar to validation_step
      ENDIF

      // --- Accumulate Overall Loss (Scalar) ---
      ADD `step_loss` TO `total_loss`

      // --- Accumulate Accuracy (if available) ---
      IF `step_accuracy` IS NOT NULL THEN
        IF `total_accuracy` IS NULL THEN // First time seeing accuracy
          INITIALIZE `total_accuracy` = 0.0
        ENDIF
        ADD `step_accuracy` TO `total_accuracy`
      ENDIF

      // --- Count Samples ---
      GET input tensor `x` FROM `data_batch`
      IF `x` IS 2D THEN INCREMENT `num_samples` by 1
      ELSE INCREMENT `num_samples` by `batch_size` of `x`
      ENDIF

    ENDFOR // Dataset loop

  **END TORCH_NO_GRAD**

  // --- Calculate Average Metrics ---
  AVERAGE `total_loss` = `total_loss` / `num_samples`
  IF `total_accuracy` IS NOT NULL THEN
    AVERAGE `total_accuracy` = `total_accuracy` / `num_samples`
  ENDIF

  // --- Construct Result Dictionary ---
  INITIALIZE `result` AS DICTIONARY with "Overall": AVERAGE `total_loss`
  IF AVERAGE `total_accuracy` IS NOT NULL THEN
    ADD "Accuracy": AVERAGE `total_accuracy` TO `result`
  ENDIF

  RETURN `result`

**END FUNCTION**

---

## File: `DCD_MUSIC/src/models_pack/subspacenet.py`

### FUNCTION `__validation_step_far_field(self, batch, batch_idx)`

**BEGIN**

  // --- Data Preparation ---
  UNPACK `batch` INTO `x`, `sources_num`, `angles`
  IF `x` IS 2D THEN ADD batch dimension to `x` ENDIF
  // MOVE data to device (Implementation detail)

  // --- Source Number Check ---
  IF `sources_num` elements are not all equal THEN
    RAISE ValueError("Number of sources must be constant within a batch for validation")
  ENDIF
  ASSIGN `M` = first element of `sources_num` // Assumed constant number of sources

  // --- Model Forward Pass ---
  // Calls self.forward(x, M) which internally:
  // 1. Computes surrogate covariance Rz
  // 2. Calls self.diff_method(Rz, M) (e.g., ESPRIT, RootMusic, MUSIC)
  ASSIGN `angles_pred`, `source_estimation`, `eigen_regularization` = CALL `self.forward`(`x`, `M`)

  // --- Loss Calculation ---
  // self.validation_loss is typically RMSPELoss for far-field validation
  ASSIGN `loss` = CALL `self.validation_loss`(`angles_pred`=\(`angles_pred`), `angles`=\(`angles`))

  // --- Accuracy Calculation ---
  ASSIGN `accuracy` = CALL `self.source_estimation_accuracy`(`M`, `source_estimation`)

  RETURN `loss`, `accuracy`

**END FUNCTION** 
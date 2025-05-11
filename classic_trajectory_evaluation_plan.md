# Plan: Classic Subspace Method Evaluation on Trajectory-Based Datasets

## 1. Goal

Implement an evaluation pipeline to assess the performance of classic subspace algorithms (e.g., MUSIC, ESPRIT, Root-MUSIC) on a **step-by-step basis** using trajectory-based datasets. This differs from traditional snapshot-based batch evaluation. **This new pipeline will be an addition to the existing evaluation of the main deep learning model, allowing for comparative analysis, not a replacement of it.**

## 2. Leveraging Existing DCD_MUSIC Components

The following components from the `DCD_MUSIC` project can be largely reused or adapted:

*   **Core Algorithm Implementations:**
    *   Located in `DCD_MUSIC/src/methods_pack/` (e.g., `music.py`, `esprit.py`, `root_music.py`).
    *   The primary processing methods (e.g., `__call__` or `forward`) that take a covariance matrix (`Rx`) and `number_of_sources` are key.
    *   **Caveat:** Their `test_step` methods are designed for snapshot batches and won't be used directly.
*   **Covariance Matrix Calculation:**
    *   `DCD_MUSIC.src.utils.calculate_covariance_tensor`: Can compute covariance matrices for each trajectory item at a given step from snapshot data `(N_traj_in_batch, num_antennas, num_snapshots_per_step)`.
*   **Loss Functions/Metrics:**
    *   `DCD_MUSIC.src.metrics.rmspe_loss.RMSPELoss`: Can be used to compare predicted DOAs from classic methods against ground truth DOAs for each step.
*   **Method Instantiation:**
    *   `DCD_MUSIC.src.evaluation.get_model_based_method`: Can still be used to create instances of the classic algorithm classes.

## 3. New Components and Adaptations Required

The following parts will need to be newly implemented or significantly adapted for trajectory-based evaluation:

*   **Trajectory-Based Evaluation Loop:**
    *   This is the core new component, likely to be implemented within `simulation/core.py` (e.g., by extending `_run_evaluation_pipeline` or creating a new helper).
    *   This loop will iterate through trajectories, then through each step of a trajectory.
*   **Step-wise Invocation of Classic Methods:**
    *   Logic to extract data for the current step.
    *   Convert step data to a covariance matrix for each trajectory item.
    *   Call the classic method with the covariance matrix and number of sources for that item/step.
*   **Batching at the Step Level:**
    *   Determine if classic methods can process a batch of covariance matrices derived from a single step across multiple trajectories (`(N_traj_in_batch, N_ant, N_ant)`).
    *   If not, an inner loop over trajectory items within a step will be necessary.
*   **`evaluate_model_based` from DCD_MUSIC:** This function is for snapshot-based dataloaders and **cannot** be used directly.

## 4. Phased Implementation Plan

### Phase 1: Configuration Setup (Similar to previous attempts)

*   **Update `config/schema.py`:**
    *   Add `subspace_methods: List[str]` to `EvaluationConfig` (if not already present and correctly defined).
*   **Update YAML Configuration File(s):**
    *   Specify the list of classic subspace methods to evaluate under the evaluation section (e.g., `simulation.evaluation.subspace_methods: ["1D-MUSIC", "Root-MUSIC"]`).

### Phase 2: Implement Trajectory-Based Evaluation Logic

*   **Modify `simulation/core.py` (e.g., `_run_evaluation_pipeline`):** This involves **extending** the existing method to include the evaluation of classic subspace methods **after** the primary deep learning model has been evaluated. The goal is to augment the results, not alter the existing DL model evaluation flow.
    *   **Imports:** Add necessary imports from `DCD_MUSIC` (e.g., `get_model_based_method`, `SystemModelParams`, `calculate_covariance_tensor`, `RMSPELoss`).
    *   **Outer Loop (Methods):** Iterate through each method name specified in `self.config.simulation.evaluation.subspace_methods`.
        *   Instantiate the classic method using `get_model_based_method`.
        *   Initialize accumulators for loss/metrics for the current method.
    *   **Middle Loop (Data Loader):** Iterate through the trajectory evaluation dataloader (e.g., `self.test_dataloader`).
        *   Extract `trajectories`, `sources_num_batch`, `labels_batch`.
    *   **Inner Loop (Steps):** Iterate through each `step_idx` in the trajectory length (`T_len`).
        *   Extract `step_snapshots` `(N_traj_in_batch, N_ant, N_snap_per_step)`.
        *   Extract `current_num_sources_at_step` and `current_labels_at_step`.
        *   **Covariance Calculation:** Compute `Rx_step_batch` `(N_traj_in_batch, N_ant, N_ant)` using `calculate_covariance_tensor`.
        *   **Innermost Loop (Trajectory Items - if needed):** Iterate `item_idx` from `0` to `N_traj_in_batch - 1`.
            *   Extract `Rx_item` `(1, N_ant, N_ant)` and `num_src_item`.
            *   **Classic Method Invocation:** Call `classic_method_instance(Rx_item, num_src_item)`. The exact call signature needs to match the DCD_MUSIC methods (usually `__call__`).
            *   Handle outputs (e.g., convert Root-MUSIC roots to angles if not done by the class).
            *   Get `true_labels_item_step`.
            *   **Loss Calculation:** Compute `step_loss` using `RMSPELoss(predicted_DOAs_item, true_labels_item_step)`.
            *   Accumulate `step_loss`.
    *   **Aggregate Results:** Calculate average loss for the method and store it.
    *   Handle errors and logging gracefully.

## 5. Conceptual Code Snippet for the New Loop (in `simulation/core.py`)

```python
# Conceptual structure within _run_evaluation_pipeline
# ...
# classic_eval_criterion = RMSPELoss().to(device)
# for method_name in self.config.simulation.evaluation.subspace_methods:
#     classic_method_instance = get_model_based_method(method_name, sm_params_for_classic).to(device).eval()
#     method_total_loss = 0.0
#     method_total_items = 0
#     with torch.no_grad():
#         for batch_data in tqdm(eval_dataloader, desc=f"Eval {method_name}"):
#             trajectories, sources_num_batch, labels_batch = batch_data
#             N_traj_in_batch, T_len = trajectories.shape[0], trajectories.shape[1]
#
#             for step_idx in range(T_len):
#                 step_snapshots = trajectories[:, step_idx]
#                 current_num_sources_at_step = sources_num_batch[:, step_idx]
#                 current_labels_at_step = labels_batch[:, step_idx]
#                 Rx_step_batch = calculate_covariance_tensor(step_snapshots)
#
#                 for item_idx in range(N_traj_in_batch):
#                     Rx_item = Rx_step_batch[item_idx].unsqueeze(0)
#                     num_src_item = current_num_sources_at_step[item_idx].item()
#                     
#                     # preds_item_step, _, _ = classic_method_instance(Rx_item, num_src_item) # Adapt this
#                     # true_labels_item_step = current_labels_at_step[item_idx, :num_src_item].unsqueeze(0)
#                     # step_loss = classic_eval_criterion(preds_item_step, true_labels_item_step)
#                     # method_total_loss += step_loss.item()
#                     # method_total_items += 1
#
#     # avg_method_loss = method_total_loss / method_total_items if method_total_items > 0 else 0.0
#     # classic_trajectory_results[method_name] = {"Overall_Loss": avg_method_loss}
# ...
```

## 6. Key Challenges and Considerations

*   **Classic Method API:** Ensuring the invocation `classic_method_instance(Rx, num_sources)` matches the exact signature and expected input/output format of DCD_MUSIC's method classes.
*   **Number of Sources (`num_src_item`):**
    *   Correctly passing the `num_src_item` for each trajectory at each step.
    *   Handling cases where a classic method might estimate a different number of sources (e.g., via AIC/MDL) than the ground truth. `RMSPELoss` behavior with mismatched counts needs to be understood.
*   **Root-MUSIC Output:** If the DCD_MUSIC `RootMusic` class does not internally convert roots to angles, this conversion step will be needed before calculating loss. (It is believed DCD_MUSIC's `RootMusic.__call__` handles this).
*   **Batching `Rx` for Classic Methods:** Investigate if DCD_MUSIC classic methods can process a batch of covariance matrices `Rx_step_batch` and a batch of `num_sources` effectively. If `num_sources` varies per item in the step-batch, a loop might be unavoidable or simplified by using a fixed `M` from `system_model_params`.
*   **Metrics:** Extend beyond simple "Overall\_Loss" to include other relevant metrics like RMSE per component, accuracy of source number estimation (if applicable), etc.
*   **Device Management:** Ensure all tensors and models are on the correct device (`cuda` or `cpu`).
*   **Error Handling:** Robust error handling for method instantiation, data processing, and individual method failures.

This plan provides a roadmap for integrating classic subspace method evaluations into the trajectory-based simulation framework. 
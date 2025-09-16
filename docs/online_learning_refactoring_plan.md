# Online Learning System Refactoring Plan

## Current Problems

The current implementation has several issues that create overhead and confusion:

1. **Inconsistent Loss Calculation**: Loss is calculated in multiple places:
   - `_process_single_step` method (for evaluation metrics)
   - `_calculate_window_training_loss` (for training)
   - `_calculate_metrics` method (for final results)
   - Outside `_calculate_metrics` for Multi-Moment loss

2. **Mixed Responsibilities**: 
   - `_process_single_step` both collects data AND calculates losses
   - Metrics are gathered both inside and outside dedicated functions
   - Loss calculation logic is scattered across multiple methods

3. **Redundant Processing**: 
   - Same data is processed multiple times for different purposes
   - Loss calculations happen at both step-level and window-level inconsistently

## Proposed Clean Architecture

### Core Principle
**Separation of Concerns**: Data collection, loss calculation, and metrics computation should be clearly separated and happen at the appropriate level.

### 1. Data Collection Phase
- **`_process_single_step`**: Only collects raw measurements and Kalman filter outputs
- **No loss calculations** in this method
- Returns all necessary data for downstream processing
- Kalman filter returns all measures needed for monitoring and loss calculation

### 2. Loss Calculation Phase
- **Training Loss**: Calculated only during online learning Phase 1 (training)
- **Evaluation Loss**: Calculated only in `_calculate_metrics` method
- **Window-level**: All losses calculated are at window level (RMSPE and RMAPE are batch-compatible)

### 3. Metrics Computation Phase
- **`_calculate_metrics`**: Responsible for ALL metrics calculation
- Encapsulates and documents all metrics properly
- Returns comprehensive results

## Detailed Refactoring Plan

### Phase 1: Clean `_process_single_step`

**Current Issues:**
```python
# Current: Mixed data collection and loss calculation
def _process_single_step(self, ...):
    # ... data collection ...
    
    # Loss calculation (WRONG PLACE)
    if loss_config is not None:
        loss = self._calculate_configured_loss(...)
    else:
        loss = rmspe_criterion(...)
    
    # More loss calculations
    step_delta_predictions_rmspe = rmspe_criterion(...)
    step_delta_predictions_rmape = rmape_criterion(...)
    
    return step_results
```

**Target State:**
```python
def _process_single_step(self, ...):
    # ONLY data collection
    # ... model prediction ...
    # ... EKF processing ...
    
    # Return raw data only
    return {
        'success': True,
        'pre_ekf_angles_pred_tensor': pre_ekf_angles_pred,
        'ekf_angles_pred_tensor': ekf_angles_pred,
        'true_angles_tensor': true_angles_tensor,
        'step_predictions': step_predictions,
        'step_covariances': step_covariances,
        'step_innovations': step_innovations,
        'step_kalman_gains': step_kalman_gains,
        'step_kalman_gain_times_innovation': step_kalman_gain_times_innovation,
        'step_y_s_inv_y': step_y_s_inv_y,
        'step_Innovation_Covariance' : step_Innovation_Covariance,
        'num_sources': num_sources_this_step
    }
```

### Phase 2: Centralize Loss Calculation in `_calculate_metrics`

**Current Issues:**
- Loss calculation scattered across multiple methods
- Multi-Moment loss calculated outside `_calculate_metrics`
- Inconsistent loss types and calculations

**Target State:**
```python
def _calculate_metrics(self, step_results_list, ...):
    # Collect all tensors for window-level loss calculation
    all_pre_ekf_preds = []
    all_ekf_preds = []
    all_true_angles = []
    
    for step_result in step_results_list:
        if step_result['success']:
            all_pre_ekf_preds.append(step_result['pre_ekf_angles_pred_tensor'])
            all_ekf_preds.append(step_result['ekf_angles_pred_tensor'])
            all_true_angles.append(step_result['true_angles_tensor'])
    
    # Stack for window-level calculation
    window_pre_ekf_preds = torch.cat(all_pre_ekf_preds, dim=0)
    window_ekf_preds = torch.cat(all_ekf_preds, dim=0)
    window_true_angles = torch.cat(all_true_angles, dim=0)
    
    # Calculate ALL losses at window level
    loss_metrics = self._calculate_all_losses(
        window_pre_ekf_preds, window_ekf_preds, window_true_angles, loss_config
    )
    
    # Calculate other metrics...
    return WindowEvaluationResult(...)
```

### Phase 3: Create Unified Loss Calculation Method

**New Method:**
```python
def _calculate_all_losses(self, pre_ekf_preds, ekf_preds, true_angles, loss_config):
    """
    Calculate all losses at window level.
    
    Returns:
        LossMetrics object with all calculated losses
    """
    # Main configured loss (what the system is optimized for)
    main_loss = self._calculate_configured_loss(ekf_preds, true_angles, loss_config)
    
    # Supervised loss (always calculated for monitoring)
    supervised_loss = rmspe_criterion(ekf_preds, true_angles)
    
    # Pre-EKF loss (raw model performance)
    pre_ekf_loss = rmspe_criterion(pre_ekf_preds, true_angles)
    
    # EKF gain losses (EKF improvement over raw predictions)
    ekf_gain_rmspe = rmspe_criterion(ekf_preds, pre_ekf_preds)
    ekf_gain_rmape = rmape_criterion(ekf_preds, pre_ekf_preds)
    
    # Multi-Moment loss (if configured)
    multimoment_loss = 0.0
    if loss_config and loss_config.training_loss_type == "multimoment":
        multimoment_loss = self._calculate_multimoment_loss(pre_ekf_preds, ekf_preds, loss_config)
    
    return LossMetrics(
        main_loss=main_loss,
        supervised_loss=supervised_loss,
        pre_ekf_loss=pre_ekf_loss,
        ekf_gain_rmspe=ekf_gain_rmspe,
        ekf_gain_rmape=ekf_gain_rmape,
        multimoment_loss=multimoment_loss
    )
```

### Phase 4: Simplify Training Flow

**Current Issues:**
- Training loss calculated in `_calculate_window_training_loss`
- Evaluation loss calculated separately
- Duplicate processing

**Target State:**
```python
def _online_training_window(self, ...):
    # Phase 1: Training (keep existing training logic)
    for gd_step in range(num_gd_steps):
        # ... existing training logic ...

    # Phase 2: Evaluation (use same flow as _evaluate_window)
    # Process window for evaluation (same as _evaluate_window)
    step_results_list = []
    for step in range(current_window_len):
        success, step_result = self._process_single_step(...)
        step_results_list.append(step_result)
    
    # Calculate metrics using unified method
    result = self._calculate_metrics(step_results_list, ...)
    return result
```

### Phase 5: Update Data Structures

**Enhanced LossMetrics:**
```python
@dataclass
class LossMetrics:
    """Encapsulates all loss-related metrics for a window evaluation."""
    main_loss: float              # Primary loss (configured)
    supervised_loss: float        # Always calculated RMSPE vs true angles
    pre_ekf_loss: float          # Raw model performance
    ekf_gain_rmspe: float        # EKF improvement (RMSPE)
    ekf_gain_rmape: float        # EKF improvement (RMAPE)
    multimoment_loss: float      # Multi-Moment loss (if applicable)
    
    # Legacy compatibility
    @property
    def ekf_loss(self) -> float:
        return self.main_loss
    
    @property
    def delta_loss(self) -> float:
        return self.ekf_gain_rmspe
    
    @property
    def delta_rmspe_loss(self) -> float:
        return self.ekf_gain_rmspe
    
    @property
    def delta_rmape_loss(self) -> float:
        return self.ekf_gain_rmape
```

## Implementation Steps

### Step 1: Remove Loss Calculations from `_process_single_step`
- Remove all loss calculation code
- Keep only data collection
- Update return structure

### Step 2: Create `_calculate_all_losses` Method
- Implement unified loss calculation
- Handle all loss types in one place
- Use window-level batch operations

### Step 3: Update `_calculate_metrics`
- Call `_calculate_all_losses` for all loss calculations
- Remove duplicate loss calculations
- Ensure comprehensive metrics

### Step 4: Simplify Training Flow
- Separate training loss calculation from evaluation
- Use same evaluation flow for both `_evaluate_window` and `_online_training_window`
- Remove duplicate processing

### Step 5: Update Configuration and Documentation
- Document new loss calculation flow
- Update configuration schema if needed
- Add comprehensive logging

## Benefits

1. **Clear Separation of Concerns**: Data collection, loss calculation, and metrics computation are clearly separated
2. **Reduced Overhead**: No duplicate processing or calculations
3. **Consistent Loss Calculation**: All losses calculated at appropriate level
4. **Better Maintainability**: Single source of truth for each type of calculation
5. **Comprehensive Metrics**: All metrics calculated and documented in one place
6. **Flexible Configuration**: Easy to add new loss types or modify existing ones

## Migration Strategy

1. **Backward Compatibility**: Maintain existing interfaces during transition
2. **Gradual Migration**: Update one method at a time
3. **Testing**: Verify each step maintains functionality
4. **Documentation**: Update all relevant documentation

This refactoring will result in a clean, maintainable, and efficient online learning system with clear separation of concerns and comprehensive metrics calculation.

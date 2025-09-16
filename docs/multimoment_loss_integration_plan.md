# Multi-Moment Innovation Consistency Loss Integration Plan

## Overview
Integrate the new `MultiMomentInnovationConsistencyLoss` into the online learning system to calculate loss over entire windows rather than individual steps, while maintaining compatibility with backpropagation and existing data collection flows.

## 1. Core Changes Required

### 1.1 Import and Initialize Loss Function
- **Location**: `Online_learning.py` imports section
- **Action**: Add import for `MultiMomentInnovationConsistencyLoss`
- **Code**:
```python
from DCD_MUSIC.src.metrics.multimoment_innovation_consistency_loss import MultiMomentInnovationConsistencyLoss
```

### 1.2 Loss Function Initialization
- **Location**: `_online_training_window` method (around line 875)
- **Action**: Initialize the new loss function alongside existing RMSPE/RMAPE criteria
- **Code**:
```python
# Initialize Multi-Moment Innovation Consistency Loss
multimoment_criterion = MultiMomentInnovationConsistencyLoss(
    alpha=1.0,  # Configurable via config
    beta=1.0,   # Configurable via config
    regularization_term=0.0  # Configurable via config
).to(device)
```

## 2. Window-Level Loss Calculation Architecture

### 2.1 Data Collection During Window Processing
- **Modify**: `_process_single_step` method to collect data instead of calculating loss
- **New Data Structure**: Collect pre-EKF predictions and EKF predictions for entire window
- **Storage**: Add to `step_results` dictionary:
```python
step_results = {
    # ... existing fields ...
    'pre_ekf_angles_pred_tensor': pre_ekf_angles_pred,  # Tensor for loss calculation
    'ekf_angles_pred_tensor': ekf_angles_pred,          # Tensor for loss calculation
    'true_angles_tensor': true_angles_tensor,           # Ground truth for loss calculation
}
```

### 2.2 Window-Level Loss Calculation Function
- **New Method**: `_calculate_window_multimoment_loss`
- **Purpose**: Calculate Multi-Moment loss over entire window
- **Input**: List of step results from window
- **Output**: Loss tensor compatible with backpropagation
- **Implementation**:
```python
def _calculate_window_multimoment_loss(self, step_results_list: List[Dict], 
                                     loss_config=None) -> torch.Tensor:
    """
    Calculate Multi-Moment Innovation Consistency Loss over entire window.
    
    Args:
        step_results_list: List of step result dictionaries from window
        loss_config: Loss configuration (optional)
    
    Returns:
        Window-level loss tensor
    """
    # Collect all predictions and targets from window
    all_pre_ekf_preds = []
    all_ekf_preds = []
    all_true_angles = []
    
    for step_result in step_results_list:
        if step_result['success']:
            all_pre_ekf_preds.append(step_result['pre_ekf_angles_pred_tensor'])
            all_ekf_preds.append(step_result['ekf_angles_pred_tensor'])
            all_true_angles.append(step_result['true_angles_tensor'])
    
    if not all_pre_ekf_preds:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Stack predictions across time steps
    window_pre_ekf_preds = torch.cat(all_pre_ekf_preds, dim=0)  # [window_size, num_sources]
    window_ekf_preds = torch.cat(all_ekf_preds, dim=0)          # [window_size, num_sources]
    window_true_angles = torch.cat(all_true_angles, dim=0)      # [window_size, num_sources]
    
    # Calculate Multi-Moment loss
    multimoment_criterion = MultiMomentInnovationConsistencyLoss().to(device)
    
    # Use pre-EKF predictions as "predictions" and EKF predictions as "targets"
    # This encourages consistency between model predictions and EKF-filtered predictions
    window_loss = multimoment_criterion(
        angles_pred=window_pre_ekf_preds,
        angles=window_ekf_preds,
        return_components=False
    )
    
    return window_loss
```

## 3. Integration Points

### 3.1 Online Training Window Method
- **Location**: `_online_training_window` method
- **Modification**: Replace individual step loss calculation with window-level loss
- **Changes**:
  1. Remove individual loss calculations in training loop
  2. Collect step results during training loop
  3. Calculate window-level loss after processing entire window
  4. Perform backpropagation on window-level loss

**Modified Training Loop Structure**:
```python
# Training phase: Run gradient descent steps per window
self.online_model.train()
total_training_loss = 0.0
num_training_steps = 0

for gd_step in range(num_gd_steps):
    self.online_optimizer.zero_grad()
    
    # Process entire window and collect results
    window_step_results = []
    for step in range(current_window_len):
        # ... existing step processing ...
        # Store results instead of calculating individual losses
        window_step_results.append(step_result)
    
    # Calculate window-level Multi-Moment loss
    window_loss = self._calculate_window_multimoment_loss(window_step_results, loss_config)
    
    # Backpropagation on window-level loss
    window_loss.backward()
    self.online_optimizer.step()
    
    total_training_loss += window_loss.item()
    num_training_steps += 1
```

### 3.2 Evaluation Window Method
- **Location**: `_evaluate_window` method
- **Modification**: Add window-level loss calculation for evaluation
- **Changes**:
  1. Collect step results during evaluation
  2. Calculate window-level Multi-Moment loss
  3. Add to metrics for tracking

**Modified Evaluation Structure**:
```python
# Process each step in window
step_results_list = []
for step in range(current_window_len):
    # ... existing step processing ...
    step_results_list.append(step_result)

# Calculate window-level Multi-Moment loss for evaluation
window_multimoment_loss = self._calculate_window_multimoment_loss(step_results_list, loss_config)

# Add to metrics
result = self._calculate_metrics(step_results_list, current_window_len, max_sources, current_eta, is_near_field)
result.window_multimoment_loss = window_multimoment_loss.item()
```

## 4. Configuration Integration

### 4.1 Config Schema Updates
- **Location**: `config/schema.py`
- **Action**: Add Multi-Moment loss configuration options
- **New Fields**:
```python
@dataclass
class LossConfig:
    # ... existing fields ...
    use_multimoment_loss: bool = False
    multimoment_alpha: float = 1.0
    multimoment_beta: float = 1.0
    multimoment_regularization_term: float = 0.0
```

### 4.2 Configuration Usage
- **Location**: Loss calculation methods
- **Action**: Use config values to initialize loss function
- **Code**:
```python
if loss_config and loss_config.use_multimoment_loss:
    multimoment_criterion = MultiMomentInnovationConsistencyLoss(
        alpha=loss_config.multimoment_alpha,
        beta=loss_config.multimoment_beta,
        regularization_term=loss_config.multimoment_regularization_term
    ).to(device)
```

## 5. Data Flow Preservation

### 5.1 Existing Metrics Collection
- **Preserve**: All existing metric collection in `_calculate_metrics`
- **Add**: Window-level Multi-Moment loss to metrics
- **Location**: `LossMetrics` dataclass
- **New Field**:
```python
@dataclass
class LossMetrics:
    # ... existing fields ...
    window_multimoment_loss: float = 0.0
```

### 5.2 Results Storage
- **Preserve**: All existing result storage and averaging
- **Add**: Multi-Moment loss to result dictionaries
- **Location**: Result dictionaries in `_run_single_trajectory_online_learning`
- **New Fields**:
```python
"window_multimoment_losses": window_multimoment_losses,
"online_multimoment_losses": online_multimoment_losses,
"training_multimoment_losses": training_multimoment_losses,
```

## 6. Backward Compatibility

### 6.1 Fallback Behavior
- **Default**: Use existing loss calculation if Multi-Moment loss not configured
- **Implementation**: Check config flag before using new loss
- **Code**:
```python
if loss_config and loss_config.use_multimoment_loss:
    # Use Multi-Moment loss
    window_loss = self._calculate_window_multimoment_loss(window_step_results, loss_config)
else:
    # Use existing loss calculation
    window_loss = self._calculate_existing_loss(window_step_results, loss_config)
```

### 6.2 Existing Loss Functions
- **Preserve**: All existing loss calculation methods
- **Maintain**: RMSPE, RMAPE, Kalman innovation, y*S^-1*y losses
- **Add**: Multi-Moment loss as additional option

## 7. Testing and Validation

### 7.1 Unit Tests
- **Test**: Window-level loss calculation
- **Test**: Backpropagation compatibility
- **Test**: Configuration integration
- **Test**: Data flow preservation

### 7.2 Integration Tests
- **Test**: Online training with Multi-Moment loss
- **Test**: Evaluation with Multi-Moment loss
- **Test**: Results averaging across trajectories
- **Test**: Comparison with existing loss functions

## 8. Implementation Order

1. **Phase 1**: Add imports and basic loss function initialization
2. **Phase 2**: Implement `_calculate_window_multimoment_loss` method
3. **Phase 3**: Modify `_process_single_step` to collect data for window-level loss
4. **Phase 4**: Update `_online_training_window` to use window-level loss
5. **Phase 5**: Update `_evaluate_window` to calculate window-level loss
6. **Phase 6**: Add configuration options
7. **Phase 7**: Update metrics collection and result storage
8. **Phase 8**: Add backward compatibility and fallback behavior
9. **Phase 9**: Testing and validation

## 9. Risk Mitigation

### 9.1 Data Flow Risks
- **Risk**: Breaking existing data collection
- **Mitigation**: Preserve all existing data structures and add new fields

### 9.2 Backpropagation Risks
- **Risk**: Gradient computation issues
- **Mitigation**: Ensure all tensors maintain gradients and test thoroughly

### 9.3 Performance Risks
- **Risk**: Increased computation time
- **Mitigation**: Profile performance and optimize if needed

### 9.4 Configuration Risks
- **Risk**: Breaking existing configurations
- **Mitigation**: Make new features optional with sensible defaults

## 10. Success Criteria

- [ ] Multi-Moment loss calculates correctly over entire windows
- [ ] Backpropagation works without gradient issues
- [ ] Existing data collection flows remain intact
- [ ] Configuration system supports new loss function
- [ ] Results averaging works across trajectories
- [ ] Backward compatibility maintained
- [ ] Performance impact is acceptable
- [ ] All existing tests pass
- [ ] New functionality is well-tested

## 11. Future Enhancements

- **Adaptive Parameters**: Make alpha and beta learnable parameters
- **Multiple Loss Combination**: Allow combining Multi-Moment with existing losses
- **Window Size Adaptation**: Adjust loss calculation based on window size
- **Source-Specific Weights**: Different weights for different sources

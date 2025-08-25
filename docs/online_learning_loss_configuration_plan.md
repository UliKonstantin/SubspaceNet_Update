# Online Learning Loss Configuration Plan

## Overview

This document outlines the plan to add configurable loss functions for online learning, allowing users to choose between different loss metrics (RMSPE/RMAPE) and supervision modes (supervised/unsupervised) for both training and evaluation phases.

## Current State Analysis

### Existing Loss Usage
- **`_online_training_window`**: Currently uses hardcoded RMAPE loss between model predictions and EKF predictions
- **`_evaluate_window`**: Currently uses hardcoded RMSPE loss for evaluation
- **Metrics**: All existing metrics are measured and stored regardless of loss choice

### Current Limitations
- Loss functions are hardcoded in both training and evaluation methods
- No flexibility to choose between different loss metrics
- No option to switch between supervised and unsupervised learning
- Inconsistent loss usage between training and evaluation phases

## Requirements

### New Loss Configuration Options
1. **Supervised RMSPE**: RMSPE between true angles and predicted angles
2. **Supervised RMAPE**: RMAPE between true angles and predicted angles  
3. **Unsupervised RMSPE**: RMSPE between pre-EKF angles and predicted angles
4. **Unsupervised RMAPE**: RMAPE between pre-EKF angles and predicted angles

### Configuration Structure
```yaml
online_learning:
  loss_config:
    metric: "rmspe" | "rmape"
    supervision: "supervised" | "unsupervised"
```

## Implementation Plan

### Phase 1: Configuration Schema Updates

#### Files to Modify
- **`config/schema.py`**

#### Changes Required
1. Add `OnlineLearningLossConfig` class with:
   - `metric`: Enum for "rmspe" or "rmape"
   - `supervision`: Enum for "supervised" or "unsupervised"
2. Update `OnlineLearningConfig` to include the new loss configuration
3. Add validation to ensure valid combinations

#### Code Structure
```python
class OnlineLearningLossConfig(BaseModel):
    metric: Literal["rmspe", "rmape"] = "rmape"
    supervision: Literal["supervised", "unsupervised"] = "unsupervised"

class OnlineLearningConfig(BaseModel):
    # ... existing fields ...
    loss_config: OnlineLearningLossConfig = OnlineLearningLossConfig()
```

### Phase 2: Core Logic Updates

#### Files to Modify
- **`simulation/runners/Online_learning.py`**

#### Method Changes

##### `_online_training_window`
- Parse `self.config.online_learning.loss_config`
- Replace hardcoded RMAPE loss calculation with configurable logic
- Implement loss calculation based on configuration:
  - **Supervised**: Compare predictions with ground truth angles
  - **Unsupervised**: Compare predictions with pre-EKF predictions (current behavior)

##### `_evaluate_window`
- Parse `self.config.online_learning.loss_config`
- Replace hardcoded RMSPE loss calculation with configurable logic
- Pass loss configuration to `_process_single_step`

##### `_process_single_step`
- Add loss configuration parameter
- Implement configurable loss calculation logic
- Support all four loss variants:
  - Supervised RMSPE
  - Supervised RMAPE
  - Unsupervised RMSPE
  - Unsupervised RMAPE

#### Loss Calculation Logic
```python
def _calculate_configured_loss(self, predictions, targets, loss_config):
    """
    Calculate loss based on configuration.
    
    Args:
        predictions: Model predictions
        targets: Ground truth or pre-EKF predictions based on supervision mode
        loss_config: Loss configuration object
    
    Returns:
        Calculated loss value
    """
    if loss_config.supervision == "supervised":
        # Use ground truth angles as targets
        target_angles = true_angles
    else:  # unsupervised
        # Use pre-EKF predictions as targets
        target_angles = pre_ekf_predictions
    
    if loss_config.metric == "rmspe":
        return rmspe_criterion(predictions, target_angles)
    else:  # rmape
        return rmape_criterion(predictions, target_angles)
```

### Phase 3: Configuration File Updates

#### Files to Modify
- **`configs/online_learning_config.yaml`**

#### Changes Required
1. Add new loss configuration section
2. Set default values for backward compatibility
3. Add documentation comments

#### Example Configuration
```yaml
online_learning:
  # ... existing configuration ...
  loss_config:
    metric: "rmape"           # Options: "rmspe", "rmape"
    supervision: "unsupervised"  # Options: "supervised", "unsupervised"
```

### Phase 4: Testing and Validation

#### Test Cases
1. **Backward Compatibility**: Ensure existing configurations work without changes
2. **Supervised RMSPE**: Test with ground truth comparison using RMSPE
3. **Supervised RMAPE**: Test with ground truth comparison using RMAPE
4. **Unsupervised RMSPE**: Test with pre-EKF comparison using RMSPE
5. **Unsupervised RMAPE**: Test with pre-EKF comparison using RMAPE (current behavior)
6. **Invalid Configuration**: Test error handling for invalid combinations

#### Validation Points
- Loss values are calculated correctly for each configuration
- All existing metrics continue to be measured and stored
- No regression in existing functionality
- Configuration validation works properly

## Design Decisions

### Backward Compatibility
- Default to current behavior: unsupervised RMAPE for training, RMSPE for evaluation
- Existing configurations without loss_config will use defaults
- No breaking changes to existing API

### Consistency
- Use same loss configuration for both training and evaluation phases
- Maintain consistency in loss calculation across all methods
- Use existing RMSPE/RMAPE implementations

### Error Handling
- Graceful fallback to defaults if configuration is invalid
- Clear error messages for invalid configurations
- Validation at configuration load time

### Performance Considerations
- Loss calculation should not significantly impact performance
- Reuse existing loss criterion instances where possible
- Minimize additional computation overhead

## Implementation Steps

### Step 1: Schema Updates
1. Add `OnlineLearningLossConfig` class to `config/schema.py`
2. Update `OnlineLearningConfig` to include loss configuration
3. Add validation logic

### Step 2: Core Logic Implementation
1. Modify `_online_training_window` to use configurable loss
2. Modify `_evaluate_window` to use configurable loss
3. Update `_process_single_step` to support loss configuration
4. Implement loss calculation helper method

### Step 3: Configuration Updates
1. Update `configs/online_learning_config.yaml` with new loss configuration
2. Set appropriate default values
3. Add documentation

### Step 4: Testing
1. Test all four loss configuration combinations
2. Verify backward compatibility
3. Test error handling
4. Validate performance impact

### Step 5: Documentation
1. Update user documentation
2. Add configuration examples
3. Document loss calculation differences

## Files Summary

### Modified Files
1. **`config/schema.py`** - Add loss configuration schema
2. **`simulation/runners/Online_learning.py`** - Implement configurable loss logic
3. **`configs/online_learning_config.yaml`** - Add default loss configuration

### Unchanged Files
1. **`config/factory.py`** - No changes needed
2. **`main.py`** - Configuration flows through existing paths
3. **`simulation/core.py`** - No changes needed

## Success Criteria

- [ ] All four loss configurations work correctly
- [ ] Backward compatibility maintained
- [ ] No regression in existing functionality
- [ ] Configuration validation works
- [ ] Performance impact is minimal
- [ ] Documentation is updated
- [ ] All tests pass

## Future Enhancements

### Potential Extensions
1. **Separate Training/Evaluation Loss**: Allow different loss configurations for training vs evaluation
2. **Custom Loss Functions**: Support for user-defined loss functions
3. **Loss Weighting**: Support for weighted combinations of multiple loss functions
4. **Dynamic Loss Selection**: Automatic loss selection based on data characteristics

### Configuration Examples
```yaml
# Supervised learning with RMSPE
loss_config:
  metric: "rmspe"
  supervision: "supervised"

# Unsupervised learning with RMAPE (current default)
loss_config:
  metric: "rmape"
  supervision: "unsupervised"

# Supervised learning with RMAPE
loss_config:
  metric: "rmape"
  supervision: "supervised"
```

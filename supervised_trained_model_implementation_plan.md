# Supervised Trained Model Implementation Plan

## Overview
Add a supervised trained model that runs in parallel with the online learning model, using the same evaluation and training methods but with potentially different loss configurations.

## Tasks Breakdown

### 1. Core Implementation Tasks

#### 1.1 Create Supervised Trained Model
- **Location**: `simulation/runners/Online_learning.py` - `run_single_trajectory` method
- **Task**: Add `supervised_trained_model` as a deepcopy of `trained_model`
- **Implementation**: Similar to how `online_model` is created
```python
supervised_trained_model = copy.deepcopy(trained_model)
```

#### 1.2 Modify Evaluation Methods
- **Location**: `simulation/runners/Online_learning.py`
- **Task**: Update `evaluate_window` and `online_training_window` methods
- **Implementation**: Add optional parameter for supervised model
- **Approach**: Add parameter like `supervised_model=None` to existing methods

#### 1.3 Parallel Model Evaluation
- **Location**: `simulation/runners/Online_learning.py` - `run_single_trajectory` method
- **Task**: Call evaluation methods for supervised model after online model
- **Implementation**: 
  - After each `evaluate_window(online_model, ...)` call
  - Add `evaluate_window(supervised_trained_model, ...)`
  - After each `online_training_window(online_model, ...)` call  
  - Add `online_training_window(supervised_trained_model, ...)`

#### 1.4 Result Storage and Logging
- **Task**: Create supervised model result storage similar to online model
- **Implementation**:
  - Add supervised model result lists (similar to online model lists)
  - Log supervised model performance
  - Store supervised model metrics

#### 1.5 Return Results
- **Task**: Return supervised model results from `run_single_trajectory`
- **Implementation**: Add supervised results to return dictionary

### 2. Schema Enhancement Tasks (Current Implementation)

#### 2.1 Add supervised_loss_type Parameter
- **Location**: `config/schema.py`
- **Task**: Add `supervised_loss_type` to `loss_config` schema
- **Implementation**: Same validation as `training_loss_type`
- **Options**: `["configured", "unsupervised_rmspe", "unsupervised_rmape", "supervised_rmspe", "supervised_rmape", "multimoment", "kalman_innovation", "y_s_inv_y"]`

#### 2.2 Update Configuration File
- **Location**: `configs/Used_for_paper/SineAccel_base_model_Online_learning_snr_sweep_config.yaml`
- **Task**: Add `supervised_loss_type` parameter
- **Default**: Use same as `training_loss_type` initially

### 3. Integration Tasks

#### 3.1 Loss Configuration Integration
- **Task**: Use `supervised_loss_type` for supervised model training
- **Implementation**: Pass supervised loss config to training methods

#### 3.2 Result Structure Integration
- **Task**: Ensure supervised results integrate with existing result structure
- **Implementation**: Add supervised results to trajectory results and averaged results

### 4. Future Tasks (Not in Current Implementation)

#### 4.1 Plotting Integration
- **Location**: `utils/plotting.py`
- **Task**: Add supervised model to all plotting functions. the supervised model label is supervised trained model 
- **Functions to Update**:
  - `plot_scenario_results`
  - `plot_online_learning_results_structured` 
  - `plot_averaged_online_learning_results`
  - `plot_performance_improvement_table`

#### 4.2 Averaging Integration
- **Location**: `utils/utils.py`
- **Task**: Add supervised model to averaging functions
- **Function**: `average_online_learning_results_across_trajectories`

## Implementation Order

### Phase 1: Core Model Implementation
1. Create supervised trained model in `run_single_trajectory`
2. Modify evaluation methods to accept supervised model parameter
3. Add parallel evaluation calls
4. Implement result storage and logging
5. Return supervised results

### Phase 2: Schema and Configuration
1. Add `supervised_loss_type` to schema
2. Update configuration file
3. Integrate loss configuration with supervised model

### Phase 3: Testing and Validation
1. Test supervised model creation and evaluation
2. Verify logging and result storage
3. Validate configuration integration

## Key Implementation Details

### Model Creation
```python
# In run_single_trajectory method
supervised_trained_model = copy.deepcopy(trained_model)
supervised_trained_model.train()  # Set to training mode
```

### Evaluation Pattern
```python
# After online model evaluation
online_results = evaluate_window(online_model, ...)
supervised_results = evaluate_window(supervised_trained_model, ...)

# After online model training
online_training_results = online_training_window(online_model, ...)  
supervised_training_results = online_training_window(supervised_trained_model, ...)
```

### Result Storage Pattern
```python
# Similar to online model results
supervised_window_losses = []
supervised_window_covariances = []
supervised_pre_ekf_losses = []
# ... other supervised result lists
```

### Return Structure
```python
return {
    "online_learning_results": {
        "pretrained_model_trajectory_results": trajectory_results,
        "online_model_trajectory_results": online_trajectory_results,
        "supervised_model_trajectory_results": supervised_trajectory_results,  # NEW
        # ... other results
    }
}
```

## Files to Modify

### Primary Files
1. `simulation/runners/Online_learning.py` - Main implementation
2. `config/schema.py` - Schema updates
3. `configs/Used_for_paper/SineAccel_base_model_Online_learning_snr_sweep_config.yaml` - Config updates

### Secondary Files (Future)
1. `utils/plotting.py` - Plotting integration
2. `utils/utils.py` - Averaging integration

## Success Criteria
- [ ] Supervised trained model created successfully
- [ ] Supervised model evaluates in parallel with online model
- [ ] Supervised model results logged and stored
- [ ] Supervised model results returned properly
- [ ] Schema supports supervised_loss_type parameter
- [ ] Configuration file updated with supervised_loss_type
- [ ] No regression in existing online learning functionality

## Risk Mitigation
- Use deepcopy to ensure model independence
- Follow existing patterns for consistency
- Add comprehensive logging for debugging
- Maintain backward compatibility with existing configurations

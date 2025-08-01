# Dual Model Online Learning Implementation Plan

## Overview

This document outlines the implementation plan for enhancing the online learning functionality with a dual model architecture. The goal is to compare the performance of a static pre-trained model against an adaptive model that learns during operation, particularly when calibration parameters (eta) change.

## Current State

- **Single Model Pipeline**: `_evaluate_window(self.trained_model)` processes each window
- **EKF Integration**: Extended Kalman Filter tracks model predictions and provides filtering
- **Eta Calibration**: Parameter can be updated every few windows
- **Metrics Logging**: Track model performance and EKF behavior
- **Drift Detection**: Logic exists but doesn't trigger model adaptation

## Proposed Architecture

### Dual Model Setup

1. **Static Model**: `self.trained_model`
   - Remains unchanged throughout simulation
   - Baseline performance reference
   
2. **Adaptive Model**: `self.online_model`
   - Starts as deep copy of trained model
   - Adapts during simulation when drift is detected

### Three-Phase System

#### Phase 1: Pre-Drift
- **State**: `drift_detected = False`
- **Behavior**: Only static model active, online model dormant
- **Processing**: Standard `_evaluate_window(self.trained_model)`

#### Phase 2: Learning Phase
- **State**: `drift_detected = True`, `learning_done = False`
- **Behavior**: Static model continues + online model training
- **Processing**: 
  - `_evaluate_window(self.trained_model)` (unchanged)
  - `_online_training_window(self.online_model)` (new)

#### Phase 3: Comparison Phase
- **State**: `drift_detected = True`, `learning_done = True`
- **Behavior**: Both models evaluated normally
- **Processing**:
  - `_evaluate_window(self.trained_model)` (unchanged)
  - `_evaluate_window(self.online_model)` (comparative evaluation)

## Processing Logic Flow

```python
For each window:
    # Always evaluate static model and update drift state
    results = _evaluate_window(self.trained_model, window_data)
    self.drift_detected = current_window_loss > loss_threshold
    
    if self.drift_detected == True:
        if self.learning_done == True:
            # Online model finished training, evaluate normally
            online_results = _evaluate_window(self.online_model, window_data)
        else:
            # Online model still learning/adapting
            _online_training_window(self.online_model, window_data)
            # This function may eventually set self.learning_done = True
```

## Class State Variables

### New Variables to Add

- `self.drift_detected`: Boolean flag updated from `_evaluate_window()` results
- `self.learning_done`: Boolean flag indicating online model training completion
- `self.online_model`: Deep copy of trained model for adaptive learning

### State Management

- `self.drift_detected` updated automatically from window evaluation results
- `self.learning_done` managed by the `_online_training_window()` function
- Both flags control the three-phase processing logic

## Implementation Execution Plan

### Step 1: Add Class Variables
- Add `self.drift_detected = False` to `__init__`
- Add `self.learning_done = False` to `__init__`
- Add `self.online_model = None` to `__init__`

### Step 2: Initialize Online Model
- In `_run_single_trajectory_online_learning()`, create deep copy:
  ```python
  self.online_model = copy.deepcopy(self.trained_model)
  ```

### Step 3: Extract Drift Detection
- In main window loop, after `_evaluate_window()` call
- Add: `self.drift_detected = current_window_loss > loss_threshold`

### Step 4: Add Conditional Logic
- After drift detection, add:
  ```python
  if self.drift_detected:
      if self.learning_done:
          online_results = self._evaluate_window(self.online_model, ...)
      else:
          self._online_training_window(self.online_model, window_data)
  ```

### Step 5: Create Placeholder Function
- Add empty `_online_training_window()` method with pass statement
- Method signature: `_online_training_window(self, model, window_data)`

### Step 6: Update Result Logging
- Modify result storage to handle both models when available
- Add online model metrics to return dictionary
- Ensure comparative data is captured for analysis

### Step 7: Test Implementation
- Verify dual model initialization works correctly
- Confirm drift detection triggers at appropriate thresholds
- Check logging includes both models post-drift
- Validate three-phase transitions

## Expected Research Outcomes

### Performance Comparisons
- Static model degradation vs adaptive model recovery post-drift
- Computational overhead of dual-model approach
- Timing analysis of when adaptive model outperforms static model

### Calibration Analysis
- Response to eta parameter changes
- Drift detection sensitivity under different conditions
- EKF behavior with both model types

### Metrics to Track
- **Both Models**: Loss, covariance, predictions, innovations
- **Comparative**: Performance delta, convergence time, stability
- **System**: Computational overhead, memory usage

## Key Design Principles

1. **No Modification to `_evaluate_window()`**: Function remains unchanged
2. **Clean State Management**: Clear separation of phases via class variables
3. **Backward Compatibility**: Existing functionality unaffected
4. **Comparative Logging**: Rich data for research analysis
5. **Conditional Processing**: Online model only active when needed

## Future Extensions

- Adaptive learning rate based on drift severity
- Multiple online models with different architectures
- Ensemble methods combining static and adaptive predictions
- Real-time performance monitoring and model selection

## File Locations

- **Implementation**: `simulation/runners/Online_learning.py`
- **Main Function**: `_run_single_trajectory_online_learning()`
- **New Method**: `_online_training_window()`
- **Configuration**: Online learning config parameters

---

*Document created for dual model online learning implementation*
*Total implementation complexity: ~7 focused steps* 
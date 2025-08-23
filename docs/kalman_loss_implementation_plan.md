# Kalman Loss Implementation Plan

## Objective
Replace RMSPE loss with Kalman gain × innovation loss for online learning model training only (keep RMSPE for static model).

## Current State
- **Current loss**: RMSPE (Root Mean Square Percentage Error)
- **Target loss**: Kalman gain × innovation (K × ν)
- **Scope**: Online learning model only (`_online_training_window`)
- **Keep unchanged**: Static model evaluation (`_evaluate_window`)

## Implementation Plan

### Phase 1: Loss Function Development
- [ ] Create custom Kalman loss function/class
- [ ] Define loss as: `loss = kalman_gain * innovation`
- [ ] Handle multiple sources aggregation (sum/mean strategy)
- [ ] Ensure differentiability for backpropagation
- [ ] Add numerical stability checks (zero innovation handling)

### Phase 2: Training Loop Restructuring
- [ ] Modify `_online_training_window` function (lines ~661-720)
- [ ] Move EKF calculations before loss computation in training loop
- [ ] Replace `rmspe_criterion(angles_pred_tensor, true_angles_tensor)` calls
- [ ] Integrate EKF step outputs (kalman_gain, innovation) into gradient computation
- [ ] Ensure gradients flow through Kalman-derived loss

### Phase 3: EKF Integration
- [ ] Initialize EKF filters before training iterations
- [ ] Manage EKF state across gradient descent steps within same window
- [ ] Handle EKF state reset/continuation between windows
- [ ] Use same EKF configuration as evaluation phase

### Phase 4: Loss Aggregation Strategy
- [ ] Define per-step aggregation: sum/average K×ν across sources
- [ ] Define per-window aggregation: accumulate across all steps
- [ ] Implement gradient descent using aggregated Kalman loss
- [ ] Validate loss magnitude compatibility with current learning rates

### Phase 5: Backward Compatibility
- [ ] Keep existing RMSPE evaluation metrics for comparison
- [ ] Maintain all existing plots and logging functionality
- [ ] Add new logging for Kalman-based loss values
- [ ] Ensure no changes to static model evaluation

### Phase 6: Testing & Validation
- [ ] Compare training convergence: RMSPE vs Kalman loss
- [ ] Verify gradients are computed correctly
- [ ] Confirm online model weights change during training
- [ ] Test numerical stability and convergence
- [ ] Validate EKF state management across windows

### Phase 7: Logging & Debugging
- [ ] Add detailed logging for Kalman loss components
- [ ] Log innovation and Kalman gain values during training
- [ ] Compare training loss vs evaluation loss values
- [ ] Add debug output for EKF state transitions

## Technical Decisions

### Loss Computation Details
- **Aggregation method**: TBD (sum vs mean across sources)
- **Loss direction**: Minimize K×ν (assuming higher innovation = worse prediction)
- **Normalization**: Consider normalizing by number of sources/steps
- **Sign handling**: Ensure consistent loss direction for optimization

### Implementation Questions
- [ ] Should we normalize Kalman loss by number of sources?
- [ ] Do we need separate EKF config for training vs evaluation?
- [ ] Should learning rate be adjusted for different loss scale?
- [ ] How to determine online training convergence criteria?
- [ ] Should we use absolute value of innovation or raw value?

## Files to Modify
- **Primary**: `simulation/runners/Online_learning.py`
  - Function: `_online_training_window` (lines ~569-924)
  - Specific area: Gradient descent loop (lines ~661-720)
- **Secondary**: Potential EKF configuration adjustments if needed

## Success Criteria
- [ ] Online model trains using Kalman gain × innovation loss
- [ ] Static model continues using RMSPE loss unchanged
- [ ] All existing functionality preserved (plots, logging, evaluation)
- [ ] Training convergence maintained or improved
- [ ] Gradient computation verified and stable

## Risk Mitigation
- [ ] Implement feature flag to switch between RMSPE and Kalman loss
- [ ] Add extensive logging for debugging
- [ ] Test on small examples first
- [ ] Verify gradients using gradient checking tools
- [ ] Compare results with baseline RMSPE implementation

# Y*S^-1*Y Loss for Online Learning

## Overview

The Y*S^-1*Y loss is a new loss criterion available for online learning in the SubspaceNet framework. This loss function encourages the model to produce predictions that result in smaller y*S^-1*y values, which indicates better measurement quality and filter performance in terms of normalized innovation squared.

## Mathematical Background

The y*S^-1*y metric represents the normalized innovation squared, where:
- `y` is the innovation (measurement residual)
- `S` is the innovation covariance matrix
- `y*S^-1*y` is the Mahalanobis distance of the innovation

Smaller values of y*S^-1*y indicate:
- Better measurement quality
- More accurate predictions
- Improved filter performance
- Reduced uncertainty in the state estimates

## Implementation

### Loss Criterion Class

The `YSInvYLoss` class is implemented in `simulation/runners/Online_learning.py`:

```python
class YSInvYLoss:
    """
    Loss function for y*S^-1*y metric.
    
    This loss encourages the model to produce predictions that result in
    smaller y*S^-1*y values, which indicates better measurement quality
    and filter performance in terms of normalized innovation squared.
    """
    
    def __init__(self):
        """Initialize the Y*S^-1*Y Loss."""
        pass
    
    def __call__(self, y_s_inv_y: torch.Tensor) -> torch.Tensor:
        """
        Calculate the absolute value of y*S^-1*y.
        
        Args:
            y_s_inv_y: Tensor of y*S^-1*y values from EKF
            
        Returns:
            Loss tensor (absolute value of y*S^-1*y)
        """
        return torch.abs(y_s_inv_y)
```

### Configuration Schema

The schema has been updated in `config/schema.py` to include the new loss type:

```python
class OnlineLearningLossConfig(BaseModel):
    """Configuration for online learning loss function."""
    metric: Literal["rmspe", "rmape"] = Field(default="rmape", description="Loss metric to use: 'rmspe' or 'rmape'")
    supervision: Literal["supervised", "unsupervised"] = Field(default="unsupervised", description="Supervision mode: 'supervised' (compare with ground truth) or 'unsupervised' (compare with pre-EKF predictions)")
    training_loss_type: Literal["configured", "kalman_innovation", "y_s_inv_y"] = Field(default="configured", description="Training loss type: 'configured' (use metric+supervision), 'kalman_innovation' (use K*innovation loss), or 'y_s_inv_y' (use y*S^-1*y loss)")
```

## Usage

### Configuration File

To use the y*S^-1*y loss for online learning, set the `training_loss_type` to `"y_s_inv_y"` in your configuration:

```yaml
online_learning:
  enabled: true
  # ... other parameters ...
  
  loss_config:
    metric: "rmspe"  # Used for evaluation, not training
    supervision: "supervised"  # Used for evaluation, not training
    training_loss_type: "y_s_inv_y"  # Use y*S^-1*y loss for training
```

### Example Configuration

A complete example configuration is provided in `configs/online_learning_y_s_inv_y_config.yaml`.

### Running with Y*S^-1*Y Loss

```bash
python3 main.py online_learning \
  -c configs/online_learning_y_s_inv_y_config.yaml \
  -m experiments/results/nonlinear_tracking_sineaccel_base_model/checkpoints/final_SubspaceNet_20250613_184942.pt \
  -o experiments/results/online_learning_y_s_inv_y_test
```

## Training Process

The training process varies depending on the loss type used:

### For y*S^-1*y Loss with Window-Averaged Backpropagation:

1. **Model Forward Pass**: The online model produces angle predictions for each step in the window
2. **EKF Processing**: Extended Kalman Filter processes the predictions and computes y*S^-1*y values for each step
3. **Loss Accumulation**: The YSInvYLoss criterion computes the loss for each step, but gradients are not computed yet
4. **Window Averaging**: Losses are averaged over the entire window to create a single window-averaged loss
5. **Gradient Computation**: Gradients are computed with respect to the model parameters using the averaged loss
6. **Parameter Update**: Model parameters are updated to minimize the window-averaged y*S^-1*y loss

### For Unsupervised RMAPE/RMSPE Loss:

1. **Model Forward Pass**: The online model produces angle predictions for each step in the window
2. **EKF Processing**: Extended Kalman Filter processes the predictions and produces filtered outputs
3. **Loss Accumulation**: The RMAPE or RMSPE criterion computes the loss between predictions and EKF outputs for each step
4. **Window Averaging**: Losses are averaged over the entire window to create a single window-averaged loss
5. **Gradient Computation**: Gradients are computed with respect to the model parameters using the averaged loss
6. **Parameter Update**: Model parameters are updated to minimize the window-averaged loss

This window-averaged approach provides more stable training by considering the entire window context rather than updating on individual steps.

## Advantages

- **Direct Filter Performance Optimization**: The loss directly optimizes for better EKF performance
- **Uncertainty-Aware**: Takes into account the innovation covariance, making it more robust than simple prediction errors
- **Theoretically Sound**: Based on well-established Kalman filter theory
- **Gradient-Preserving**: Maintains gradients through the EKF computation for proper backpropagation
- **Window-Averaged Training**: Uses window-averaged backpropagation for more stable training by considering the entire window context

## Window-Averaged Backpropagation

The online learning implementation now uses window-averaged backpropagation, which provides several benefits:

- **Stable Gradients**: By averaging losses over the entire window, gradients are more stable and less noisy
- **Context Awareness**: The model learns from the entire window context rather than individual steps
- **Better Convergence**: Window-averaged updates typically lead to better convergence properties
- **Reduced Variance**: Averaging reduces the variance in gradient estimates

The training process now:
1. Processes all steps in a window without computing gradients
2. Averages the losses across all steps in the window
3. Performs backpropagation on the averaged loss
4. Updates model parameters based on the window-averaged gradients

## Unsupervised Loss Types

The framework now supports two additional unsupervised loss types that don't require ground truth labels:

### Unsupervised RMAPE Loss
- **Type**: `unsupervised_rmape`
- **Description**: Uses RMAPE (Root Mean Absolute Percentage Error) between model predictions and EKF outputs
- **Use Case**: When you want to train the model to produce predictions that are close to the EKF-filtered outputs
- **Advantage**: No ground truth required, learns from the EKF's filtered estimates

### Unsupervised RMSPE Loss
- **Type**: `unsupervised_rmspe`
- **Description**: Uses RMSPE (Root Mean Square Percentage Error) between model predictions and EKF outputs
- **Use Case**: When you want to train the model to produce predictions that are close to the EKF-filtered outputs
- **Advantage**: No ground truth required, penalizes larger errors more heavily than RMAPE

Both unsupervised loss types are particularly useful when:
- Ground truth labels are not available during online learning
- You want the model to learn from the EKF's filtered estimates
- You want to maintain consistency between model predictions and EKF outputs

## Comparison with Other Loss Types

| Loss Type | Description | Use Case |
|-----------|-------------|----------|
| `configured` | Uses metric (RMSPE/RMAPE) with supervision mode | Standard supervised/unsupervised learning |
| `kalman_innovation` | Uses K*y (Kalman gain × innovation) | Optimize for measurement quality |
| `y_s_inv_y` | Uses y*S^-1*y (normalized innovation squared) | Optimize for filter performance and uncertainty |
| `unsupervised_rmape` | Uses RMAPE between predictions and EKF outputs | Unsupervised learning with RMAPE metric |
| `unsupervised_rmspe` | Uses RMSPE between predictions and EKF outputs | Unsupervised learning with RMSPE metric |

## Monitoring

When using the y*S^-1*y loss, you can monitor:
- Training loss values (should decrease over time)
- EKF performance metrics
- Innovation statistics
- Filter convergence behavior

The loss values are logged during training:
```
Training step 0, GD 0: Y*S^-1*Y Loss = 0.123456
```

## Notes

- The y*S^-1*y loss is only used during the training phase of online learning
- Evaluation metrics (RMSPE/RMAPE) are still computed for performance monitoring
- The loss requires gradients to flow through the EKF computation, which is handled automatically
- This loss type is particularly effective when you want to optimize for filter stability and performance rather than just prediction accuracy

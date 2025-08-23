"""
Kalman-based loss functions for online learning.
"""

import torch
from typing import List

# Device setup for evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KalmanInnovationLoss:
    """
    Custom loss function based on Kalman gain × innovation for online learning.
    
    Loss = mean(|kalman_gain * innovation|) across all sources and steps.
    Higher innovation indicates worse prediction, so we minimize this value.
    
    This loss is designed to be used instead of RMSPE for online model training,
    leveraging the Kalman filter's innovation (prediction error) and gain values
    to provide a more principled training signal based on estimation theory.
    """
    
    def __init__(self, reduction='mean'):
        """
        Initialize the Kalman Innovation Loss.
        
        Args:
            reduction (str): 'mean' or 'sum' for aggregating across sources.
                           'mean' is recommended for stable gradients across 
                           different numbers of sources.
        """
        self.reduction = reduction
        if reduction not in ['mean', 'sum']:
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction}")
    
    def __call__(self, kalman_gains: List, innovations: List) -> torch.Tensor:
        """
        Calculate Kalman loss from gains and innovations.
        
        The loss is computed as the aggregated magnitude of kalman_gain * innovation
        across all sources. This represents how much the Kalman filter needs to 
        correct the prediction based on the measurement innovation.
        
        Args:
            kalman_gains (List): List of Kalman gains for each source (tensors or floats).
                               Higher gain = less confidence in prediction.
            innovations (List): List of innovations (measurement - prediction) 
                              for each source (tensors or floats). Higher innovation = worse prediction.
            
        Returns:
            torch.Tensor: Scalar tensor representing the loss value.
                         Always positive, suitable for minimization.
                         
        Raises:
            ValueError: If kalman_gains and innovations have different lengths.
        """
        if len(kalman_gains) != len(innovations):
            raise ValueError(
                f"Kalman gains ({len(kalman_gains)}) and innovations "
                f"({len(innovations)}) must have same length"
            )
        
        # Handle edge case of no sources
        if len(kalman_gains) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Calculate K * innovation for each source
        # Use absolute value to ensure we're minimizing innovation magnitude
        k_times_innovation = []
        for k, innov in zip(kalman_gains, innovations):
            # Handle both tensor and scalar inputs
            if isinstance(k, torch.Tensor):
                k_tensor = k.to(device).float()
            else:
                k_tensor = torch.tensor(float(k), device=device, dtype=torch.float32, requires_grad=False)
            
            if isinstance(innov, torch.Tensor):
                innov_tensor = innov.to(device).float()
            else:
                innov_tensor = torch.tensor(float(innov), device=device, dtype=torch.float32, requires_grad=False)
            
            # Calculate |K * innovation| - this is the core Kalman loss
            k_innov = torch.abs(k_tensor * innov_tensor)
            k_times_innovation.append(k_innov)
        
        # Stack tensors and require gradients for the final loss
        if k_times_innovation:
            loss_tensor = torch.stack(k_times_innovation)
            # Make the final loss require gradients
            loss_tensor.requires_grad_(True)
        else:
            loss_tensor = torch.tensor([], device=device, dtype=torch.float32, requires_grad=True)
        
        # Aggregate according to reduction method
        if self.reduction == 'mean':
            return torch.mean(loss_tensor)
        elif self.reduction == 'sum':
            return torch.sum(loss_tensor)
        else:
            # This should never happen due to __init__ validation
            raise ValueError(f"Unknown reduction method: {self.reduction}")
    
    def to(self, device_target):
        """
        Move loss function to specified device (for compatibility with other PyTorch losses).
        
        Args:
            device_target: Target device (cpu/cuda)
            
        Returns:
            self: Returns self for method chaining compatibility
        """
        global device
        device = device_target
        return self
    
    def __repr__(self):
        return f"KalmanInnovationLoss(reduction='{self.reduction}')"

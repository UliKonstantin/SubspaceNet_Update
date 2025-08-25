"""
Sine acceleration non-linear state evolution model.

This module implements the sine acceleration model:
θ_{k+1} = θ_k + (ω0 + κ sin θ_k)T + η_k
"""

import numpy as np
import torch
import logging
from .base import StateEvolutionModel

logger = logging.getLogger("SubspaceNet.kalman_filter.models.sine_accel")

class SineAccelStateModel(StateEvolutionModel):
    """
    Implements the sine acceleration non-linear model:
    θ_{k+1} = θ_k + (ω0 + κ sin θ_k)T + η_k
    
    This model adds a sinusoidal acceleration component to the angle evolution,
    creating non-linear dynamics that depend on the current angle.
    """
    
    def __init__(self, omega0, kappa, noise_std, time_step=1.0, device=None):
        """
        Initialize the model.
        
        Args:
            omega0: Base angular velocity (rad/s)
            kappa: Sine acceleration coefficient (rad/s²)
            noise_std: Standard deviation of process noise (rad)
            time_step: Time step between measurements (s)
            device: Device for tensor operations (cuda/cpu)
        """
        # Set device for tensor operations
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Convert parameters to tensors with consistent dtype (float32 for training compatibility)
        if isinstance(omega0, torch.Tensor):
            self.omega0 = omega0.to(device=device, dtype=torch.float32)
        else:
            self.omega0 = torch.tensor(omega0, dtype=torch.float32, device=device)
            
        if isinstance(kappa, torch.Tensor):
            self.kappa = kappa.to(device=device, dtype=torch.float32)
        else:
            self.kappa = torch.tensor(kappa, dtype=torch.float32, device=device)
            
        if isinstance(noise_std, torch.Tensor):
            self.base_noise_variance = (noise_std.to(device=device, dtype=torch.float32)) ** 2
        else:
            self.base_noise_variance = torch.tensor(noise_std**2, dtype=torch.float32, device=device)
            
        if isinstance(time_step, torch.Tensor):
            self.time_step = time_step.to(device=device, dtype=torch.float32)
        else:
            self.time_step = torch.tensor(time_step, dtype=torch.float32, device=device)
        
        logger.debug(f"Created SineAccelStateModel with ω₀={omega0}, κ={kappa}, σ={noise_std}, T={time_step}, device={device}")
    
    def f(self, x):
        """
        State transition: θ_{k+1} = θ_k + (ω0 + κ sin θ_k)T
        
        Args:
            x: Current angle (tensor or scalar)
            
        Returns:
            Predicted next angle (tensor)
        """
        # Convert to tensor if needed, ensuring proper dtype and device
        if isinstance(x, torch.Tensor):
            x_tensor = x.to(device=self.device, dtype=torch.float32)
        else:
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        # x is already in radians
        x_rad = x_tensor
        
        # Calculate acceleration term using tensor operations
        # Ensure all operations maintain gradients
        delta = (self.omega0 + self.kappa * torch.sin(x_rad)) * self.time_step
        
        # Apply state transition (simplified version without degree conversion)
        # Use direct radian calculation to maintain gradient flow
        result = x_rad + delta
        
        logger.debug(f"SineAccel state transition: {float(x) if hasattr(x, 'item') else x} -> {float(result) if hasattr(result, 'item') else result}")
        return result
    
    def F_jacobian(self, x):
        """
        Jacobian: ∂f/∂x = 1 + κT·cos(θ)
        
        Args:
            x: Current angle (tensor or scalar)
            
        Returns:
            Derivative of state transition with respect to state (tensor)
        """
        # Convert to tensor if needed, ensuring proper dtype and device
        if isinstance(x, torch.Tensor):
            x_tensor = x.to(device=self.device, dtype=torch.float32)
        else:
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        # x is already in radians
        x_rad = x_tensor
        
        # Calculate the derivative using tensor operations
        # Ensure all operations maintain gradients
        jacobian = torch.tensor(1.0, dtype=torch.float32, device=self.device) + self.kappa * torch.cos(x_rad) * self.time_step
        
        logger.debug(f"SineAccel Jacobian at x={float(x) if hasattr(x, 'item') else x}: {float(jacobian) if hasattr(jacobian, 'item') else jacobian}")
        return jacobian
    
    def noise_variance(self, x):
        """
        Process noise variance (constant for this model).
        
        Args:
            x: Current angle (tensor or scalar)
            
        Returns:
            Noise variance (tensor)
        """
        # Return constant noise variance as tensor
        # Ensure it has the same shape as input for broadcasting
        if isinstance(x, torch.Tensor):
            return torch.full_like(x, self.base_noise_variance, dtype=torch.float32, device=self.device)
        else:
            return self.base_noise_variance
    
    def f_batch(self, x_batch):
        """
        Batch version of state transition function.
        
        Args:
            x_batch: Tensor or array of current angles
            
        Returns:
            Tensor of predicted next angles
        """
        # Convert to tensor if needed, ensuring proper dtype and device
        if isinstance(x_batch, torch.Tensor):
            x_tensor = x_batch.to(device=self.device, dtype=torch.float32)
        else:
            x_tensor = torch.tensor(x_batch, dtype=torch.float32, device=self.device)
        
        # Apply state transition function element-wise using tensor operations
        # Ensure all operations maintain gradients
        delta = (self.omega0 + self.kappa * torch.sin(x_tensor)) * self.time_step
        result = x_tensor + delta
        
        logger.debug(f"SineAccel batch state transition for {x_tensor.numel()} states")
        return result
    
    def F_jacobian_batch(self, x_batch):
        """
        Batch version of Jacobian computation.
        
        Args:
            x_batch: Tensor or array of current angles
            
        Returns:
            Tensor of Jacobians
        """
        # Convert to tensor if needed, ensuring proper dtype and device
        if isinstance(x_batch, torch.Tensor):
            x_tensor = x_batch.to(device=self.device, dtype=torch.float32)
        else:
            x_tensor = torch.tensor(x_batch, dtype=torch.float32, device=self.device)
        
        # Calculate Jacobians element-wise using tensor operations
        # Ensure all operations maintain gradients
        jacobian = torch.tensor(1.0, dtype=torch.float32, device=self.device) + self.kappa * torch.cos(x_tensor) * self.time_step
        
        logger.debug(f"SineAccel batch Jacobian for {x_tensor.numel()} states")
        return jacobian
    
    def noise_variance_batch(self, x_batch):
        """
        Batch version of noise variance computation.
        
        Args:
            x_batch: Tensor or array of current angles
            
        Returns:
            Tensor of noise variances (constant for this model)
        """
        # Convert to tensor if needed, ensuring proper dtype and device
        if isinstance(x_batch, torch.Tensor):
            x_tensor = x_batch.to(device=self.device, dtype=torch.float32)
        else:
            x_tensor = torch.tensor(x_batch, dtype=torch.float32, device=self.device)
        
        # Return constant noise variance for all states using tensor operations
        return torch.full_like(x_tensor, self.base_noise_variance, dtype=torch.float32, device=self.device) 
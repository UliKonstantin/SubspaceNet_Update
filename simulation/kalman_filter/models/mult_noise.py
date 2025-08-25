"""
Multiplicative noise non-linear state evolution model.

This module implements the multiplicative noise model:
θ_{k+1} = θ_k + ω0 T + σ(θ_k) η_k
where σ(θ) = base_std * (1 + amp * sin²(θ))
"""

import numpy as np
import torch
import logging
from .base import StateEvolutionModel

logger = logging.getLogger("SubspaceNet.kalman_filter.models.mult_noise")

class MultNoiseStateModel(StateEvolutionModel):
    """
    Implements the multiplicative noise non-linear model:
    θ_{k+1} = θ_k + ω0 T + σ(θ_k) η_k
    where σ(θ) = base_std * (1 + amp * sin²(θ))
    
    This model has constant deterministic evolution but state-dependent
    noise amplitude, creating a different kind of non-linearity.
    """
    
    def __init__(self, omega0, amp, base_std, time_step=1.0, device=None):
        """
        Initialize the model.
        
        Args:
            omega0: Base angular velocity (rad/s)
            amp: Amplitude of multiplicative term (unitless)
            base_std: Base noise standard deviation (rad)
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
            
        if isinstance(amp, torch.Tensor):
            self.amp = amp.to(device=device, dtype=torch.float32)
        else:
            self.amp = torch.tensor(amp, dtype=torch.float32, device=device)
            
        if isinstance(base_std, torch.Tensor):
            self.base_std = base_std.to(device=device, dtype=torch.float32)
            self.base_noise_variance = base_std ** 2
        else:
            self.base_std = torch.tensor(base_std, dtype=torch.float32, device=device)
            self.base_noise_variance = torch.tensor(base_std**2, dtype=torch.float32, device=device)
            
        if isinstance(time_step, torch.Tensor):
            self.time_step = time_step.to(device=device, dtype=torch.float32)
        else:
            self.time_step = torch.tensor(time_step, dtype=torch.float32, device=device)
        
        logger.debug(f"Created MultNoiseStateModel with ω₀={omega0}, amp={amp}, σ={base_std}, T={time_step}, device={device}")
    
    def f(self, x):
        """
        Deterministic part: θ_{k+1} = θ_k + ω0 T
        
        Args:
            x: Current angle (tensor or scalar)
            
        Returns:
            Predicted next angle (tensor) without noise
        """
        # Convert to tensor if needed, ensuring proper dtype and device
        if isinstance(x, torch.Tensor):
            x_tensor = x.to(device=self.device, dtype=torch.float32)
        else:
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        # Calculate deterministic part (constant velocity model)
        # Ensure all operations maintain gradients
        result = x_tensor + self.omega0 * self.time_step
        
        logger.debug(f"MultNoise state transition: {float(x) if hasattr(x, 'item') else x} -> {float(result) if hasattr(result, 'item') else result}")
        return result
    
    def F_jacobian(self, x):
        """
        Jacobian: ∂f/∂x = 1 (constant velocity model)
        
        The deterministic part is linear, so the Jacobian is 1.
        The non-linearity comes from the state-dependent noise.
        
        Args:
            x: Current angle (tensor or scalar)
            
        Returns:
            Derivative of state transition with respect to state (always 1)
        """
        # For the constant velocity model, the Jacobian is always 1
        # Return as tensor with proper dtype and device
        if isinstance(x, torch.Tensor):
            return torch.tensor(1.0, dtype=torch.float32, device=self.device)
        else:
            return torch.tensor(1.0, dtype=torch.float32, device=self.device)
    
    def noise_variance(self, x):
        """
        State-dependent noise variance: σ²(θ) = base_std² * (1 + amp * sin²(θ))²
        
        Args:
            x: Current angle (tensor or scalar)
            
        Returns:
            Process noise variance at the given state (tensor)
        """
        # Convert to tensor if needed, ensuring proper dtype and device
        if isinstance(x, torch.Tensor):
            x_tensor = x.to(device=self.device, dtype=torch.float32)
        else:
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        # Calculate state-dependent standard deviation using tensor operations
        # Ensure all operations maintain gradients
        std = self.base_std * (torch.tensor(1.0, dtype=torch.float32, device=self.device) + self.amp * torch.sin(x_tensor)**2)
        
        # Return variance
        variance = std**2
        
        logger.debug(f"MultNoise variance at x={float(x) if hasattr(x, 'item') else x}: {float(variance) if hasattr(variance, 'item') else variance}")
        return variance
    
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
        result = x_tensor + self.omega0 * self.time_step
        
        logger.debug(f"MultNoise batch state transition for {x_tensor.numel()} states")
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
        
        # For the constant velocity model, the Jacobian is always 1
        # Return tensor of ones with same shape as input
        jacobian = torch.ones_like(x_tensor, dtype=torch.float32, device=self.device)
        
        logger.debug(f"MultNoise batch Jacobian for {x_tensor.numel()} states")
        return jacobian
    
    def noise_variance_batch(self, x_batch):
        """
        Batch version of noise variance computation.
        
        Args:
            x_batch: Tensor or array of current angles
            
        Returns:
            Tensor of noise variances
        """
        # Convert to tensor if needed, ensuring proper dtype and device
        if isinstance(x_batch, torch.Tensor):
            x_tensor = x_batch.to(device=self.device, dtype=torch.float32)
        else:
            x_tensor = torch.tensor(x_batch, dtype=torch.float32, device=self.device)
        
        # Calculate state-dependent standard deviation using tensor operations
        # Ensure all operations maintain gradients
        std = self.base_std * (torch.tensor(1.0, dtype=torch.float32, device=self.device) + self.amp * torch.sin(x_tensor)**2)
        
        # Return variance
        variance = std**2
        
        logger.debug(f"MultNoise batch variance for {x_tensor.numel()} states")
        return variance 
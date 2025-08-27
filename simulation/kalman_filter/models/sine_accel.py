"""
Sine acceleration non-linear state evolution model.

This module implements the sine acceleration model:
 θ_{k+1} = θ_k + κ sin(ω0 * t) + η_k
"""

import numpy as np
import torch
import logging
from .base import StateEvolutionModel

logger = logging.getLogger("SubspaceNet.kalman_filter.models.sine_accel")

class SineAccelStateModel(StateEvolutionModel):
    """
    Implements the oscillatory model:
    θ_{k+1} = θ_k + κ sin(ω0 * t) + η_k
    
    This model creates oscillatory behavior around the current angle instead of drifting
    in one direction. The oscillation is time-based rather than state-based.
    """
    
    def __init__(self, omega0, kappa, noise_std, time_step=1.0, device=None, initial_time=0.0):
        """
        Initialize the model.
        
        Args:
            omega0: Frequency of oscillation (rad/s) - can be single value or array per source
            kappa: Amplitude of oscillation (rad) - can be single value or array per source
            noise_std: Standard deviation of process noise (rad)
            time_step: Time step between measurements (s)
            device: Device for tensor operations (cuda/cpu)
            initial_time: Initial time value (default 0.0) - used to set the starting time
        """
        # Set device for tensor operations
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Convert parameters to tensors with consistent dtype (float32 for training compatibility)
        # Handle both single values and arrays
        if isinstance(omega0, (list, tuple, np.ndarray)):
            self.omega0 = torch.tensor(omega0, dtype=torch.float32, device=device)
            self.num_sources = len(omega0)
        else:
            self.omega0 = torch.tensor(omega0, dtype=torch.float32, device=device)
            self.num_sources = 1
            
        if isinstance(kappa, (list, tuple, np.ndarray)):
            self.kappa = torch.tensor(kappa, dtype=torch.float32, device=device)
            if len(kappa) != self.num_sources:
                raise ValueError(f"kappa array length ({len(kappa)}) must match omega0 length ({self.num_sources})")
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
        
        # Initialize time counter for oscillatory model with the provided initial time
        if isinstance(initial_time, torch.Tensor):
            self.current_time = initial_time.to(device=device, dtype=torch.float32)
        else:
            self.current_time = torch.tensor(initial_time, dtype=torch.float32, device=device)
        
        logger.debug(f"Created OscillatoryStateModel with {self.num_sources} sources:")
        if self.num_sources == 1:
            logger.debug(f"  ω₀={omega0}, κ={kappa}, σ={noise_std}, T={time_step}, initial_time={initial_time}, device={device}")
        else:
            for i in range(self.num_sources):
                logger.debug(f"  Source {i}: ω₀={self.omega0[i]}, κ={self.kappa[i]}, σ={noise_std}, T={time_step}, initial_time={initial_time}, device={device}")
    
    def f(self, x, source_idx=0):
        """
        State transition: θ_{k+1} = θ_k + κ sin(ω0 * t)
        
        Args:
            x: Current angle (tensor or scalar) - single source value
            source_idx: Index of the source (default 0) - used to select source-specific parameters
            
        Returns:
            Predicted next angle (tensor)
        """
        # Convert to tensor if needed, ensuring proper dtype and device
        if isinstance(x, torch.Tensor):
            x_tensor = x.to(device=self.device, dtype=torch.float32)
        else:
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        # Get source-specific parameters
        if self.num_sources == 1:
            # Single source case - use the single parameter values
            omega0_source = self.omega0
            kappa_source = self.kappa
        else:
            # Multiple sources case - use parameters for the specific source
            if source_idx >= self.num_sources:
                raise ValueError(f"Source index {source_idx} out of bounds for {self.num_sources} sources")
            omega0_source = self.omega0[source_idx]
            kappa_source = self.kappa[source_idx]
        
        # Calculate oscillatory term for this specific source: κ_source sin(ω0_source * t)
        oscillation = kappa_source * torch.sin(omega0_source * self.current_time)
        
        # Apply state transition: θ_{k+1} = θ_k + oscillation
        result_deg=0.99*(x_tensor*180/np.pi)+oscillation
        result = result_deg*np.pi/180
        
        logger.debug(f"Oscillatory state transition for source {source_idx}: {x_tensor} -> {result}")
        return result
    
    def F_jacobian(self, x, source_idx=0):
        """
        Jacobian: ∂f/∂x = 1 (simplified for oscillatory model)
        
        Args:
            x: Current angle (tensor or scalar)
            source_idx: Index of the source (default 0) - for consistency with f() method
            
        Returns:
            Jacobian matrix (tensor)
        """
        # For oscillatory model, the Jacobian is simply 1
        # since f(x) = x + κ sin(ω0 * t), so ∂f/∂x = 1
        
        if isinstance(x, torch.Tensor):
            x_tensor = x.to(device=self.device, dtype=torch.float32)
        else:
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        # Return identity matrix (scalar 1 for 1D case)
        return torch.ones_like(x_tensor)
    
    def noise_variance(self, x, source_idx=0):
        """
        Process noise variance (constant for this model).
        
        Args:
            x: Current angle (tensor or scalar)
            source_idx: Index of the source (default 0) - for consistency with f() method
            
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
            x_batch: Tensor or array of current angles - shape should be (batch_size, num_sources)
            
        Returns:
            Tensor of predicted next angles
        """
        # Convert to tensor if needed, ensuring proper dtype and device
        if isinstance(x_batch, torch.Tensor):
            x_tensor = x_batch.to(device=self.device, dtype=torch.float32)
        else:
            x_tensor = torch.tensor(x_batch, dtype=torch.float32, device=self.device)
        
        # Handle single source vs multiple sources
        if self.num_sources == 1:
            # Single source case - apply same parameters to all batch elements
            oscillation = self.kappa * torch.sin(self.omega0 * self.current_time)
            result = x_tensor + oscillation
        else:
            # Multiple sources case - apply source-specific parameters
            if x_tensor.shape[-1] != self.num_sources:
                # If we have multiple source parameters but batch doesn't match,
                # use the first source's parameters for all elements (for backward compatibility)
                if x_tensor.shape[-1] == 1:
                    oscillation = self.kappa[0] * torch.sin(self.omega0[0] * self.current_time)
                    result = x_tensor + oscillation
                else:
                    raise ValueError(f"Last dimension of batch ({x_tensor.shape[-1]}) must match number of sources ({self.num_sources})")
            else:
                # Calculate oscillatory term for each source: κ_i sin(ω0_i * t)
                # Expand to match batch dimensions
                oscillation = self.kappa * torch.sin(self.omega0 * self.current_time)
                oscillation = oscillation.expand(x_tensor.shape[0], -1)  # Expand to batch size
                
                # Apply state transition: θ_{k+1} = θ_k + oscillation
                result = x_tensor + oscillation
        
        logger.debug(f"Oscillatory batch state transition for {x_tensor.numel()} states")
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
        jacobian = torch.ones_like(x_tensor)
        
        logger.debug(f"Oscillatory batch Jacobian for {x_tensor.numel()} states")
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
    
    def advance_time(self):
        """
        Advance the time counter by one time step.
        This should be called after each prediction step.
        """
        self.current_time += self.time_step 
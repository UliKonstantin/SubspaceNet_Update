"""
Sine acceleration non-linear state evolution model.

This module implements the sine acceleration model:
 θ_{k+1} = damping * θ_k + (1-damping) * b + κ sin(ω0 * t) + η_k
"""

import numpy as np
import torch
import logging
from .base import StateEvolutionModel

logger = logging.getLogger("SubspaceNet.kalman_filter.models.sine_accel")

_DEG2RAD = np.pi / 180.0

class SineAccelStateModel(StateEvolutionModel):
    """
    Implements the oscillatory model:
    θ_{k+1} = damping * θ_k + (1-damping) * b + κ sin(ω0 * t) + η_k
    
    When dc_offset (b) is zero, oscillation converges around 0.
    """
    
    def __init__(self, omega0, kappa, noise_std, time_step=1.0, device=None, initial_time=1.0, damping=0.99,
                 angle_params_in_degrees=True, dc_offset=0.0):
        """
        Initialize the model.
        
        Args:
            omega0: Frequency of oscillation — same units as trajectory config (step index scale)
            kappa: Oscillation amplitude in degrees (matches trajectory_physics)
            noise_std: Process noise standard deviation in degrees
            time_step: Time step between measurements (s)
            device: Device for tensor operations (cuda/cpu)
            initial_time: Initial time index for sin(ω0·t); default 1.0 matches GT step indexing
            damping: Per-step damping on angle (default 0.99, matches trajectory_physics)
            angle_params_in_degrees: If True, κ and σ are in degrees (trajectory config).
                If False, σ is in the same units as filter state (radians); used for RW fallback.
            dc_offset: Per-source DC attractor (degrees if angle_params_in_degrees). Scalar or array.
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
            
        self.angle_params_in_degrees = angle_params_in_degrees
        noise_scale = _DEG2RAD if angle_params_in_degrees else 1.0
        if isinstance(noise_std, torch.Tensor):
            noise_std_state = noise_std.to(device=device, dtype=torch.float32) * noise_scale
            self.base_noise_variance = noise_std_state ** 2
        else:
            noise_std_state = float(noise_std) * noise_scale
            self.base_noise_variance = torch.tensor(noise_std_state ** 2, dtype=torch.float32, device=device)

        self.deg2rad = torch.tensor(_DEG2RAD, dtype=torch.float32, device=device)
            
        if isinstance(time_step, torch.Tensor):
            self.time_step = time_step.to(device=device, dtype=torch.float32)
        else:
            self.time_step = torch.tensor(time_step, dtype=torch.float32, device=device)
        
        # Initialize time counter for oscillatory model with the provided initial time
        if isinstance(damping, torch.Tensor):
            self.damping = damping.to(device=device, dtype=torch.float32)
        else:
            self.damping = torch.tensor(damping, dtype=torch.float32, device=device)

        if isinstance(initial_time, torch.Tensor):
            self.current_time = initial_time.to(device=device, dtype=torch.float32)
        else:
            self.current_time = torch.tensor(initial_time, dtype=torch.float32, device=device)

        if isinstance(dc_offset, (list, tuple, np.ndarray)):
            self.dc_offset = torch.tensor(dc_offset, dtype=torch.float32, device=device)
            if len(dc_offset) != self.num_sources:
                raise ValueError(
                    f"dc_offset length ({len(dc_offset)}) must match num_sources ({self.num_sources})"
                )
        else:
            self.dc_offset = torch.tensor(float(dc_offset), dtype=torch.float32, device=device)
        
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
            omega0_source = self.omega0
            kappa_source = self.kappa
            dc_source = self.dc_offset
        else:
            if source_idx >= self.num_sources:
                raise ValueError(f"Source index {source_idx} out of bounds for {self.num_sources} sources")
            omega0_source = self.omega0[source_idx]
            kappa_source = self.kappa[source_idx]
            dc_source = self.dc_offset[source_idx] if self.dc_offset.ndim > 0 else self.dc_offset
        
        oscillation = kappa_source * torch.sin(omega0_source * self.current_time) * self.deg2rad
        dc_rad = dc_source * self.deg2rad if self.angle_params_in_degrees else dc_source
        leak = (1.0 - self.damping) * dc_rad
        
        result = self.damping * x_tensor + leak + oscillation
        
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
        
        # ∂f/∂x = damping (leak and oscillation terms do not depend on x)
        return torch.full_like(x_tensor, self.damping)
    
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

        # Flat [num_sources] from batch KF mask indexing → single trajectory row
        squeeze_output = False
        if self.num_sources > 1 and x_tensor.ndim == 1 and x_tensor.shape[0] == self.num_sources:
            x_tensor = x_tensor.unsqueeze(0)
            squeeze_output = True
        
        leak_rad = (1.0 - self.damping) * (
            self.dc_offset * self.deg2rad if self.angle_params_in_degrees else self.dc_offset
        )

        # Handle single source vs multiple sources
        if self.num_sources == 1:
            oscillation = self.kappa * torch.sin(self.omega0 * self.current_time) * self.deg2rad
            leak = leak_rad if isinstance(leak_rad, torch.Tensor) and leak_rad.ndim == 0 else leak_rad
            result = self.damping * x_tensor + leak + oscillation
        else:
            if x_tensor.shape[-1] != self.num_sources:
                if x_tensor.shape[-1] == 1:
                    oscillation = self.kappa[0] * torch.sin(self.omega0[0] * self.current_time) * self.deg2rad
                    leak = (1.0 - self.damping) * (
                        self.dc_offset[0] * self.deg2rad if self.angle_params_in_degrees else self.dc_offset[0]
                    )
                    result = self.damping * x_tensor + leak + oscillation
                else:
                    raise ValueError(f"Last dimension of batch ({x_tensor.shape[-1]}) must match number of sources ({self.num_sources})")
            else:
                oscillation = self.kappa * torch.sin(self.omega0 * self.current_time) * self.deg2rad
                oscillation = oscillation.expand(x_tensor.shape[0], -1)
                if leak_rad.ndim == 0:
                    leak = leak_rad.expand_as(x_tensor)
                else:
                    leak = leak_rad.unsqueeze(0).expand(x_tensor.shape[0], -1)
                result = self.damping * x_tensor + leak + oscillation
        
        if squeeze_output:
            result = result.squeeze(0)

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

        if self.num_sources > 1 and x_tensor.ndim == 1 and x_tensor.shape[0] == self.num_sources:
            x_tensor = x_tensor.unsqueeze(0)
        
        jacobian = torch.full_like(x_tensor, self.damping)

        if jacobian.ndim == 2 and jacobian.shape[0] == 1:
            jacobian = jacobian.squeeze(0)
        
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

        if self.num_sources > 1 and x_tensor.ndim == 1 and x_tensor.shape[0] == self.num_sources:
            x_tensor = x_tensor.unsqueeze(0)

        out = torch.full_like(x_tensor, self.base_noise_variance, dtype=torch.float32, device=self.device)
        if out.ndim == 2 and out.shape[0] == 1:
            out = out.squeeze(0)
        return out
    
    def advance_time(self):
        """
        Advance the time counter by one time step.
        This should be called after each prediction step.
        """
        self.current_time += self.time_step

    def reset_time(self, initial_time=1.0):
        """Reset the time index (call when a new trajectory batch starts)."""
        if isinstance(initial_time, torch.Tensor):
            self.current_time = initial_time.to(device=self.device, dtype=torch.float32)
        else:
            self.current_time = torch.tensor(initial_time, dtype=torch.float32, device=self.device)
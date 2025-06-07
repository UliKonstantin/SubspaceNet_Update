"""
Multiplicative noise non-linear state evolution model.

This module implements the multiplicative noise model:
θ_{k+1} = θ_k + ω0 T + σ(θ_k) η_k
where σ(θ) = base_std * (1 + amp * sin²(θ))
"""

import numpy as np
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
    
    def __init__(self, omega0, amp, base_std, time_step=1.0):
        """
        Initialize the model.
        
        Args:
            omega0: Base angular velocity (rad/s)
            amp: Amplitude of multiplicative term (unitless)
            base_std: Base noise standard deviation (rad)
            time_step: Time step between measurements (s)
        """
        self.omega0 = omega0
        self.amp = amp
        self.base_std = base_std
        self.base_noise_variance = base_std**2
        self.time_step = time_step
        
        logger.debug(f"Created MultNoiseStateModel with ω₀={omega0}, amp={amp}, σ={base_std}, T={time_step}")
    
    def f(self, x):
        """
        Deterministic part: θ_{k+1} = θ_k + ω0 T
        
        Args:
            x: Current angle (degrees)
            
        Returns:
            Predicted next angle (degrees) without noise
        """
        # Calculate deterministic part (constant velocity model)
        result = x + np.degrees(self.omega0 * self.time_step)
        
        logger.debug(f"MultNoise state transition: {x} -> {result}")
        return result
    
    def F_jacobian(self, x):
        """
        Jacobian: ∂f/∂x = 1 (constant velocity model)
        
        The deterministic part is linear, so the Jacobian is 1.
        The non-linearity comes from the state-dependent noise.
        
        Args:
            x: Current angle (degrees)
            
        Returns:
            Derivative of state transition with respect to state (always 1)
        """
        # For the constant velocity model, the Jacobian is always 1
        return 1.0
    
    def noise_variance(self, x):
        """
        State-dependent noise variance: σ²(θ) = base_std² * (1 + amp * sin²(θ))²
        
        Args:
            x: Current angle (degrees)
            
        Returns:
            Process noise variance at the given state
        """
        # Convert to radians for trigonometric function
        x_rad = np.radians(x) if isinstance(x, (int, float)) else np.radians(x.copy())
        
        # Calculate state-dependent standard deviation
        std = self.base_std * (1.0 + self.amp * np.sin(x_rad)**2)
        
        # Return variance
        variance = std**2
        
        logger.debug(f"MultNoise variance at x={x}: {variance}")
        return variance 
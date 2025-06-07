"""
Sine acceleration non-linear state evolution model.

This module implements the sine acceleration model:
θ_{k+1} = θ_k + (ω0 + κ sin θ_k)T + η_k
"""

import numpy as np
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
    
    def __init__(self, omega0, kappa, noise_std, time_step=1.0):
        """
        Initialize the model.
        
        Args:
            omega0: Base angular velocity (rad/s)
            kappa: Sine acceleration coefficient (rad/s²)
            noise_std: Standard deviation of process noise (rad)
            time_step: Time step between measurements (s)
        """
        self.omega0 = omega0
        self.kappa = kappa
        self.base_noise_variance = noise_std**2
        self.time_step = time_step
        
        logger.debug(f"Created SineAccelStateModel with ω₀={omega0}, κ={kappa}, σ={noise_std}, T={time_step}")
    
    def f(self, x):
        """
        State transition: θ_{k+1} = θ_k + (ω0 + κ sin θ_k)T
        
        Args:
            x: Current angle (degrees)
            
        Returns:
            Predicted next angle (degrees)
        """
        # Convert to radians for trigonometric function
        x_rad = np.radians(x) if isinstance(x, (int, float)) else np.radians(x.copy())
        
        # Calculate acceleration term
        delta = (self.omega0 + self.kappa * np.sin(x_rad)) * self.time_step
        
        # Return in same units as input (degrees)
        result = x + np.degrees(delta)
        
        logger.debug(f"SineAccel state transition: {x} -> {result}")
        return result
    
    def F_jacobian(self, x):
        """
        Jacobian: ∂f/∂x = 1 + κT·cos(θ)
        
        Args:
            x: Current angle (degrees)
            
        Returns:
            Derivative of state transition with respect to state
        """
        # Convert to radians for trigonometric function
        x_rad = np.radians(x) if isinstance(x, (int, float)) else np.radians(x.copy())
        
        # Calculate the derivative
        # Note: need to account for degrees->radians conversion in the derivative
        jacobian = 1 + self.kappa * np.cos(x_rad) * self.time_step * np.pi/180.0
        
        logger.debug(f"SineAccel Jacobian at x={x}: {jacobian}")
        return jacobian 
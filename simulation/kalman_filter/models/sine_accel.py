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
        #x_rad = np.radians(x) if isinstance(x, (int, float)) else np.radians(x.copy())
        x_rad = x
        x_deg = x_rad * 180/np.pi
        # Calculate acceleration term
        delta = (self.omega0 + self.kappa * np.sin(x_rad)) * self.time_step
        
        # Return in same units as input (degrees)
        #result = x + np.radians(delta)
        result_deg = 0.99*x_deg + delta
        result = np.radians(result_deg) # Convert back to radians
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
        #x_rad = np.radians(x) if isinstance(x, (int, float)) else np.radians(x.copy())
        x_rad = x
        # Calculate the derivative
        # Note: need to account for degrees->radians conversion in the derivative
        #jacobian = 1 + self.kappa * np.cos(x_rad) * self.time_step * np.pi/180.0
        jacobian = 0.99 + self.kappa * np.cos(x_rad) * self.time_step
        logger.debug(f"SineAccel Jacobian at x={x}: {jacobian}")
        return jacobian
    
    def f_batch(self, x_batch):
        """
        Batch version of state transition function.
        
        Args:
            x_batch: Array of current angles
            
        Returns:
            Array of predicted next angles
        """
        # Convert to numpy array if needed
        x_array = np.asarray(x_batch)
        
        # Apply state transition function element-wise
        delta = (self.omega0 + self.kappa * np.sin(x_array)) * self.time_step
        result = x_array + delta
        
        logger.debug(f"SineAccel batch state transition for {len(x_array)} states")
        return result
    
    def F_jacobian_batch(self, x_batch):
        """
        Batch version of Jacobian computation.
        
        Args:
            x_batch: Array of current angles
            
        Returns:
            Array of Jacobians
        """
        # Convert to numpy array if needed
        x_array = np.asarray(x_batch)
        
        # Calculate Jacobians element-wise
        jacobian = 1 + self.kappa * np.cos(x_array) * self.time_step
        
        logger.debug(f"SineAccel batch Jacobian for {len(x_array)} states")
        return jacobian
    
    def noise_variance_batch(self, x_batch):
        """
        Batch version of noise variance computation.
        
        Args:
            x_batch: Array of current angles
            
        Returns:
            Array of noise variances (constant for this model)
        """
        # Convert to numpy array if needed
        x_array = np.asarray(x_batch)
        
        # Return constant noise variance for all states
        return np.full_like(x_array, self.base_noise_variance) 
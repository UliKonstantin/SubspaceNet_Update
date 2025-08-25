"""
Abstract base class for state evolution models used by Extended Kalman Filter.

This module defines the StateEvolutionModel interface that all concrete
state evolution models must implement for use with the ExtendedKalmanFilter1D.
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
import logging

logger = logging.getLogger("SubspaceNet.kalman_filter.models")

class StateEvolutionModel(ABC):
    """
    Abstract base class for state evolution models used by EKF.
    
    This class defines the interface for state evolution models which describe
    how the state variables evolve over time. Concrete implementations must
    provide the non-linear state transition function and its Jacobian.
    """
    
    @abstractmethod
    def f(self, x):
        """
        Non-linear state transition function.
        
        Args:
            x: Current state (tensor or scalar)
            
        Returns:
            Next state without noise (tensor)
        """
        pass
    
    @abstractmethod
    def F_jacobian(self, x):
        """
        Jacobian of the state transition function.
        
        Args:
            x: Current state (tensor or scalar)
            
        Returns:
            Derivative of f with respect to x (tensor)
        """
        pass
    
    def noise_variance(self, x):
        """
        State-dependent process noise variance.
        
        Args:
            x: Current state (tensor or scalar)
            
        Returns:
            Process noise variance at state x (tensor)
        """
        # Default implementation returns constant noise variance
        # Subclasses should override this for state-dependent noise
        if hasattr(self, 'base_noise_variance'):
            if isinstance(x, torch.Tensor):
                return torch.full_like(x, self.base_noise_variance, dtype=torch.float32, device=x.device)
            else:
                return self.base_noise_variance
        else:
            raise NotImplementedError("Subclasses must implement noise_variance or provide base_noise_variance") 
"""
Abstract base class for state evolution models used by Extended Kalman Filter.

This module defines the StateEvolutionModel interface that all concrete
state evolution models must implement for use with the ExtendedKalmanFilter1D.
"""

from abc import ABC, abstractmethod
import numpy as np
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
            x: Current state
            
        Returns:
            Next state without noise
        """
        pass
    
    @abstractmethod
    def F_jacobian(self, x):
        """
        Jacobian of the state transition function.
        
        Args:
            x: Current state
            
        Returns:
            Derivative of f with respect to x
        """
        pass
    
    def noise_variance(self, x):
        """
        State-dependent process noise variance.
        
        Args:
            x: Current state
            
        Returns:
            Process noise variance at state x
        """
        return self.base_noise_variance 
"""
Kalman Filter implementation for the SubspaceNet framework.

This module provides Kalman Filter implementations for tracking state variables
like angle and range in direction-of-arrival (DOA) estimation.
"""

import numpy as np
import logging

logger = logging.getLogger('SubspaceNet.kalman_filter')

class KalmanFilter1D:
    """
    1D Kalman Filter for tracking a single state variable (e.g., angle).
    
    Implements a simple Kalman Filter for tracking a scalar variable, assuming:
    - State transition model: x_k = x_{k-1} + w_k (F = 1, B = 0)
    - Observation model: z_k = x_k + v_k (H = 1)
    - w_k ~ N(0, Q), v_k ~ N(0, R)
    
    This is suitable for tracking angles that follow a random walk model.
    """
    
    def __init__(self, Q, R, P0=1.0):
        """
        Initialize the Kalman Filter.
        
        Args:
            Q: Process noise variance (scalar)
            R: Measurement noise variance (scalar)
            P0: Initial state covariance (scalar)
        """
        # Prevent exactly zero measurement noise (numerical stability)
        if R == 0:
            R = 1e-6
            
        self.Q = Q  # Process noise variance
        self.R = R  # Measurement noise variance
        self.P0 = P0  # Initial state covariance
        
        # State will be initialized later
        self.x = None  # State estimate
        self.P = P0  # State estimate covariance
        
        logger.debug(f"Initialized 1D Kalman Filter with Q={Q}, R={R}, P0={P0}")
    
    def initialize_state(self, x0):
        """
        Initialize the state estimate.
        
        Args:
            x0: Initial state estimate (scalar)
        """
        self.x = x0
        self.P = self.P0
        logger.debug(f"Initialized state to x0={x0} with covariance P0={self.P0}")
    
    def predict(self):
        """
        Perform the time update (prediction) step.
        
        Returns:
            Predicted state (scalar)
        """
        if self.x is None:
            raise ValueError("State must be initialized before prediction")
        
        # State prediction (x̂_k|k-1 = F * x̂_k-1|k-1 + B * u_k)
        # For our model, F = 1, B = 0, so x̂_k|k-1 = x̂_k-1|k-1
        x_pred = self.x
        
        # Covariance prediction (P_k|k-1 = F * P_k-1|k-1 * F^T + Q)
        # For our model, F = 1, so P_k|k-1 = P_k-1|k-1 + Q
        P_pred = self.P + self.Q
        
        # Update state and covariance
        self.x = x_pred
        self.P = P_pred
        
        logger.debug(f"Prediction: x={x_pred}, P={P_pred}")
        return x_pred
    
    def update(self, z):
        """
        Perform the measurement update step.
        
        Args:
            z: Measurement (scalar)
            
        Returns:
            Updated state estimate (scalar)
        """
        if self.x is None:
            raise ValueError("State must be initialized before update")
        
        # Innovation / measurement residual (y = z - H * x̂_k|k-1)
        # For our model, H = 1, so y = z - x̂_k|k-1
        y = z - self.x
        
        # Innovation covariance (S = H * P_k|k-1 * H^T + R)
        # For our model, H = 1, so S = P_k|k-1 + R
        S = self.P + self.R
        
        # Kalman gain (K = P_k|k-1 * H^T * S^-1)
        # For our model, H = 1, so K = P_k|k-1 / S
        K = self.P / S
        
        # State update (x̂_k|k = x̂_k|k-1 + K * y)
        x_new = self.x + K * y
        
        # Covariance update (P_k|k = (I - K * H) * P_k|k-1)
        # For our model, H = 1, so P_k|k = (1 - K) * P_k|k-1
        P_new = (1 - K) * self.P
        
        # Update state and covariance
        self.x = x_new
        self.P = P_new
        
        logger.debug(f"Update with z={z}: y={y}, K={K}, x={x_new}, P={P_new}")
        return x_new 
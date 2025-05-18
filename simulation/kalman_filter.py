"""
Kalman Filter implementation for the SubspaceNet framework.

This module provides Kalman Filter implementations for tracking state variables
like angle and range in direction-of-arrival (DOA) estimation.
"""

import numpy as np
import logging
import torch

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
    
    @classmethod
    def from_config(cls, config):
        """
        Create KalmanFilter1D parameters from configuration.
        
        Args:
            config: Configuration object containing kalman_filter and trajectory settings
            
        Returns:
            tuple: (Q, R, P0) - Parameters needed for KalmanFilter1D initialization
        """
        # Configure Kalman Filter parameters
        kf_process_noise_std_dev = config.kalman_filter.process_noise_std_dev
        if kf_process_noise_std_dev is None:
            kf_process_noise_std_dev = config.trajectory.random_walk_std_dev
            logger.debug(f"Using trajectory.random_walk_std_dev ({kf_process_noise_std_dev}) for KF process noise")
        
        # Calculate variances from standard deviations
        kf_Q = kf_process_noise_std_dev ** 2
        kf_R = config.kalman_filter.measurement_noise_std_dev ** 2
        kf_P0 = config.kalman_filter.initial_covariance
        
        logger.info(f"Kalman Filter parameters: Q={kf_Q}, R={kf_R}, P0={kf_P0}")
        
        return kf_Q, kf_R, kf_P0
    
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

    @classmethod
    def create_filters_from_config(cls, config, num_instances=1):
        """
        Create multiple KalmanFilter1D instances directly from config.
        
        Args:
            config: Configuration object containing kalman_filter and trajectory settings
            num_instances: Number of filter instances to create
            
        Returns:
            list: List of KalmanFilter1D instances
        """
        params = cls.from_config(config)
        filters = [cls(Q=params[0], R=params[1], P0=params[2]) for _ in range(num_instances)]
        return filters


class BatchKalmanFilter1D:
    """
    Batch-enabled 1D Kalman Filter for tracking multiple sources across multiple trajectories.
    
    This extends the KalmanFilter1D concept to work efficiently with batches of data
    by vectorizing the Kalman Filter operations.
    
    Notes:
        - Uses PyTorch tensors for efficient batch operations
        - Can handle variable number of sources per trajectory using masks
    """
    
    def __init__(self, batch_size, max_sources, Q, R, P0=1.0, device=None):
        """
        Initialize the batch Kalman Filter.
        
        Args:
            batch_size: Number of trajectories in the batch
            max_sources: Maximum number of sources per trajectory
            Q: Process noise variance (scalar or batch)
            R: Measurement noise variance (scalar or batch)
            P0: Initial state covariance (scalar or batch)
            device: PyTorch device (for GPU acceleration)
        """
        self.batch_size = batch_size
        self.max_sources = max_sources
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Prevent exactly zero measurement noise (numerical stability)
        if isinstance(R, (int, float)) and R == 0:
            R = 1e-6
            
        # Convert scalars to tensors if needed
        if isinstance(Q, (int, float)):
            self.Q = torch.full((batch_size, max_sources), Q, device=self.device)
        else:
            self.Q = torch.as_tensor(Q, device=self.device)
            
        if isinstance(R, (int, float)):
            self.R = torch.full((batch_size, max_sources), R, device=self.device)
        else:
            self.R = torch.as_tensor(R, device=self.device)
            
        if isinstance(P0, (int, float)):
            self.P0 = torch.full((batch_size, max_sources), P0, device=self.device)
        else:
            self.P0 = torch.as_tensor(P0, device=self.device)
        
        # Initialize state and masks
        self.x = torch.zeros((batch_size, max_sources), device=self.device)  # State estimates
        self.P = self.P0.clone()  # State covariances
        self.state_initialized = torch.zeros((batch_size, max_sources), dtype=torch.bool, device=self.device)
        
        # Create source mask (to handle variable sources per trajectory)
        self.source_mask = torch.zeros((batch_size, max_sources), dtype=torch.bool, device=self.device)
        
        logger.info(f"Initialized Batch KF with batch_size={batch_size}, max_sources={max_sources} on {self.device}")
        
    @classmethod
    def from_config(cls, config, batch_size, max_sources, device=None):
        """
        Create BatchKalmanFilter1D from configuration.
        
        Args:
            config: Configuration object containing kalman_filter settings
            batch_size: Number of trajectories in the batch
            max_sources: Maximum number of sources
            device: PyTorch device
            
        Returns:
            BatchKalmanFilter1D instance
        """
        # Get parameters from config
        kf_process_noise_std_dev = config.kalman_filter.process_noise_std_dev
        if kf_process_noise_std_dev is None:
            kf_process_noise_std_dev = config.trajectory.random_walk_std_dev
            logger.debug(f"Using trajectory.random_walk_std_dev ({kf_process_noise_std_dev}) for KF process noise")
        
        # Calculate variances from standard deviations
        kf_Q = kf_process_noise_std_dev ** 2
        kf_R = config.kalman_filter.measurement_noise_std_dev ** 2
        kf_P0 = config.kalman_filter.initial_covariance
        
        logger.info(f"Batch Kalman Filter parameters: Q={kf_Q}, R={kf_R}, P0={kf_P0}")
        
        return cls(batch_size, max_sources, kf_Q, kf_R, kf_P0, device)
    
    def initialize_states(self, initial_states, num_sources):
        """
        Initialize all state estimates at once.
        
        Args:
            initial_states: Initial state estimates tensor [batch_size, max_sources]
                            or list of tensors with variable length
            num_sources: Tensor of source counts [batch_size] or list/array
        """
        # Convert inputs to appropriate tensor type if necessary
        if not isinstance(initial_states, torch.Tensor):
            # Handle list/array of initial states
            temp_states = torch.zeros((self.batch_size, self.max_sources), device=self.device)
            for i, states in enumerate(initial_states):
                if i < self.batch_size and len(states) > 0:
                    temp_states[i, :len(states)] = torch.as_tensor(states, device=self.device)
            initial_states = temp_states
        
        if not isinstance(num_sources, torch.Tensor):
            num_sources = torch.as_tensor(num_sources, device=self.device)
            
        # Update the source mask based on num_sources
        self.source_mask.fill_(False)
        for i in range(self.batch_size):
            if i < len(num_sources):
                n = min(int(num_sources[i].item()), self.max_sources)
                self.source_mask[i, :n] = True
        
        # Set initial states where we have valid sources
        self.x = torch.where(self.source_mask, initial_states, self.x)
        self.P = torch.where(self.source_mask, self.P0, self.P)
        self.state_initialized = self.source_mask.clone()
        
        logger.debug(f"Initialized batch states for {self.source_mask.sum().item()} sources")
    
    def predict(self):
        """
        Perform the time update (prediction) step for all states.
        
        Returns:
            Predicted states tensor [batch_size, max_sources]
        """
        # Only predict for initialized states
        active_mask = self.state_initialized & self.source_mask
        
        # For identity state transition model (F=1), predicted state is the same
        # x_pred = F * x + B * u, but with F=1, B=0, so x_pred = x
        x_pred = self.x
        
        # Covariance prediction: P_pred = F * P * F^T + Q = P + Q
        self.P = torch.where(active_mask, self.P + self.Q, self.P)
        
        return x_pred.clone()
    
    def update(self, measurements, measurement_mask=None):
        """
        Perform the measurement update step for all states.
        
        Args:
            measurements: Measurement tensor [batch_size, max_sources]
            measurement_mask: Boolean mask for valid measurements [batch_size, max_sources] 
                             (defaults to self.source_mask)
        
        Returns:
            Updated state estimates tensor [batch_size, max_sources]
        """
        if measurement_mask is None:
            measurement_mask = self.source_mask
        
        # Combine masks to only update initialized states with valid measurements
        active_mask = self.state_initialized & self.source_mask & measurement_mask
        
        # Innovation (y = z - H*x), where H=1
        y = torch.where(active_mask, measurements - self.x, torch.zeros_like(self.x))
        
        # Innovation covariance (S = H*P*H^T + R), where H=1
        S = torch.where(active_mask, self.P + self.R, torch.ones_like(self.P))
        
        # Kalman gain (K = P*H^T*S^-1), where H=1
        K = torch.where(active_mask, self.P / S, torch.zeros_like(self.P))
        
        # State update (x_new = x + K*y)
        self.x = torch.where(active_mask, self.x + K * y, self.x)
        
        # Covariance update (P_new = (I - K*H)*P), where H=1
        self.P = torch.where(active_mask, (1.0 - K) * self.P, self.P)
        
        return self.x.clone()
    
    def predict_and_update(self, measurements, measurement_mask=None):
        """
        Perform both prediction and update steps at once.
        
        Args:
            measurements: Measurement tensor [batch_size, max_sources]
            measurement_mask: Boolean mask for valid measurements
        
        Returns:
            Both predicted and updated states (before_update, after_update)
        """
        predicted = self.predict()
        updated = self.update(measurements, measurement_mask)
        return predicted, updated 
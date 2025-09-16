"""
Extended Kalman Filter implementation for non-linear dynamics.

This module provides an Extended Kalman Filter implementation capable of
handling non-linear state evolution models for angle tracking in DOA estimation.
"""

import numpy as np
import torch
import logging
from config.schema import TrajectoryType
from simulation.kalman_filter.models import SineAccelStateModel, MultNoiseStateModel
from pathlib import Path
import datetime

logger = logging.getLogger("SubspaceNet.kalman_filter.extended")

# Add file handler for debug logs
def setup_debug_file_logging():
    """Setup file logging for EKF debug messages."""
    # Create debug logs directory
    debug_dir = Path("experiments/debug_logs")
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = debug_dir / f"ekf_debug_{timestamp}.txt"
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)  # Ensure this logger accepts DEBUG messages
    
    return log_file

# Setup debug logging on module import
debug_log_file = setup_debug_file_logging()
logger.info(f"EKF debug logging enabled. Logs will be saved to: {debug_log_file}")

class ExtendedKalmanFilter1D:
    """
    Extended Kalman Filter for non-linear state evolution models.
    
    Implements EKF for tracking a scalar variable with non-linear dynamics:
    - State transition: x_k = f(x_{k-1}) + w_k
    - Observation model: z_k = x_k + v_k (H = 1)
    - w_k ~ N(0, Q(x)), v_k ~ N(0, R)
    
    This filter can handle various non-linear state evolution models and
    automatically selects the appropriate model based on the trajectory type.
    """
    
    def __init__(self, state_model, R, P0=0.001, device=None, source_idx=0):
        """
        Initialize the Extended Kalman Filter.
        
        Args:
            state_model: StateEvolutionModel instance
            R: Measurement noise variance (scalar or tensor)
            P0: Initial state covariance (scalar or tensor)
            device: Device for tensor operations (cuda/cpu)
            source_idx: Index of the source this filter tracks (default 0)
        """
        # Set device for tensor operations
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Store source index
        self.source_idx = source_idx
        
        # Convert parameters to tensors with consistent dtype (float32 for training compatibility)
        if isinstance(R, torch.Tensor):
            R_tensor = R.to(device=device, dtype=torch.float32)
        else:
            R_tensor = torch.tensor(R, dtype=torch.float32, device=device)
            
        if isinstance(P0, torch.Tensor):
            P0_tensor = P0.to(device=device, dtype=torch.float32)
        else:
            P0_tensor = torch.tensor(P0, dtype=torch.float32, device=device)
        
        # Prevent exactly zero measurement noise (numerical stability)
        if R_tensor == 0:
            R_tensor = torch.tensor(1e-6, dtype=torch.float32, device=device)
            
        self.state_model = state_model
        self.R = R_tensor  # Measurement noise variance (tensor)
        self.P0 = P0_tensor  # Initial state covariance (tensor)
        
        # State will be initialized later
        self.x = None  # State estimate (tensor)
        self.P = P0_tensor.clone()  # State estimate covariance (tensor)
        self.Q = torch.tensor(0.0, dtype=torch.float32, device=device)  # Process noise variance (tensor)
        logger.debug(f"Initialized Extended Kalman Filter for source {source_idx} with R={R}, P0={P0}, device={device}")
    
    @classmethod
    def from_config(cls, config, trajectory_type=None, device=None, initial_time=0.0):
        """
        Get ExtendedKalmanFilter1D parameters from configuration.
        
        Args:
            config: Configuration object with kalman_filter and trajectory settings
            trajectory_type: Type of trajectory to model (defaults to config value)
            device: Device for tensor operations (cuda/cpu)
            initial_time: Initial time value for oscillatory models (default 0.0)
            
        Returns:
            tuple: (state_model, R, P0) - Parameters needed for ExtendedKalmanFilter1D initialization
        """
        # Get trajectory type from config if not specified
        if trajectory_type is None:
            trajectory_type = config.trajectory.trajectory_type
        
        # Get process noise from Kalman filter config (prioritize KF config over trajectory config)
        if hasattr(config.kalman_filter, 'process_noise_std_dev') and config.kalman_filter.process_noise_std_dev is not None:
            kf_process_noise_std = config.kalman_filter.process_noise_std_dev
            logger.info(f"Using Kalman filter process noise: {kf_process_noise_std}")
        else:
            # Fallback to trajectory-specific noise if KF process noise not specified
            if trajectory_type == TrajectoryType.SINE_ACCEL_NONLINEAR:
                kf_process_noise_std = config.trajectory.sine_accel_noise_std
            elif trajectory_type == TrajectoryType.MULT_NOISE_NONLINEAR:
                kf_process_noise_std = config.trajectory.mult_noise_base_std
            else:
                kf_process_noise_std = config.trajectory.random_walk_std_dev
            logger.info(f"Using trajectory-based process noise as fallback: {kf_process_noise_std}")
        
        # Create appropriate state evolution model based on trajectory type
        if trajectory_type == TrajectoryType.SINE_ACCEL_NONLINEAR:
            # Get sine acceleration model parameters (support both single values and arrays)
            omega0 = config.trajectory.sine_accel_omega0
            kappa = config.trajectory.sine_accel_kappa
            # Use KF process noise instead of trajectory noise
            noise_std = kf_process_noise_std
            
            # Convert single values to arrays for source-specific parameters
            if isinstance(omega0, (int, float)):
                omega0 = [omega0] * config.system_model.M
            if isinstance(kappa, (int, float)):
                kappa = [kappa] * config.system_model.M
            
            # Ensure arrays have correct length
            if len(omega0) != config.system_model.M:
                raise ValueError(f"sine_accel_omega0 array length ({len(omega0)}) must match number of sources ({config.system_model.M})")
            if len(kappa) != config.system_model.M:
                raise ValueError(f"sine_accel_kappa array length ({len(kappa)}) must match number of sources ({config.system_model.M})")
            
            # Create state model with source-specific parameters and initial time
            state_model = SineAccelStateModel(omega0, kappa, noise_std, device=device, initial_time=initial_time)
            
            logger.info(f"Using source-specific sine acceleration model with initial_time={initial_time}:")
            for i, (om, kap) in enumerate(zip(omega0, kappa)):
                logger.info(f"  Source {i}: ω₀={om}, κ={kap}, σ={noise_std}")
            
        elif trajectory_type == TrajectoryType.MULT_NOISE_NONLINEAR:
            # Get multiplicative noise model parameters
            omega0 = config.trajectory.mult_noise_omega0
            amp = config.trajectory.mult_noise_amp
            # Use KF process noise instead of trajectory noise
            base_std = kf_process_noise_std
            
            state_model = MultNoiseStateModel(omega0, amp, base_std, device=device)
            logger.info(f"Using multiplicative noise model with ω₀={omega0}, amp={amp}, σ={base_std}")
            
        else:
            # Default to random walk model for other types
            # Use KF process noise instead of trajectory noise
            noise_std = kf_process_noise_std
            state_model = SineAccelStateModel(0.0, 0.0, noise_std, device=device, initial_time=initial_time)
            logger.info(f"Using random walk model with σ={noise_std}")
        
        # Get measurement noise and initial covariance
        kf_R = config.kalman_filter.measurement_noise_std_dev ** 2
        kf_P0 = config.kalman_filter.initial_covariance
        
        return state_model, kf_R, kf_P0

    @classmethod
    def create_from_config(cls, config, trajectory_type=None, device=None, source_idx=0, initial_time=0.0):
        """
        Create ExtendedKalmanFilter1D instance directly from config.
        
        Args:
            config: Configuration object with kalman_filter and trajectory settings
            trajectory_type: Type of trajectory to model (defaults to config value)
            device: Device for tensor operations (cuda/cpu)
            source_idx: Index of the source this filter tracks (default 0)
            initial_time: Initial time value for oscillatory models (default 0.0)
            
        Returns:
            ExtendedKalmanFilter1D instance
        """
        params = cls.from_config(config, trajectory_type, device=device, initial_time=initial_time)
        return cls(state_model=params[0], R=params[1], P0=params[2], device=device, source_idx=source_idx)
    
    @classmethod
    def create_filters_from_config(cls, config, num_instances=1, trajectory_type=None, device=None, initial_time=0.0):
        """
        Create multiple ExtendedKalmanFilter1D instances directly from config.
        
        Args:
            config: Configuration object with kalman_filter and trajectory settings
            num_instances: Number of filter instances to create
            trajectory_type: Type of trajectory to model (defaults to config value)
            device: Device for tensor operations (cuda/cpu)
            initial_time: Initial time value for oscillatory models (default 0.0)
            
        Returns:
            list: List of ExtendedKalmanFilter1D instances, each with source_idx=i
        """
        params = cls.from_config(config, trajectory_type, device=device, initial_time=initial_time)
        filters = [cls(state_model=params[0], R=params[1], P0=params[2], device=device, source_idx=i) for i in range(num_instances)]
        return filters
    
    def initialize_state(self, x0):
        """
        Initialize the state estimate.
        
        Args:
            x0: Initial state estimate (scalar or tensor)
        """
        # Convert to tensor if needed, ensuring proper dtype and device
        if isinstance(x0, torch.Tensor):
            self.x = x0.to(device=self.device, dtype=torch.float32)
        else:
            self.x = torch.tensor(x0, dtype=torch.float32, device=self.device)
        
        # Ensure P is properly initialized as tensor
        self.P = self.P0.clone()
        logger.debug(f"Initialized state to x0={x0} with covariance P0={self.P0}")
    
    def predict(self):
        """
        Perform the time update (prediction) step.
        
        Returns:
            Predicted state (tensor)
        """
        if self.x is None:
            raise ValueError("State must be initialized before prediction")
        # print(f"state before prediction: {self.x}")
        # State prediction using non-linear function
        x_pred = self.state_model.f(self.x, self.source_idx)

        # Get Jacobian at current state
        F = self.state_model.F_jacobian(self.x, self.source_idx)
        
        # Get state-dependent process noise
        self.Q = self.state_model.noise_variance(self.x, self.source_idx)
        
        # Covariance prediction using linearization
        # Use proper tensor operations to maintain gradients
        P_pred = F * self.P * F + self.Q
        
        # Update state and covariance
        self.x = x_pred
        self.P = P_pred
        
        # Advance time for oscillatory models
        if hasattr(self.state_model, 'advance_time'):
            self.state_model.advance_time()
        
        return x_pred
    
    def update(self, z):
        """
        Perform the measurement update step.
        
        Args:
            z: Measurement (tensor)
            
        Returns:
            Tuple of (updated state estimate (tensor), innovation (tensor), kalman_gain (tensor), 
                     kalman_gain_times_innovation (tensor), y_s_inv_y (tensor))
        """
        if self.x is None:
            raise ValueError("State must be initialized before update")
        
        # Convert measurement to tensor if needed, ensuring proper dtype and device
        if isinstance(z, torch.Tensor):
            z_tensor = z.to(device=self.device, dtype=torch.float32)
        else:
            z_tensor = torch.tensor(z, dtype=torch.float32, device=self.device)
            
        # Measurement model is linear (H=1) for our problem
        # Innovation / measurement residual (y = z - H * x̂_k|k-1)
        y = z_tensor - self.x
        
        # Innovation covariance (S = H * P_k|k-1 * H^T + R)
        S = self.P + self.R
        
        # Kalman gain (K = P_k|k-1 * H^T * S^-1)
        K = self.P / S
        
        # State update (x̂_k|k = x̂_k|k-1 + K * y)
        x_new = self.x + K * y
        
        # Covariance update (P_k|k = (I - K * H) * P_k|k-1)
        # Use proper tensor operations to maintain gradients
        P_new = (torch.tensor(1.0, dtype=torch.float32, device=self.device) - K) * self.P
        
        # Calculate y*(S^-1)*y (innovation covariance metric)
        y_s_inv_y = y * (torch.tensor(1.0, dtype=torch.float32, device=self.device) / S) * y
        
        # Update state and covariance
        self.x = x_new
        self.P = P_new
        
        # Return all values as tensors
        return x_new, y, K, K * y, y_s_inv_y,S

    def predict_and_update(self, measurement, true_state=None):
        """
        Perform a complete cycle of prediction and update.
        
        This method:
        1. Predicts the next state
        2. Updates the filter state using provided measurement
        
        Args:
            measurement: Measurement to use for update (tensor or scalar)
            true_state: Optional ground truth state for logging (tensor or scalar)
        
        Returns:
            Tuple of (predicted_state, updated_state, innovation, kalman_gain, 
                     kalman_gain_times_innovation, y_s_inv_y) - all tensors
        """
        # Perform prediction step
        predicted_state = self.predict()
        
        # Perform update step with measurement
        updated_state, innovation, kalman_gain, kalman_gain_times_innovation, y_s_inv_y,Innovation_Covariance = self.update(z=measurement)
        
        return predicted_state, updated_state, innovation, kalman_gain, kalman_gain_times_innovation, y_s_inv_y,Innovation_Covariance 
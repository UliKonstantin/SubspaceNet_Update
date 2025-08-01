"""
Extended Kalman Filter implementation for non-linear dynamics.

This module provides an Extended Kalman Filter implementation capable of
handling non-linear state evolution models for angle tracking in DOA estimation.
"""

import numpy as np
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
    
    def __init__(self, state_model, R, P0=0.001):
        """
        Initialize the Extended Kalman Filter.
        
        Args:
            state_model: StateEvolutionModel instance
            R: Measurement noise variance (scalar)
            P0: Initial state covariance (scalar)
        """
        # Prevent exactly zero measurement noise (numerical stability)
        if R == 0:
            R = 1e-6
            
        self.state_model = state_model
        self.R = R  # Measurement noise variance
        self.P0 = P0  # Initial state covariance
        
        # State will be initialized later
        self.x = None  # State estimate
        self.P = P0  # State estimate covariance
        self.Q = 0 # Process noise variance
        logger.debug(f"Initialized Extended Kalman Filter with R={R}, P0={P0}")
    
    @classmethod
    def from_config(cls, config, trajectory_type=None):
        """
        Get ExtendedKalmanFilter1D parameters from configuration.
        
        Args:
            config: Configuration object with kalman_filter and trajectory settings
            trajectory_type: Type of trajectory to model (defaults to config value)
            
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
            # Get sine acceleration model parameters
            omega0 = config.trajectory.sine_accel_omega0
            kappa = config.trajectory.sine_accel_kappa
            # Use KF process noise instead of trajectory noise
            noise_std = kf_process_noise_std
            
            state_model = SineAccelStateModel(omega0, kappa, noise_std)
            logger.info(f"Using sine acceleration model with ω₀={omega0}, κ={kappa}, σ={noise_std}")
            
        elif trajectory_type == TrajectoryType.MULT_NOISE_NONLINEAR:
            # Get multiplicative noise model parameters
            omega0 = config.trajectory.mult_noise_omega0
            amp = config.trajectory.mult_noise_amp
            # Use KF process noise instead of trajectory noise
            base_std = kf_process_noise_std
            
            state_model = MultNoiseStateModel(omega0, amp, base_std)
            logger.info(f"Using multiplicative noise model with ω₀={omega0}, amp={amp}, σ={base_std}")
            
        else:
            # Default to random walk model for other types
            # Use KF process noise instead of trajectory noise
            noise_std = kf_process_noise_std
            state_model = SineAccelStateModel(0.0, 0.0, noise_std)
            logger.info(f"Using random walk model with σ={noise_std}")
        
        # Get measurement noise and initial covariance
        kf_R = config.kalman_filter.measurement_noise_std_dev ** 2
        kf_P0 = config.kalman_filter.initial_covariance
        
        return state_model, kf_R, kf_P0

    @classmethod
    def create_from_config(cls, config, trajectory_type=None):
        """
        Create ExtendedKalmanFilter1D instance directly from config.
        
        Args:
            config: Configuration object with kalman_filter and trajectory settings
            trajectory_type: Type of trajectory to model (defaults to config value)
            
        Returns:
            ExtendedKalmanFilter1D instance
        """
        params = cls.from_config(config, trajectory_type)
        return cls(state_model=params[0], R=params[1], P0=params[2])
    
    @classmethod
    def create_filters_from_config(cls, config, num_instances=1, trajectory_type=None):
        """
        Create multiple ExtendedKalmanFilter1D instances directly from config.
        
        Args:
            config: Configuration object with kalman_filter and trajectory settings
            num_instances: Number of filter instances to create
            trajectory_type: Type of trajectory to model (defaults to config value)
            
        Returns:
            list: List of ExtendedKalmanFilter1D instances
        """
        params = cls.from_config(config, trajectory_type)
        filters = [cls(state_model=params[0], R=params[1], P0=params[2]) for _ in range(num_instances)]
        return filters
    
    def initialize_state(self, x0):
        """
        Initialize the state estimate.
        
        Args:
            x0: Initial state estimate (scalar)
        """
        self.x = x0
        self.P = self.P0
        logger.debug(f"Initialized state to x0={x0} with covariance P0={self.P0}")
    
    def predict(self, generate_measurement=False):
        """
        Perform the time update (prediction) step.
        
        Args:
            generate_measurement: If True, also generates a noisy measurement for the
                                predicted state and returns it along with the prediction
        
        Returns:
            If generate_measurement is False: Predicted state (scalar)
            If generate_measurement is True: Tuple of (predicted state, measurement)
        """
        if self.x is None:
            raise ValueError("State must be initialized before prediction")
        
        # State prediction using non-linear function
        x_pred = self.state_model.f(self.x)

        # Get Jacobian at current state
        F = self.state_model.F_jacobian(self.x)
        
        # Get state-dependent process noise
        self.Q = self.state_model.noise_variance(self.x)
        
        # Covariance prediction using linearization
        P_pred = F * self.P * F + self.Q
        
        # Update state and covariance
        self.x = x_pred
        self.P = P_pred
        
        # Optionally generate measurement for the predicted state
        if generate_measurement:
            measurement = self.simulate_measurement(true_state=x_pred)
            return x_pred, measurement
        else:
            return x_pred
    
    def simulate_measurement(self, true_state=None):
        """
        Generate a simulated noisy measurement based on the current state.
        
        Args:
            true_state: Optional true state to use instead of the filter's state estimate
                        (useful for simulation when we know the ground truth)
        
        Returns:
            Simulated noisy measurement (scalar)
        """
        if true_state is None:
            if self.x is None:
                raise ValueError("State must be initialized before simulating measurements")
            state = self.x
        else:
            state = true_state
            
        # Generate measurement noise with variance R
        measurement_noise = np.random.normal(0, np.sqrt(self.R))
        
        # Measurement model: z = x + v where v ~ N(0, R)
        measurement = state + measurement_noise
        
        return measurement
    
    def update(self, z=None):
        """
        Perform the measurement update step.
        
        Args:
            z: Measurement (scalar). If None, a measurement will be simulated
               based on the current state estimate.
            
        Returns:
            Tuple of (updated state estimate (scalar), innovation (scalar))
        """
        if self.x is None:
            raise ValueError("State must be initialized before update")
        
        # If no measurement provided, simulate one based on current state
        if z is None:
            z = self.simulate_measurement()
            logger.debug(f"Using simulated measurement: {z}")
        # Measurement model is linear (H=1) for our problem
        # Innovation / measurement residual (y = z - H * x̂_k|k-1)
        y = z - self.x
        
        # Innovation covariance (S = H * P_k|k-1 * H^T + R)
        S = self.P + self.R
        
        # Kalman gain (K = P_k|k-1 * H^T * S^-1)
        K = self.P / S
        
        # State update (x̂_k|k = x̂_k|k-1 + K * y)
        x_new = self.x + K * y
        
        # Covariance update (P_k|k = (I - K * H) * P_k|k-1)
        P_new = (1 - K) * self.P
        
        # Calculate y*(S^-1)*y (innovation covariance metric)
        y_s_inv_y = y * (1 / S) * y
        
        # Update state and covariance
        self.x = x_new
        self.P = P_new
        
        return x_new, y, K, K * y, y_s_inv_y

    def predict_and_update(self, measurement=None, true_state=None):
        """
        Perform a complete cycle of prediction and update.
        
        This method:
        1. Predicts the next state
        2. Updates the filter state using provided measurement or simulated measurement
        
        Args:
            measurement: Optional measurement to use for update. If None, generates
                        a noisy measurement based on true_state or predicted state.
            true_state: Optional ground truth state to use for measurement generation
                       when measurement is None. If None, uses the filter's predicted state.
        
        Returns:
            Tuple of (predicted_state, updated_state, innovation, kalman_gain, 
                     kalman_gain_times_innovation, y_s_inv_y)
        """
        # Perform prediction step
        predicted_state = self.predict()
        
        # Use provided measurement or generate one
        if measurement is not None:
            z = measurement
        else:
            # Generate measurement (either from true state or from prediction)
            z = self.simulate_measurement(true_state=true_state if true_state is not None else predicted_state)
        
        # Perform update step with measurement
        updated_state, innovation, kalman_gain, kalman_gain_times_innovation, y_s_inv_y = self.update(z=z)
        distance_to_measurement = updated_state - z
        distance_to_evolution_model = updated_state - predicted_state
        
        # Log results in table format
        table_header = "| Metric                                               | Value       |"
        table_separator = "|------------------------------------------------------|-------------|"
        # Calculate distance to ground truth if available
        distance_to_ground_truth = updated_state - true_state if true_state is not None else None
        
        # Format values with proper conditionals
        true_state_str = f"{true_state:11.6f}" if true_state is not None else "        N/A"
        distance_to_ground_truth_str = f"{distance_to_ground_truth:11.6f}" if distance_to_ground_truth is not None else "        N/A"
        
        table_rows = [
            f"| Predict Step State (State Evolution Model)           | {predicted_state:11.6f} |",
            f"| Measurement     (SubspaceNet)                        | {z:11.6f} |",
            f"| EKF State   (EKF)                                    | {updated_state:11.6f} |",
            f"| True State (Ground Truth)                            | {true_state_str} |",
            f"| Distance to Measurement (SubspaceNet)                | {distance_to_measurement:11.6f} |",
            f"| Distance to State Evolution Model                    | {distance_to_evolution_model:11.6f} |",
            f"| Distance to Ground Truth                             | {distance_to_ground_truth_str} |",
            f"| Innovation (EKF)                                     | {innovation:11.6f} |",
            f"| Kalman Gain (SubspaceNet-State Evolution Model)      | {kalman_gain:11.6f} |"
        ]
        
        table_output = "\n".join([
            "EKF Prediction and Update Results:",
            table_header,
            table_separator
        ] + table_rows)
        
        logger.debug(f"\n{table_output}")
        return predicted_state, updated_state, innovation, kalman_gain, kalman_gain_times_innovation, y_s_inv_y 
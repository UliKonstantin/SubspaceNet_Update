"""
Batch Extended Kalman Filter implementation for non-linear dynamics.

This module provides a batch-enabled Extended Kalman Filter implementation 
capable of handling non-linear state evolution models for tracking multiple 
sources across multiple trajectories efficiently.
"""

import numpy as np
import torch
import logging
from config.schema import TrajectoryType
from simulation.kalman_filter.models import SineAccelStateModel, MultNoiseStateModel

logger = logging.getLogger("SubspaceNet.kalman_filter.batch_extended")

class BatchExtendedKalmanFilter1D:
    """
    Batch-enabled Extended Kalman Filter for non-linear state evolution models.
    
    This combines the efficiency of batch processing with the capability to handle
    non-linear dynamics using Extended Kalman Filter algorithms.
    
    Implements EKF for tracking scalar variables with non-linear dynamics:
    - State transition: x_k = f(x_{k-1}) + w_k
    - Observation model: z_k = x_k + v_k (H = 1)
    - w_k ~ N(0, Q(x)), v_k ~ N(0, R)
    
    Notes:
        - Uses PyTorch tensors for efficient batch operations
        - Can handle variable number of sources per trajectory using masks
        - Supports various non-linear state evolution models
    """
    
    def __init__(self, batch_size, max_sources, state_model, R, P0=1.0, device=None):
        """
        Initialize the batch Extended Kalman Filter.
        
        Args:
            batch_size: Number of trajectories in the batch
            max_sources: Maximum number of sources per trajectory
            state_model: StateEvolutionModel instance for non-linear dynamics
            R: Measurement noise variance (scalar or batch)
            P0: Initial state covariance (scalar or batch)
            device: PyTorch device (for GPU acceleration)
        """
        self.batch_size = batch_size
        self.max_sources = max_sources
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_model = state_model
        
        # Prevent exactly zero measurement noise (numerical stability)
        if isinstance(R, (int, float)) and R == 0:
            R = 1e-6
        
        # Convert scalars to tensors if needed
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
        
        logger.info(f"Initialized Batch Extended KF with batch_size={batch_size}, max_sources={max_sources} on {self.device}")
        
    @classmethod
    def from_config(cls, config, batch_size, max_sources, device=None, trajectory_type=None):
        """
        Create BatchExtendedKalmanFilter1D from configuration.
        
        Args:
            config: Configuration object containing kalman_filter and trajectory settings
            batch_size: Number of trajectories in the batch
            max_sources: Maximum number of sources
            device: PyTorch device
            trajectory_type: Type of trajectory to model (defaults to config value)
            
        Returns:
            BatchExtendedKalmanFilter1D instance
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
            omega0 = config.trajectory.sine_accel_omega0
            kappa = config.trajectory.sine_accel_kappa
            noise_std = kf_process_noise_std
            if isinstance(omega0, (int, float)):
                omega0 = [omega0] * max_sources
            if isinstance(kappa, (int, float)):
                kappa = [kappa] * max_sources
            from simulation.kalman_filter.extended import resolve_ekf_dc_offsets

            dc_arr = resolve_ekf_dc_offsets(config)
            state_model = SineAccelStateModel(omega0, kappa, noise_std, dc_offset=dc_arr)
            logger.info(
                f"Using batch sine acceleration model with ω₀={omega0}, κ={kappa}, "
                f"σ={noise_std}, dc_offset={dc_arr.tolist()}"
            )
            
        elif trajectory_type == TrajectoryType.MULT_NOISE_NONLINEAR:
            # Get multiplicative noise model parameters
            omega0 = config.trajectory.mult_noise_omega0
            amp = config.trajectory.mult_noise_amp
            # Use KF process noise instead of trajectory noise
            base_std = kf_process_noise_std
            
            state_model = MultNoiseStateModel(omega0, amp, base_std)
            logger.info(f"Using batch multiplicative noise model with ω₀={omega0}, amp={amp}, σ={base_std}")
            
        else:
            # Default to random walk model for other types
            # Use KF process noise instead of trajectory noise
            noise_std = kf_process_noise_std
            state_model = SineAccelStateModel(0.0, 0.0, noise_std, damping=1.0, angle_params_in_degrees=False)
            logger.info(f"Using batch random walk model with σ={noise_std}")
        
        # Get measurement noise and initial covariance
        kf_R = config.kalman_filter.measurement_noise_std_dev ** 2
        kf_P0 = config.kalman_filter.initial_covariance
        
        logger.info(f"Batch Extended Kalman Filter parameters: R={kf_R}, P0={kf_P0}")
        
        return cls(batch_size, max_sources, state_model, kf_R, kf_P0, device)
    
    def initialize_states(self, initial_states, num_sources):
        """
        Initialize the state estimates for multiple trajectories.
        
        Args:
            initial_states: Tensor of initial states [batch_size, max_sources]
            num_sources: Tensor or list with number of sources per trajectory [batch_size]
        """
        # Convert to tensor if needed
        if not isinstance(num_sources, torch.Tensor):
            num_sources = torch.tensor(num_sources, device=self.device)
            
        # Create source mask based on number of sources per trajectory
        self.source_mask.fill_(False)
        for i in range(self.batch_size):
            self.source_mask[i, :num_sources[i]] = True
        
        # Initialize states and mark as initialized
        self.x = initial_states.clone().to(self.device)
        self.P = self.P0.clone()
        self.state_initialized = self.source_mask.clone()

        if hasattr(self.state_model, "reset_time"):
            self.state_model.reset_time(1.0)
        
        logger.debug(f"Initialized batch states with shape {initial_states.shape}")
    
    def predict(self):
        """
        Perform the time update (prediction) step for all trajectories and sources.
        
        Uses non-linear state evolution with Jacobian linearization.
        
        Returns:
            Tensor of predicted states [batch_size, max_sources]
        """
        # Only update states that have been initialized
        mask = self.state_initialized & self.source_mask
        
        if mask.sum() == 0:
            return self.x

        # Apply state model on the full [batch_size, max_sources] grid.
        # Boolean mask indexing would flatten to 1D and break vectorized f_batch.
        x_pred = self._batch_state_function(self.x)
        F = self._batch_jacobian(self.x)
        Q = self._batch_noise_variance(self.x)

        P_pred = F * self.P * F + Q

        self.x[mask] = x_pred[mask]
        self.P[mask] = P_pred[mask]

        if hasattr(self.state_model, "advance_time"):
            self.state_model.advance_time()
        
        logger.debug(f"Batch prediction complete for {mask.sum().item()} states")
        return self.x
    
    def update(self, measurements, measurement_mask=None):
        """
        Perform the measurement update step for all trajectories and sources.
        
        Args:
            measurements: Tensor of measurements [batch_size, max_sources]
            measurement_mask: Boolean mask indicating valid measurements [batch_size, max_sources]
            
        Returns:
            Tuple of (updated states [batch_size, max_sources], error covariances [batch_size, max_sources])
        """
        if measurement_mask is None:
            measurement_mask = self.source_mask
        else:
            # Ensure we only update initialized and active sources
            measurement_mask = measurement_mask & self.state_initialized & self.source_mask
        
        # Only process valid measurements
        if measurement_mask.sum() == 0:
            return self.x, self.P
        
        # Innovation: y = z - H*x (H=1 for our measurement model)
        y = torch.zeros_like(self.x)
        y[measurement_mask] = measurements[measurement_mask] - self.x[measurement_mask]
        
        # Innovation covariance: S = H*P*H^T + R (H=1)
        S = self.P.clone()
        S[measurement_mask] = S[measurement_mask] + self.R[measurement_mask]
        
        # Kalman gain: K = P*H^T*S^(-1) (H=1)
        K = torch.zeros_like(self.P)
        K[measurement_mask] = self.P[measurement_mask] / S[measurement_mask]
        
        # State update: x_new = x + K * y
        x_new = self.x.clone()
        x_new[measurement_mask] = self.x[measurement_mask] + K[measurement_mask] * y[measurement_mask]
        
        # Covariance update: P_new = (I - K*H) * P (H=1)
        P_new = self.P.clone()
        P_new[measurement_mask] = (1 - K[measurement_mask]) * self.P[measurement_mask]
        
        # Update state and covariance
        self.x = x_new
        self.P = P_new
        
        logger.debug(f"Batch update complete for {measurement_mask.sum().item()} measurements")
        return self.x, self.P
    
    def predict_and_update(self, measurements, measurement_mask=None):
        """
        Perform both prediction and update steps in sequence.
        
        Args:
            measurements: Tensor of measurements [batch_size, max_sources]
            measurement_mask: Boolean mask indicating valid measurements [batch_size, max_sources]
            
        Returns:
            Tuple of (updated states [batch_size, max_sources], error covariances [batch_size, max_sources])
        """
        self.predict()
        return self.update(measurements, measurement_mask)
    
    def _as_state_batch(self, x_batch: torch.Tensor) -> torch.Tensor:
        """Ensure state tensor is 2D [batch_rows, max_sources] for vectorized models."""
        if x_batch.ndim == 2:
            return x_batch
        if x_batch.ndim == 1 and self.max_sources > 0 and x_batch.numel() % self.max_sources == 0:
            return x_batch.view(-1, self.max_sources)
        return x_batch.unsqueeze(0)

    def _call_state_model_batch(self, method_name: str, x_batch: torch.Tensor):
        """Invoke a batch method on the state model, preserving tensor device/dtype."""
        x_2d = self._as_state_batch(x_batch)
        method = getattr(self.state_model, method_name, None)
        if method is not None:
            result = method(x_2d)
        else:
            scalar_method = getattr(
                self.state_model,
                method_name.replace("_batch", ""),
            )
            flat = x_2d.reshape(-1)
            values = [scalar_method(x_i.item()) for x_i in flat]
            result = np.array(values).reshape(x_2d.shape)

        if isinstance(result, torch.Tensor):
            return result.to(device=self.device, dtype=x_batch.dtype)
        return torch.tensor(result, device=self.device, dtype=x_batch.dtype)

    def _batch_state_function(self, x_batch):
        """
        Apply the non-linear state evolution function to a batch of states.
        
        Args:
            x_batch: Tensor [batch_size, max_sources] or flattened active states [N]
            
        Returns:
            Tensor of predicted states with the same shape as x_batch
        """
        x_2d = self._as_state_batch(x_batch)
        f_x = self._call_state_model_batch("f_batch", x_2d)
        if x_batch.ndim == 1:
            return f_x.reshape(-1)
        return f_x
    
    def _batch_jacobian(self, x_batch):
        """
        Compute Jacobians for a batch of states.
        
        Args:
            x_batch: Tensor [batch_size, max_sources] or flattened active states [N]
            
        Returns:
            Tensor of Jacobians matching x_batch shape
        """
        x_2d = self._as_state_batch(x_batch)
        F_x = self._call_state_model_batch("F_jacobian_batch", x_2d)
        if x_batch.ndim == 1:
            return F_x.reshape(-1)
        return F_x
    
    def _batch_noise_variance(self, x_batch):
        """
        Compute noise variances for a batch of states.
        
        Args:
            x_batch: Tensor [batch_size, max_sources] or flattened active states [N]
            
        Returns:
            Tensor of noise variances matching x_batch shape
        """
        x_2d = self._as_state_batch(x_batch)
        Q_x = self._call_state_model_batch("noise_variance_batch", x_2d)
        if x_batch.ndim == 1:
            return Q_x.reshape(-1)
        return Q_x
    
    def simulate_measurements(self, true_states=None, measurement_mask=None):
        """
        Generate simulated noisy measurements for batch of states.
        
        Args:
            true_states: Optional tensor of true states [batch_size, max_sources].
                        If None, uses filter's current state estimates.
            measurement_mask: Boolean mask indicating which measurements to generate [batch_size, max_sources]
            
        Returns:
            Tensor of simulated noisy measurements [batch_size, max_sources]
        """
        if measurement_mask is None:
            measurement_mask = self.source_mask
        
        if true_states is None:
            if self.x is None:
                raise ValueError("States must be initialized before simulating measurements")
            states = self.x
        else:
            states = true_states.to(self.device)
        
        # Initialize measurements tensor
        measurements = torch.zeros_like(states)
        
        # Generate measurement noise for valid measurements
        if measurement_mask.sum() > 0:
            # Extract relevant R values for active measurements
            R_active = self.R[measurement_mask]
            
            # Generate noise: v ~ N(0, R)
            noise = torch.normal(0, torch.sqrt(R_active))
            
            # Measurement model: z = x + v
            measurements[measurement_mask] = states[measurement_mask] + noise
        
        logger.debug(f"Generated {measurement_mask.sum().item()} simulated measurements")
        return measurements 
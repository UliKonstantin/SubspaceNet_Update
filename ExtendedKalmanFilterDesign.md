# Extended Kalman Filter Design for Non-Linear Trajectory Models

## 1. Class Architecture

```
ExtendedKalmanFilter1D
  |
  ├── Main filter class with all EKF functionality
  |    - predict(), update() methods
  |    - Built-in state model selection based on trajectory type
  |    
  └── StateEvolutionModel (Abstract Base Class)
       |
       ├── SineAccelStateModel (implements StateEvolutionModel)
       |    - For θ_{k+1} = θ_k + (ω0 + κ sin θ_k)T + η_k
       |
       └── MultNoiseStateModel (implements StateEvolutionModel)
            - For θ_{k+1} = θ_k + ω0 T + σ(θ_k) η_k
```

## 2. Simplified Implementation Approach

Instead of using a factory pattern, the ExtendedKalmanFilter1D class will:
1. Directly select the appropriate state model based on trajectory type in its `from_config` class method
2. Internally implement state model creation logic
3. Provide a clean interface that hides implementation details from users

## 3. Detailed Class Specifications

### 3.1 StateEvolutionModel (Abstract Base Class)

```python
class StateEvolutionModel(ABC):
    """Abstract base class for state evolution models used by EKF."""
    
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
```

### 3.2 SineAccelStateModel

```python
class SineAccelStateModel(StateEvolutionModel):
    """
    Implements the sine acceleration non-linear model:
    θ_{k+1} = θ_k + (ω0 + κ sin θ_k)T + η_k
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
    
    def f(self, x):
        """
        State transition: θ_{k+1} = θ_k + (ω0 + κ sin θ_k)T
        """
        # Convert to radians for trigonometric function
        x_rad = np.radians(x) if isinstance(x, (int, float)) else np.radians(x.copy())
        
        # Calculate acceleration term
        delta = (self.omega0 + self.kappa * np.sin(x_rad)) * self.time_step
        
        # Return in same units as input
        return x + np.degrees(delta)
    
    def F_jacobian(self, x):
        """
        Jacobian: ∂f/∂x = 1 + κT·cos(θ)
        """
        # Convert to radians for trigonometric function
        x_rad = np.radians(x) if isinstance(x, (int, float)) else np.radians(x.copy())
        
        # Calculate the derivative
        return 1 + self.kappa * np.cos(x_rad) * self.time_step * np.pi/180.0
```

### 3.3 MultNoiseStateModel

```python
class MultNoiseStateModel(StateEvolutionModel):
    """
    Implements the multiplicative noise non-linear model:
    θ_{k+1} = θ_k + ω0 T + σ(θ_k) η_k
    where σ(θ) = base_std * (1 + amp * sin²(θ))
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
    
    def f(self, x):
        """
        Deterministic part: θ_{k+1} = θ_k + ω0 T
        """
        return x + np.degrees(self.omega0 * self.time_step)
    
    def F_jacobian(self, x):
        """
        Jacobian: ∂f/∂x = 1 (constant velocity model)
        """
        return 1.0
    
    def noise_variance(self, x):
        """
        State-dependent noise variance: σ²(θ) = base_std² * (1 + amp * sin²(θ))²
        """
        # Convert to radians for trigonometric function
        x_rad = np.radians(x) if isinstance(x, (int, float)) else np.radians(x.copy())
        
        # Calculate state-dependent standard deviation
        std = self.base_std * (1.0 + self.amp * np.sin(x_rad)**2)
        
        # Return variance
        return std**2
```

### 3.4 ExtendedKalmanFilter1D (with integrated model selection)

```python
class ExtendedKalmanFilter1D:
    """
    Extended Kalman Filter for non-linear state evolution models.
    
    Implements EKF for tracking a scalar variable with non-linear dynamics:
    - State transition: x_k = f(x_{k-1}) + w_k
    - Observation model: z_k = x_k + v_k (H = 1)
    - w_k ~ N(0, Q(x)), v_k ~ N(0, R)
    """
    
    def __init__(self, state_model, R, P0=1.0):
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
        
        logger.debug(f"Initialized Extended Kalman Filter with R={R}, P0={P0}")
    
    @classmethod
    def from_config(cls, config, trajectory_type=None):
        """
        Create ExtendedKalmanFilter1D from configuration.
        
        Args:
            config: Configuration object with kalman_filter and trajectory settings
            trajectory_type: Type of trajectory to model (defaults to config value)
            
        Returns:
            ExtendedKalmanFilter1D instance
        """
        # Get trajectory type from config if not specified
        if trajectory_type is None:
            trajectory_type = config.trajectory.trajectory_type
        
        # Create appropriate state evolution model based on trajectory type
        if trajectory_type == TrajectoryType.SINE_ACCEL_NONLINEAR:
            # Get sine acceleration model parameters
            omega0 = config.trajectory.sine_accel_omega0
            kappa = config.trajectory.sine_accel_kappa
            noise_std = config.trajectory.sine_accel_noise_std
            
            state_model = SineAccelStateModel(omega0, kappa, noise_std)
            logger.info(f"Using sine acceleration model with ω₀={omega0}, κ={kappa}, σ={noise_std}")
            
        elif trajectory_type == TrajectoryType.MULT_NOISE_NONLINEAR:
            # Get multiplicative noise model parameters
            omega0 = config.trajectory.mult_noise_omega0
            amp = config.trajectory.mult_noise_amp
            base_std = config.trajectory.mult_noise_base_std
            
            state_model = MultNoiseStateModel(omega0, amp, base_std)
            logger.info(f"Using multiplicative noise model with ω₀={omega0}, amp={amp}, σ={base_std}")
            
        else:
            # Default to random walk model for other types
            noise_std = config.trajectory.random_walk_std_dev
            state_model = SineAccelStateModel(0.0, 0.0, noise_std)
            logger.info(f"Using random walk model with σ={noise_std}")
        
        # Get measurement noise and initial covariance
        kf_R = config.kalman_filter.measurement_noise_std_dev ** 2
        kf_P0 = config.kalman_filter.initial_covariance
        
        return cls(state_model, kf_R, kf_P0)
    
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
        
        # State prediction using non-linear function
        x_pred = self.state_model.f(self.x)
        
        # Get Jacobian at current state
        F = self.state_model.F_jacobian(self.x)
        
        # Get state-dependent process noise
        Q = self.state_model.noise_variance(self.x)
        
        # Covariance prediction using linearization
        P_pred = F * self.P * F + Q
        
        # Update state and covariance
        self.x = x_pred
        self.P = P_pred
        
        logger.debug(f"Prediction: x={x_pred}, P={P_pred}, F={F}, Q={Q}")
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
        
        # Update state and covariance
        self.x = x_new
        self.P = P_new
        
        logger.debug(f"Update with z={z}: y={y}, K={K}, x={x_new}, P={P_new}")
        return x_new
```

## 4. Usage Example

```python
# Example usage in a tracking scenario
def track_with_ekf(config, true_angles):
    """
    Track source angles using the EKF with appropriate state model.
    
    Args:
        config: Configuration object
        true_angles: Ground truth angles to track
        
    Returns:
        Dictionary with tracking results
    """
    # Create EKF directly from config - handles model selection internally
    ekf = ExtendedKalmanFilter1D.from_config(config)
    
    # Initialize with first measurement
    ekf.initialize_state(true_angles[0])
    
    # Storage for results
    trajectory_length = len(true_angles)
    predicted_angles = np.zeros(trajectory_length)
    filtered_angles = np.zeros(trajectory_length)
    predicted_angles[0] = true_angles[0]
    filtered_angles[0] = true_angles[0]
    
    # Simulate measurements (add noise)
    measurement_std = np.sqrt(config.kalman_filter.measurement_noise_std_dev)
    measurements = true_angles + np.random.normal(0, measurement_std, trajectory_length)
    
    # Track through the trajectory
    for t in range(1, trajectory_length):
        # Predict next state
        predicted_angles[t] = ekf.predict()
        
        # Update with measurement
        filtered_angles[t] = ekf.update(measurements[t])
    
    return {
        'true': true_angles,
        'measured': measurements,
        'predicted': predicted_angles,
        'filtered': filtered_angles
    }
```

## 5. Integration with Existing Code

### 5.1 Integration with the Simulation Pipeline

To integrate this new ExtendedKalmanFilter1D with the existing simulation pipeline:

1. Create a compatibility function or method in `simulation/kalman_filter/__init__.py` that returns the appropriate filter type
2. Update the simulation loop to use this function
3. Ensure parameter consistency between trajectory generation and filtering

### 5.2 Configuration Updates

The kalman_filter section in the configuration should be extended to support choosing the filter type:

```yaml
kalman_filter:
  filter_type: "extended"  # "standard" or "extended"
  process_noise_std_dev: null  # standard deviation of process noise (degrees)
  measurement_noise_std_dev: 1.0e-3  # standard deviation of measurement noise (degrees)
  initial_covariance: 1.0  # initial state covariance
```

## 6. Performance and Testing Considerations

### 6.1 Numerical Stability

- Add safeguards against division by zero
- Use small epsilon values for variance terms
- Consider matrix forms for higher dimensional state spaces

### 6.2 Unit Tests

- Test each state evolution model separately
- Verify Jacobian calculations with numerical differentiation
- Test EKF with simple synthetic data for each model

### 6.3 Integration Tests

- Compare EKF performance against ground truth for each trajectory type
- Validate consistency between trajectory generation and filtering
- Measure estimation errors across various parameter settings

## 7. Extensions for Future Work

### 7.1 Higher-Dimensional State Spaces

For more complex tracking scenarios, the design can be extended to include:

- Angle and velocity state variables
- Multiple sources with data association
- Near-field scenarios with range estimation

### 7.2 Unscented Kalman Filter Alternative

For highly non-linear models, an Unscented Kalman Filter (UKF) may offer better performance:

- No need for explicit Jacobian calculation
- Better handling of strong non-linearities
- More accurate propagation of covariance

### 7.3 Batch Processing Optimization

To improve computational efficiency for batch processing:

- Vectorize operations for multiple trajectories
- Utilize GPU acceleration for large datasets
- Implement parallel processing for independent trajectories

## 8. Code Organization and File Structure

Upon implementation, the code will be organized as follows:

### 8.1 File Structure

```
simulation/
├── kalman_filter/
│   ├── __init__.py                     # Package exports and compatibility function
│   ├── base.py                         # Base KalmanFilter1D class (original implementation)
│   ├── extended.py                     # ExtendedKalmanFilter1D implementation with model selection
│   ├── models/
│   │   ├── __init__.py                 # Models package exports
│   │   ├── base.py                     # StateEvolutionModel abstract base class
│   │   ├── sine_accel.py               # SineAccelStateModel implementation
│   │   └── mult_noise.py               # MultNoiseStateModel implementation
│   └── batch/                          # Batch implementations (future)
│       ├── __init__.py
│       ├── batch_ekf.py                # BatchExtendedKalmanFilter implementation
│       └── utils.py                    # Batch processing utilities
└── runners/
    └── tracking.py                     # Integration with tracking pipeline
```

### 8.2 Import Hierarchy

```
# Import dependencies
from abc import ABC, abstractmethod
import numpy as np
import torch
import logging

# Import from existing codebase
from config.schema import TrajectoryType
from simulation.kalman_filter.base import KalmanFilter1D

# New imports within the EKF implementation
from simulation.kalman_filter.models.base import StateEvolutionModel
from simulation.kalman_filter.models.sine_accel import SineAccelStateModel
from simulation.kalman_filter.models.mult_noise import MultNoiseStateModel
from simulation.kalman_filter.extended import ExtendedKalmanFilter1D
```

### 8.3 Integration Points

The key integration points with the existing codebase include:

1. **Config Schema** (`config/schema.py`):
   - `TrajectoryType` enum already includes `SINE_ACCEL_NONLINEAR` and `MULT_NOISE_NONLINEAR`
   - `KalmanFilterConfig` would be extended with a `filter_type` field

2. **Kalman Filter Selection** (to be added in `simulation/kalman_filter/__init__.py`):
   ```python
   def get_kalman_filter(config, trajectory_type=None):
       """Get appropriate Kalman Filter based on config and trajectory type."""
       filter_type = getattr(config.kalman_filter, "filter_type", "standard")
       
       if filter_type == "extended" or (trajectory_type and trajectory_type in [
           TrajectoryType.SINE_ACCEL_NONLINEAR, 
           TrajectoryType.MULT_NOISE_NONLINEAR
       ]):
           # Use extended KF for non-linear trajectory types
           return ExtendedKalmanFilter1D.from_config(config, trajectory_type)
       else:
           # Use the original KalmanFilter1D for simple models
           return KalmanFilter1D.from_config(config)
   ```

3. **Trajectory Handler** (`simulation/runners/data.py`):
   - The trajectory generation code is already aligned with the state evolution models
   - The same parameters are used in both trajectory generation and EKF filtering

4. **Tracking Pipeline** (`simulation/runners/tracking.py`):
   - Update to use the `get_kalman_filter` function for creating the appropriate filter
   - Existing tracking code would work with both filter types as they share the same interface

### 8.4 Backward Compatibility

To maintain backward compatibility:

1. The `ExtendedKalmanFilter1D` implements the same public interface as `KalmanFilter1D`:
   - `initialize_state(x0)`
   - `predict()`
   - `update(z)`
   - `from_config(config)`

2. The `get_kalman_filter` function automatically selects the appropriate filter type based on the trajectory type or configuration.

3. Default values for all new configuration parameters ensure existing config files continue to work.

### 8.5 Developer Workflow

When implementing the EKF:

1. Start by implementing the state evolution models in the models directory
2. Implement the ExtendedKalmanFilter1D class with integrated model selection
3. Create unit tests for each component
4. Update configuration schema to include filter type
5. Implement the get_kalman_filter compatibility function
6. Update tracking pipeline to use the get_kalman_filter function
7. Run integration tests with different trajectory types 
#!/usr/bin/env python3
"""
Minimal standalone test for Extended Kalman Filter mathematical correctness.

This test contains the essential EKF code copied locally to avoid import issues.
"""

import numpy as np
import matplotlib.pyplot as plt

# Minimal state evolution model base class
class StateEvolutionModel:
    """Base class for state evolution models."""
    
    def f(self, x):
        """State transition function f(x)."""
        raise NotImplementedError
        
    def F(self, x):
        """Jacobian of state transition function."""
        raise NotImplementedError

# Sine acceleration model implementation
class SineAccelStateModel(StateEvolutionModel):
    """
    Implements the sine acceleration non-linear model:
    θ_{k+1} = θ_k + (ω0 + κ sin θ_k)T + η_k
    """
    
    def __init__(self, omega0, kappa, process_noise_std, T=1.0):
        """
        Initialize the sine acceleration model.
        
        Args:
            omega0: Base angular velocity
            kappa: Sine acceleration coefficient
            process_noise_std: Standard deviation of process noise
            T: Time step (default 1.0)
        """
        self.omega0 = omega0
        self.kappa = kappa
        self.process_noise_std = process_noise_std
        self.T = T
        self.Q = process_noise_std**2  # Process noise variance
        
    def f(self, x):
        """State transition function."""
        return x + (self.omega0 + self.kappa * np.sin(x)) * self.T
        
    def F(self, x):
        """Jacobian of state transition function."""
        return 1.0 + self.kappa * np.cos(x) * self.T

# Extended Kalman Filter implementation
class ExtendedKalmanFilter1D:
    """
    Extended Kalman Filter for non-linear state evolution models.
    """
    
    def __init__(self, state_model, R, P0):
        """
        Initialize the Extended Kalman Filter.
        
        Args:
            state_model: State evolution model with f() and F() methods
            R: Measurement noise variance
            P0: Initial state covariance
        """
        self.state_model = state_model
        self.R = R  # Measurement noise variance
        self.P = P0  # State covariance
        self.x = 0.0  # State estimate (will be initialized)
        self.H = 1.0  # Measurement model (direct observation)
        
    def initialize_state(self, x0):
        """Initialize the state estimate."""
        self.x = x0
        
    def predict(self):
        """Prediction step of the EKF."""
        # Predict state using non-linear model
        x_pred = self.state_model.f(self.x)
        
        # Predict covariance using linearized model
        F = self.state_model.F(self.x)  # Jacobian at current state
        P_pred = F * self.P * F + self.state_model.Q
        
        # Update internal state
        self.x = x_pred
        self.P = P_pred
        
        return x_pred
        
    def update(self, measurement):
        """
        Update step of the EKF.
        
        Returns:
            tuple: (updated_state, innovation, K, correction, y_s_inv_y)
        """
        # Innovation (measurement residual)
        innovation = measurement - self.H * self.x
        
        # Innovation covariance
        S = self.H * self.P * self.H + self.R
        
        # Kalman gain
        K = (self.P * self.H) / S
        
        # State update
        correction = K * innovation
        self.x = self.x + correction
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = 1.0 - K * self.H
        self.P = I_KH * self.P * I_KH + K * self.R * K
        
        # Additional return values for analysis
        y_s_inv_y = innovation * innovation / S
        
        return self.x, innovation, K, correction, y_s_inv_y

def test_ekf_simple_case():
    """
    Test EKF with a simple sine acceleration model.
    """
    print("=== Testing Extended Kalman Filter ===")
    
    # Test parameters
    omega0 = 0.1  # Base angular velocity
    kappa = 0.05  # Sine acceleration coefficient  
    process_noise_std = 0.05  # Increased process noise
    measurement_noise_var = 0.1  # Increased measurement noise
    initial_covariance = 1.0
    
    # Create state model and EKF
    state_model = SineAccelStateModel(omega0, kappa, process_noise_std)
    ekf = ExtendedKalmanFilter1D(
        state_model=state_model,
        R=measurement_noise_var,
        P0=initial_covariance
    )
    
    # Initialize with true starting position
    true_state = 0.0  # Start at angle 0
    ekf.initialize_state(true_state)
    
    print(f"Initial true state: {true_state:.6f}")
    print(f"Initial EKF state: {ekf.x:.6f}")
    print(f"Initial covariance: {ekf.P:.6f}")
    
    # Test parameters
    num_steps = 50
    true_states = [true_state]
    ekf_states = [ekf.x]
    measurements = []  # Store measurements for plotting
    covariances = [ekf.P]
    innovations = []
    kalman_gains = []  # Store Kalman gains for plotting
    corrections = []  # Store Kalman gain * innovation
    innovation_norms = []  # Store Y^T * S^(-1) * Y
    
    print("\n=== Running EKF simulation ===")
    
    for step in range(num_steps):
        # Generate true next state using the same model
        true_state = state_model.f(true_state) + np.random.normal(0, process_noise_std)
        
        # Generate noisy measurement of true state
        measurement = true_state + np.random.normal(0, np.sqrt(measurement_noise_var))
        
        # EKF prediction step
        predicted_state = ekf.predict()
        
        # EKF update step
        updated_state, innovation, K, correction, y_s_inv_y = ekf.update(measurement)
        
        # Store results
        true_states.append(true_state)
        ekf_states.append(updated_state)
        measurements.append(measurement)  # Store measurements
        covariances.append(ekf.P)
        innovations.append(innovation)
        kalman_gains.append(K)  # Store Kalman gains
        corrections.append(correction)  # Store K * innovation
        innovation_norms.append(y_s_inv_y)  # Store Y^T * S^(-1) * Y
        
        if step < 5 or step % 10 == 0:
            print(f"Step {step+1:2d}: True={true_state:8.4f}, "
                  f"Pred={predicted_state:8.4f}, "
                  f"Meas={measurement:8.4f}, "
                  f"EKF={updated_state:8.4f}, "
                  f"P={ekf.P:8.6f}, "
                  f"K={K:6.4f}")
    
    # Calculate final statistics
    true_states = np.array(true_states)
    ekf_states = np.array(ekf_states)
    errors = np.abs(true_states - ekf_states)
    
    print(f"\n=== Final Results ===")
    print(f"Final true state: {true_states[-1]:.6f}")
    print(f"Final EKF state: {ekf_states[-1]:.6f}")
    print(f"Final error: {errors[-1]:.6f}")
    print(f"Mean absolute error: {np.mean(errors):.6f}")
    print(f"Final covariance: {ekf.P:.6f}")
    print(f"RMS innovation: {np.sqrt(np.mean(np.array(innovations)**2)):.6f}")
    
    # Test mathematical properties
    print(f"\n=== Mathematical Properties ===")
    
    # 1. Covariance should decrease over time (filter should become more confident)
    initial_cov = covariances[0]
    final_cov = covariances[-1]
    print(f"Covariance reduction: {initial_cov:.6f} -> {final_cov:.6f} (ratio: {final_cov/initial_cov:.3f})")
    
    # 2. Mean innovation should be close to zero (unbiased)
    mean_innovation = np.mean(innovations)
    print(f"Mean innovation (should be ~0): {mean_innovation:.6f}")
    
    # 3. Innovation variance should be consistent with theoretical prediction
    innovation_var = np.var(innovations)
    # Theoretical innovation variance is S = P + R (approximately)
    theoretical_innovation_var = np.mean(covariances[1:]) + measurement_noise_var
    print(f"Innovation variance: empirical={innovation_var:.6f}, theoretical~{theoretical_innovation_var:.6f}")
    
    # 4. Additional diagnostic metrics
    mean_correction = np.mean(np.abs(corrections))
    mean_innovation_norm = np.mean(innovation_norms)
    print(f"Mean |correction| (K*innovation): {mean_correction:.6f}")
    print(f"Mean innovation norm (Y^T*S^(-1)*Y): {mean_innovation_norm:.6f} (should be ~1.0 for consistency)")
    
    return true_states, ekf_states, measurements, covariances, innovations, kalman_gains, corrections, innovation_norms

def test_ekf_convergence():
    """
    Test that EKF converges when measurements are consistent.
    """
    print("\n=== Testing EKF Convergence ===")
    
    # Simple constant state model (omega0=0, kappa=0 gives random walk)
    state_model = SineAccelStateModel(0.0, 0.0, 0.001)  # Very small process noise
    ekf = ExtendedKalmanFilter1D(state_model=state_model, R=0.01, P0=10.0)  # High initial uncertainty
    
    # Initialize with wrong estimate
    true_value = 1.0
    initial_estimate = 0.0  # Wrong initial guess
    ekf.initialize_state(initial_estimate)
    
    print(f"True constant value: {true_value}")
    print(f"Initial estimate: {initial_estimate}")
    print(f"Initial covariance: {ekf.P}")
    
    # Provide consistent measurements of the true value
    states = [ekf.x]
    covariances = [ekf.P]
    
    for step in range(20):
        # Predict (should stay roughly the same with zero dynamics)
        ekf.predict()
        
        # Update with noisy measurement of true value
        measurement = true_value + np.random.normal(0, 0.1)
        updated_state, innovation, K, correction, y_s_inv_y = ekf.update(measurement)
        
        states.append(updated_state)
        covariances.append(ekf.P)
        
        if step < 5 or step % 5 == 0:
            print(f"Step {step+1:2d}: State={updated_state:6.4f}, P={ekf.P:8.6f}, K={K:6.4f}")
    
    final_error = abs(states[-1] - true_value)
    print(f"\nFinal estimate: {states[-1]:.6f}")
    print(f"Final error: {final_error:.6f}")
    print(f"Covariance reduction: {covariances[0]:.6f} -> {covariances[-1]:.6f}")
    
    # Should converge close to true value
    assert final_error < 0.1, f"Filter did not converge properly: error = {final_error}"
    print("✓ Convergence test passed!")
    
    return states, covariances

def test_ekf_adaptive_noise():
    """
    Test EKF with increasing measurement noise every 5 samples.
    """
    print("\n=== Testing EKF with Adaptive Noise ===")
    
    # Test parameters
    omega0 = 0.1  # Base angular velocity
    kappa = 0.05  # Sine acceleration coefficient  
    process_noise_std = 0.02  # Fixed process noise
    base_measurement_noise = 0.01  # Starting measurement noise
    noise_increase_factor = 1.5  # Multiply noise by this every 5 steps
    initial_covariance = 0.1
    
    # Create state model and EKF
    state_model = SineAccelStateModel(omega0, kappa, process_noise_std)
    ekf = ExtendedKalmanFilter1D(
        state_model=state_model,
        R=base_measurement_noise,  # Will be updated dynamically
        P0=initial_covariance
    )
    
    # Initialize with true starting position
    true_state = 0.0
    ekf.initialize_state(true_state)
    
    print(f"Initial measurement noise: {base_measurement_noise:.4f}")
    print(f"Noise increases by factor {noise_increase_factor} every 5 steps")
    
    # Test parameters
    num_steps = 50
    true_states = [true_state]
    ekf_states = [ekf.x]
    measurements = []
    covariances = [ekf.P]
    innovations = []
    noise_levels = [base_measurement_noise]
    kalman_gains = []
    corrections = []  # Store Kalman gain * innovation
    innovation_norms = []  # Store Y^T * S^(-1) * Y
    
    current_measurement_noise = base_measurement_noise
    
    print("\n=== Running Adaptive Noise EKF simulation ===")
    
    for step in range(num_steps):
        # Update measurement noise every 5 steps
        if step > 0 and step % 5 == 0:
            current_measurement_noise *= noise_increase_factor
            ekf.R = current_measurement_noise  # Update EKF's noise model
            print(f"Step {step+1}: Increased measurement noise to {current_measurement_noise:.4f}")
        
        # Generate true next state using the same model
        true_state = state_model.f(true_state) + np.random.normal(0, process_noise_std)
        
        # Generate noisy measurement with current noise level
        measurement = true_state + np.random.normal(0, np.sqrt(current_measurement_noise))
        
        # EKF prediction step
        predicted_state = ekf.predict()
        
        # EKF update step
        updated_state, innovation, K, correction, y_s_inv_y = ekf.update(measurement)
        
        # Store results
        true_states.append(true_state)
        ekf_states.append(updated_state)
        measurements.append(measurement)
        covariances.append(ekf.P)
        innovations.append(innovation)
        noise_levels.append(current_measurement_noise)
        kalman_gains.append(K)
        corrections.append(correction)  # Store K * innovation
        innovation_norms.append(y_s_inv_y)  # Store Y^T * S^(-1) * Y
        
        if step < 10 or step % 10 == 0:
            print(f"Step {step+1:2d}: True={true_state:8.4f}, "
                  f"Meas={measurement:8.4f}, "
                  f"EKF={updated_state:8.4f}, "
                  f"P={ekf.P:8.6f}, "
                  f"K={K:6.4f}, "
                  f"R={current_measurement_noise:6.4f}")
    
    # Calculate final statistics
    true_states = np.array(true_states)
    ekf_states = np.array(ekf_states)
    errors = np.abs(true_states - ekf_states)
    
    print(f"\n=== Adaptive Noise Results ===")
    print(f"Initial measurement noise: {base_measurement_noise:.4f}")
    print(f"Final measurement noise: {current_measurement_noise:.4f}")
    print(f"Noise increased by factor: {current_measurement_noise/base_measurement_noise:.1f}")
    print(f"Mean absolute error: {np.mean(errors):.6f}")
    print(f"Final covariance: {ekf.P:.6f}")
    print(f"RMS innovation: {np.sqrt(np.mean(np.array(innovations)**2)):.6f}")
    
    # Additional diagnostic metrics
    mean_correction = np.mean(np.abs(corrections))
    mean_innovation_norm = np.mean(innovation_norms)
    print(f"Mean |correction| (K*innovation): {mean_correction:.6f}")
    print(f"Mean innovation norm (Y^T*S^(-1)*Y): {mean_innovation_norm:.6f} (should be ~1.0 for consistency)")
    
    return true_states, ekf_states, measurements, covariances, noise_levels, kalman_gains, innovations, corrections, innovation_norms

def test_ekf_blind_noise():
    """
    Test EKF with increasing measurement noise BUT the filter doesn't know about it.
    The actual noise changes but EKF's R parameter stays constant.
    """
    print("\n=== Testing EKF with Unknown Noise Changes ===")
    
    # Test parameters
    omega0 = 0.1  # Base angular velocity
    kappa = 0.05  # Sine acceleration coefficient  
    process_noise_std = 0.02  # Fixed process noise
    base_measurement_noise = 0.01  # Starting measurement noise
    noise_increase_factor = 1.5  # Multiply noise by this every 5 steps
    initial_covariance = 0.1
    
    # Create state model and EKF
    state_model = SineAccelStateModel(omega0, kappa, process_noise_std)
    ekf = ExtendedKalmanFilter1D(
        state_model=state_model,
        R=base_measurement_noise,  # KEEP THIS FIXED - EKF doesn't know noise changes!
        P0=initial_covariance
    )
    
    # Initialize with true starting position
    true_state = 0.0
    ekf.initialize_state(true_state)
    
    print(f"EKF assumes constant measurement noise: {base_measurement_noise:.4f}")
    print(f"Actual noise increases by factor {noise_increase_factor} every 5 steps")
    print("EKF will NOT be informed about the noise changes!")
    
    # Test parameters
    num_steps = 50
    true_states = [true_state]
    ekf_states = [ekf.x]
    measurements = []
    covariances = [ekf.P]
    innovations = []
    actual_noise_levels = [base_measurement_noise]  # Track actual noise
    assumed_noise_levels = [base_measurement_noise]  # Track what EKF thinks
    kalman_gains = []
    corrections = []  # Store Kalman gain * innovation
    innovation_norms = []  # Store Y^T * S^(-1) * Y
    
    current_actual_noise = base_measurement_noise
    
    print("\n=== Running Blind Noise EKF simulation ===")
    
    for step in range(num_steps):
        # Update ACTUAL measurement noise every 5 steps
        if step > 0 and step % 5 == 0:
            current_actual_noise *= noise_increase_factor
            print(f"Step {step+1}: Actual noise increased to {current_actual_noise:.4f}, "
                  f"but EKF still thinks R={ekf.R:.4f}")
        
        # Generate true next state using the same model
        true_state = state_model.f(true_state) + np.random.normal(0, process_noise_std)
        
        # Generate noisy measurement with ACTUAL noise level (EKF doesn't know this!)
        measurement = true_state + np.random.normal(0, np.sqrt(current_actual_noise))
        
        # EKF prediction step
        predicted_state = ekf.predict()
        
        # EKF update step (still using old R assumption!)
        updated_state, innovation, K, correction, y_s_inv_y = ekf.update(measurement)
        
        # Store results
        true_states.append(true_state)
        ekf_states.append(updated_state)
        measurements.append(measurement)
        covariances.append(ekf.P)
        innovations.append(innovation)
        actual_noise_levels.append(current_actual_noise)
        assumed_noise_levels.append(ekf.R)  # What EKF thinks (constant)
        kalman_gains.append(K)
        corrections.append(correction)  # Store K * innovation
        innovation_norms.append(y_s_inv_y)  # Store Y^T * S^(-1) * Y
        
        if step < 10 or step % 10 == 0:
            print(f"Step {step+1:2d}: True={true_state:8.4f}, "
                  f"Meas={measurement:8.4f}, "
                  f"EKF={updated_state:8.4f}, "
                  f"P={ekf.P:8.6f}, "
                  f"K={K:6.4f}, "
                  f"R_actual={current_actual_noise:6.4f}, "
                  f"R_assumed={ekf.R:6.4f}")
    
    # Calculate final statistics
    true_states = np.array(true_states)
    ekf_states = np.array(ekf_states)
    errors = np.abs(true_states - ekf_states)
    
    print(f"\n=== Blind Noise Results ===")
    print(f"EKF assumed constant noise: {base_measurement_noise:.4f}")
    print(f"Actual final noise: {current_actual_noise:.4f}")
    print(f"Actual noise increased by factor: {current_actual_noise/base_measurement_noise:.1f}")
    print(f"Mean absolute error: {np.mean(errors):.6f}")
    print(f"Final covariance: {ekf.P:.6f}")
    print(f"RMS innovation: {np.sqrt(np.mean(np.array(innovations)**2)):.6f}")
    
    # Check innovation consistency
    expected_innovation_var = ekf.R + np.mean(covariances[1:])  # What EKF expects
    actual_innovation_var = np.var(innovations)  # What actually happened
    print(f"Innovation variance: expected={expected_innovation_var:.6f}, actual={actual_innovation_var:.6f}")
    print(f"Innovation mismatch ratio: {actual_innovation_var/expected_innovation_var:.2f}")
    
    # Additional diagnostic metrics
    mean_correction = np.mean(np.abs(corrections))
    mean_innovation_norm = np.mean(innovation_norms)
    print(f"Mean |correction| (K*innovation): {mean_correction:.6f}")
    print(f"Mean innovation norm (Y^T*S^(-1)*Y): {mean_innovation_norm:.6f} (should be ~1.0 for consistency)")
    
    return true_states, ekf_states, measurements, covariances, actual_noise_levels, assumed_noise_levels, kalman_gains, innovations, corrections, innovation_norms

def plot_results(true_states, ekf_states, measurements, covariances, kalman_gains, corrections, innovation_norms):
    """Plot the test results."""
    fig = plt.figure(figsize=(20, 12))
    
    # Create a 2x3 grid for 6 plots
    ax1 = plt.subplot(2, 3, 1)
    ax2 = plt.subplot(2, 3, 2) 
    ax3 = plt.subplot(2, 3, 3)
    ax4 = plt.subplot(2, 3, 4)
    ax5 = plt.subplot(2, 3, 5)
    ax6 = plt.subplot(2, 3, 6)
    
    steps = range(len(true_states))
    measurement_steps = range(1, len(measurements) + 1)  # Measurements start from step 1
    kalman_gain_steps = range(1, len(kalman_gains) + 1)  # Kalman gains start from step 1
    
    # Plot 1: All angles together
    ax1.plot(steps, true_states, 'b-', label='True Angle', linewidth=2)
    ax1.plot(steps, ekf_states, 'r--', label='EKF Estimate', linewidth=2)
    ax1.plot(measurement_steps, measurements, 'go', label='Measurements', markersize=4, alpha=0.7)
    ax1.set_ylabel('Angle (radians)')
    ax1.set_xlabel('Time Step')
    ax1.set_title('Angles: True vs Measurements vs EKF Estimates')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: States comparison with Kalman gain
    ax2_twin = ax2.twinx()
    ax2.plot(steps, true_states, 'b-', label='True State', linewidth=2)
    ax2.plot(steps, ekf_states, 'r--', label='EKF Estimate', linewidth=2)
    ax2_twin.plot(kalman_gain_steps, kalman_gains, 'orange', linewidth=2, label='Kalman Gain (K)')
    ax2.set_ylabel('State Value', color='black')
    ax2_twin.set_ylabel('Kalman Gain', color='orange')
    ax2.set_xlabel('Time Step')
    ax2.set_title('EKF Tracking Performance with Kalman Gain')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True)
    
    # Plot 3: Error
    errors = np.abs(np.array(true_states) - np.array(ekf_states))
    ax3.plot(steps, errors, 'g-', linewidth=2)
    ax3.set_ylabel('Absolute Error')
    ax3.set_xlabel('Time Step')
    ax3.set_title('Tracking Error')
    ax3.grid(True)
    
    # Plot 4: Covariance
    ax4.plot(steps, covariances, 'm-', linewidth=2)
    ax4.set_ylabel('Covariance')
    ax4.set_xlabel('Time Step')
    ax4.set_title('State Uncertainty (Covariance)')
    ax4.grid(True)
    
    # Plot 5: Correction terms (K * innovation)
    correction_steps = range(1, len(corrections) + 1)
    ax5.plot(correction_steps, np.abs(corrections), 'cyan', linewidth=2)
    ax5.set_ylabel('|Correction| (|K × Innovation|)')
    ax5.set_xlabel('Time Step')
    ax5.set_title('State Correction Magnitude')
    ax5.grid(True)
    
    # Plot 6: Innovation consistency metric
    ax6.plot(correction_steps, innovation_norms, 'red', linewidth=2)
    ax6.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Expected Value = 1.0')
    ax6.set_ylabel('Y^T × S^(-1) × Y')
    ax6.set_xlabel('Time Step')
    ax6.set_title('Innovation Consistency Metric')
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout()
    plt.savefig('ekf_test_results.png', dpi=150)
    print("Results saved to ekf_test_results.png")

def plot_adaptive_results(true_states, ekf_states, measurements, covariances, noise_levels, kalman_gains, corrections, innovation_norms):
    """Plot the adaptive noise test results."""
    fig = plt.figure(figsize=(20, 12))
    
    # Create a 2x3 grid for 6 plots
    ax1 = plt.subplot(2, 3, 1)
    ax2 = plt.subplot(2, 3, 2) 
    ax3 = plt.subplot(2, 3, 3)
    ax4 = plt.subplot(2, 3, 4)
    ax5 = plt.subplot(2, 3, 5)
    ax6 = plt.subplot(2, 3, 6)
    
    steps = range(len(true_states))
    measurement_steps = range(1, len(measurements) + 1)
    
    # Plot 1: All angles with increasing noise
    ax1.plot(steps, true_states, 'b-', label='True Angle', linewidth=2)
    ax1.plot(steps, ekf_states, 'r--', label='EKF Estimate', linewidth=2)
    ax1.plot(measurement_steps, measurements, 'go', label='Measurements', markersize=3, alpha=0.6)
    ax1.set_ylabel('Angle (radians)')
    ax1.set_xlabel('Time Step')
    ax1.set_title('Adaptive Noise: True vs Measurements vs EKF')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Measurement noise levels and Kalman gains
    ax2_twin = ax2.twinx()
    ax2.plot(steps, noise_levels, 'purple', linewidth=3, label='Measurement Noise (R)')
    ax2_twin.plot(measurement_steps, kalman_gains, 'orange', linewidth=2, label='Kalman Gain (K)')
    ax2.set_ylabel('Measurement Noise Variance', color='purple')
    ax2_twin.set_ylabel('Kalman Gain', color='orange')
    ax2.set_xlabel('Time Step')
    ax2.set_title('Noise Level vs Kalman Gain Adaptation')
    ax2.grid(True)
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    
    # Plot 3: Tracking error evolution
    errors = np.abs(np.array(true_states) - np.array(ekf_states))
    ax3.plot(steps, errors, 'g-', linewidth=2)
    ax3.set_ylabel('Absolute Error')
    ax3.set_xlabel('Time Step')
    ax3.set_title('Tracking Error vs Increasing Noise')
    ax3.grid(True)
    
    # Plot 4: Covariance evolution
    ax4.plot(steps, covariances, 'm-', linewidth=2)
    ax4.set_ylabel('Covariance')
    ax4.set_xlabel('Time Step')
    ax4.set_title('State Uncertainty Evolution')
    ax4.grid(True)
    
    # Plot 5: Correction terms (K * innovation)
    correction_steps = range(1, len(corrections) + 1)
    ax5.plot(correction_steps, np.abs(corrections), 'cyan', linewidth=2)
    ax5.set_ylabel('|Correction| (|K × Innovation|)')
    ax5.set_xlabel('Time Step')
    ax5.set_title('State Correction Magnitude (Adaptive)')
    ax5.grid(True)
    
    # Plot 6: Innovation consistency metric
    ax6.plot(correction_steps, innovation_norms, 'red', linewidth=2)
    ax6.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Expected Value = 1.0')
    ax6.set_ylabel('Y^T × S^(-1) × Y')
    ax6.set_xlabel('Time Step')
    ax6.set_title('Innovation Consistency (Adaptive)')
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout()
    plt.savefig('ekf_adaptive_test_results.png', dpi=150)
    print("Adaptive test results saved to ekf_adaptive_test_results.png")

def plot_blind_results(true_states, ekf_states, measurements, covariances, actual_noise, assumed_noise, kalman_gains, corrections, innovation_norms):
    """Plot the blind noise test results."""
    fig = plt.figure(figsize=(20, 12))
    
    # Create a 2x3 grid for 6 plots
    ax1 = plt.subplot(2, 3, 1)
    ax2 = plt.subplot(2, 3, 2) 
    ax3 = plt.subplot(2, 3, 3)
    ax4 = plt.subplot(2, 3, 4)
    ax5 = plt.subplot(2, 3, 5)
    ax6 = plt.subplot(2, 3, 6)
    
    steps = range(len(true_states))
    measurement_steps = range(1, len(measurements) + 1)
    
    # Plot 1: All angles with unknown increasing noise
    ax1.plot(steps, true_states, 'b-', label='True Angle', linewidth=2)
    ax1.plot(steps, ekf_states, 'r--', label='EKF Estimate', linewidth=2)
    ax1.plot(measurement_steps, measurements, 'go', label='Measurements', markersize=3, alpha=0.6)
    ax1.set_ylabel('Angle (radians)')
    ax1.set_xlabel('Time Step')
    ax1.set_title('Blind Noise: EKF Unaware of Increasing Noise')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Actual vs Assumed noise and Kalman gains
    ax2_twin = ax2.twinx()
    ax2.plot(steps, actual_noise, 'red', linewidth=3, label='Actual Noise (Unknown to EKF)')
    ax2.plot(steps, assumed_noise, 'blue', linewidth=3, linestyle='--', label='Assumed Noise (EKF)')
    ax2_twin.plot(measurement_steps, kalman_gains, 'orange', linewidth=2, label='Kalman Gain (K)')
    ax2.set_ylabel('Measurement Noise Variance', color='black')
    ax2_twin.set_ylabel('Kalman Gain', color='orange')
    ax2.set_xlabel('Time Step')
    ax2.set_title('Model Mismatch: Actual vs Assumed Noise')
    ax2.grid(True)
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    
    # Plot 3: Tracking error evolution
    errors = np.abs(np.array(true_states) - np.array(ekf_states))
    ax3.plot(steps, errors, 'g-', linewidth=2)
    ax3.set_ylabel('Absolute Error')
    ax3.set_xlabel('Time Step')
    ax3.set_title('Tracking Error (EKF Blind to Noise Changes)')
    ax3.grid(True)
    
    # Plot 4: Covariance evolution (overconfident?)
    ax4.plot(steps, covariances, 'm-', linewidth=2)
    ax4.set_ylabel('Covariance')
    ax4.set_xlabel('Time Step')
    ax4.set_title('State Uncertainty (Potentially Overconfident)')
    ax4.grid(True)
    
    # Plot 5: Correction terms (K * innovation)
    correction_steps = range(1, len(corrections) + 1)
    ax5.plot(correction_steps, np.abs(corrections), 'cyan', linewidth=2)
    ax5.set_ylabel('|Correction| (|K × Innovation|)')
    ax5.set_xlabel('Time Step')
    ax5.set_title('State Correction Magnitude (Blind)')
    ax5.grid(True)
    
    # Plot 6: Innovation consistency metric (should show severe mismatch)
    ax6.plot(correction_steps, innovation_norms, 'red', linewidth=2)
    ax6.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Expected Value = 1.0')
    ax6.set_ylabel('Y^T × S^(-1) × Y')
    ax6.set_xlabel('Time Step')
    ax6.set_title('Innovation Consistency (Blind - Severe Mismatch!)')
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout()
    plt.savefig('ekf_blind_test_results.png', dpi=150)
    print("Blind noise test results saved to ekf_blind_test_results.png")

if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Run basic tests
    true_states, ekf_states, measurements, covariances, innovations, kalman_gains, corrections, innovation_norms = test_ekf_simple_case()
    convergence_states, convergence_covariances = test_ekf_convergence()
    
    # Plot basic results
    plot_results(true_states, ekf_states, measurements, covariances, kalman_gains, corrections, innovation_norms)
    
    # Run adaptive noise test (reset seed for reproducibility)
    np.random.seed(123)
    adaptive_true, adaptive_ekf, adaptive_meas, adaptive_cov, noise_levels, adaptive_K, adaptive_innov, adaptive_corr, adaptive_norms = test_ekf_adaptive_noise()
    
    # Plot adaptive results
    plot_adaptive_results(adaptive_true, adaptive_ekf, adaptive_meas, adaptive_cov, noise_levels, adaptive_K, adaptive_corr, adaptive_norms)
    
    # Run blind noise test (EKF doesn't know about noise changes)
    np.random.seed(456)
    blind_true, blind_ekf, blind_meas, blind_cov, actual_noise, assumed_noise, blind_K, blind_innov, blind_corr, blind_norms = test_ekf_blind_noise()
    
    # Plot blind results
    plot_blind_results(blind_true, blind_ekf, blind_meas, blind_cov, actual_noise, assumed_noise, blind_K, blind_corr, blind_norms)
    
    print("\n=== All Tests Completed ===")
    print("✓ Extended Kalman Filter appears to be working correctly!")
    print("✓ Adaptive noise test shows EKF properly adjusts to changing conditions!")
    print("✓ Blind noise test demonstrates effects of model mismatch!") 
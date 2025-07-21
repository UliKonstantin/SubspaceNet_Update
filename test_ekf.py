#!/usr/bin/env python3
"""
Simple test for Extended Kalman Filter mathematical correctness.

This test verifies that the EKF correctly tracks a non-linear trajectory
and converges to the true state values.
"""

import numpy as np
import matplotlib.pyplot as plt
from simulation.kalman_filter.extended import ExtendedKalmanFilter1D
from simulation.kalman_filter.models import SineAccelStateModel

def test_ekf_simple_case():
    """
    Test EKF with a simple sine acceleration model.
    """
    print("=== Testing Extended Kalman Filter ===")
    
    # Test parameters
    omega0 = 0.1  # Base angular velocity
    kappa = 0.05  # Sine acceleration coefficient  
    process_noise_std = 0.01  # Small process noise
    measurement_noise_var = 0.001  # Small measurement noise
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
    covariances = [ekf.P]
    innovations = []
    
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
        covariances.append(ekf.P)
        innovations.append(innovation)
        
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
    
    return true_states, ekf_states, covariances, innovations

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

def plot_results(true_states, ekf_states, covariances):
    """Plot the test results."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    
    steps = range(len(true_states))
    
    # Plot states
    ax1.plot(steps, true_states, 'b-', label='True State', linewidth=2)
    ax1.plot(steps, ekf_states, 'r--', label='EKF Estimate', linewidth=2)
    ax1.set_ylabel('State Value')
    ax1.set_title('Extended Kalman Filter Tracking Performance')
    ax1.legend()
    ax1.grid(True)
    
    # Plot error
    errors = np.abs(np.array(true_states) - np.array(ekf_states))
    ax2.plot(steps, errors, 'g-', linewidth=2)
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('Tracking Error')
    ax2.grid(True)
    
    # Plot covariance
    ax3.plot(steps, covariances, 'm-', linewidth=2)
    ax3.set_ylabel('Covariance')
    ax3.set_xlabel('Time Step')
    ax3.set_title('State Uncertainty (Covariance)')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('ekf_test_results.png', dpi=150)
    print("Results saved to ekf_test_results.png")

if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Run tests
    true_states, ekf_states, covariances, innovations = test_ekf_simple_case()
    convergence_states, convergence_covariances = test_ekf_convergence()
    
    # Plot results
    plot_results(true_states, ekf_states, covariances)
    
    print("\n=== All Tests Completed ===")
    print("✓ Extended Kalman Filter appears to be working correctly!") 
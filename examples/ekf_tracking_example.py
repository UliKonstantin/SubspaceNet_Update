#!/usr/bin/env python3
"""
Example of using the Extended Kalman Filter for tracking non-linear angle trajectories.

This example demonstrates how to use the Extended Kalman Filter (EKF) to track
angles that follow non-linear trajectories, including:
1. Sine acceleration model: θ_{k+1} = θ_k + (ω0 + κ sin θ_k)T + η_k
2. Multiplicative noise model: θ_{k+1} = θ_k + ω0 T + σ(θ_k) η_k

The example creates synthetic trajectories using the same models, adds
measurement noise, and then compares the tracking performance of the EKF
with the standard Kalman Filter (KF).
"""

import numpy as np
import matplotlib.pyplot as plt
from pydantic import create_model_from_dict
from config.schema import TrajectoryType, Config
from simulation.kalman_filter import get_kalman_filter
from simulation.kalman_filter.models import SineAccelStateModel, MultNoiseStateModel


def generate_trajectory(model, trajectory_length=100, initial_angle=0.0):
    """
    Generate a synthetic trajectory using a state evolution model.
    
    Args:
        model: StateEvolutionModel instance
        trajectory_length: Length of trajectory
        initial_angle: Starting angle
        
    Returns:
        Tuple of (true_angles, noisy_measurements)
    """
    # Generate true trajectory
    true_angles = np.zeros(trajectory_length)
    true_angles[0] = initial_angle
    
    for t in range(1, trajectory_length):
        # Deterministic part
        true_angles[t] = model.f(true_angles[t-1])
        
        # Add process noise
        process_noise = np.random.normal(0, np.sqrt(model.noise_variance(true_angles[t-1])))
        true_angles[t] += process_noise
    
    # Generate noisy measurements
    measurement_noise_std = 0.5  # Standard deviation of measurement noise
    noisy_measurements = true_angles + np.random.normal(0, measurement_noise_std, trajectory_length)
    
    return true_angles, noisy_measurements


def track_with_kalman_filter(kf, measurements):
    """
    Track a trajectory using a Kalman filter.
    
    Args:
        kf: KalmanFilter1D or ExtendedKalmanFilter1D instance
        measurements: Noisy angle measurements
        
    Returns:
        Array of filtered angles
    """
    trajectory_length = len(measurements)
    filtered_angles = np.zeros(trajectory_length)
    
    # Initialize with first measurement
    kf.initialize_state(measurements[0])
    filtered_angles[0] = measurements[0]
    
    # Track through the trajectory
    for t in range(1, trajectory_length):
        # Predict next state
        kf.predict()
        
        # Update with measurement
        filtered_angles[t] = kf.update(measurements[t])
    
    return filtered_angles


def run_sine_accel_example():
    """Run example with sine acceleration model."""
    print("Running sine acceleration model example...")
    
    # Create a sine acceleration model
    omega0 = 0.1  # rad/s
    kappa = 0.5   # rad/s²
    noise_std = 0.2  # rad
    model = SineAccelStateModel(omega0, kappa, noise_std)
    
    # Generate trajectory
    trajectory_length = 100
    initial_angle = 0.0
    true_angles, measurements = generate_trajectory(model, trajectory_length, initial_angle)
    
    # Create config for KF and EKF
    config_dict = {
        "trajectory": {
            "trajectory_type": TrajectoryType.SINE_ACCEL_NONLINEAR,
            "sine_accel_omega0": omega0,
            "sine_accel_kappa": kappa,
            "sine_accel_noise_std": noise_std
        },
        "kalman_filter": {
            "measurement_noise_std_dev": 0.5,
            "initial_covariance": 1.0
        }
    }
    config = create_model_from_dict(Config, config_dict)
    
    # Get standard and extended KF
    config.kalman_filter.filter_type = "standard"
    standard_kf = get_kalman_filter(config)
    
    config.kalman_filter.filter_type = "extended"
    extended_kf = get_kalman_filter(config)
    
    # Track with both filters
    standard_filtered = track_with_kalman_filter(standard_kf, measurements)
    extended_filtered = track_with_kalman_filter(extended_kf, measurements)
    
    # Calculate errors
    standard_rmse = np.sqrt(np.mean((standard_filtered - true_angles)**2))
    extended_rmse = np.sqrt(np.mean((extended_filtered - true_angles)**2))
    
    print(f"Standard KF RMSE: {standard_rmse:.4f}")
    print(f"Extended KF RMSE: {extended_rmse:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(true_angles, 'k-', label='True Angles')
    plt.plot(measurements, 'r.', alpha=0.3, label='Measurements')
    plt.plot(standard_filtered, 'g-', label=f'Standard KF (RMSE: {standard_rmse:.4f})')
    plt.plot(extended_filtered, 'b-', label=f'Extended KF (RMSE: {extended_rmse:.4f})')
    plt.legend()
    plt.xlabel('Time Step')
    plt.ylabel('Angle (degrees)')
    plt.title('Sine Acceleration Model: Extended vs Standard KF')
    plt.grid(True)
    plt.savefig('sine_accel_example.png')
    plt.close()


def run_mult_noise_example():
    """Run example with multiplicative noise model."""
    print("Running multiplicative noise model example...")
    
    # Create a multiplicative noise model
    omega0 = 0.2  # rad/s
    amp = 0.8     # unitless
    base_std = 0.2  # rad
    model = MultNoiseStateModel(omega0, amp, base_std)
    
    # Generate trajectory
    trajectory_length = 100
    initial_angle = 0.0
    true_angles, measurements = generate_trajectory(model, trajectory_length, initial_angle)
    
    # Create config for KF and EKF
    config_dict = {
        "trajectory": {
            "trajectory_type": TrajectoryType.MULT_NOISE_NONLINEAR,
            "mult_noise_omega0": omega0,
            "mult_noise_amp": amp,
            "mult_noise_base_std": base_std
        },
        "kalman_filter": {
            "measurement_noise_std_dev": 0.5,
            "initial_covariance": 1.0
        }
    }
    config = create_model_from_dict(Config, config_dict)
    
    # Get standard and extended KF
    config.kalman_filter.filter_type = "standard"
    standard_kf = get_kalman_filter(config)
    
    config.kalman_filter.filter_type = "extended"
    extended_kf = get_kalman_filter(config)
    
    # Track with both filters
    standard_filtered = track_with_kalman_filter(standard_kf, measurements)
    extended_filtered = track_with_kalman_filter(extended_kf, measurements)
    
    # Calculate errors
    standard_rmse = np.sqrt(np.mean((standard_filtered - true_angles)**2))
    extended_rmse = np.sqrt(np.mean((extended_filtered - true_angles)**2))
    
    print(f"Standard KF RMSE: {standard_rmse:.4f}")
    print(f"Extended KF RMSE: {extended_rmse:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(true_angles, 'k-', label='True Angles')
    plt.plot(measurements, 'r.', alpha=0.3, label='Measurements')
    plt.plot(standard_filtered, 'g-', label=f'Standard KF (RMSE: {standard_rmse:.4f})')
    plt.plot(extended_filtered, 'b-', label=f'Extended KF (RMSE: {extended_rmse:.4f})')
    plt.legend()
    plt.xlabel('Time Step')
    plt.ylabel('Angle (degrees)')
    plt.title('Multiplicative Noise Model: Extended vs Standard KF')
    plt.grid(True)
    plt.savefig('mult_noise_example.png')
    plt.close()


if __name__ == "__main__":
    # Run examples
    run_sine_accel_example()
    run_mult_noise_example() 
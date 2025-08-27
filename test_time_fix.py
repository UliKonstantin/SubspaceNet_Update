#!/usr/bin/env python3
"""
Test to verify the time fix works correctly.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from simulation.kalman_filter.models.sine_accel import SineAccelStateModel
from simulation.kalman_filter.extended import ExtendedKalmanFilter1D


def test_time_fix():
    """Test that the time fix works correctly."""
    
    print("Testing Time Fix")
    print("=" * 40)
    
    # Test parameters
    omega0 = [0.2, 0.3, 0.15]  # Different frequencies
    kappa = [0.1, 0.15, 0.08]  # Different amplitudes
    noise_std = 0.01  # Noise
    device = torch.device("cpu")
    
    print(f"Source-specific parameters:")
    for i in range(len(omega0)):
        print(f"  Source {i}: ω₀={omega0[i]}, κ={kappa[i]}")
    print()
    
    # Test 1: Verify that initial_time is correctly set
    print("Test 1: Initial Time Setting")
    print("-" * 50)
    
    # Test different initial times
    test_times = [0, 10, 20, 50]
    
    for initial_time in test_times:
        print(f"\nTesting initial_time = {initial_time}:")
        
        # Create state model with specific initial time
        state_model = SineAccelStateModel(omega0, kappa, noise_std, device=device, initial_time=initial_time)
        
        print(f"  State model current_time: {state_model.current_time}")
        
        # Test prediction for each source
        for source_idx in range(len(omega0)):
            current_angle = 0.0
            current_tensor = torch.tensor([current_angle], dtype=torch.float32)
            
            # Get prediction
            predicted_angle = state_model.f(current_tensor, source_idx=source_idx)
            
            # Calculate expected oscillation manually
            expected_oscillation = kappa[source_idx] * np.sin(omega0[source_idx] * initial_time)
            expected_angle = current_angle + expected_oscillation
            
            print(f"    Source {source_idx}: predicted={predicted_angle.item():.4f}, expected={expected_angle:.4f}")
            
            # Verify they match
            assert abs(predicted_angle.item() - expected_angle) < 1e-6, f"Prediction mismatch for source {source_idx}"
    
    # Test 2: Verify EKF filters with different initial times
    print("\n\nTest 2: EKF Filters with Different Initial Times")
    print("-" * 50)
    
    # Create a simple config-like object for testing
    class MockConfig:
        def __init__(self):
            self.trajectory = type('obj', (object,), {
                'trajectory_type': 'sine_accel_nonlinear',
                'sine_accel_omega0': omega0,
                'sine_accel_kappa': kappa,
                'sine_accel_noise_std': noise_std
            })()
            self.system_model = type('obj', (object,), {'M': len(omega0)})()
            self.kalman_filter = type('obj', (object,), {
                'process_noise_std_dev': noise_std,
                'measurement_noise_std_dev': 0.001,
                'initial_covariance': 1.0
            })()
    
    config = MockConfig()
    
    # Test different window and step combinations
    test_cases = [
        (0, 0, 0),    # Window 0, Step 0 -> initial_time = 0
        (1, 0, 10),   # Window 1, Step 0 -> initial_time = 10
        (2, 5, 25),   # Window 2, Step 5 -> initial_time = 25
    ]
    
    for window_idx, step_idx, expected_initial_time in test_cases:
        print(f"\nTesting window {window_idx}, step {step_idx} (expected initial_time = {expected_initial_time}):")
        
        # Create EKF filter with calculated initial time
        window_size = 10  # Mock window size
        calculated_initial_time = window_idx * window_size + step_idx
        
        ekf_filter = ExtendedKalmanFilter1D.create_from_config(
            config, 
            trajectory_type='sine_accel_nonlinear',
            device=device,
            source_idx=0,  # Test with first source
            initial_time=calculated_initial_time
        )
        
        print(f"  EKF filter source_idx: {ekf_filter.source_idx}")
        print(f"  State model current_time: {ekf_filter.state_model.current_time}")
        
        # Verify the time is set correctly
        assert abs(ekf_filter.state_model.current_time.item() - calculated_initial_time) < 1e-6, f"Time mismatch for window {window_idx}, step {step_idx}"
        
        # Test prediction
        ekf_filter.initialize_state(0.0)
        predicted_state = ekf_filter.predict()
        
        # Calculate expected oscillation manually
        expected_oscillation = kappa[0] * np.sin(omega0[0] * calculated_initial_time)
        expected_state = 0.0 + expected_oscillation
        
        print(f"  Predicted state: {predicted_state.item():.4f}")
        print(f"  Expected state: {expected_state:.4f}")
        
        # Verify they match
        assert abs(predicted_state.item() - expected_state) < 1e-6, f"State prediction mismatch for window {window_idx}, step {step_idx}"
    
    print("\n✅ Time fix test completed successfully!")
    print("EKF filters now correctly initialize with the proper time for oscillatory models.")


if __name__ == "__main__":
    test_time_fix()

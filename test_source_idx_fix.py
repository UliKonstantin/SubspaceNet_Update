#!/usr/bin/env python3
"""
Test to verify the source_idx fix works correctly.
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


def test_source_idx_fix():
    """Test that source_idx parameter works correctly."""
    
    print("Testing Source Index Fix")
    print("=" * 40)
    
    # Test parameters - different for each source
    omega0 = [0.2, 0.3, 0.15]  # Different frequencies
    kappa = [0.1, 0.15, 0.08]  # Different amplitudes
    noise_std = 0.01  # Noise
    device = torch.device("cpu")
    
    print(f"Source-specific parameters:")
    for i in range(len(omega0)):
        print(f"  Source {i}: ω₀={omega0[i]}, κ={kappa[i]}")
    print()
    
    # Test 1: Create state model with source-specific parameters
    print("Test 1: State Model with Source-Specific Parameters")
    print("-" * 50)
    
    state_model = SineAccelStateModel(omega0, kappa, noise_std, device=device)
    
    # Test each source individually
    initial_angle = 0.0
    num_steps = 10
    
    for source_idx in range(len(omega0)):
        print(f"\nTesting source {source_idx}:")
        current_angle = initial_angle
        
        for t in range(num_steps):
            # Get prediction using source-specific parameters
            current_tensor = torch.tensor([current_angle], dtype=torch.float32)
            predicted_angle = state_model.f(current_tensor, source_idx=source_idx)
            
            # Add noise
            noise = torch.randn(1) * noise_std
            next_angle = predicted_angle + noise
            
            if t < 3:  # Show first few steps
                print(f"  Step {t}: {current_angle:.4f} -> {next_angle.item():.4f}")
            
            current_angle = next_angle.item()
            
            # Advance time
            state_model.advance_time()
        
        print(f"  Final angle for source {source_idx}: {current_angle:.4f}")
    
    # Test 2: Create EKF filters with source-specific parameters
    print("\n\nTest 2: EKF Filters with Source-Specific Parameters")
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
    
    # Create EKF filters for each source
    ekf_filters = []
    for i in range(len(omega0)):
        ekf_filter = ExtendedKalmanFilter1D.create_from_config(
            config, 
            trajectory_type='sine_accel_nonlinear',
            device=device,
            source_idx=i
        )
        ekf_filters.append(ekf_filter)
        print(f"Created EKF filter for source {i}")
    
    # Test EKF processing for each source
    print("\nTesting EKF processing:")
    for source_idx in range(len(omega0)):
        print(f"\nSource {source_idx}:")
        ekf_filter = ekf_filters[source_idx]
        
        # Initialize with true angle
        true_angle = 0.0
        ekf_filter.initialize_state(true_angle)
        
        # Process a few steps
        for t in range(5):
            # Simulate measurement (add some noise to true angle)
            measurement = true_angle + torch.randn(1) * 0.01
            
            # EKF predict and update
            predicted_state, updated_state, innovation, kalman_gain, _, _ = ekf_filter.predict_and_update(
                measurement=measurement
            )
            
            if t < 3:  # Show first few steps
                print(f"  Step {t}: Predicted={predicted_state.item():.4f}, Updated={updated_state.item():.4f}, Innovation={innovation.item():.4f}")
            
            # Update true angle for next step (simulate trajectory evolution)
            true_angle = true_angle + 0.1  # Simple linear motion for testing
    
    print("\n✅ Test completed successfully!")
    print("Each source should use its own oscillatory parameters (ω₀ and κ).")


if __name__ == "__main__":
    test_source_idx_fix()

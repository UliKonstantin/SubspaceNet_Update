#!/usr/bin/env python3
"""
Integration test for source-specific oscillatory model with online learning.
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
from config.schema import Config


def test_source_specific_integration():
    """Test that source-specific parameters work correctly in the online learning context."""
    
    print("Testing Source-Specific Integration with Online Learning")
    print("=" * 60)
    
    # Test parameters - different for each source
    omega0 = [0.2, 0.3, 0.15]  # Different frequencies
    kappa = [0.1, 0.15, 0.08]  # Different amplitudes
    noise_std = 0.01  # Noise
    device = torch.device("cpu")
    
    print(f"Source-specific parameters:")
    for i in range(len(omega0)):
        print(f"  Source {i}: ω₀={omega0[i]}, κ={kappa[i]}")
    print()
    
    # Test 1: Single source processing (like in online learning)
    print("Test 1: Single Source Processing (Online Learning Style)")
    print("-" * 50)
    
    # Create separate state models for each source (like EKF filters in online learning)
    source_models = []
    for i in range(len(omega0)):
        # Each source gets its own model with its specific parameters
        source_model = SineAccelStateModel([omega0[i]], [kappa[i]], noise_std, device=device)
        source_models.append(source_model)
    
    # Simulate online learning processing: one source at a time
    initial_angles = [0.0, 0.0, 0.0]
    num_steps = 20
    
    # Store trajectories for each source
    single_source_trajectories = {i: [initial_angles[i]] for i in range(len(omega0))}
    
    print("Processing each source individually (like online learning):")
    for t in range(num_steps - 1):
        for source_idx in range(len(omega0)):
            current_angle = single_source_trajectories[source_idx][-1]
            
            # Get prediction from source-specific model
            current_tensor = torch.tensor([current_angle], dtype=torch.float32)
            predicted_angle = source_models[source_idx].f(current_tensor)
            
            # Add noise
            noise = torch.randn(1) * noise_std
            next_angle = predicted_angle + noise
            
            single_source_trajectories[source_idx].append(next_angle.item())
            
            # Advance time for this source's model
            source_models[source_idx].advance_time()
        
        if t < 3:  # Show first few steps
            print(f"  Step {t}: {[single_source_trajectories[i][-1] for i in range(len(omega0))]}")
    
    print(f"  ... (showing first 3 steps)")
    print(f"  Final angles: {[single_source_trajectories[i][-1] for i in range(len(omega0))]}")
    print()
    
    # Test 2: Multi-source processing (like in data generation)
    print("Test 2: Multi-Source Processing (Data Generation Style)")
    print("-" * 50)
    
    # Create single model that handles all sources
    multi_source_model = SineAccelStateModel(omega0, kappa, noise_std, device=device)
    
    # Simulate data generation processing: all sources at once
    multi_source_trajectories = {i: [initial_angles[i]] for i in range(len(omega0))}
    
    print("Processing all sources together (like data generation):")
    for t in range(num_steps - 1):
        current_angles = [multi_source_trajectories[i][-1] for i in range(len(omega0))]
        
        # Get prediction from multi-source model
        current_tensor = torch.tensor(current_angles, dtype=torch.float32)
        predicted_angles = multi_source_model.f(current_tensor)
        
        # Add noise
        noise = torch.randn(len(omega0)) * noise_std
        next_angles = predicted_angles + noise
        
        # Store results
        for i in range(len(omega0)):
            multi_source_trajectories[i].append(next_angles[i].item())
        
        # Advance time for the multi-source model
        multi_source_model.advance_time()
        
        if t < 3:  # Show first few steps
            print(f"  Step {t}: {[multi_source_trajectories[i][-1] for i in range(len(omega0))]}")
    
    print(f"  ... (showing first 3 steps)")
    print(f"  Final angles: {[multi_source_trajectories[i][-1] for i in range(len(omega0))]}")
    print()
    
    # Test 3: Compare results
    print("Test 3: Comparison of Processing Methods")
    print("-" * 50)
    
    print("Final angles comparison:")
    for i in range(len(omega0)):
        single_final = single_source_trajectories[i][-1]
        multi_final = multi_source_trajectories[i][-1]
        diff = abs(single_final - multi_final)
        print(f"  Source {i}: Single={single_final:.4f}, Multi={multi_final:.4f}, Diff={diff:.6f}")
    
    # Check if results are similar (they should be)
    max_diff = max(abs(single_source_trajectories[i][-1] - multi_source_trajectories[i][-1]) 
                   for i in range(len(omega0)))
    print(f"\nMaximum difference between methods: {max_diff:.6f}")
    
    if max_diff < 1e-5:
        print("✅ PASS: Single-source and multi-source processing produce identical results")
    else:
        print("❌ FAIL: Single-source and multi-source processing produce different results")
    
    # Test 4: Verify source-specific behavior
    print("\nTest 4: Verify Source-Specific Behavior")
    print("-" * 50)
    
    # Calculate oscillation ranges for each source
    for i in range(len(omega0)):
        angles = single_source_trajectories[i]
        angle_range = max(angles) - min(angles)
        print(f"  Source {i} (ω₀={omega0[i]}, κ={kappa[i]}): Range={angle_range:.4f} rad")
    
    # The source with higher kappa should have larger oscillation range
    expected_order = sorted(range(len(kappa)), key=lambda x: kappa[x], reverse=True)
    print(f"\nExpected oscillation order (by amplitude): {expected_order}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Single source processing results
    plt.subplot(2, 3, 1)
    colors = ['blue', 'red', 'green']
    times = list(range(num_steps))
    for i in range(len(omega0)):
        plt.plot(times, single_source_trajectories[i], color=colors[i], linewidth=2, 
                label=f'Source {i} (Single)', linestyle='-')
    plt.xlabel('Time Step')
    plt.ylabel('Angle (radians)')
    plt.title('Single Source Processing (Online Learning Style)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Multi source processing results
    plt.subplot(2, 3, 2)
    for i in range(len(omega0)):
        plt.plot(times, multi_source_trajectories[i], color=colors[i], linewidth=2, 
                label=f'Source {i} (Multi)', linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel('Angle (radians)')
    plt.title('Multi Source Processing (Data Generation Style)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 3: Comparison
    plt.subplot(2, 3, 3)
    for i in range(len(omega0)):
        plt.plot(times, single_source_trajectories[i], color=colors[i], linewidth=2, 
                label=f'Source {i} Single', linestyle='-')
        plt.plot(times, multi_source_trajectories[i], color=colors[i], linewidth=1, 
                label=f'Source {i} Multi', linestyle='--', alpha=0.7)
    plt.xlabel('Time Step')
    plt.ylabel('Angle (radians)')
    plt.title('Comparison: Single vs Multi Processing')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 4: Source-specific oscillation patterns
    plt.subplot(2, 3, 4)
    for i in range(len(omega0)):
        angles_degrees = [angle * 180 / np.pi for angle in single_source_trajectories[i]]
        plt.plot(times, angles_degrees, color=colors[i], linewidth=2, 
                label=f'Source {i} (ω₀={omega0[i]}, κ={kappa[i]})')
    plt.xlabel('Time Step')
    plt.ylabel('Angle (degrees)')
    plt.title('Source-Specific Oscillation Patterns')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 5: Phase space for each source
    plt.subplot(2, 3, 5)
    for i in range(len(omega0)):
        angular_velocities = []
        for t in range(1, len(single_source_trajectories[i])):
            velocity = single_source_trajectories[i][t] - single_source_trajectories[i][t-1]
            angular_velocities.append(velocity)
        
        plt.plot(single_source_trajectories[i][:-1], angular_velocities, color=colors[i], 
                linewidth=1, alpha=0.7, label=f'Source {i}')
    
    plt.xlabel('Angle (radians)')
    plt.ylabel('Angular Velocity (rad/step)')
    plt.title('Phase Space - Source-Specific Patterns')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 6: Radar view
    plt.subplot(2, 3, 6)
    
    # Generate incremental distances
    distances = [20.0 + i * 1.0 for i in range(len(times))]
    
    for source_idx in range(len(omega0)):
        # Convert to Cartesian coordinates for this source
        x_coords = []
        y_coords = []
        for angle, distance in zip(single_source_trajectories[source_idx], distances):
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            x_coords.append(x)
            y_coords.append(y)
        
        # Plot trajectory for this source
        plt.plot(x_coords, y_coords, '-o', markersize=3, linewidth=2, 
                color=colors[source_idx], label=f'Source {source_idx}', alpha=0.8)
        
        # Mark start and end points
        plt.plot(x_coords[0], y_coords[0], 'o', markersize=8, color=colors[source_idx], 
                markeredgecolor='black', markeredgewidth=1)
        plt.plot(x_coords[-1], y_coords[-1], 's', markersize=8, color=colors[source_idx], 
                markeredgecolor='black', markeredgewidth=1)
    
    # Plot radar location
    plt.plot(0, 0, 'bD', markersize=15, label='Radar')
    
    # Add distance circles
    for d in [20, 25, 30, 35]:
        circle = plt.Circle((0, 0), d, fill=False, linestyle='--', alpha=0.3, color='gray')
        plt.gca().add_patch(circle)
        plt.text(0, d, f'{d}m', va='bottom', ha='center', fontsize=10)
    
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Radar View - Source-Specific Trajectories')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('test_source_specific_integration.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nTest completed! Check 'test_source_specific_integration.png' for visualization.")
    print("Each source should show different oscillatory patterns based on their unique parameters.")
    print("The single-source and multi-source processing methods should produce identical results.")


if __name__ == "__main__":
    test_source_specific_integration()

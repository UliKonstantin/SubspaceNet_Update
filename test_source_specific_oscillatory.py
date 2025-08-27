#!/usr/bin/env python3
"""
Test script for the source-specific oscillatory model implementation.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from simulation.kalman_filter.models.sine_accel import SineAccelStateModel


def test_source_specific_oscillatory_model():
    """Test the source-specific oscillatory model implementation."""
    
    print("Testing Source-Specific Oscillatory Model Implementation")
    print("=" * 60)
    
    # Test parameters - different for each source
    omega0 = [0.2, 0.3, 0.15]  # Different frequencies
    kappa = [0.1, 0.15, 0.08]  # Different amplitudes
    noise_std = 0.01  # Noise
    device = torch.device("cpu")
    
    # Create source-specific oscillatory model
    model = SineAccelStateModel(omega0, kappa, noise_std, device=device)
    
    print(f"Model parameters for {model.num_sources} sources:")
    for i in range(model.num_sources):
        print(f"  Source {i}: ω₀={omega0[i]}, κ={kappa[i]}")
    print(f"  noise_std: {noise_std}")
    print()
    
    # Test state evolution for multiple sources
    initial_angles = [0.0, 0.0, 0.0]  # Start all sources at 0
    num_steps = 100
    
    # Store trajectories for each source
    trajectories = {i: [initial_angles[i]] for i in range(model.num_sources)}
    times = list(range(num_steps))
    
    print("Testing state evolution for multiple sources:")
    for t in range(num_steps - 1):
        current_angles = [trajectories[i][-1] for i in range(model.num_sources)]
        
        # Get prediction from model (expects tensor with shape [num_sources])
        current_tensor = torch.tensor(current_angles, dtype=torch.float32)
        predicted_angles = model.f(current_tensor)
        
        # Add noise
        noise = torch.randn(model.num_sources) * noise_std
        next_angles = predicted_angles + noise
        
        # Store results
        for i in range(model.num_sources):
            trajectories[i].append(next_angles[i].item())
        
        # Advance time
        model.advance_time()
        
        if t < 5:  # Show first few steps
            print(f"  Step {t}: {current_angles} -> {[next_angles[i].item() for i in range(model.num_sources)]}")
    
    print(f"  ... (showing first 5 steps)")
    print(f"  Final angles: {[trajectories[i][-1] for i in range(model.num_sources)]}")
    print()
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Angle vs Time for all sources
    plt.subplot(2, 3, 1)
    colors = ['blue', 'red', 'green']
    for i in range(model.num_sources):
        plt.plot(times, trajectories[i], color=colors[i], linewidth=2, label=f'Source {i}')
    plt.xlabel('Time Step')
    plt.ylabel('Angle (radians)')
    plt.title('Source-Specific Oscillatory Model - Angle vs Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Angle vs Time (degrees)
    plt.subplot(2, 3, 2)
    for i in range(model.num_sources):
        angles_degrees = [angle * 180 / np.pi for angle in trajectories[i]]
        plt.plot(times, angles_degrees, color=colors[i], linewidth=2, label=f'Source {i} (degrees)')
    plt.xlabel('Time Step')
    plt.ylabel('Angle (degrees)')
    plt.title('Source-Specific Oscillatory Model - Angle vs Time (degrees)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 3: Phase space for each source
    plt.subplot(2, 3, 3)
    for i in range(model.num_sources):
        angular_velocities = []
        for t in range(1, len(trajectories[i])):
            velocity = trajectories[i][t] - trajectories[i][t-1]
            angular_velocities.append(velocity)
        
        plt.plot(trajectories[i][:-1], angular_velocities, color=colors[i], linewidth=1, alpha=0.7, label=f'Source {i}')
    
    plt.xlabel('Angle (radians)')
    plt.ylabel('Angular Velocity (rad/step)')
    plt.title('Source-Specific Oscillatory Model - Phase Space')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 4: Radar view for all sources
    plt.subplot(2, 3, (4, 6))  # Span 2 columns for the main radar view
    
    # Generate incremental distances
    distances = [20.0 + i * 1.0 for i in range(len(times))]
    
    for source_idx in range(model.num_sources):
        # Convert to Cartesian coordinates for this source
        x_coords = []
        y_coords = []
        for angle, distance in zip(trajectories[source_idx], distances):
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
    for d in [20, 30, 40, 50, 60, 70]:
        circle = plt.Circle((0, 0), d, fill=False, linestyle='--', alpha=0.3, color='gray')
        plt.gca().add_patch(circle)
        plt.text(0, d, f'{d}m', va='bottom', ha='center', fontsize=10)
    
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Source-Specific Oscillatory Model - Radar View')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('test_source_specific_oscillatory.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Test completed! Check 'test_source_specific_oscillatory.png' for visualization.")
    print("Each source should show different oscillatory patterns based on their unique parameters.")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for i in range(model.num_sources):
        angles = trajectories[i]
        print(f"  Source {i} (ω₀={omega0[i]}, κ={kappa[i]}):")
        print(f"    Final angle: {angles[-1]:.4f} rad ({angles[-1] * 180 / np.pi:.2f}°)")
        print(f"    Range: {max(angles) - min(angles):.4f} rad")
        print(f"    Mean: {np.mean(angles):.4f} rad")


if __name__ == "__main__":
    test_source_specific_oscillatory_model()

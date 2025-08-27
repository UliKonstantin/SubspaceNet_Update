#!/usr/bin/env python3
"""
Test script for the new oscillatory model implementation.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from simulation.kalman_filter.models.sine_accel import SineAccelStateModel
from config.schema import TrajectoryType


def test_oscillatory_model():
    """Test the oscillatory model implementation."""
    
    print("Testing Oscillatory Model Implementation")
    print("=" * 50)
    
    # Test parameters
    omega0 = 0.2  # Frequency
    kappa = 0.1   # Amplitude
    noise_std = 0.01  # Noise
    device = torch.device("cpu")
    
    # Create oscillatory model
    model = SineAccelStateModel(omega0, kappa, noise_std, device=device)
    
    print(f"Model parameters:")
    print(f"  omega0 (frequency): {omega0}")
    print(f"  kappa (amplitude): {kappa}")
    print(f"  noise_std: {noise_std}")
    print()
    
    # Test state evolution
    initial_angle = 0.0
    num_steps = 50
    
    angles = [initial_angle]
    times = list(range(num_steps))
    
    print("Testing state evolution:")
    for t in range(num_steps - 1):
        current_angle = angles[-1]
        
        # Get prediction from model
        predicted_angle = model.f(torch.tensor(current_angle))
        
        # Add noise
        noise = torch.randn(1) * noise_std
        next_angle = predicted_angle + noise
        
        angles.append(next_angle.item())
        
        # Advance time
        model.advance_time()
        
        if t < 5:  # Show first few steps
            print(f"  Step {t}: {current_angle:.4f} -> {next_angle.item():.4f}")
    
    print(f"  ... (showing first 5 steps)")
    print(f"  Final angle: {angles[-1]:.4f}")
    print()
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Angle vs Time
    plt.subplot(2, 2, 1)
    plt.plot(times, angles, 'b-', linewidth=2, label='Angle')
    plt.xlabel('Time Step')
    plt.ylabel('Angle (radians)')
    plt.title('Oscillatory Model - Angle vs Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Angle vs Time (degrees)
    plt.subplot(2, 2, 2)
    angles_degrees = [angle * 180 / np.pi for angle in angles]
    plt.plot(times, angles_degrees, 'r-', linewidth=2, label='Angle (degrees)')
    plt.xlabel('Time Step')
    plt.ylabel('Angle (degrees)')
    plt.title('Oscillatory Model - Angle vs Time (degrees)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 3: Phase space
    plt.subplot(2, 2, 3)
    angular_velocities = []
    for i in range(1, len(angles)):
        velocity = angles[i] - angles[i-1]
        angular_velocities.append(velocity)
    
    plt.plot(angles[:-1], angular_velocities, 'g-', linewidth=1, alpha=0.7)
    plt.xlabel('Angle (radians)')
    plt.ylabel('Angular Velocity (rad/step)')
    plt.title('Oscillatory Model - Phase Space')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Radar view
    plt.subplot(2, 2, 4)
    
    # Generate incremental distances
    distances = [20.0 + i * 1.0 for i in range(len(angles))]
    
    # Convert to Cartesian coordinates
    x_coords = []
    y_coords = []
    for angle, distance in zip(angles, distances):
        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        x_coords.append(x)
        y_coords.append(y)
    
    # Plot trajectory
    plt.plot(x_coords, y_coords, '-o', markersize=4, linewidth=2, label='Trajectory')
    
    # Mark start and end points
    plt.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
    plt.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='End')
    
    # Plot radar location
    plt.plot(0, 0, 'bD', markersize=15, label='Radar')
    
    # Add distance circles
    for d in [20, 30, 40, 50]:
        circle = plt.Circle((0, 0), d, fill=False, linestyle='--', alpha=0.3, color='gray')
        plt.gca().add_patch(circle)
        plt.text(0, d, f'{d}m', va='bottom', ha='center', fontsize=10)
    
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Oscillatory Model - Radar View')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('test_oscillatory_model.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Test completed! Check 'test_oscillatory_model.png' for visualization.")
    print("The trajectory should show oscillatory behavior instead of drifting in one direction.")


if __name__ == "__main__":
    test_oscillatory_model()

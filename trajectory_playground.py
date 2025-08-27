#!/usr/bin/env python3
"""
Trajectory Playground - Standalone trajectory generation and testing

This file provides a playground environment for testing different trajectory
generation functions. It's completely disconnected from the main codebase.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import math


class TrajectoryGenerator:
    """Standalone trajectory generator for testing different trajectory functions."""
    
    def __init__(self, omega: float = 0.1, kappa: float = 0.05, noise_std: float = 0.01):
        """
        Initialize trajectory generator.
        
        Args:
            omega: Base angular velocity
            kappa: Sine acceleration coefficient
            noise_std: Standard deviation of Gaussian noise
        """
        self.omega = omega
        self.kappa = kappa
        self.noise_std = noise_std
        
    def generate_sine_accel_trajectory(self, initial_angle: float, num_steps: int = 100) -> Tuple[List[float], List[float]]:
        """
        Generate trajectory using the model: theta_{t+1} = theta_t + omega + kappa*sin(theta_t) + noise
        
        Args:
            initial_angle: Starting angle in radians
            num_steps: Number of steps to generate
            
        Returns:
            Tuple of (angles, time_steps)
        """
        angles = [initial_angle]
        time_steps = list(range(num_steps))
        
        for t in range(num_steps - 1):
            current_angle = angles[-1]
            
            # Calculate next angle: theta_{t+1} = theta_t + omega + kappa*sin(theta_t) + noise
            deterministic_part = current_angle + self.omega + self.kappa * math.sin(current_angle)
            noise = np.random.normal(0, self.noise_std)
            next_angle = deterministic_part + noise
            
            angles.append(next_angle)
        
        return angles, time_steps
    
    def generate_oscillatory_trajectory(self, initial_angle: float, num_steps: int = 100) -> Tuple[List[float], List[float]]:
        """
        Generate oscillatory trajectory using the model: theta_{t+1} = theta_t + kappa*sin(omega*t) + noise
        
        This creates oscillatory behavior around the initial angle instead of drifting in one direction.
        
        Args:
            initial_angle: Starting angle in radians
            num_steps: Number of steps to generate
            
        Returns:
            Tuple of (angles, time_steps)
        """
        angles = [initial_angle]
        time_steps = list(range(num_steps))
        
        for t in range(num_steps - 1):
            current_angle = angles[-1]
            
            # Calculate next angle: theta_{t+1} = theta_t + kappa*sin(omega*t) + noise
            # This creates oscillation around the current angle
            oscillation = self.kappa * math.sin(self.omega * t)
            noise = np.random.normal(0, self.noise_std)
            next_angle = current_angle + oscillation + noise
            
            angles.append(next_angle)
        
        return angles, time_steps
    
    def generate_centered_oscillatory_trajectory(self, center_angle: float, num_steps: int = 100) -> Tuple[List[float], List[float]]:
        """
        Generate oscillatory trajectory that oscillates around a center angle.
        Model: theta_{t+1} = center + amplitude*sin(omega*t + phase) + noise
        
        Args:
            center_angle: Center angle around which to oscillate (radians)
            num_steps: Number of steps to generate
            
        Returns:
            Tuple of (angles, time_steps)
        """
        angles = []
        time_steps = list(range(num_steps))
        
        for t in range(num_steps):
            # Oscillate around center_angle with amplitude kappa
            oscillation = self.kappa * math.sin(self.omega * t)
            noise = np.random.normal(0, self.noise_std)
            angle = center_angle + oscillation + noise
            
            angles.append(angle)
        
        return angles, time_steps
    
    def plot_trajectory(self, angles: List[float], time_steps: List[int], 
                       title: str = "Trajectory", save_path: Optional[str] = None):
        """
        Plot the generated trajectory using radar view with incremental distance.
        
        Args:
            angles: List of angles over time (in radians)
            time_steps: List of time steps
            title: Plot title
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Angle vs Time (radians)
        plt.subplot(2, 3, 1)
        plt.plot(time_steps, angles, 'b-', linewidth=2, label='Angle')
        plt.xlabel('Time Step')
        plt.ylabel('Angle (radians)')
        plt.title(f'{title} - Angle vs Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 2: Angle vs Time (degrees)
        plt.subplot(2, 3, 2)
        angles_degrees = [angle * 180 / math.pi for angle in angles]
        plt.plot(time_steps, angles_degrees, 'r-', linewidth=2, label='Angle (degrees)')
        plt.xlabel('Time Step')
        plt.ylabel('Angle (degrees)')
        plt.title(f'{title} - Angle vs Time (degrees)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 3: Phase space (angle vs angular velocity)
        plt.subplot(2, 3, 3)
        angular_velocities = []
        for i in range(1, len(angles)):
            velocity = angles[i] - angles[i-1]
            angular_velocities.append(velocity)
        
        plt.plot(angles[:-1], angular_velocities, 'g-', linewidth=1, alpha=0.7)
        plt.xlabel('Angle (radians)')
        plt.ylabel('Angular Velocity (rad/step)')
        plt.title(f'{title} - Phase Space')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Radar View with Incremental Distance (Main Plot)
        plt.subplot(2, 3, (4, 6))  # Span 2 columns for the main radar view
        
        # Generate incremental distances (start at 20m, increment by 1m each step)
        distances = [20.0 + i * 1.0 for i in range(len(angles))]
        
        # Convert from polar to Cartesian coordinates
        x_coords = []
        y_coords = []
        for angle, distance in zip(angles, distances):
            x = distance * math.cos(angle)
            y = distance * math.sin(angle)
            x_coords.append(x)
            y_coords.append(y)
        
        # Plot trajectory
        plt.plot(x_coords, y_coords, '-o', markersize=4, linewidth=2, label='Trajectory')
        
        # Mark start and end points
        if len(x_coords) > 0:
            plt.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')  # Green for start
            plt.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='End')  # Red for end
        
        # Plot radar location
        plt.plot(0, 0, 'bD', markersize=15, label='Radar')
        
        # Add distance circles
        for d in [20, 30, 40, 50, 60]:
            circle = plt.Circle((0, 0), d, fill=False, linestyle='--', alpha=0.3, color='gray')
            plt.gca().add_patch(circle)
            plt.text(0, d, f'{d}m', va='bottom', ha='center', fontsize=10)
        
        # Add angle lines
        for a in range(-90, 91, 30):
            a_rad = a * (math.pi/180)
            plt.plot([0, 70*math.cos(a_rad)], [0, 70*math.sin(a_rad)], 'k:', alpha=0.2)
            plt.text(65*math.cos(a_rad), 65*math.sin(a_rad), f'{a}°', 
                    va='center', ha='center', bbox=dict(facecolor='white', alpha=0.7), fontsize=9)
        
        # Add some intermediate trajectory points with step numbers
        step_interval = max(1, len(angles) // 10)  # Show every 10th step or so
        for i in range(0, len(angles), step_interval):
            if i < len(x_coords) and i < len(y_coords):
                plt.annotate(f'{i}', (x_coords[i], y_coords[i]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.7)
        
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.title(f'{title} - Radar View (T={len(angles)} steps)')
        plt.legend()
        
        # Set axis limits to show the full trajectory
        max_range = max(distances) + 5
        plt.xlim(-max_range, max_range)
        plt.ylim(-max_range, max_range)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show(block=False)  # Non-blocking show so execution continues
        plt.pause(0.1)  # Small pause to allow plot to render
        
        # Close the plot to free memory
        plt.close()
    
    def analyze_trajectory(self, angles: List[float], time_steps: List[int]):
        """
        Analyze the generated trajectory and print statistics.
        
        Args:
            angles: List of angles over time
            time_steps: List of time steps
        """
        print("=" * 60)
        print("TRAJECTORY ANALYSIS")
        print("=" * 60)
        
        # Basic statistics
        angles_array = np.array(angles)
        angles_degrees = angles_array * 180 / math.pi
        
        print(f"Parameters:")
        print(f"  omega: {self.omega:.4f}")
        print(f"  kappa: {self.kappa:.4f}")
        print(f"  noise_std: {self.noise_std:.4f}")
        print()
        
        print(f"Trajectory Statistics:")
        print(f"  Number of steps: {len(angles)}")
        print(f"  Initial angle: {angles[0]:.4f} rad ({angles_degrees[0]:.2f}°)")
        print(f"  Final angle: {angles[-1]:.4f} rad ({angles_degrees[-1]:.2f}°)")
        print(f"  Total change: {angles[-1] - angles[0]:.4f} rad ({(angles_degrees[-1] - angles_degrees[0]):.2f}°)")
        print()
        
        print(f"Angle Statistics (radians):")
        print(f"  Mean: {np.mean(angles_array):.4f}")
        print(f"  Std: {np.std(angles_array):.4f}")
        print(f"  Min: {np.min(angles_array):.4f}")
        print(f"  Max: {np.max(angles_array):.4f}")
        print()
        
        print(f"Angle Statistics (degrees):")
        print(f"  Mean: {np.mean(angles_degrees):.2f}°")
        print(f"  Std: {np.std(angles_degrees):.2f}°")
        print(f"  Min: {np.min(angles_degrees):.2f}°")
        print(f"  Max: {np.max(angles_degrees):.2f}°")
        print()
        
        # Angular velocity statistics
        angular_velocities = []
        for i in range(1, len(angles)):
            velocity = angles[i] - angles[i-1]
            angular_velocities.append(velocity)
        
        angular_velocities_array = np.array(angular_velocities)
        angular_velocities_degrees = angular_velocities_array * 180 / math.pi
        
        print(f"Angular Velocity Statistics (rad/step):")
        print(f"  Mean: {np.mean(angular_velocities_array):.4f}")
        print(f"  Std: {np.std(angular_velocities_array):.4f}")
        print(f"  Min: {np.min(angular_velocities_array):.4f}")
        print(f"  Max: {np.max(angular_velocities_array):.4f}")
        print()
        
        print(f"Angular Velocity Statistics (degrees/step):")
        print(f"  Mean: {np.mean(angular_velocities_degrees):.2f}°")
        print(f"  Std: {np.std(angular_velocities_degrees):.2f}°")
        print(f"  Min: {np.min(angular_velocities_degrees):.2f}°")
        print(f"  Max: {np.max(angular_velocities_degrees):.2f}°")
        print("=" * 60)


def main():
    """Main function to demonstrate trajectory generation and testing."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("Trajectory Playground - Multiple Models")
    print("=" * 50)
    
    # Create trajectory generator with oscillatory parameters
    test_cases = [
        {
            'name': 'Oscillatory Model 1',
            'omega': 0.2,  # Higher frequency for more oscillation
            'kappa': 0.1,  # Amplitude of oscillation
            'noise_std': 0.01
        },
        {
            'name': 'Oscillatory Model 2', 
            'omega': 0.3,
            'kappa': 0.15,
            'noise_std': 0.015
        },
        {
            'name': 'Centered Oscillatory',
            'omega': 0.25,
            'kappa': 0.2,
            'noise_std': 0.02
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {test_case['name']}")
        print("-" * 40)
        
        # Create generator
        generator = TrajectoryGenerator(
            omega=test_case['omega'],
            kappa=test_case['kappa'],
            noise_std=test_case['noise_std']
        )
        
        # Generate different types of trajectories
        initial_angle = 0.0  # Start at 0 radians
        center_angle = math.pi / 4  # 45 degrees for centered oscillation
        
        if i == 2:  # Centered oscillatory
            angles, time_steps = generator.generate_centered_oscillatory_trajectory(
                center_angle=center_angle,
                num_steps=100
            )
        else:  # Regular oscillatory
            angles, time_steps = generator.generate_oscillatory_trajectory(
                initial_angle=initial_angle,
                num_steps=100
            )
        
        # Analyze trajectory
        generator.analyze_trajectory(angles, time_steps)
        
        # Plot trajectory
        generator.plot_trajectory(
            angles=angles,
            time_steps=time_steps,
            title=f"{test_case['name']}",
            save_path=f"oscillatory_test_{i+1}.png"
        )
        
        print(f"Generated trajectory with {len(angles)} steps")
        print(f"Final angle: {angles[-1]:.4f} rad ({angles[-1] * 180 / math.pi:.2f}°)")
        print()
    
    # Also test the original sine acceleration model for comparison
    print("\n" + "="*50)
    print("COMPARISON: Original Sine Acceleration Model")
    print("="*50)
    
    generator = TrajectoryGenerator(omega=0.1, kappa=0.05, noise_std=0.01)
    angles, time_steps = generator.generate_sine_accel_trajectory(
        initial_angle=0.0, num_steps=100
    )
    
    generator.analyze_trajectory(angles, time_steps)
    generator.plot_trajectory(
        angles=angles,
        time_steps=time_steps,
        title="Original Sine Acceleration (for comparison)",
        save_path="sine_accel_comparison.png"
    )


if __name__ == "__main__":
    main()

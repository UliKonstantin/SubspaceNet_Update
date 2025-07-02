"""
Example demonstrating BatchExtendedKalmanFilter1D usage.

This example shows how to use the batch Extended Kalman Filter for
tracking multiple angle trajectories with non-linear dynamics.
"""

import torch
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from simulation.kalman_filter import BatchExtendedKalmanFilter1D
from simulation.kalman_filter.models import SineAccelStateModel
from config.schema import Config

def create_sample_config():
    """Create a sample configuration for testing."""
    config_dict = {
        "trajectory": {
            "trajectory_type": "sine_accel_nonlinear",
            "sine_accel_omega0": 0.1,
            "sine_accel_kappa": 0.2,
            "sine_accel_noise_std": 0.05,
            "random_walk_std_dev": 0.1
        },
        "kalman_filter": {
            "filter_type": "extended",
            "measurement_noise_std_dev": 0.01,
            "process_noise_std_dev": 0.1,
            "initial_covariance": 1.0
        },
        "system_model": {
            "N": 9,
            "M": 3,
            "T": 200,
            "snr": 10,
            "field_type": "far",
            "signal_nature": "non-coherent",
            "signal_type": "narrowband",
            "wavelength": 1,
            "eta": 0.0,
            "bias": 0,
            "sv_noise_var": 0.0,
            "doa_range": 180,
            "doa_resolution": 1,
            "max_range_ratio_to_limit": 0.5,
            "range_resolution": 1,
            "nominal": True
        },
        "dataset": {
            "samples_size": 500,
            "test_validation_train_split": [0.8, 0.1, 0.1],
            "create_data": True,
            "save_dataset": False,
            "true_doa_test": None,
            "true_range_test": None
        },
        "model": {
            "type": "SubspaceNet",
            "params": {
                "diff_method": "esprit",
                "train_loss_type": "rmspe",
                "tau": 8,
                "field_type": "Far",
                "regularization": "null",
                "variant": "small",
                "norm_layer": False,
                "batch_norm": False
            }
        },
        "training": {
            "enabled": True,
            "batch_size": 500,
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau",
            "learning_rate": 0.001,
            "weight_decay": 1e-9,
            "step_size": 50,
            "gamma": 0.5,
            "training_objective": "angle",
            "use_wandb": False,
            "simulation_name": "batch_extended_kf_example"
        },
        "simulation": {
            "train_model": False,
            "evaluate_model": False,
            "load_model": False,
            "save_model": False,
            "plot_results": False,
            "save_plots": False,
            "model_path": None,
            "subspace_methods": []
        },
        "evaluation": {
            "save_results": False,
            "results_format": "json",
            "detailed_metrics": True,
            "kalman_filter": True
        }
    }
    
    return Config(**config_dict)

def main():
    """Main function demonstrating BatchExtendedKalmanFilter1D usage."""
    print("BatchExtendedKalmanFilter1D Example")
    print("=" * 50)
    
    # Configuration
    config = create_sample_config()
    batch_size = 4  # Number of trajectories
    max_sources = 3  # Maximum sources per trajectory  
    num_time_steps = 20  # Length of simulation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Max sources: {max_sources}")
    print(f"  Time steps: {num_time_steps}")
    print(f"  Device: {device}")
    print()
    
    # Create the batch Extended Kalman Filter
    batch_ekf = BatchExtendedKalmanFilter1D.from_config(
        config, batch_size, max_sources, device
    )
    
    print(f"Created BatchExtendedKalmanFilter1D:")
    print(f"  State model: {type(batch_ekf.state_model).__name__}")
    print(f"  Measurement noise variance: {batch_ekf.R[0, 0].item():.6f}")
    print()
    
    # Initialize states with random angles (in degrees)
    initial_states = torch.rand(batch_size, max_sources, device=device) * 60 - 30  # [-30, 30] degrees
    num_sources = [2, 3, 1, 3]  # Variable number of sources per trajectory
    
    batch_ekf.initialize_states(initial_states, num_sources)
    
    print("Initial states:")
    for i in range(batch_size):
        active_sources = num_sources[i]
        states = initial_states[i, :active_sources].cpu().numpy()
        print(f"  Trajectory {i}: {states} degrees")
    print()
    
    # Simulate trajectory and filtering
    print("Running simulation...")
    states_history = []
    measurements_history = []
    covariances_history = []
    
    for t in range(num_time_steps):
        # Generate true measurements (simulate perfect measurements with noise)
        true_measurements = batch_ekf.simulate_measurements()
        
        # Perform prediction and update
        filtered_states, covariances = batch_ekf.predict_and_update(true_measurements)
        
        # Store results
        states_history.append(filtered_states.clone())
        measurements_history.append(true_measurements.clone())
        covariances_history.append(covariances.clone())
        
        if t % 5 == 0:  # Print every 5 steps
            print(f"  Step {t:2d}: Mean state = {filtered_states[batch_ekf.source_mask].mean().item():.3f}°, "
                  f"Mean covariance = {covariances[batch_ekf.source_mask].mean().item():.6f}")
    
    print()
    
    # Display final results
    print("Final Results:")
    print("-" * 30)
    final_states = states_history[-1]
    final_covariances = covariances_history[-1]
    
    for i in range(batch_size):
        active_sources = num_sources[i]
        initial = initial_states[i, :active_sources].cpu().numpy()
        final = final_states[i, :active_sources].cpu().numpy()
        covs = final_covariances[i, :active_sources].cpu().numpy()
        
        print(f"Trajectory {i}:")
        print(f"  Initial:  {initial}")
        print(f"  Final:    {final}")
        print(f"  Drift:    {final - initial}")
        print(f"  Cov:      {covs}")
        print()
    
    # Compute some statistics
    all_initial = initial_states[batch_ekf.source_mask].cpu().numpy()
    all_final = final_states[batch_ekf.source_mask].cpu().numpy()
    all_drift = all_final - all_initial
    all_covs = final_covariances[batch_ekf.source_mask].cpu().numpy()
    
    print("Summary Statistics:")
    print(f"  Total sources tracked: {len(all_initial)}")
    print(f"  Mean initial angle: {np.mean(all_initial):.3f}° ± {np.std(all_initial):.3f}°")
    print(f"  Mean final angle: {np.mean(all_final):.3f}° ± {np.std(all_final):.3f}°")
    print(f"  Mean drift: {np.mean(all_drift):.3f}° ± {np.std(all_drift):.3f}°")
    print(f"  Mean final covariance: {np.mean(all_covs):.6f} ± {np.std(all_covs):.6f}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main() 
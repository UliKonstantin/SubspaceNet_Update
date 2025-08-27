"""
Data handling for trajectory-based DOA simulations.

This module handles the creation and processing of synthetic datasets
with trajectory support for direction-of-arrival estimation.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
import h5py
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import logging
import datetime

# These would be imported from the DCD_MUSIC module
from DCD_MUSIC.src.signal_creation import Samples
from DCD_MUSIC.src.system_model import SystemModelParams
from DCD_MUSIC.src.utils import device

# Import from configuration module
from config.schema import TrajectoryType, TrajectoryConfig, Config

logger = logging.getLogger('SubspaceNet.data')

class TrajectoryDataHandler:
    """
    Handles trajectory-based data creation and loading for DOA estimation.
    
    This class provides methods to generate synthetic datasets with trajectory
    support, where angles (DOAs) follow specific patterns over time.
    """
    
    def __init__(self, system_model_params, config=None):
        """
        Initialize the data handler with system model parameters.
        
        Args:
            system_model_params: SystemModelParams instance
            config: Optional configuration dictionary
        """
        self.system_model_params = system_model_params
        self.config = config
        self.samples_model = Samples(system_model_params)
    
    def create_dataset(self, 
                     samples_size: int,
                     trajectory_length: int,
                     trajectory_type: Union[TrajectoryType, str] = TrajectoryType.RANDOM,
                     save_dataset: bool = False,
                     dataset_path: Optional[Path] = None,
                     custom_trajectory_fn: Optional[Callable] = None) -> Tuple[Dataset, Any]:
        """
        Create a synthetic dataset with trajectory support.
        
        Args:
            samples_size: Number of distinct samples/trajectories
            trajectory_length: Length of each trajectory
            trajectory_type: Type of trajectory to generate
            save_dataset: Whether to save the dataset
            dataset_path: Path where to save the dataset
            custom_trajectory_fn: Custom function for trajectory generation
            
        Returns:
            Tuple of (dataset, samples_model)
        """
        logger.info(f"Creating dataset with {samples_size} trajectories of length {trajectory_length}")
        
        # Generate the trajectories
        trajectory_data = self._generate_trajectories(
            samples_size, 
            trajectory_length, 
            trajectory_type,
            custom_trajectory_fn
        )
        
        # Create observations and labels based on trajectories
        time_series, labels, sources_num = self._create_observations(trajectory_data)
        
        # Create the dataset object
        dataset = TrajectoryDataset(time_series, labels, sources_num)
        
        # Save dataset if requested
        if save_dataset and dataset_path:
            filename = self._get_dataset_filename(samples_size, trajectory_length, trajectory_type)
            dataset.save(dataset_path / filename)
            logger.info(f"Dataset saved to {dataset_path / filename}")
            
        return dataset, self.samples_model
    
    def load_dataset(self, 
                   filename: Path,
                   dataset_path: Path) -> Dataset:
        """
        Load a dataset from file.
        
        Args:
            filename: Name of the dataset file
            dataset_path: Path to the dataset
            
        Returns:
            Loaded dataset
        """
        logger.info(f"Loading dataset from {dataset_path / filename}")
        dataset = TrajectoryDataset(None, None, None)
        dataset.load(dataset_path / filename)
        return dataset
    
    def _generate_trajectories(
        self,
        samples_size: int,
        trajectory_length: int,
        trajectory_type: Union[TrajectoryType, str],
        custom_trajectory_fn: Optional[Callable] = None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], List[int]]:
        """
        Generate angle and distance trajectories.
        
        Args:
            samples_size: Number of distinct trajectories
            trajectory_length: Length of each trajectory
            trajectory_type: Type of trajectory to generate
            custom_trajectory_fn: Custom function for trajectory generation
            
        Returns:
            Tuple containing:
            - Tuple of (angle_trajectories, distance_trajectories)
            - List of sources per trajectory
        """
        # Convert string to enum if needed
        if isinstance(trajectory_type, str):
            trajectory_type = TrajectoryType(trajectory_type)
                
        # Determine number of sources
        if isinstance(self.system_model_params.M, tuple):
            low_M, high_M = self.system_model_params.M
            high_M = min(high_M, self.system_model_params.N-1)
            sources_per_trajectory = [
                np.random.randint(low_M, high_M+1) 
                for _ in range(samples_size)
            ]
        else:
            sources_per_trajectory = [self.system_model_params.M] * samples_size
        
        # Get angle limits
        doa_range = self.system_model_params.doa_range
        angle_min = -doa_range/2
        angle_max = doa_range/2
        
        logger.info(f"Generating trajectories with angle range: [{angle_min}, {angle_max}]")
        
        # Get parameters from config
        cfg = self.config if self.config else None
        
        # Get random walk step size from config if we're using random walk
        random_walk_std_dev = 1.0  # Default value
        if trajectory_type == TrajectoryType.RANDOM_WALK and cfg and hasattr(cfg.trajectory, 'random_walk_std_dev'):
            random_walk_std_dev = cfg.trajectory.random_walk_std_dev
            logger.info(f"Using random walk with std dev: {random_walk_std_dev}")
        
        # Parameters for sine acceleration non-linear model
        sa_omega0 = 0.0
        sa_kappa = 3.0
        sa_noise_sd = 0.1
        if trajectory_type == TrajectoryType.SINE_ACCEL_NONLINEAR and cfg and hasattr(cfg.trajectory, 'sine_accel_omega0'):
            sa_omega0 = cfg.trajectory.sine_accel_omega0
            sa_kappa = cfg.trajectory.sine_accel_kappa
            sa_noise_sd = cfg.trajectory.sine_accel_noise_std
            logger.info(f"Using sine acceleration non-linear model with ω₀={sa_omega0}, κ={sa_kappa}, σ={sa_noise_sd}")
        
        # Parameters for multiplicative noise non-linear model
        mn_omega0 = 0.0
        mn_amp = 0.5
        mn_base_sd = 0.1
        if trajectory_type == TrajectoryType.MULT_NOISE_NONLINEAR and cfg and hasattr(cfg.trajectory, 'mult_noise_omega0'):
            mn_omega0 = cfg.trajectory.mult_noise_omega0
            mn_amp = cfg.trajectory.mult_noise_amp
            mn_base_sd = cfg.trajectory.mult_noise_base_std
            logger.info(f"Using multiplicative noise non-linear model with ω₀={mn_omega0}, amp={mn_amp}, σ={mn_base_sd}")
        
        # Log trajectory type information before starting generation
        if trajectory_type == TrajectoryType.RANDOM:
            logger.info("Using RANDOM trajectory type: completely random angles at each step")
        elif trajectory_type == TrajectoryType.STATIC:
            logger.info("Using STATIC trajectory type: fixed angles throughout trajectory")
        elif trajectory_type == TrajectoryType.LINEAR:
            logger.info("Using LINEAR trajectory type: linear interpolation between random start and end angles")
        elif trajectory_type == TrajectoryType.CIRCULAR:
            logger.info("Using CIRCULAR trajectory type: angles follow sinusoidal motion around center points")
        elif trajectory_type == TrajectoryType.CUSTOM:
            logger.info("Using CUSTOM trajectory type with provided trajectory function")
        elif trajectory_type == TrajectoryType.FULL_RANDOM:
            logger.info("Using FULL_RANDOM trajectory type: completely independent random angles for each source and step")
        elif trajectory_type == TrajectoryType.SINE_ACCEL_NONLINEAR:
            logger.info("Using SINE_ACCEL_NONLINEAR trajectory type: θ_{k+1} = θ_k + κ sin(ω0 * t) + η_k (oscillatory model)")
        elif trajectory_type == TrajectoryType.MULT_NOISE_NONLINEAR:
            logger.info("Using MULT_NOISE_NONLINEAR trajectory type: θ_{k+1} = θ_k + ω0 T + σ(θ_k) η_k")
        
        # Create empty trajectories arrays
        max_sources = max(sources_per_trajectory)
        angle_trajectories = torch.zeros(
            (samples_size, trajectory_length, max_sources), 
            dtype=torch.float32
        )
        distance_trajectories = torch.zeros(
            (samples_size, trajectory_length, max_sources), 
            dtype=torch.float32
        )
        
        # Generate trajectories
        for i, num_sources in enumerate(sources_per_trajectory):
            # Initialize all sources at distance 20
            for s in range(num_sources):
                distance_trajectories[i, 0, s] = 20.0
            
            # Each step increases distance by 1
            for t in range(1, trajectory_length):
                distance_trajectories[i, t, :num_sources] = distance_trajectories[i, t-1, :num_sources] + 1.0
            
            # Generate angle trajectories based on selected type
            if trajectory_type == TrajectoryType.SINE_ACCEL_NONLINEAR:
                # Start with random angles
                angle_trajectories[i, 0, :num_sources] = torch.FloatTensor(
                    np.random.uniform(-30, 30, size=num_sources)
                )
                
                # Get oscillatory model parameters (support both single values and arrays)
                sa_omega0 = config.trajectory.sine_accel_omega0
                sa_kappa = config.trajectory.sine_accel_kappa
                sa_noise_sd = config.trajectory.sine_accel_noise_std
                
                # Convert single values to arrays for source-specific parameters
                if isinstance(sa_omega0, (int, float)):
                    sa_omega0 = [sa_omega0] * num_sources
                if isinstance(sa_kappa, (int, float)):
                    sa_kappa = [sa_kappa] * num_sources
                
                # Ensure arrays have correct length
                if len(sa_omega0) != num_sources:
                    raise ValueError(f"sine_accel_omega0 array length ({len(sa_omega0)}) must match number of sources ({num_sources})")
                if len(sa_kappa) != num_sources:
                    raise ValueError(f"sine_accel_kappa array length ({len(sa_kappa)}) must match number of sources ({num_sources})")
                
                # Convert to tensors for efficient computation
                sa_omega0_tensor = torch.tensor(sa_omega0, dtype=torch.float32)
                sa_kappa_tensor = torch.tensor(sa_kappa, dtype=torch.float32)
                
                # Apply oscillatory model: θ_{k+1} = θ_k + κ sin(ω0 * t) + η_k
                # This creates oscillatory behavior instead of drifting in one direction
                for t in range(1, trajectory_length):
                    theta_prev = angle_trajectories[i, t-1, :num_sources]
                    
                    # Calculate oscillatory term: κ sin(ω0 * t) for each source
                    # Each source has its own frequency and amplitude
                    oscillation = sa_kappa_tensor * torch.sin(sa_omega0_tensor * t) * 1.0  # T = 1 s
                    
                    # Add noise
                    noise = torch.randn(num_sources) * sa_noise_sd
                    
                    # Update angle: θ_{k+1} = θ_k + oscillation + noise
                    theta_new = 0.99*theta_prev + oscillation + noise
                    
                    # Ensure angles stay within bounds
                    angle_trajectories[i, t, :num_sources] = torch.clamp(
                        theta_new, 
                        min=angle_min, 
                        max=angle_max
                    )
            elif trajectory_type == TrajectoryType.MULT_NOISE_NONLINEAR:
                # Start with random angles
                angle_trajectories[i, 0, :num_sources] = torch.FloatTensor(
                    np.random.uniform(angle_min, angle_max, size=num_sources)
                )
                
                # Apply multiplicative noise model: θ_{k+1} = θ_k + ω0 T + σ(θ_k) η_k
                for t in range(1, trajectory_length):
                    theta_prev = angle_trajectories[i, t-1, :num_sources]
                    # Convert to radians for trigonometric functions
                    theta_prev_rad = theta_prev * (np.pi / 180.0)
                    
                    # Deterministic part
                    deterministic = mn_omega0 * 1.0  # T = 1 s
                    
                    # State-dependent noise standard deviation
                    std = mn_base_sd * (1.0 + mn_amp * torch.sin(theta_prev_rad)**2)
                    
                    # Generate noise
                    noise = torch.randn(num_sources) * std
                    
                    # Update angle
                    theta_new = theta_prev + deterministic + noise
                    
                    # Ensure angles stay within bounds
                    angle_trajectories[i, t, :num_sources] = torch.clamp(
                        theta_new, 
                        min=angle_min, 
                        max=angle_max
                    )
                    
            elif trajectory_type == TrajectoryType.RANDOM:
                # Completely random angles at each step
                for t in range(trajectory_length):
                    angle_trajectories[i, t, :num_sources] = torch.FloatTensor(
                        np.random.uniform(
                            angle_min, 
                            angle_max, 
                            size=num_sources
                        )
                    )
                    
            elif trajectory_type == TrajectoryType.RANDOM_WALK:
                # Start with random angles
                angle_trajectories[i, 0, :num_sources] = torch.FloatTensor(
                    np.random.uniform(
                        angle_min, 
                        angle_max, 
                        size=num_sources
                    )
                )
                
                # Apply state-space model: θ_k = θ_{k-1} + w_k
                # where w_k is zero-mean Gaussian noise with std_dev = random_walk_std_dev
                for t in range(1, trajectory_length):
                    # Generate process noise
                    w_k = torch.randn(num_sources) * random_walk_std_dev
                    
                    # Update angle using state transition: θ_k = θ_{k-1} + w_k
                    angle_trajectories[i, t, :num_sources] = angle_trajectories[i, t-1, :num_sources] + w_k
                    
                    # Ensure angles stay within bounds
                    angle_trajectories[i, t, :num_sources] = torch.clamp(
                        angle_trajectories[i, t, :num_sources], 
                        min=angle_min, 
                        max=angle_max
                    )
                    
            elif trajectory_type == TrajectoryType.LINEAR:
                # Start with random angles
                angle_trajectories[i, 0, :num_sources] = torch.FloatTensor(
                    np.random.uniform(
                        angle_min, 
                        angle_max, 
                        size=num_sources
                    )
                )
                
                # Set random end points
                end_angles = torch.FloatTensor(
                    np.random.uniform(
                        angle_min, 
                        angle_max, 
                        size=num_sources
                    )
                )
                
                # Linear interpolation
                for t in range(1, trajectory_length):
                    alpha = t / (trajectory_length - 1)
                    angle_trajectories[i, t, :num_sources] = (1 - alpha) * angle_trajectories[i, 0, :num_sources] + alpha * end_angles
            
            elif trajectory_type == TrajectoryType.CIRCULAR:
                # Center points of circular motion
                centers = torch.FloatTensor(
                    np.random.uniform(
                        angle_min + 0.2, 
                        angle_max - 0.2, 
                        size=num_sources
                    )
                )
                
                # Radii of circular motion (bounded to stay in range)
                max_radius = min(centers.min() - angle_min, angle_max - centers.max()) * 0.8
                radii = torch.FloatTensor(
                    np.random.uniform(
                        0.05, 
                        max_radius, 
                        size=num_sources
                    )
                )
                
                # Phase offsets for variety
                phases = torch.FloatTensor(
                    np.random.uniform(
                        0, 
                        2*np.pi, 
                        size=num_sources
                    )
                )
                
                # Generate circular trajectories
                for t in range(trajectory_length):
                    angle = 2 * np.pi * t / trajectory_length
                    angle_trajectories[i, t, :num_sources] = centers + radii * torch.sin(angle + phases)
            
            elif trajectory_type == TrajectoryType.STATIC:
                # Fixed angles throughout trajectory
                static_angles = torch.FloatTensor(
                    np.random.uniform(
                        angle_min, 
                        angle_max, 
                        size=num_sources
                    )
                )
                
                for t in range(trajectory_length):
                    angle_trajectories[i, t, :num_sources] = static_angles
            
            elif trajectory_type == TrajectoryType.CUSTOM and custom_trajectory_fn:
                # Use custom function to generate trajectories
                custom_trajectories = custom_trajectory_fn(
                    num_sources, 
                    trajectory_length, 
                    angle_min, 
                    angle_max
                )
                angle_trajectories[i, :, :num_sources] = torch.FloatTensor(custom_trajectories)
            
            elif trajectory_type == TrajectoryType.FULL_RANDOM:
                # Generate completely independent random angles for each step and each source
                # Unlike RANDOM which assigns the same random pattern to each source
                for t in range(trajectory_length):
                    for s in range(num_sources):
                        angle_trajectories[i, t, s] = torch.FloatTensor([
                            np.random.uniform(angle_min, angle_max)
                        ])
            
        # Plot the last trajectory for verification
        self._plot_trajectory_verification(
            angle_trajectories[-1, :, :sources_per_trajectory[-1]],
            distance_trajectories[-1, :, :sources_per_trajectory[-1]]
        )
        
        return (angle_trajectories, distance_trajectories), sources_per_trajectory
    
    def _plot_trajectory_verification(self, angles, distances):
        """
        Plot a trajectory for verification purposes.
        
        Args:
            angles: Tensor of shape [trajectory_length, num_sources]
            distances: Tensor of shape [trajectory_length, num_sources]
        """
        try:
            import matplotlib.pyplot as plt
            
            # Convert polar coordinates (angle, distance) to Cartesian (x, y)
            angles_rad = angles * (np.pi / 180)  # Convert to radians if in degrees
            
            num_sources = angles.shape[1]
            trajectory_length = angles.shape[0]
            
            plt.figure(figsize=(10, 8))
            
            # Plot each source trajectory
            for s in range(num_sources):
                # Convert from polar to Cartesian coordinates
                x = distances[:, s] * torch.cos(angles_rad[:, s])
                y = distances[:, s] * torch.sin(angles_rad[:, s])
                
                # Plot trajectory
                plt.plot(x.numpy(), y.numpy(), '-o', label=f'Source {s+1}')
                
                # Mark start and end points
                plt.plot(x[0].item(), y[0].item(), 'go', markersize=10)  # Green for start
                plt.plot(x[-1].item(), y[-1].item(), 'ro', markersize=10)  # Red for end
            
            # Plot radar location
            plt.plot(0, 0, 'bD', markersize=12, label='Radar')
            
            # Add distance circles
            for d in [20, 30, 40, 50]:
                circle = plt.Circle((0, 0), d, fill=False, linestyle='--', alpha=0.3)
                plt.gca().add_patch(circle)
                plt.text(0, d, f'{d}m', va='bottom', ha='center')
            
            # Add angle lines
            for a in range(-90, 91, 30):
                a_rad = a * (np.pi/180)
                plt.plot([0, 60*np.cos(a_rad)], [0, 60*np.sin(a_rad)], 'k:', alpha=0.2)
                plt.text(55*np.cos(a_rad), 55*np.sin(a_rad), f'{a}°', 
                        va='center', ha='center', bbox=dict(facecolor='white', alpha=0.5))
            
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.xlabel('X (meters)')
            plt.ylabel('Y (meters)')
            plt.title(f'Trajectory Verification (T={trajectory_length}, Sources={num_sources})')
            plt.legend()
            
            # Save the plot - Fixed the f-string syntax issue
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectory_verification_{timestamp}.png"
            output_dir = Path('experiments/results/trajectory_plots')
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / filename
            plt.savefig(output_path)
            logger.info(f"Trajectory plot saved to: {output_path}")
            
        except ImportError:
            logger.warning("matplotlib not available for trajectory verification plot")
        except Exception as e:
            logger.error(f"Error generating trajectory verification plot: {e}")
    
    def _create_observations(self, 
                           trajectory_data: Tuple[Tuple[torch.Tensor, torch.Tensor], List[int]]) -> Tuple[List, List, List]:
        """
        Create observations and labels from angle and distance trajectories.
        
        Args:
            trajectory_data: Tuple containing:
                - Tuple of (angle_trajectories, distance_trajectories)
                - List of sources per trajectory
                
        Returns:
            Tuple of (time_series, labels, sources_num)
        """
        (angle_trajectories, distance_trajectories), sources_per_trajectory = trajectory_data
        
        samples_size, trajectory_length, _ = angle_trajectories.shape
        
        # Initialize containers
        time_series = []
        labels = []
        sources_num = []
        
        # For each sample
        for i in range(samples_size):
            sample_time_series = []
            sample_labels = []
            sample_sources = []
            
            # For each step in trajectory
            for t in range(trajectory_length):
                num_sources = sources_per_trajectory[i]
                angles = angle_trajectories[i, t, :num_sources].numpy()
                distances = distance_trajectories[i, t, :num_sources].numpy()
                
                # Set the angles and distances for this step
                if self.system_model_params.field_type.lower() == "far":
                    self.samples_model.set_labels(num_sources, angles=angles.tolist(), distances=None)
                else:  # near or full field
                    self.samples_model.set_labels(num_sources, angles=angles.tolist(), distances=distances.tolist())
                
                # Generate observation matrix
                X = self.samples_model.samples_creation(
                    noise_mean=0,
                    noise_variance=1,
                    signal_mean=0,
                    signal_variance=1,
                    source_number=num_sources
                )[0]
                
                # Use as_tensor instead of tensor to avoid copy warnings
                X_tensor = torch.as_tensor(X)
                
                # Get ground truth labels
                Y = self.samples_model.get_labels()
                
                # Store data for this step
                sample_time_series.append(X_tensor)
                sample_labels.append(Y)
                sample_sources.append(num_sources)
            
            # Store data for this sample
            time_series.append(sample_time_series)
            labels.append(sample_labels)
            sources_num.append(sample_sources)
        
        # Convert lists to tensors where appropriate (no need to convert again since we already created tensors)
        time_series = [torch.stack(traj) for traj in time_series]
        
        return time_series, labels, sources_num
    
    def _get_dataset_filename(self, 
                            samples_size: int, 
                            trajectory_length: int,
                            trajectory_type: Union[TrajectoryType, str]) -> str:
        """
        Generate a filename for the dataset.
        
        Args:
            samples_size: Number of distinct trajectories
            trajectory_length: Length of each trajectory
            trajectory_type: Type of trajectory used
            
        Returns:
            Filename string
        """
        if isinstance(trajectory_type, str):
            trajectory_type = TrajectoryType(trajectory_type)
            
        if isinstance(self.system_model_params.M, tuple):
            low_M, high_M = self.system_model_params.M
            M = f"random_{low_M}_{high_M}"
        else:
            M = self.system_model_params.M
            
        return (
            f"Trajectory_DataSet_"
            f"{trajectory_type.value}_"
            f"traj{trajectory_length}_"
            f"{self.system_model_params.field_type}_field_"
            f"{self.system_model_params.signal_type}_"
            f"{self.system_model_params.signal_nature}_"
            f"{samples_size}_M={M}_"
            f"N={self.system_model_params.N}_"
            f"T={self.system_model_params.T}_"
            f"SNR={self.system_model_params.snr}_"
            f"eta={self.system_model_params.eta}_"
            f"sv_noise_var{self.system_model_params.sv_noise_var}_"
            f"bias={self.system_model_params.bias}"
            f".h5"
        )


class TrajectoryDataset(Dataset):
    """
    Dataset class for trajectory-based DOA estimation.
    
    Extends the original TimeSeriesDataset to support trajectories.
    """
    
    def __init__(self, time_series, labels, sources_num):
        """
        Initialize the dataset.
        
        Args:
            time_series: List of trajectories, each trajectory is a tensor of 
                         shape [trajectory_length, N, T]
            labels: List of label trajectories
            sources_num: List of source counts for each step in trajectories
        """
        self.time_series = time_series
        self.labels = labels
        self.sources_num = sources_num
        self.path = None
        self.len = None
        self.h5f = None
        
        if time_series is not None:
            self.trajectory_length = len(time_series[0])
        else:
            self.trajectory_length = None
    
    def __len__(self):
        """Return the number of trajectories."""
        if self.len is None:
            return len(self.time_series) if self.time_series is not None else 0
        return self.len
    
    def __getitem__(self, idx):
        """
        Get a trajectory.
        
        Args:
            idx: Index of the trajectory
            
        Returns:
            Tuple of (time_series, sources_num, labels) for the trajectory
        """
        if self.path is None:
            return self.time_series[idx], self.sources_num[idx], self.labels[idx]
        
        self._open_h5_file()
        
        # Load from H5 file
        trajectory_length = self.h5f[f'trajectory_length'][idx]
        
        # Load time series for all steps in trajectory
        time_series = [
            torch.as_tensor(self.h5f[f'X/{idx}/step_{step}'][:]) 
            for step in range(trajectory_length)
        ]
        
        # Load labels for all steps
        labels = [
            torch.as_tensor(self.h5f[f'Y/{idx}/label_{step}'][:]) 
            for step in range(trajectory_length)
        ]
        
        # Load sources count for all steps
        sources_num = [
            int(self.h5f[f'M/{idx}/sources_{step}']) 
            for step in range(trajectory_length)
        ]
        
        return torch.stack(time_series), sources_num, labels
    
    def get_dataloaders(self, batch_size, validation_split=0.1):
        """
        Create training and validation dataloaders.
        
        Args:
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            
        Returns:
            Tuple of (train_dataloader, valid_dataloader)
        """
        # Split indices for training and validation
        indices = range(len(self))
        total_samples = len(indices)
        
        # Safety check for small datasets
        if total_samples <= 3:  # For very small datasets
            adjusted_split = min(validation_split, 1.0/total_samples)
            if adjusted_split != validation_split:
                logger.warning(f"Adjusting validation_split from {validation_split} to {adjusted_split} due to small dataset size ({total_samples} samples)")
                validation_split = adjusted_split
        
        train_indices, val_indices = train_test_split(
            indices, 
            test_size=validation_split,
            shuffle=True
        )
        
        # Create subset datasets
        train_dataset = Subset(self, train_indices)
        valid_dataset = Subset(self, val_indices)
        
        logger.info(f"Training dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(valid_dataset)}")
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_trajectories
        )
        
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_trajectories
        )
        
        return train_dataloader, valid_dataloader
    
    def save(self, path):
        """
        Save the dataset to an H5 file.
        
        Args:
            path: Path where to save the dataset
        """
        with h5py.File(path, 'w') as h5f:
            # Store dataset size and trajectory length
            h5f.create_dataset('dataset_size', data=len(self))
            if self.trajectory_length:
                h5f.create_dataset('trajectory_length', data=[self.trajectory_length] * len(self))
            
            # For each trajectory
            for i in range(len(self)):
                # Create groups for this trajectory
                x_grp = h5f.create_group(f'X/{i}')
                y_grp = h5f.create_group(f'Y/{i}')
                m_grp = h5f.create_group(f'M/{i}')
                
                # For each step in the trajectory
                for t in range(len(self.time_series[i])):
                    # Store time series
                    x_grp.create_dataset(
                        f'step_{t}', 
                        data=self.time_series[i][t].numpy()
                    )
                    
                    # Store labels
                    y_grp.create_dataset(
                        f'label_{t}', 
                        data=np.array(self.labels[i][t])
                    )
                    
                    # Store source counts
                    m_grp.create_dataset(
                        f'sources_{t}', 
                        data=self.sources_num[i][t]
                    )
    
    def load(self, path):
        """
        Load the dataset from an H5 file.
        
        Args:
            path: Path to the H5 file
            
        Returns:
            Self for chaining
        """
        self.path = path
        self._open_h5_file()
        
        # Get dataset size
        self.len = len(self.h5f['dataset_size'])
        
        # Get trajectory length for first sample
        self.trajectory_length = int(self.h5f['trajectory_length'][0])
        
        return self
    
    def _open_h5_file(self):
        """Open the H5 file if not already open."""
        if self.h5f is None and self.path is not None:
            self.h5f = h5py.File(self.path, 'r')
    
    def close(self):
        """Close the H5 file if open."""
        if self.h5f is not None:
            self.h5f.close()
            self.h5f = None
    
    def __del__(self):
        """Ensure H5 file is closed when object is deleted."""
        self.close()
    
    def _collate_trajectories(self, batch):
        """
        Custom collate function for trajectory batches.
        
        Args:
            batch: List of (time_series, sources_num, labels) tuples
            
        Returns:
            Collated batch with padded trajectories
        """
        time_series, sources_num, labels = zip(*batch)
        
        # Find maximum values for padding
        max_trajectory_length = max(ts.shape[0] for ts in time_series)
        max_sources = max(max(src) for src in sources_num)
        
        # Batch size
        batch_size = len(time_series)
        
        # Get dimensions of time series
        _, n_sensors, n_snapshots = time_series[0].shape
        
        # Initialize padded tensors
        padded_time_series = torch.zeros(
            (batch_size, max_trajectory_length, n_sensors, n_snapshots),
            dtype=torch.complex64
        )
        
        padded_labels = torch.zeros(
            (batch_size, max_trajectory_length, max_sources),
            dtype=torch.float32
        )
        
        padded_sources_num = torch.zeros(
            (batch_size, max_trajectory_length),
            dtype=torch.long
        )
        
        # Fill padded tensors
        for i, (ts, src, lbl) in enumerate(zip(time_series, sources_num, labels)):
            traj_len = ts.shape[0]
            
            # Copy time series
            padded_time_series[i, :traj_len] = ts
            
            # Copy sources_num
            padded_sources_num[i, :traj_len] = torch.as_tensor(src)
            
            # Copy labels (with padding for varying number of sources)
            for t in range(traj_len):
                n_src = src[t]
                padded_labels[i, t, :n_src] = lbl[t]
        
        return padded_time_series, padded_sources_num, padded_labels


# NEW Class: OnlineLearningTrajectoryGenerator
class OnlineLearningTrajectoryGenerator:
    """
    Generates a single, continuous trajectory on-demand for online learning.
    Maintains the state of the true underlying trajectory and allows for
    dynamic updates to system parameters like 'eta' which affect newly
    generated noisy observations.
    """
    def __init__(self, system_model_params: SystemModelParams, 
                 trajectory_config: TrajectoryConfig, 
                 initial_eta: float,
                 num_sources: Union[int, Tuple[int, int]]):
        """
        Initializes the generator.

        Args:
            system_model_params: The shared SystemModelParams instance. 
                                 Changes to this instance (e.g., eta) will affect generation.
            trajectory_config: Configuration for the trajectory generation (e.g., type, std_dev).
            initial_eta: The initial value for eta.
            num_sources: Number of sources (M). Can be an int or a tuple (min_M, max_M) for variable sources.
        """
        self.system_model_params = system_model_params
        self.trajectory_config = trajectory_config
        
        # Set initial eta directly on the shared system_model_params
        self.system_model_params.eta = initial_eta
        
        self.samples_model = Samples(self.system_model_params) # Uses the shared system_model_params

        if isinstance(num_sources, tuple):
            self.min_M, self.max_M = num_sources
            self.M_is_variable = True
            self.current_M = torch.randint(self.min_M, self.max_M + 1, (1,)).item()
        else:
            self.current_M = num_sources
            self.M_is_variable = False

        self.angle_min = -self.system_model_params.doa_range / 2
        self.angle_max = self.system_model_params.doa_range / 2
        
        # State for the true underlying trajectory
        angles = (torch.rand(self.current_M) * 60 - 30).numpy()  # Uniform in [-30, 30]
        angles = sorted(angles)
        
        # Check if minimum 4° separation exists, otherwise push edges away from middle
        if self.current_M > 1:
            min_sep = min(angles[i+1] - angles[i] for i in range(len(angles)-1))
            if min_sep < 4.0:
                # Push edges away from middle to ensure 4° separation
                middle_idx = self.current_M // 2
                
                # Start from middle and expand outward
                for i in range(1, self.current_M):
                    if middle_idx - i >= 0:  # Left side
                        angles[middle_idx - i] = angles[middle_idx - i + 1] - 4.0
                    if middle_idx + i < self.current_M:  # Right side
                        angles[middle_idx + i] = angles[middle_idx + i - 1] + 4.0
        
        self.last_true_angles = np.array(angles)
        # TODO: Add self.last_true_ranges for near-field if needed and make it dynamic based on field_type

        self.current_step_in_session = 0
        logger.info(f"OnlineLearningTrajectoryGenerator initialized. eta={self.system_model_params.eta:.4f}, M={self.current_M}, type={self.trajectory_config.trajectory_type.value}")

    def update_eta(self, new_eta: float):
        """Updates the eta value in the shared SystemModelParams."""
        old_eta = self.system_model_params.eta
        self.system_model_params.eta = new_eta
        
        # Regenerate distance noise with new eta value
        self.samples_model.eta = self.samples_model._SystemModel__set_eta()
        #if not getattr(self.system_model_params, 'nominal', True):
        self.samples_model.location_noise = self.samples_model.get_distance_noise(True)
        
        logger.info(f"Generator eta updated from {old_eta:.4f} to {self.system_model_params.eta:.4f} with new distance noise pattern.")

    def _generate_next_true_step(self) -> Tuple[np.ndarray, int]:
        """Generates the next set of true angles (and potentially ranges) and the number of sources for this step."""
        if self.M_is_variable: # Potentially change M at each step if configured
             # Simple heuristic: 10% chance to change M
            if torch.rand(1).item() < 0.1:
                self.current_M = torch.randint(self.min_M, self.max_M + 1, (1,)).item()
                # If M changes, re-initialize angles for the new number of sources
                self.last_true_angles = (torch.rand(self.current_M) * (self.angle_max - self.angle_min) + self.angle_min).numpy()


        # Trajectory generation logic (e.g., random walk)
        # This should be more sophisticated, using self.trajectory_config.trajectory_type
        # and other parameters like self.trajectory_config.random_walk_std_dev
        
        # Simplified Random Walk for now:
        if self.trajectory_config.trajectory_type == TrajectoryType.RANDOM_WALK:
            std_dev = self.trajectory_config.random_walk_std_dev if self.trajectory_config.random_walk_std_dev is not None else 1.0
            noise = (torch.randn(self.current_M) * std_dev).numpy()
            next_angles = self.last_true_angles + noise
            self.last_true_angles = np.clip(next_angles, self.angle_min, self.angle_max)
        elif self.trajectory_config.trajectory_type == TrajectoryType.STATIC:
            # Angles remain the same as self.last_true_angles (initialized once)
            pass # No change needed for static
        elif self.trajectory_config.trajectory_type == TrajectoryType.SINE_ACCEL_NONLINEAR:
            # Get parameters for oscillatory model (support both single values and arrays)
            sa_omega0 = self.trajectory_config.sine_accel_omega0 if hasattr(self.trajectory_config, 'sine_accel_omega0') else 0.2
            sa_kappa = self.trajectory_config.sine_accel_kappa if hasattr(self.trajectory_config, 'sine_accel_kappa') else 0.1
            sa_noise_sd = self.trajectory_config.sine_accel_noise_std if hasattr(self.trajectory_config, 'sine_accel_noise_std') else 0.01
            
            # Convert single values to arrays for source-specific parameters
            if isinstance(sa_omega0, (int, float)):
                sa_omega0 = [sa_omega0] * self.current_M
            if isinstance(sa_kappa, (int, float)):
                sa_kappa = [sa_kappa] * self.current_M
            
            # Ensure arrays have correct length
            if len(sa_omega0) != self.current_M:
                raise ValueError(f"sine_accel_omega0 array length ({len(sa_omega0)}) must match number of sources ({self.current_M})")
            if len(sa_kappa) != self.current_M:
                raise ValueError(f"sine_accel_kappa array length ({len(sa_kappa)}) must match number of sources ({self.current_M})")
            
            # Convert to numpy arrays for computation
            sa_omega0_array = np.array(sa_omega0)
            sa_kappa_array = np.array(sa_kappa)
            
            # Apply oscillatory model: θ_{k+1} = θ_k + κ sin(ω0 * t) + η_k
            # Each source has its own frequency and amplitude
            oscillation = sa_kappa_array * np.sin(sa_omega0_array * self.current_step_in_session) * 1.0  # T = 1 s
            
            # Add noise (in degrees)
            noise = (torch.randn(self.current_M) * sa_noise_sd).numpy()
            
            # Update angle: θ_{k+1} = θ_k + oscillation + noise
            next_angles = 0.99*self.last_true_angles + oscillation + noise
            
            # Ensure angles stay within bounds
            self.last_true_angles = np.clip(next_angles, self.angle_min, self.angle_max)
        elif self.trajectory_config.trajectory_type == TrajectoryType.MULT_NOISE_NONLINEAR:
            # Get parameters for multiplicative noise model
            mn_omega0 = self.trajectory_config.mult_noise_omega0 if hasattr(self.trajectory_config, 'mult_noise_omega0') else 0.0
            mn_amp = self.trajectory_config.mult_noise_amp if hasattr(self.trajectory_config, 'mult_noise_amp') else 0.5
            mn_base_sd = self.trajectory_config.mult_noise_base_std if hasattr(self.trajectory_config, 'mult_noise_base_std') else 0.1
            
            # Apply multiplicative noise model: θ_{k+1} = θ_k + ω0 T + σ(θ_k) η_k
            # Convert to radians for trigonometric functions only
            theta_prev_rad = self.last_true_angles * (np.pi / 180.0)
            
            # Deterministic part (in degrees)
            deterministic = mn_omega0 * 1.0  # T = 1 s
            
            # State-dependent noise standard deviation
            std = mn_base_sd * (1.0 + mn_amp * np.sin(theta_prev_rad)**2)
            
            # Generate noise (in degrees)
            noise = (torch.randn(self.current_M) * std).numpy()
            
            # Update angle (all in degrees)
            next_angles = self.last_true_angles + deterministic + noise
            
            # Ensure angles stay within bounds
            self.last_true_angles = np.clip(next_angles, self.angle_min, self.angle_max)
        else: # Default to RANDOM if not specified or other types not implemented here yet
            self.last_true_angles = (torch.rand(self.current_M) * (self.angle_max - self.angle_min) + self.angle_min).numpy()
            
        # TODO: Implement other trajectory types (LINEAR, CIRCULAR etc.) if needed for online generation
        # TODO: Handle near-field ranges generation based on self.system_model_params.field_type
        
        return self.last_true_angles.copy(), self.current_M

    def get_next_window(self, window_size: int) -> Tuple[torch.Tensor, List[int], List[np.ndarray]]:
        """
        Generates the next window of trajectory data.

        Args:
            window_size: The number of time steps in the window.

        Returns:
            A tuple containing:
            - observations_tensor: Tensor of shape [window_size, N_antennas, T_snapshots_per_step]
            - sources_nums_list: List of source counts for each step in the window.
            - true_labels_list: List of true angle (and range) np.arrays for each step.
        """
        window_observations_list = []
        window_sources_nums_list = []
        window_true_labels_list = []

        for step_in_window in range(window_size):
            current_true_angles, num_sources_for_step = self._generate_next_true_step()
            
            # Set labels for the Samples model (which uses the current eta from shared system_model_params)
            if self.system_model_params.field_type.lower() == "far":
                self.samples_model.set_labels(num_sources_for_step, angles=current_true_angles.tolist(), distances=None)
            else: # near or full field
                # TODO: Add actual distance generation for near-field
                # For now, using placeholder distances if near-field
                placeholder_distances = np.array([20.0] * num_sources_for_step) 
                self.samples_model.set_labels(num_sources_for_step, angles=current_true_angles.tolist(), distances=placeholder_distances.tolist())
            
            # Generate noisy observation matrix using the current system_model_params.eta
            # samples_creation returns (array_output, true_clean_signal, true_noise, sources_positions)
            observation_matrix, _, _, _ = self.samples_model.samples_creation(source_number=num_sources_for_step)
            
            window_observations_list.append(torch.as_tensor(observation_matrix, dtype=torch.complex64))
            window_sources_nums_list.append(num_sources_for_step)
            
            # Store ground truth for this step using the actual labels from samples_model (in radians)
            # This ensures consistency with the units used by the model
            true_label_for_step = self.samples_model.get_labels().cpu().numpy()
            window_true_labels_list.append(true_label_for_step)

            self.current_step_in_session +=1

        # Stack observations for the window: [window_size, N_antennas, T_snapshots_per_step]
        observations_tensor = torch.stack(window_observations_list)
        
        # logger.debug(f"Generated window: {self.current_step_in_session - window_size +1} to {self.current_step_in_session}")
        return observations_tensor, window_sources_nums_list, window_true_labels_list

# REFACTORED OnlineLearningDataset class
class OnlineLearningDataset(Dataset):
    """
    Dataset for online learning that generates data on-the-fly using a
    trajectory generator. It does not pre-compute or store the entire trajectory.
    """
    
    def __init__(self, generator: OnlineLearningTrajectoryGenerator, 
                 total_num_windows: int, window_size: int):
        """
        Args:
            generator: An instance of OnlineLearningTrajectoryGenerator.
            total_num_windows: The total number of windows this dataset will yield.
            window_size: The size of each window (number of steps).
        """
        self.generator = generator
        self.total_num_windows = total_num_windows
        self.window_size = window_size
        self._generated_windows_count = 0 # Internal counter

        if self.total_num_windows <= 0:
            raise ValueError(f"total_num_windows must be positive, got {total_num_windows}")
        logger.info(f"OnlineLearningDataset initialized. Expecting to generate {total_num_windows} windows of size {window_size}.")
    
    def __len__(self):
        """Return the total number of windows to be generated."""
        return self.total_num_windows
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[int], List[np.ndarray]]:
        """
        Generates and returns the next window of data.
        The `idx` is used to ensure we don't generate more windows than `total_num_windows`.
        """
        if idx < 0 or idx >= self.total_num_windows:
            raise IndexError(f"Index {idx} out of bounds for OnlineLearningDataset of size {self.total_num_windows}")

        # This check ensures that __getitem__ is somewhat stateful regarding the number of windows yielded
        # in a typical iteration. For random access (less common for online learning), this might behave unexpectedly
        # if not iterated sequentially. However, DataLoader usually iterates sequentially.
        if self._generated_windows_count >= self.total_num_windows:
            # This case should ideally not be reached if DataLoader respects __len__
            raise IndexError(f"Attempted to generate window beyond total_num_windows ({self.total_num_windows}).")

        # time_series_window_tensor, sources_num_list, labels_list_of_arrays
        window_data = self.generator.get_next_window(self.window_size)
        self._generated_windows_count += 1
        return window_data

    def update_eta(self, new_eta: float):
        """Delegates eta update to the underlying generator."""
        self.generator.update_eta(new_eta)

    def get_dataloader(self, batch_size: int, shuffle: bool = True, collate_fn: Optional[Callable] = None) -> DataLoader:
        """
        Creates a DataLoader for this dataset.

        Args:
            batch_size: How many samples per batch to load.
            shuffle: Whether to shuffle the data at every epoch.
            collate_fn: Custom collate function. If None, uses a default.

        Returns:
            A DataLoader instance.
        """
        effective_collate_fn = collate_fn if collate_fn is not None else self._collate_windows
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=effective_collate_fn
        )

    def _collate_windows(self, batch: List[Tuple[torch.Tensor, List[int], List[np.ndarray]]]) -> Tuple[torch.Tensor, torch.Tensor, List[List[np.ndarray]]]:
        """
        Custom collate function for batches of windows.
        Typically, batch_size will be 1 for online learning.
        
        Args:
            batch: A list of tuples, where each tuple is (time_series_window_tensor, sources_num_list, labels_list_of_arrays).
                   - time_series_window_tensor: [window_size, N, T]
                   - sources_num_list: List[int] of M for each step in the window
                   - labels_list_of_arrays: List[np.ndarray] of true labels for each step
            
        Returns:
            Collated batch:
            - Batched time_series: Tensor[batch_size, window_size, N, T]
            - Batched sources_num: Tensor[batch_size, window_size] (long)
            - Batched labels: List of lists of np.ndarray, outer list for batch, inner for window steps.
        """
        # Unzip the batch
        time_series_windows, sources_num_lists, labels_lists = zip(*batch)
        
        # Stack time_series tensors along a new batch dimension
        # Each time_series_window is already [window_size, N, T]
        batched_time_series = torch.stack(time_series_windows) # -> [batch_size, window_size, N, T]
        
        # Pad and stack sources_num lists
        # Assuming all windows in a batch have the same window_size (which they should from __getitem__)
        # max_window_len = max(len(s_list) for s_list in sources_num_lists) # Should be self.window_size
        batched_sources_num_tensors = [torch.tensor(s_list, dtype=torch.long) for s_list in sources_num_lists]
        batched_sources_num = torch.stack(batched_sources_num_tensors) # -> [batch_size, window_size]

        # Labels (labels_lists) remain a list of lists of arrays because the number of sources (and thus label size)
        # can vary per step, making direct tensor stacking complex if M changes.
        # For online learning evaluation, this structure is often processed step-by-step.
        batched_labels = list(labels_lists) # -> List[batch_size] of List[window_size] of np.ndarray

        return batched_time_series, batched_sources_num, batched_labels

# REFACTORED create_online_learning_dataset factory function
def create_online_learning_dataset(
    system_model_params: SystemModelParams, 
    config: Config, # Full config for access to online_learning and trajectory sections
    window_size: int, 
    stride: int # Stride determines the "granularity" or "density" of windows over the total duration
) -> OnlineLearningDataset:
    """
    Creates an OnlineLearningDataset that generates data on-the-fly.

    Args:
        system_model_params: The shared SystemModelParams instance.
        config: The main simulation configuration object.
        window_size: The number of time steps in each generated window.
        stride: The step size between the start of consecutive windows. This, along with
                trajectory_length and window_size, determines the total number of windows.
    Returns:
        An OnlineLearningDataset instance.
    """
    online_config = config.online_learning
    
    # total_duration_steps is the total number of individual simulation steps planned for the online learning session.
    # The generator will be capable of producing up to this many steps in sequence.
    total_duration_steps = online_config.trajectory_length
    if total_duration_steps is None: # Fallback if not specified in online_learning
        total_duration_steps = config.trajectory.trajectory_length
    
    if not isinstance(total_duration_steps, int) or total_duration_steps <= 0:
        raise ValueError(f"online_learning.trajectory_length must be a positive integer, got {total_duration_steps}")

    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    if total_duration_steps < window_size:
        raise ValueError(f"Online learning total_duration_steps ({total_duration_steps}) must be >= window_size ({window_size}).")

    # Calculate the total number of unique windows that can be formed from the total_duration_steps
    # This defines how many times we can call __getitem__ before exhausting the dataset.
    num_possible_windows = (total_duration_steps - window_size) // stride + 1
    
    if num_possible_windows <= 0:
        # This can happen if, e.g., total_duration_steps = 10, window_size = 10, stride = 1 -> 1 window
        # total_duration_steps = 10, window_size = 11, stride = 1 -> 0 windows
        raise ValueError(
            f"Cannot form any windows. total_duration_steps={total_duration_steps}, "
            f"window_size={window_size}, stride={stride}. Results in {num_possible_windows} windows."
        )

    logger.info(
        f"Creating on-demand OnlineLearningDataset: "
        f"total_duration_steps={total_duration_steps}, window_size={window_size}, stride={stride}. "
        f"This will yield {num_possible_windows} windows."
    )

    # Pass the shared system_model_params instance. Eta updates on this instance
    # will be reflected in the generator.
    # Also pass M (number of sources) which can be fixed or a range from system_model_params
    num_sources_M = config.system_model.M 

    generator = OnlineLearningTrajectoryGenerator(
        system_model_params=system_model_params,
        trajectory_config=config.trajectory, # For type, random_walk_std_dev etc.
        initial_eta=system_model_params.eta, # Start with current eta from config (or last dynamic update)
        num_sources=num_sources_M
    )
    
    return OnlineLearningDataset(
        generator=generator, 
        total_num_windows=num_possible_windows, 
        window_size=window_size
    ) 
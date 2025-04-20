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
from config.schema import TrajectoryType

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
    
    def _generate_trajectories(self, 
                             samples_size: int, 
                             trajectory_length: int,
                             trajectory_type: Union[TrajectoryType, str],
                             custom_trajectory_fn: Optional[Callable] = None) -> Tuple[Tuple[torch.Tensor, torch.Tensor], List[int]]:
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
            if trajectory_type == TrajectoryType.RANDOM:
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
                
                # Get random walk step size from config
                random_walk_std_dev = 1.0  # Default value
                if self.config and hasattr(self.config.trajectory, 'random_walk_std_dev'):
                    random_walk_std_dev = self.config.trajectory.random_walk_std_dev
                    
                logger.info(f"Using random walk with std dev: {random_walk_std_dev}")
                
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
            print(f"Trajectory plot saved to: {output_path}")
            plt.show()
            
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
                
                # Simply convert to tensor without forcing any particular type
                # Let PyTorch handle tensor types internally
                X_tensor = torch.tensor(X)
                
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
            torch.tensor(self.h5f[f'X/{idx}/step_{step}'][:]) 
            for step in range(trajectory_length)
        ]
        
        # Load labels for all steps
        labels = [
            torch.tensor(self.h5f[f'Y/{idx}/label_{step}'][:]) 
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
            dtype=torch.float32
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
            padded_sources_num[i, :traj_len] = torch.tensor(src)
            
            # Copy labels (with padding for varying number of sources)
            for t in range(traj_len):
                n_src = src[t]
                padded_labels[i, t, :n_src] = lbl[t]
        
        return padded_time_series, padded_sources_num, padded_labels 
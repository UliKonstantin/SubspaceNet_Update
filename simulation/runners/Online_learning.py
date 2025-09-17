from ast import Not
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import logging
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import time
import copy
import yaml
from itertools import permutations
import itertools
from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any

from config.schema import Config
from simulation.runners.data import TrajectoryDataHandler, create_online_learning_dataset
from simulation.runners.training import Trainer, TrainingConfig, TrajectoryTrainer, OnlineTrainer
from simulation.runners.evaluation import Evaluator
from utils.plotting import plot_online_learning_results, plot_online_learning_trajectory
from utils.utils import log_window_summary, save_model_state, log_online_learning_window_summary
from simulation.kalman_filter import KalmanFilter1D, BatchKalmanFilter1D, BatchExtendedKalmanFilter1D
from DCD_MUSIC.src.metrics.rmspe_loss import RMSPELoss
from DCD_MUSIC.src.metrics.rmape_loss import RMAPELoss
from DCD_MUSIC.src.metrics.multimoment_innovation_consistency_loss import MultiMomentInnovationConsistencyLoss
from DCD_MUSIC.src.signal_creation import Samples
from DCD_MUSIC.src.evaluation import get_model_based_method, evaluate_model_based
from simulation.kalman_filter.extended import ExtendedKalmanFilter1D


class KalmanInnovationLoss:
    """
    Loss function for Kalman gain times innovation.
    
    This loss encourages the model to produce predictions that result in
    smaller Kalman gain times innovation values, which indicates better
    measurement quality and filter performance.
    """
    
    def __init__(self):
        """Initialize the Kalman Innovation Loss."""
        pass
    
    def __call__(self, kalman_gain_times_innovation: torch.Tensor) -> torch.Tensor:
        """
        Calculate the absolute value of Kalman gain times innovation.
        
        Args:
            kalman_gain_times_innovation: Tensor of K*y values from EKF
            
        Returns:
            Loss tensor (absolute value of K*y)
        """
        return torch.abs(kalman_gain_times_innovation)


class YSInvYLoss:
    """
    Loss function for y*S^-1*y metric.
    
    This loss encourages the model to produce predictions that result in
    smaller y*S^-1*y values, which indicates better measurement quality
    and filter performance in terms of normalized innovation squared.
    """
    
    def __init__(self):
        """Initialize the Y*S^-1*Y Loss."""
        pass
    
    def __call__(self, y_s_inv_y: torch.Tensor) -> torch.Tensor:
        """
        Calculate the absolute value of y*S^-1*y.
        
        Args:
            y_s_inv_y: Tensor of y*S^-1*y values from EKF
            
        Returns:
            Loss tensor (absolute value of y*S^-1*y)
        """
        return torch.abs(y_s_inv_y)


logger = logging.getLogger(__name__)

# Device setup for evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class StepMetrics:
    """Encapsulates step-level metrics for a window evaluation."""
    covariances: torch.Tensor  # Shape: [window_size, max_sources]
    innovations: torch.Tensor  # Shape: [window_size, max_sources]
    kalman_gains: torch.Tensor  # Shape: [window_size, max_sources]
    kalman_gain_times_innovation: torch.Tensor  # Shape: [window_size, max_sources]
    y_s_inv_y: torch.Tensor  # Shape: [window_size, max_sources]


@dataclass
class DOAMetrics:
    """Encapsulates DOA (Direction of Arrival) prediction metrics for a window evaluation."""
    ekf_predictions: torch.Tensor  # Shape: [window_size, max_sources] - EKF angle predictions
    pre_ekf_predictions: torch.Tensor  # Shape: [window_size, max_sources] - Pre-EKF angle predictions
    true_angles: torch.Tensor  # Shape: [window_size, max_sources] - Ground truth angles
    avg_ekf_angle_pred: List[float] = None  # Averaged EKF angle predictions per source
    avg_pre_ekf_angle_pred: List[float] = None  # Averaged pre-EKF angle predictions per source


@dataclass
class LossMetrics:
    """Encapsulates all loss-related metrics for a window evaluation."""
    main_loss: float              # Primary loss (uses supervision + metric)
    main_loss_db: float           # Primary loss in dB units (20 * log10(main_loss))
    main_loss_config: str         # Configuration string for main loss (e.g., "supervised_rmspe")
    online_training_reference_loss: float  # Training reference loss (uses training_loss_type)
    online_training_reference_loss_config: str  # Configuration string for training reference loss (e.g., "multimoment")
    pre_ekf_loss: float          # Raw model performance
    ekf_gain_rmspe: float        # EKF improvement (RMSPE)
    ekf_gain_rmape: float        # EKF improvement (RMAPE)
    


@dataclass
class WindowMetrics:
    """Encapsulates window-level metrics and state."""
    window_size: int
    num_sources: int
    avg_covariance: float  # Average covariance across all sources and steps
    eta_value: float  # Current eta value from system model
    is_near_field: bool  # Whether processing near-field or far-field
    # Averaged metrics across time steps
    avg_ekf_angle_pred: List[float] = None  # Averaged EKF angle predictions per source
    avg_pre_ekf_angle_pred: List[float] = None  # Averaged pre-EKF angle predictions per source
    avg_ekf_covariances: List[float] = None  # Averaged EKF covariances per source
    avg_ekf_innovations: List[float] = None  # Averaged EKF innovations per source
    avg_ekf_kalman_gains: List[float] = None  # Averaged EKF Kalman gains per source
    avg_ekf_kalman_gain_times_innovation: List[float] = None  # Averaged EKF Kalman gain times innovation per source
    avg_ekf_y_s_inv_y: List[float] = None  # Averaged EKF y_s_inv_y per source
    avg_step_innovation_covariances: List[float] = None  # Averaged step innovation covariances per source


@dataclass
class TrajectoryResults:
    """Encapsulates all results for a single trajectory using structured approach."""
    # Window-level results
    window_results: List['WindowEvaluationResult']
    
    # Basic trajectory tracking
    window_indices: List[int]
    window_eta_values: List[float]
    
    # Labels
    window_labels: List[List[np.ndarray]]
    
    def __init__(self):
        # Initialize all lists
        self.window_results = []
        self.window_indices = []
        self.window_eta_values = []
        self.window_labels = []
    
    def add_window_result(self, window_idx: int, window_result: 'WindowEvaluationResult', eta_value: float, labels: List[np.ndarray]):
        """Add a window result to the trajectory results."""
        self.window_results.append(window_result)
        self.window_indices.append(window_idx)
        self.window_eta_values.append(eta_value)
        self.window_labels.append(labels)


@dataclass
class WindowEvaluationResult:
    """Main return type for window evaluation containing all metrics and data."""
    # Core metrics
    loss_metrics: LossMetrics
    window_metrics: WindowMetrics
    step_metrics: StepMetrics
    doa_metrics: DOAMetrics
    
    
    # Success indicators
    is_valid: bool = True
    error_message: Optional[str] = None
    
    def to_tuple(self) -> Tuple:
        """Convert to the original tuple format for backward compatibility."""
        return (
            self.loss_metrics.main_loss,
            self.window_metrics.avg_covariance,
            self.doa_metrics.ekf_predictions,
            self.step_metrics.covariances,
            self.loss_metrics.pre_ekf_loss,
            self.step_metrics.innovations,
            self.step_metrics.kalman_gains,
            self.step_metrics.kalman_gain_times_innovation,
            self.step_metrics.y_s_inv_y,
            self.loss_metrics.ekf_gain_rmspe,
            self.doa_metrics.pre_ekf_predictions,
            self.loss_metrics.ekf_gain_rmape,
            self.window_metrics.avg_ekf_angle_pred,
            self.window_metrics.avg_pre_ekf_angle_pred
        )
    
    @classmethod
    def create_error_result(cls, error_message: str) -> 'WindowEvaluationResult':
        """Create an error result with default values."""
        return cls(
            loss_metrics=LossMetrics(
                main_loss=float('inf'),
                main_loss_db=float('inf'),
                main_loss_config="error",
                online_training_reference_loss=float('inf'),
                online_training_reference_loss_config="error",
                pre_ekf_loss=float('inf'),
                ekf_gain_rmspe=0.0,
                ekf_gain_rmape=0.0
            ),
            window_metrics=WindowMetrics(
                window_size=0,
                num_sources=0,
                avg_covariance=float('nan'),
                eta_value=0.0,
                is_near_field=False,
                avg_ekf_angle_pred=[],
                avg_pre_ekf_angle_pred=[],
                avg_ekf_covariances=[],
                avg_ekf_innovations=[],
                avg_ekf_kalman_gains=[],
                avg_ekf_kalman_gain_times_innovation=[],
                avg_ekf_y_s_inv_y=[],
                avg_step_innovation_covariances=[]
            ),
            step_metrics=StepMetrics(
                covariances=torch.empty(0, 0, dtype=torch.float64),
                innovations=torch.empty(0, 0, dtype=torch.float64),
                kalman_gains=torch.empty(0, 0, dtype=torch.float64),
                kalman_gain_times_innovation=torch.empty(0, 0, dtype=torch.float64),
                y_s_inv_y=torch.empty(0, 0, dtype=torch.float64)
            ),
            doa_metrics=DOAMetrics(
                ekf_predictions=torch.empty(0, 0, dtype=torch.float64),
                pre_ekf_predictions=torch.empty(0, 0, dtype=torch.float64),
                true_angles=torch.empty(0, 0, dtype=torch.float64),
                avg_ekf_angle_pred=[],
                avg_pre_ekf_angle_pred=[]
            ),
            is_valid=False,
            error_message=error_message
        )


class OnlineLearning:
    """
    Handles all online learning functionality for the simulation.
    
    This class encapsulates the online learning pipeline including:
    - Multi-trajectory online learning
    - EKF-based state estimation
    - Dynamic eta updates
    - Results plotting and analysis
    """
    
    def __init__(self, config: Config, system_model, trained_model, output_dir: Path, results: Dict[str, Any]):
        """
        Initialize the OnlineLearning handler.
        
        Args:
            config: Configuration object
            system_model: System model instance
            trained_model: Trained neural network model
            output_dir: Directory for saving results
            results: Results dictionary to store outputs
        """
        self.config = config
        self.system_model = system_model
        self.trained_model = trained_model
        self.output_dir = output_dir
        self.results = results
        print(self.system_model.params.snr)
        # Dual model online learning state variables
        self.drift_detected = False
        self.learning_done = False
        self.online_model = None
        self.online_training_count = 0
        self.first_eta_change = True  # Track if this is the first eta change
        self.learning_start_window = None  # Track when learning started
        self.training_window_indices = []  # Track which windows were used for training
        
        # Get time_to_learn from configuration
        online_config = self.config.online_learning
        self.time_to_learn = getattr(online_config, 'time_to_learn', None)
        if self.time_to_learn is None:
            logger.warning("time_to_learn not specified in config, online learning will not start automatically")
        else:
            logger.info(f"Online learning will start at window {self.time_to_learn}")
        
        logger.info("OnlineLearning handler initialized")
    
    def run_online_learning(self) -> Dict[str, Any]:
        """
        Run online learning pipeline over multiple trajectories and average results.
        
        This method runs the online learning process over a dataset of trajectories and
        averages the results for more robust analysis.
        
        Returns:
            Dict containing online learning results.
        """
        logger.info("Starting online learning pipeline")
        
        try:
            # Check if model is available for online learning
            if self.trained_model is None:
                logger.error("No model available for online learning")
                return {"status": "error", "message": "No model available for online learning"}
            
            # Get dataset size from config
            online_config = self.config.online_learning
            dataset_size = getattr(online_config, 'dataset_size', 1)
            
            logger.info(f"Starting online learning over {dataset_size} trajectories")
            
            all_results = []
            
            # Run online learning for each trajectory
            for trajectory_idx in range(dataset_size):
                logger.info(f"Processing trajectory {trajectory_idx + 1}/{dataset_size}")
                
                # Reset eta to initial value for each new trajectory
                initial_eta = 0
                logger.info(f"Resetting eta to initial value {initial_eta:.4f} for trajectory {trajectory_idx + 1}")
                self.system_model.params.eta = initial_eta
                
                # Reset the system model's distance noise and eta scaling
                self.system_model.eta = self.system_model._SystemModel__set_eta()
                if not getattr(self.system_model.params, 'nominal', True):
                    self.system_model.location_noise = self.system_model.get_distance_noise(True)
                
                # Run single trajectory online learning
                trajectory_result = self._run_single_trajectory_online_learning(trajectory_idx)
                
                
                if trajectory_result.get("status") == "error":
                    logger.error(f"Error in trajectory {trajectory_idx + 1}: {trajectory_result.get('message')}")
                    continue
                    
                all_results.append(trajectory_result)
            
            if not all_results:
                logger.error("No successful trajectory results")
                self.results["online_learning_error"] = "No successful trajectory results"
                return {"status": "error", "message": "No successful trajectory results"}
                
            # Average results across all trajectories using the new utility method
            from utils.utils import average_online_learning_results_across_trajectories
            averaged_results_across_trajectories = average_online_learning_results_across_trajectories(all_results)
            
            # Store averaged results
            if averaged_results_across_trajectories.get("status") == "success":
                self.results["online_learning_averaged"] = averaged_results_across_trajectories["averaged_results"]
                logger.info(f"Successfully averaged results across {averaged_results_across_trajectories['averaged_results']['trajectory_count']} trajectories")
            else:
                logger.warning(f"Failed to average trajectory results: {averaged_results_across_trajectories.get('message')}")
            

            # Use structured plotting approach with AVERAGED results
            from utils.plotting import plot_online_learning_results_structured
            
            # Extract individual trajectory results for the old plotting method (will be removed later)
            pretrained_trajectory_results = [result["online_learning_results"]["pretrained_model_trajectory_results"] for result in all_results]
            online_trajectory_results = [result["online_learning_results"]["online_model_trajectory_results"] for result in all_results]
            
            # Get loss configurations from the first result
            if all_results and all_results[0]["online_learning_results"]["pretrained_model_trajectory_results"].window_results:
                first_window_result = all_results[0]["online_learning_results"]["pretrained_model_trajectory_results"].window_results[0]
                main_loss_config = first_window_result.loss_metrics.main_loss_config
                training_reference_loss_config = first_window_result.loss_metrics.online_training_reference_loss_config
            else:
                main_loss_config = "unknown"
                training_reference_loss_config = "unknown"
            
            # Get training and eta change info from results
            training_start_window = None
            training_end_window = None
            eta_change_windows = []
            if all_results and len(all_results) > 0:
                # Get training info from first trajectory result
                first_result = all_results[0]["online_learning_results"]
                training_start_window = first_result.get("training_start_window")
                training_end_window = first_result.get("training_end_window")
                eta_change_windows = first_result.get("eta_change_windows", [])
            
            #plot_online_learning_results_structured(
            #    self.output_dir,
            #    pretrained_trajectory_results,
            #    online_trajectory_results,
            #    main_loss_config,
            #    training_reference_loss_config,
            #    training_start_window,
            #    training_end_window,
            #    eta_change_windows
            #)
            
            # ALSO call the new direct averaged plotting function
            if averaged_results_across_trajectories.get("status") == "success":
                from utils.plotting import plot_averaged_online_learning_results
                
                averaged_data = averaged_results_across_trajectories["averaged_results"]
                plot_averaged_online_learning_results(
                    self.output_dir,
                    averaged_data["averaged_pretrained_trajectory"],
                    averaged_data["averaged_online_trajectory"],
                    main_loss_config,
                    training_reference_loss_config,
                    training_start_window,
                    training_end_window,
                    eta_change_windows,
                    averaged_data.get("averaged_supervised_trajectory")
                )
            
            # Calculate summary statistics from structured results
            total_drift_detected = sum(result["online_learning_results"].get("drift_detected_count", 0) for result in all_results)
            total_model_updated = sum(result["online_learning_results"].get("model_updated_count", 0) for result in all_results)
            avg_drift_detected = total_drift_detected / len(all_results) if all_results else 0
            avg_model_updated = total_model_updated / len(all_results) if all_results else 0
            
            logger.info(f"Online learning completed over {dataset_size} trajectories: "
                       f"{avg_drift_detected:.1f} avg drifts detected, "
                       f"{avg_model_updated:.1f} avg model updates")
            
            # Prepare return results
            return_results = {
                "status": "success", 
                "online_learning_results": {
                    "pretrained_trajectory_results": pretrained_trajectory_results,
                    "online_trajectory_results": online_trajectory_results,
                    "avg_drift_detected": avg_drift_detected,
                    "avg_model_updated": avg_model_updated,
                    "dataset_size": dataset_size
                }
            }
            
            # Add averaged results if available
            if averaged_results_across_trajectories.get("status") == "success":
                return_results["averaged_results"] = averaged_results_across_trajectories["averaged_results"]
            
            return return_results
            
        except Exception as e:
            logger.exception(f"Error running online learning: {e}")
            self.results["online_learning_error"] = str(e)
            return {"status": "error", "message": str(e), "exception": type(e).__name__}

    def _create_averaged_trajectory_result(self, averaged_metrics: dict) -> TrajectoryResults:
        """
        Create a TrajectoryResults object from averaged metrics for plotting.
        
        Args:
            averaged_metrics: Dictionary containing averaged metrics from utils.average_online_learning_results_across_trajectories
            
        Returns:
            TrajectoryResults object containing the averaged data
        """
        import torch
        import numpy as np
        
        # Create a TrajectoryResults object
        trajectory_result = TrajectoryResults()
        
        # Get the number of windows from the averaged data
        num_windows = len(averaged_metrics.get("main_losses", []))
        
        # Create window results for each averaged window
        for window_idx in range(num_windows):
            # Create averaged loss metrics
            loss_metrics = LossMetrics(
                main_loss=averaged_metrics["main_losses"][window_idx],
                main_loss_db=averaged_metrics["main_losses_db"][window_idx],
                main_loss_config="averaged_supervised_rmspe",  # Placeholder
                online_training_reference_loss=averaged_metrics["training_reference_losses"][window_idx],
                online_training_reference_loss_config="averaged_multimoment",  # Placeholder
                pre_ekf_loss=averaged_metrics["main_losses"][window_idx] + averaged_metrics["ekf_gain_rmspe"][window_idx],  # Approximate
                ekf_gain_rmspe=averaged_metrics["ekf_gain_rmspe"][window_idx],
                ekf_gain_rmape=averaged_metrics["ekf_gain_rmape"][window_idx]
            )
            
            # Create averaged window metrics
            window_metrics = WindowMetrics(
                window_size=10,  # Placeholder
                num_sources=3,   # Placeholder
                avg_covariance=averaged_metrics["avg_covariances"][window_idx],
                eta_value=averaged_metrics["window_eta_values"][window_idx],
                is_near_field=False,  # Placeholder
                avg_ekf_angle_pred=None,  # Not averaged (meaningless)
                avg_pre_ekf_angle_pred=None,  # Not averaged (meaningless)
                avg_ekf_covariances=None,  # Could add if needed
                avg_ekf_innovations=[averaged_metrics["avg_innovations"][window_idx]] if averaged_metrics["avg_innovations"][window_idx] > 0 else None,
                avg_ekf_kalman_gains=[averaged_metrics["avg_kalman_gains"][window_idx]] if averaged_metrics["avg_kalman_gains"][window_idx] > 0 else None,
                avg_ekf_kalman_gain_times_innovation=[averaged_metrics["avg_kalman_gain_times_innovation"][window_idx]] if averaged_metrics["avg_kalman_gain_times_innovation"][window_idx] > 0 else None,
                avg_ekf_y_s_inv_y=[averaged_metrics["avg_y_s_inv_y"][window_idx]] if averaged_metrics["avg_y_s_inv_y"][window_idx] > 0 else None,
                avg_step_innovation_covariances=None
            )
            
            # Create placeholder step metrics (not used in structured plotting)
            step_metrics = StepMetrics(
                covariances=torch.zeros(1, 3),  # Placeholder
                innovations=torch.zeros(1, 3),  # Placeholder
                kalman_gains=torch.zeros(1, 3),  # Placeholder
                kalman_gain_times_innovation=torch.zeros(1, 3),  # Placeholder
                y_s_inv_y=torch.zeros(1, 3)  # Placeholder
            )
            
            # Create placeholder DOA metrics (not used in structured plotting)
            doa_metrics = DOAMetrics(
                ekf_predictions=torch.zeros(1, 3),  # Placeholder
                pre_ekf_predictions=torch.zeros(1, 3),  # Placeholder
                true_angles=torch.zeros(1, 3),  # Placeholder
                avg_ekf_angle_pred=None,  # Not meaningful for averaged data
                avg_pre_ekf_angle_pred=None  # Not meaningful for averaged data
            )
            
            # Create window evaluation result
            window_result = WindowEvaluationResult(
                loss_metrics=loss_metrics,
                window_metrics=window_metrics,
                step_metrics=step_metrics,
                doa_metrics=doa_metrics,
                is_valid=True,
                error_message=None
            )
            
            # Add to trajectory results
            trajectory_result.add_window_result(
                window_idx=averaged_metrics["window_indices"][window_idx],
                window_result=window_result,
                eta_value=averaged_metrics["window_eta_values"][window_idx],
                labels=[]  # Empty labels for averaged data
            )
        
        return trajectory_result

    def _run_single_trajectory_online_learning(self, trajectory_idx: int = 0) -> Dict[str, Any]:
        """
        Run online learning pipeline for a single trajectory.
        """
        try:
            # Access online learning configuration
            online_config = self.config.online_learning
            
            # Check if model is available for online learning
            if self.trained_model is None:
                logger.error("No model available for online learning")
                return {"status": "error", "message": "No model available for online learning"}
            
            # Initialize online model as copy of trained model using the same factory function
            import torch
            from copy import deepcopy
            
            # Create a new model instance using the same factory function that created the trained model
            try:
                # Method 1: Use the same factory function to ensure identical architecture
                from config.factory import create_model
                self.online_model = create_model(self.config, self.system_model)
                self.online_model.load_state_dict(self.trained_model.state_dict())
                self.online_model.eval()  # Set to eval mode like the trained model
                logger.info("Initialized online model as copy of trained model using factory function")
            except Exception as e:
                logger.warning(f"Factory function copy failed ({e}), trying clone approach")
                try:
                    # Method 2: Use torch.clone() for parameters
                    self.online_model = deepcopy(self.trained_model.cpu())
                    if torch.cuda.is_available() and next(self.trained_model.parameters()).is_cuda:
                        self.online_model = self.online_model.cuda()
                    logger.info("Initialized online model using CPU deepcopy then moved to GPU")
                except Exception as e2:
                    logger.error(f"All model copying methods failed: {e2}")
                    raise RuntimeError(f"Failed to create online model copy. Factory function failed: {e}, Deepcopy failed: {e2}. Cannot proceed with online learning.")
            
            # Create supervised trained model as copy of trained model (same approach as online model)
            try:
                # Method 1: Use the same factory function to ensure identical architecture
                self.supervised_trained_model = create_model(self.config, self.system_model)
                self.supervised_trained_model.load_state_dict(self.trained_model.state_dict())
                self.supervised_trained_model.train()  # Set to training mode for supervised learning
                logger.info("Initialized supervised trained model as copy of trained model using factory function")
            except Exception as e:
                logger.warning(f"Factory function copy failed for supervised model ({e}), trying clone approach")
                try:
                    # Method 2: Use torch.clone() for parameters
                    self.supervised_trained_model = deepcopy(self.trained_model.cpu())
                    if torch.cuda.is_available() and next(self.trained_model.parameters()).is_cuda:
                        self.supervised_trained_model = self.supervised_trained_model.cuda()
                    self.supervised_trained_model.train()  # Set to training mode for supervised learning
                    logger.info("Initialized supervised trained model using CPU deepcopy then moved to GPU")
                except Exception as e2:
                    logger.error(f"All supervised model copying methods failed: {e2}")
                    raise RuntimeError(f"Failed to create supervised trained model copy. Factory function failed: {e}, Deepcopy failed: {e2}. Cannot proceed with supervised learning.")
            
            # Reset dual model state for new trajectory
            self.drift_detected = False
            self.learning_done = False
            self.online_training_count = 0
            self.first_eta_change = True
            # Reset online optimizer to start fresh for new trajectory
            if hasattr(self, 'online_optimizer'):
                delattr(self, 'online_optimizer')
            # Reset supervised optimizer to start fresh for new trajectory
            if hasattr(self, 'supervised_optimizer'):
                delattr(self, 'supervised_optimizer')
            logger.info("Reset dual model state for new trajectory")
            
            # Get online learning parameters
            window_size = getattr(online_config, 'window_size', 10)
            stride = getattr(online_config, 'stride', 5)
            
            # Create an on-demand dataset and dataloader for online learning
            # This will generate data in windows with the current system_model.params.
            # This allows dynamic eta updates within the generator to be reflected.
            system_model_params = self.system_model.params
            
            # Create on-demand dataset using the factory function
            logger.info("Online learning is enabled, creating on-demand dataset and dataloader.")
                # SET BREAKPOINT HERE - Before creating dataset
            print(f"Creating dataset with SNR: {system_model_params.snr}")
            online_learning_dataset = create_online_learning_dataset(
                system_model_params=system_model_params,
                config=self.config,
                window_size=window_size,
                stride=stride
            )
            
            # Ensure the dataset generator uses the current (reset) eta value
            current_eta = system_model_params.eta
            online_learning_dataset.update_eta(current_eta)
            logger.info(f"Initialized trajectory with eta = {current_eta:.4f}")
            
            # Create dataloader from on-demand dataset
            from torch.utils.data import DataLoader
            online_learning_dataloader = DataLoader(
                online_learning_dataset,
                batch_size=1,  # Process one window at a time
                shuffle=False,
                num_workers=0,  # On-demand dataset must use 0 workers
                drop_last=False
            )
            logger.info(f"Created on-demand online learning dataloader for {len(online_learning_dataloader)} windows.")
            
            # Create online trainer
            online_trainer = OnlineTrainer(
                model=self.trained_model,
                config=self.config,
                device=device
            )
            
            # Set loss threshold for drift detection
            loss_threshold = getattr(online_config, 'loss_threshold', 0.5)
            max_iterations = getattr(online_config, 'max_iterations', 10)
            
            # Initialize trajectory results structure
            trajectory_results = TrajectoryResults()
            window_update_flags = []
            drift_detected_count = 0
            model_updated_count = 0
            last_ekf_predictions = None  # Track last window's EKF predictions
            last_ekf_covariances = None  # Track last window's EKF covariances
            
            # Initialize online model results structure
            online_trajectory_results = TrajectoryResults()
            
            # Initialize supervised model results structure
            supervised_trajectory_results = TrajectoryResults()
            
            # EKF state tracking for online learning
            online_last_ekf_predictions = None
            online_last_ekf_covariances = None
            
            # EKF state tracking for supervised learning
            supervised_last_ekf_predictions = None
            supervised_last_ekf_covariances = None
            
            # Track training and eta change events
            eta_change_windows = []  # List of window indices where eta changed
            training_start_window = None  # Window where training started
            training_end_window = None  # Window where training ended
            
            # Process each window of online data
            for window_idx, (time_series_batch, sources_num_batch, labels_batch) in enumerate(tqdm(online_learning_dataloader, desc="Online Learning")):
                # --- Dynamic Eta Update Logic ---
                if online_config.eta_update_interval_windows and online_config.eta_update_interval_windows > 0 and \
                   window_idx > 0 and window_idx % online_config.eta_update_interval_windows == 0:
                    current_eta = self.system_model.params.eta # Get current eta from the shared SystemModelParams
                    eta_increment = online_config.eta_increment if online_config.eta_increment is not None else 0.01
                    new_eta = current_eta + eta_increment
                    
                    # Apply min/max constraints if specified
                    if online_config.max_eta is not None:
                        new_eta = min(new_eta, online_config.max_eta)
                    if online_config.min_eta is not None:
                        new_eta = max(new_eta, online_config.min_eta)
                    
                    # Only update if there's an actual change
                    if abs(new_eta - current_eta) > 1e-6:
                        logger.info(f"Online Learning: Dynamically updating eta at window {window_idx}. From {current_eta:.4f} to {new_eta:.4f}")
                        # Track eta change window
                        eta_change_windows.append(window_idx)
                    if self.first_eta_change:
                        self.first_eta_change = False
                        logger.info(f"First eta modification at window {window_idx}")
                        # The dataset holds the generator, which updates the shared self.system_model.params.eta
                        online_learning_dataloader.dataset.update_eta(new_eta)
                                            # Set drift detected on first eta change only
                if self.time_to_learn is not None and window_idx == self.time_to_learn:
                    self.drift_detected = True
                    logger.info(f"Drift detected at window {window_idx} (configured time_to_learn)")
                    # Initialize online EKF state with static model's current state
                    online_last_ekf_predictions = last_ekf_predictions
                    online_last_ekf_covariances = last_ekf_covariances
                    logger.info(f"Initialized online EKF state with static model's state at window {window_idx}")
                    
                    
                # --- End Dynamic Eta Update Logic ---
                
                # Unpack batch data
                # The batch data shapes are:
                # time_series_batch: [1, window_size, N, T] = [1, 10, 8, 200]
                # sources_num_batch: [10, 1] (not [1, 10] as expected)
                # labels_batch: [10, 1, 3] (not [1, 10, 3] as expected)
                
                # Extract the content properly from the batch
                time_series_single_window = time_series_batch[0]  # Shape: [window_size, N, T]
                
                # Reshape sources_num to be a list of integers
                sources_num_single_window_list = sources_num_batch[:, 0].tolist() if isinstance(sources_num_batch, torch.Tensor) else [s[0] for s in sources_num_batch]
                
                # Reshape labels to be a list of arrays
                if isinstance(labels_batch, torch.Tensor):
                    # If labels_batch is a tensor [10, 1, 3]
                    labels_single_window_list_of_arrays = [arr[0].cpu().numpy() if isinstance(arr, torch.Tensor) else arr[0] for arr in labels_batch]
                else:
                    # If labels_batch is a list of tensors or arrays
                    labels_single_window_list_of_arrays = [arr[0].cpu().numpy() if isinstance(arr, torch.Tensor) else arr[0] for arr in labels_batch]
                
                # Calculate loss on the current window (generated with current eta)
                window_result = self._evaluate_window(
                    time_series_single_window, 
                    sources_num_single_window_list, 
                    labels_single_window_list_of_arrays,
                    trajectory_idx,
                    window_idx,
                    is_first_window=(window_idx == 0),
                    last_ekf_predictions=last_ekf_predictions,
                    last_ekf_covariances=last_ekf_covariances
                )
               
                # Add window result to trajectory results
                trajectory_results.add_window_result(window_idx, window_result, self.system_model.params.eta, labels_single_window_list_of_arrays)
                
                # Update last predictions and covariances for next window
                last_ekf_predictions = window_result.doa_metrics.ekf_predictions
                last_ekf_covariances = window_result.step_metrics.covariances
                
                logger.info(f"Window {window_idx}: Main Loss = {window_result.loss_metrics.main_loss:.6f} ({window_result.loss_metrics.main_loss_config}), Cov = {window_result.window_metrics.avg_covariance:.6f} (current eta={self.system_model.params.eta:.4f})")
                
                # Log all loss metrics for comparison
                logger.info(f"Window {window_idx}: Pre-EKF Loss = {window_result.loss_metrics.pre_ekf_loss:.6f}, Main Loss = {window_result.loss_metrics.main_loss:.6f} ({window_result.loss_metrics.main_loss_config}), Training Ref Loss = {window_result.loss_metrics.online_training_reference_loss:.6f} ({window_result.loss_metrics.online_training_reference_loss_config}), Cov = {window_result.window_metrics.avg_covariance:.6f} (eta={self.system_model.params.eta:.4f})")
                
                # Store trained model results for comparison
                trained_subspacenet_loss = window_result.loss_metrics.pre_ekf_loss
                trained_ekf_loss = window_result.loss_metrics.main_loss
                
                # Check if loss exceeds threshold (drift detected)
                if window_result.loss_metrics.main_loss > loss_threshold:
                    logger.info(f"Drift detected in window {window_idx} (loss: {window_result.loss_metrics.main_loss:.6f} > threshold: {loss_threshold:.6f})")
                    # window_update_flags.append(False)
                    # self.drift_detected = True
                    # Track when learning started
                    # if self.learning_start_window is None:
                    #     self.learning_start_window = window_idx
                    #     logger.info(f"Online learning started at window {window_idx}")
                else:
                    logger.info(f"No drift detected in window {window_idx} (loss: {window_result.loss_metrics.main_loss:.6f} <= threshold: {loss_threshold:.6f})")
                    window_update_flags.append(False)
                    # Keep previous drift_detected state if no drift in current window
                
                # Dual model processing logic
                if self.drift_detected:
                    if self.learning_done:
                        # Online model finished training, evaluate it normally
                        logger.info(f"Evaluating online model (post-training) for window {window_idx}")
                        online_window_result = self._evaluate_window(
                            time_series_single_window, 
                            sources_num_single_window_list, 
                            labels_single_window_list_of_arrays,
                            trajectory_idx,
                            window_idx,
                            is_first_window=(window_idx == 0),
                            last_ekf_predictions=online_last_ekf_predictions,
                            last_ekf_covariances=online_last_ekf_covariances,
                            model=self.online_model
                        )
                        
                        # Add online model result to trajectory results
                        online_trajectory_results.add_window_result(window_idx, online_window_result, self.system_model.params.eta, labels_single_window_list_of_arrays)
                        
                        logger.info(f"Online model - Window {window_idx}: Main Loss = {online_window_result.loss_metrics.main_loss:.6f} ({online_window_result.loss_metrics.main_loss_config}), Cov = {online_window_result.window_metrics.avg_covariance:.6f}")
                        
                        # Log online learning window summary (post-learning evaluation)
                        log_online_learning_window_summary(
                            subspacenet_loss=trained_subspacenet_loss,
                            ekf_loss=trained_ekf_loss,
                            online_ekf_loss=online_window_result.loss_metrics.main_loss,
                            current_eta=self.system_model.params.eta,
                            is_near_field=hasattr(self.trained_model, 'field_type') and self.trained_model.field_type.lower() == "near",
                            trajectory_idx=trajectory_idx,
                            window_idx=window_idx,
                            is_learning=False
                        )
                        
                        # Evaluate supervised trained model (post-training)
                        logger.info(f"Evaluating supervised trained model (post-training) for window {window_idx}")
                        supervised_window_result = self._evaluate_window(
                            time_series_single_window, 
                            sources_num_single_window_list, 
                            labels_single_window_list_of_arrays,
                            trajectory_idx,
                            window_idx,
                            is_first_window=(window_idx == 0),
                            last_ekf_predictions=supervised_last_ekf_predictions,
                            last_ekf_covariances=supervised_last_ekf_covariances,
                            model=self.supervised_trained_model
                        )
                        
                        # Add supervised model result to trajectory results
                        supervised_trajectory_results.add_window_result(window_idx, supervised_window_result, self.system_model.params.eta, labels_single_window_list_of_arrays)
                        
                        logger.info(f"Supervised trained model - Window {window_idx}: Main Loss = {supervised_window_result.loss_metrics.main_loss:.6f} ({supervised_window_result.loss_metrics.main_loss_config}), Cov = {supervised_window_result.window_metrics.avg_covariance:.6f}")
                        
                        # Update supervised EKF state for next window
                        supervised_last_ekf_predictions = supervised_window_result.doa_metrics.ekf_predictions
                        supervised_last_ekf_covariances = supervised_window_result.step_metrics.covariances
                        
                        # Update online EKF state for next window
                        # online_window_result.doa_metrics.ekf_predictions is already in tensor format
                        online_last_ekf_predictions = online_window_result.doa_metrics.ekf_predictions
                        online_last_ekf_covariances = online_window_result.step_metrics.covariances
                    else:
                        # Online model still learning/adapting
                        logger.info(f"Training online model for window {window_idx}")
                        
                        # Track training start on first training call
                        if training_start_window is None:
                            training_start_window = window_idx
                            logger.info(f"Training started at window {window_idx}")
                        
                        # Call the refactored _online_training_window method
                        training_result = self._online_training_window(
                            time_series_single_window, 
                            sources_num_single_window_list, 
                            labels_single_window_list_of_arrays,
                            trajectory_idx,
                            window_idx,
                            is_first_window=(window_idx == 0),
                            last_ekf_predictions=online_last_ekf_predictions,
                            last_ekf_covariances=online_last_ekf_covariances
                        )
                        
                        # Add training result to online trajectory results
                        online_trajectory_results.add_window_result(window_idx, training_result, self.system_model.params.eta, labels_single_window_list_of_arrays)
                        
                        logger.info(f"Online training - Window {window_idx}: Main Loss = {training_result.loss_metrics.main_loss:.6f} ({training_result.loss_metrics.main_loss_config}), Cov = {training_result.window_metrics.avg_covariance:.6f}, Pre-EKF Loss = {training_result.loss_metrics.pre_ekf_loss:.6f}")
                        
                        # Log online learning window summary (learning phase)
                        log_online_learning_window_summary(
                            subspacenet_loss=trained_subspacenet_loss,
                            ekf_loss=trained_ekf_loss,
                            online_ekf_loss=training_result.loss_metrics.main_loss,
                            current_eta=self.system_model.params.eta,
                            is_near_field=hasattr(self.trained_model, 'field_type') and self.trained_model.field_type.lower() == "near",
                            trajectory_idx=trajectory_idx,
                            window_idx=window_idx,
                            is_learning=True
                        )
                        
                        # Check if training just ended (learning_done became True)
                        if self.learning_done and training_end_window is None:
                            training_end_window = window_idx
                            logger.info(f"Training ended at window {window_idx}")
                        
                        # Update online EKF state for next window
                        # training_result.doa_metrics.ekf_predictions is already in tensor format
                        online_last_ekf_predictions = training_result.doa_metrics.ekf_predictions
                        online_last_ekf_covariances = training_result.step_metrics.covariances
                        
                        # Train supervised model with supervised loss configuration
                        logger.info(f"Training supervised trained model for window {window_idx}")
                        
                        # Create supervised loss config object
                        from copy import deepcopy
                        supervised_loss_config = deepcopy(self.config.online_learning.loss_config)
                        # Override the training_loss_type with supervised_loss_type
                        supervised_loss_config.training_loss_type = self.config.online_learning.loss_config.supervised_loss_type
                        
                        supervised_training_result = self._online_training_window(
                            time_series_single_window, 
                            sources_num_single_window_list, 
                            labels_single_window_list_of_arrays,
                            trajectory_idx,
                            window_idx,
                            is_first_window=(window_idx == 0),
                            last_ekf_predictions=supervised_last_ekf_predictions,
                            last_ekf_covariances=supervised_last_ekf_covariances,
                            model=self.supervised_trained_model,
                            loss_config_override=supervised_loss_config
                        )
                        
                        # Add supervised training result to supervised trajectory results
                        supervised_trajectory_results.add_window_result(window_idx, supervised_training_result, self.system_model.params.eta, labels_single_window_list_of_arrays)
                        
                        logger.info(f"Supervised training - Window {window_idx}: Main Loss = {supervised_training_result.loss_metrics.main_loss:.6f} ({supervised_training_result.loss_metrics.main_loss_config}), Cov = {supervised_training_result.window_metrics.avg_covariance:.6f}, Pre-EKF Loss = {supervised_training_result.loss_metrics.pre_ekf_loss:.6f}")
                        
                        # Update supervised EKF state for next window
                        supervised_last_ekf_predictions = supervised_training_result.doa_metrics.ekf_predictions
                        supervised_last_ekf_covariances = supervised_training_result.step_metrics.covariances
            
            # Save final model if it was updated
            if model_updated_count > 0:
                model_save_path = save_model_state(
                    self.trained_model,
                    self.output_dir,
                    model_type=f"{self.config.model.type}_online_updated"
                )
                logger.info(f"Saved final online-updated model to {model_save_path}")
            
            # Return results
            return {
                "status": "success",
                "online_learning_results": {
                    # Model results
                    "pretrained_model_trajectory_results": trajectory_results,
                    "online_model_trajectory_results": online_trajectory_results,
                    "supervised_model_trajectory_results": supervised_trajectory_results,
                    
                    # Learning metadata
                    "window_updates": window_update_flags,
                    "drift_detected_count": drift_detected_count,
                    "model_updated_count": model_updated_count,
                    "window_count": len(online_learning_dataloader),
                    "window_size": online_config.window_size,
                    "stride": online_config.stride,
                    "loss_threshold": loss_threshold,
                    "learning_start_window": self.learning_start_window,
                    
                    # Learning state tracking
                    "drift_detected_final": self.drift_detected,
                    "learning_done_final": self.learning_done,
                    "first_eta_change_final": self.first_eta_change,
                    "online_training_count_final": self.online_training_count,
                    
                    # Training and eta change tracking
                    "eta_change_windows": eta_change_windows,
                    "training_start_window": training_start_window,
                    "training_end_window": training_end_window
                }
            }
            
        except Exception as e:
            logger.exception(f"Error during online learning: {e}")
            return {"status": "error", "message": str(e), "exception": type(e).__name__}

    def _online_training_window(self, window_time_series, window_sources_num, window_labels, trajectory_idx: int = 0, window_idx: int = 0, 
                               is_first_window: bool = True, last_ekf_predictions: Optional[torch.Tensor] = None, 
                               last_ekf_covariances: Optional[torch.Tensor] = None, model=None, loss_config_override=None) -> WindowEvaluationResult:
        """
        Train the provided model on a single window, then evaluate it like _evaluate_window.
        
        This function performs the same evaluation as _evaluate_window but adds a training step
        after the model forward pass and before the Kalman filter processing.
        
        Args:
            model: Model to train (defaults to self.online_model if None)
            loss_config_override: Optional loss configuration override (defaults to online loss config if None)
        
        Args:
            window_time_series: Time series data for window
            window_sources_num: Source counts for window  
            window_labels: Labels for window
            trajectory_idx: Index of the current trajectory
            window_idx: Index of the current window within the trajectory
            is_first_window: Whether this is the first window
            last_ekf_predictions: Last EKF predictions from previous window (tensor format)
            last_ekf_covariances: Last EKF covariances from previous window (tensor format)
            
        Returns:
            WindowEvaluationResult containing all metrics and data
        """
        # Increment training counter
        self.online_training_count += 1
        logger.info(f"Online training step {self.online_training_count} called for trajectory {trajectory_idx}, window {window_idx}")
        
        # Set learning done after 7 training calls
        if self.online_training_count >= 10:
            self.learning_done = True
            logger.info(f"Online model training completed after {self.online_training_count} training windows")
        
        # Debug: Log input shapes
        logger.debug(f"_online_training_window input shapes: "
                     f"window_time_series={window_time_series.shape if hasattr(window_time_series, 'shape') else 'not tensor'}, "
                     f"window_sources_num={len(window_sources_num)}, "
                     f"window_labels={len(window_labels)}")
        
        # Unpack arguments directly, assuming they are for a single window
        time_series_steps = window_time_series  # Already [window_size, N, T]
        sources_num_per_step = window_sources_num  # List[int]
        labels_per_step_list = window_labels  # List[np.ndarray]
        
        # Validate inputs
        current_window_len, error_message = self._validate_inputs(time_series_steps, sources_num_per_step, labels_per_step_list)
        if error_message:
            logger.error(error_message)
            return WindowEvaluationResult.create_error_result(error_message)
        
        # Use provided model or default to online_model
        training_model = model if model is not None else self.online_model
        
        # Check if we're dealing with far-field or near-field
        is_near_field = hasattr(training_model, 'field_type') and training_model.field_type.lower() == "near"
        
        # Use RMSPE loss for evaluation
        rmspe_criterion = RMSPELoss().to(device)
        
        # Use RMAPE loss for online training
        rmape_criterion = RMAPELoss().to(device)
        
        # Set up optimizer for training
        if training_model is self.online_model:
            # Use online optimizer for online model
            if not hasattr(self, 'online_optimizer'):
                self.online_optimizer = optim.Adam(self.online_model.parameters(), lr=1e-3)
            optimizer = self.online_optimizer
        elif training_model is self.supervised_trained_model:
            # Use supervised optimizer for supervised model
            if not hasattr(self, 'supervised_optimizer'):
                self.supervised_optimizer = optim.Adam(self.supervised_trained_model.parameters(), lr=1e-3)
            optimizer = self.supervised_optimizer
        else:
            # Fallback: create a temporary optimizer
            optimizer = optim.Adam(training_model.parameters(), lr=1e-3)
        
        # Initialize Extended Kalman Filters
        max_sources = self.config.system_model.M
        ekf_filters = self._initialize_ekf_filters(max_sources, window_idx, 0)
        
        # Get current eta value from system model
        current_eta = self.system_model.params.eta
        logger.info(f"Online training: Initialized {max_sources} EKF instances for window (eta={current_eta:.4f})")
        
        # Training phase: Run gradient descent steps per window
        training_model.train()  # Set to training mode
        total_training_loss = 0.0
        num_training_steps = 0
        
        # Number of gradient descent steps per window
        num_gd_steps = 5
        
        # Get loss configuration for training (use override if provided)
        base_loss_config = getattr(self.config.online_learning, 'loss_config', None)
        if loss_config_override is not None:
            loss_config = loss_config_override
        else:
            loss_config = base_loss_config
        windows_last_ekf_covariances = last_ekf_covariances
        windows_last_ekf_predictions = last_ekf_predictions
        for gd_step in range(num_gd_steps):
            # Zero gradients at the start of each GD step
            optimizer.zero_grad()
            
            # Collect step results for window-level loss calculation
            # Each GD iteration processes the entire window and calculates window-level loss
            window_step_results = []  # Store step results for window-level loss calculation
            step_count = 0
            last_ekf_covariances = windows_last_ekf_covariances
            last_ekf_predictions = windows_last_ekf_predictions
            # Process each step for training in this gradient descent iteration
            for step in range(current_window_len):
                try:
                    # Extract data for this step
                    step_data_tensor = time_series_steps[step:step+1].to(device)  # Shape: [1, N, T]
                    num_sources_this_step = sources_num_per_step[step]
                    
                    # Skip if no sources
                    if num_sources_this_step <= 0:
                        continue
                    if step ==0:
                        last_ekf_predictions = windows_last_ekf_predictions
                        last_ekf_covariances = windows_last_ekf_covariances

                    # Get ground truth labels for this step
                    true_angles_this_step = labels_per_step_list[step][:num_sources_this_step]
                    
                    if not is_near_field:
                        # Forward pass through training model (with gradients for training)
                        angles_pred, _, _ = training_model(step_data_tensor, num_sources_this_step)
                        
                        # Convert predictions and true angles to proper format
                        # Ensure angles_pred_tensor has shape [1, num_sources] for loss functions
                        if angles_pred.dim() == 3:  # [batch, channels, sources] -> [batch, sources]
                            angles_pred_tensor = angles_pred.squeeze(1)[:, :num_sources_this_step]
                        elif angles_pred.dim() == 2:  # [batch, sources] or [batch, features]
                            angles_pred_tensor = angles_pred.view(1, -1)[:, :num_sources_this_step]
                        else:  # [sources] -> [1, sources]
                            angles_pred_tensor = angles_pred.view(1, -1)[:, :num_sources_this_step]
                        
                        # Ensure true_angles_tensor has shape [1, num_sources]
                        true_angles_tensor = torch.tensor(true_angles_this_step, device=device).unsqueeze(0)  # Shape: [1, num_sources]
                        
                        # Get optimal permutation for training
                        angles_pred_np = angles_pred.detach().cpu().numpy().flatten()[:num_sources_this_step]
                        model_perm = self._get_optimal_permutation(angles_pred_np, true_angles_this_step)
                        angles_pred_tensor = angles_pred_tensor[:, model_perm]
                        angles_pred_np = angles_pred_np[model_perm].flatten()
                        
                        # ============ EKF Processing for Training ============
                        # Initialize EKF for this training step (create fresh filters for each step)
                        # Calculate initial time for training EKF filters
                        window_size = self.config.online_learning.window_size
                        training_initial_time = window_idx * window_size + step
                        
                        # Initialize training EKF filters (separate from evaluation EKF)
                        self.training_ekf_filters = []
                        for i in range(num_sources_this_step):
                            training_ekf = ExtendedKalmanFilter1D.create_from_config(
                                self.config, 
                                trajectory_type=self.config.trajectory.trajectory_type,
                                device=device,
                                source_idx=i,  # Pass source index to use source-specific parameters
                                initial_time=training_initial_time  # Pass initial time for correct oscillatory behavior
                            )
                            self.training_ekf_filters.append(training_ekf)
                        
                        # Use the _initialize_ekf_state method to properly initialize state and covariance
                        # This ensures we use the last predictions and covariances from the previous window
                        self._initialize_ekf_state(
                            step=0, 
                            num_sources_this_step=num_sources_this_step, 
                            true_angles_this_step=true_angles_this_step,
                            ekf_filters=self.training_ekf_filters, 
                            is_first_window=False,  # Not first window since we're in online training
                            last_ekf_predictions=last_ekf_predictions, 
                            last_ekf_covariances=last_ekf_covariances
                        )
                        
                        # Apply EKF to each source prediction using tensor inputs to preserve gradients
                        ekf_angles_pred = []
                        ekf_covariances_pred = []
                        kalman_gain_times_innovation_list = []  # Collect K*y for training
                        y_s_inv_y_list = []  # Collect y*S^-1*y for training
                        step_Innovation_Covariance_list = []  # Collect Innovation Covariance for training
                        for i in range(num_sources_this_step):
                            if i < len(self.training_ekf_filters) and i < angles_pred_tensor.size(2):
                                ekf_filter = self.training_ekf_filters[i]
                                
                                # Use tensor measurement to preserve gradients (shape: [batch, seq, sources])
                                tensor_measurement = angles_pred_tensor[0, 0, i]
                                
                                # EKF predict and update with tensor measurement
                                _, updated_state, _, kalman_gain, kalman_gain_times_innovation, y_s_inv_y,Innovation_Covariance = ekf_filter.predict_and_update(
                                    measurement=tensor_measurement, 
                                    true_state=true_angles_this_step[i]
                                )
                                
                                # Verify tensors maintain gradients - fail hard if not
                                if not isinstance(updated_state, torch.Tensor):
                                    raise RuntimeError(f"EKF updated_state is not a tensor: {type(updated_state)}. EKF must return tensors for gradient computation.")
                                if not updated_state.requires_grad:
                                    raise RuntimeError(f"EKF updated_state tensor does not require gradients. This breaks the computation graph.")
                                
                                # Verify kalman_gain_times_innovation maintains gradients
                                if not isinstance(kalman_gain_times_innovation, torch.Tensor):
                                    raise RuntimeError(f"EKF kalman_gain_times_innovation is not a tensor: {type(kalman_gain_times_innovation)}. EKF must return tensors for gradient computation.")
                                if not kalman_gain_times_innovation.requires_grad:
                                    raise RuntimeError(f"EKF kalman_gain_times_innovation tensor does not require gradients. This breaks the computation graph.")
                                
                                # Verify y_s_inv_y maintains gradients
                                if not isinstance(y_s_inv_y, torch.Tensor):
                                    raise RuntimeError(f"EKF y_s_inv_y is not a tensor: {type(y_s_inv_y)}. EKF must return tensors for gradient computation.")
                                if not y_s_inv_y.requires_grad:
                                    raise RuntimeError(f"EKF y_s_inv_y tensor does not require gradients. This breaks the computation graph.")
                                
                                # Ensure updated_state is a scalar tensor for consistent stacking
                                if updated_state.dim() > 0:
                                    updated_state_scalar = updated_state.flatten()[0]  # Take first element if multi-dimensional
                                else:
                                    updated_state_scalar = updated_state
                                ekf_angles_pred.append(updated_state_scalar)
                                ekf_covariances_pred.append(ekf_filter.P)
                                kalman_gain_times_innovation_list.append(kalman_gain_times_innovation)
                                y_s_inv_y_list.append(y_s_inv_y)
                                step_Innovation_Covariance_list.append(Innovation_Covariance)
                            else:
                                # No fallback - fail hard if EKF not available or index out of bounds
                                if i >= len(self.training_ekf_filters):
                                    raise RuntimeError(f"EKF filter index {i} out of bounds. Expected {len(self.training_ekf_filters)} training EKF filters but got {num_sources_this_step} sources.")
                                if i >= angles_pred_tensor.size(2):
                                    raise RuntimeError(f"Source index {i} out of bounds for tensor shape {angles_pred_tensor.shape}. Cannot access source {i} from {angles_pred_tensor.size(2)} sources.")
                                raise RuntimeError(f"EKF filter {i} not available but should be. This indicates a serious configuration error.")
                        
                        # Create EKF predictions tensor
                        # Ensure ekf_angles_pred_tensor has shape [1, num_sources] for loss functions
                        ekf_angles_pred_tensor = torch.stack(ekf_angles_pred).unsqueeze(0)  # Shape: [1, num_sources]
                        logger.debug(f"EKF predictions tensor shape: {ekf_angles_pred_tensor.shape} (expected [1, {num_sources_this_step}])")
                        ekf_covariances_tensor = torch.stack(ekf_covariances_pred).view(1, -1)  # Shape: [1, num_sources]
                        step_Innovation_Covariance_tensor = torch.stack(step_Innovation_Covariance_list).view(1, -1)  # Shape: [1, num_sources]
                        last_ekf_predictions = ekf_angles_pred_tensor
                        last_ekf_covariances = ekf_covariances_tensor
                        # Store step results for window-level loss calculation
                        step_result = {
                            'success': True,
                            'pre_ekf_angles_pred_tensor': angles_pred_tensor,  # Tensor for window-level loss calculation
                            'ekf_angles_pred_tensor': ekf_angles_pred_tensor,  # Tensor for window-level loss calculation
                            'true_angles_tensor': true_angles_tensor,  # Tensor for window-level loss calculation
                            'num_sources': num_sources_this_step,
                            'Innovation_Covariance_tensor': step_Innovation_Covariance_tensor  # Tensor for window-level loss calculation
                        }
                        
                        # Store step result for window-level loss calculation
                        window_step_results.append(step_result)
                        step_count += 1
                        
                        logger.debug(f"Training step {step}, GD {gd_step}: Collected step result for window-level loss calculation")
                    
                    else:
                        # Near-field case - not supported
                        error_msg = "Near-field processing is not supported in this project. Please use far-field models only."
                        logger.error(error_msg)
                        raise NotImplementedError(error_msg)
                    
                except Exception as e:
                    logger.warning(f"Error during online training step {step} in GD iteration {gd_step}: {e}")
                    continue
            
            # Calculate window-level loss and perform backpropagation
            if step_count > 0 and len(window_step_results) > 0:
                # Unified window-level loss calculation
                window_loss = self._calculate_window_training_loss(window_step_results, loss_config, rmspe_criterion, rmape_criterion)
                logger.info(f"GD step {gd_step + 1}: Window loss = {window_loss.item():.6f} over {step_count} steps")
                
                # Backward pass on the window loss
                window_loss.backward()
                
                # Check if gradients were computed properly
                gradients_ok = self._check_gradients(training_model, step_count, gd_step)
                
                # Update model parameters
                optimizer.step()
                # Zero gradients after optimizer step to prepare for next GD iteration
                optimizer.zero_grad()
                
                total_training_loss += window_loss.item()
                num_training_steps += 1  # Count each GD step, not each individual step
                logger.info(f"Online training GD step {gd_step + 1}/{num_gd_steps}: Updated model with window loss = {window_loss.item():.6f} over {step_count} steps")
            else:
                logger.warning(f"No valid training steps in GD iteration {gd_step} for window {window_idx}")
        
        # Calculate overall average training loss across all GD iterations
        # Note: num_training_steps now counts GD iterations, not individual steps
        if num_training_steps > 0:
            avg_training_loss = total_training_loss / num_training_steps
            logger.info(f"Online training step {self.online_training_count}: Completed {num_gd_steps} GD iterations with overall avg loss = {avg_training_loss:.6f} over {num_training_steps} GD steps")
        else:
            avg_training_loss = float('inf')  # Set default value if no training steps
            logger.warning(f"No valid training steps in window {window_idx}")
        
        # Set model back to eval mode for EKF evaluation
        training_model.eval()
        
        # Clean up training EKF filters to free memory
        if hasattr(self, 'training_ekf_filters'):
            delattr(self, 'training_ekf_filters')
            logger.debug(f"Cleaned up training EKF filters after window {window_idx}")
        
        # Now evaluate the trained model with EKF using the same pattern as _evaluate_window
        # Process each step in window
        step_results_list = []
        last_ekf_predictions = windows_last_ekf_predictions
        last_ekf_covariances = windows_last_ekf_covariances
        for step in range(current_window_len):
            # Initialize EKF state if this is the first step
            num_sources_this_step = sources_num_per_step[step]
            true_angles_this_step = labels_per_step_list[step][:num_sources_this_step]
            
            self._initialize_ekf_state(step, num_sources_this_step, true_angles_this_step, 
                                     ekf_filters, is_first_window, last_ekf_predictions, last_ekf_covariances)
            
            # Process single step
            success, step_result = self._process_single_step(
                step, time_series_steps, sources_num_per_step, labels_per_step_list,
                ekf_filters, training_model, is_near_field, False
            )
            
            step_results_list.append(step_result)
        
        # Use loss config override if provided, otherwise use default
        effective_loss_config = loss_config_override if loss_config_override is not None else loss_config
        
        # Calculate aggregated metrics using the same helper method
        result = self._calculate_metrics(step_results_list, current_window_len, max_sources, current_eta, is_near_field, effective_loss_config)
        
        # Log window summary
        if result.is_valid:
            logger.info(f"Online training window {window_idx}: "
                       f"Pre-EKF Loss = {result.loss_metrics.pre_ekf_loss:.6f}, "
                       f"Main Loss = {result.loss_metrics.main_loss:.6f} ({result.loss_metrics.main_loss_config}), "
                       f"Avg Cov = {result.window_metrics.avg_covariance:.6f}, "
                       f"Training Loss = {avg_training_loss:.6f}")
        
        return result

    def _check_gradients(self, model, step: int, gd_step: int) -> bool:
        """
        Check if gradients were properly computed after backward pass.
        
        Args:
            model: The model to check gradients for
            step: Current training step
            gd_step: Current gradient descent step
            
        Returns:
            bool: True if gradients exist and are finite, False otherwise
        """
        has_gradients = False
        total_norm = 0
        num_params_with_grad = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_gradients = True
                num_params_with_grad += 1
                
                # Check for NaN or inf gradients
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    logger.error(f"NaN/Inf gradients detected in {name} at step {step}, GD {gd_step}")
                    return False
                
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        if has_gradients:
            total_norm = total_norm ** (1. / 2)
            logger.debug(f"Step {step}, GD {gd_step}: {num_params_with_grad} params have gradients, norm: {total_norm:.6f}")
            return True
        else:
            logger.warning(f"No gradients computed in step {step}, GD {gd_step}")
            return False

    def _validate_inputs(self, window_time_series: torch.Tensor, window_sources_num: List[int], 
                        window_labels: List[np.ndarray]) -> Tuple[int, str]:
        """
        Validate input data for window evaluation.
        
        Args:
            window_time_series: Time series data for window [window_size, N, T]
            window_sources_num: Source counts for window [window_size]
            window_labels: Labels for window [window_size]
            
        Returns:
            Tuple of (valid_window_size, error_message). If valid, error_message is empty string.
        """
        current_window_len = window_time_series.shape[0]
        
        # Sanity check the input lengths
        if len(window_sources_num) < current_window_len:
            logger.warning(f"Window source count list length ({len(window_sources_num)}) is less than time series length ({current_window_len}). Truncating window.")
            current_window_len = len(window_sources_num)
        
        if len(window_labels) < current_window_len:
            logger.warning(f"Window labels list length ({len(window_labels)}) is less than time series length ({current_window_len}). Truncating window.")
            current_window_len = len(window_labels)
        
        if current_window_len == 0:
            return 0, "Window has zero valid steps. Cannot evaluate."
        
        return current_window_len, ""

    def _initialize_ekf_filters(self, max_sources: int, window_idx: int = 0, step_idx: int = 0) -> List[ExtendedKalmanFilter1D]:
        """
        Initialize Extended Kalman Filters for window evaluation.
        
        Args:
            max_sources: Maximum number of sources to track
            window_idx: Index of the current window (used to calculate initial time)
            step_idx: Index of the current step within the window (used to calculate initial time)
            
        Returns:
            List of initialized EKF filter instances, each with source-specific parameters
        """
        # Calculate the initial time based on window and step indices
        # This ensures the EKF filters start with the correct time for oscillatory models
        window_size = self.config.online_learning.window_size
        initial_time = window_idx * window_size + step_idx
        
        ekf_filters = []
        
        # Create EKF filters for each source with source-specific parameters
        for i in range(max_sources):
            # Create EKF filter with source index i - the filter will use source-specific parameters
            ekf_filter = ExtendedKalmanFilter1D.create_from_config(
                self.config, 
                trajectory_type=self.config.trajectory.trajectory_type,
                device=device,
                source_idx=i,  # Pass source index to the filter
                initial_time=initial_time  # Pass initial time for correct oscillatory behavior
            )
            ekf_filters.append(ekf_filter)
        
        current_eta = self.system_model.params.eta
        logger.info(f"Initialized {max_sources} EKF instances for window {window_idx}, step {step_idx} (initial_time={initial_time}, eta={current_eta:.4f})")
        
        return ekf_filters

    def _initialize_ekf_state(self, step: int, num_sources_this_step: int, true_angles_this_step: np.ndarray,
                            ekf_filters: List[ExtendedKalmanFilter1D], is_first_window: bool,
                            last_ekf_predictions: Optional[List], last_ekf_covariances: Optional[List]) -> None:
        """
        Initialize EKF state for the current step.
        
        Args:
            step: Current step index
            num_sources_this_step: Number of sources in current step
            true_angles_this_step: Ground truth angles for current step
            ekf_filters: List of EKF filter instances
            is_first_window: Whether this is the first window
            last_ekf_predictions: Last EKF predictions from previous window
            last_ekf_covariances: Last EKF covariances from previous window
        """
        if step == 0:
            if is_first_window:
                # Initialize with true angles for first window
                for i in range(num_sources_this_step):
                    ekf_filters[i].initialize_state(true_angles_this_step[i])
            else:
                # Initialize with last predictions from previous window
                if (last_ekf_predictions is not None and last_ekf_covariances is not None and 

 
                     last_ekf_predictions.shape[0] > 0 and last_ekf_covariances.shape[0] > 0 and
                    last_ekf_predictions.shape[1] >= num_sources_this_step and 
                    last_ekf_covariances.shape[1] >= num_sources_this_step):
                    
                    # Get the last predictions (last row of the tensor) and calculate their optimal permutation
                    last_predictions_pre_perm = last_ekf_predictions[-1, :num_sources_this_step]
                    true_angles_tensor = torch.tensor(true_angles_this_step, device=last_predictions_pre_perm.device)
                    last_perm = self._get_optimal_permutation_tensor(last_predictions_pre_perm, true_angles_tensor)
                    last_predictions = last_predictions_pre_perm[last_perm]
                    
                    # Get the last covariances (last row of the tensor) and apply the same permutation
                    last_covariances_pre_perm = last_ekf_covariances[-1, :num_sources_this_step]
                    last_covariances = last_covariances_pre_perm[last_perm]
                    
                    for i in range(num_sources_this_step):
                        ekf_filters[i].initialize_state(last_predictions.flatten()[i].item())
                        ekf_filters[i].P = last_covariances.flatten()[i].item()
                else:
                    # Fallback to true angles if no valid last predictions
                    logger.warning("No valid last predictions or covariances available, falling back to true angles")
                    for i in range(num_sources_this_step):
                        ekf_filters[i].initialize_state(true_angles_this_step.flatten()[i].flatten())

    def _process_single_step(self, step: int, time_series_steps: torch.Tensor, sources_num_per_step: List[int],
                           labels_per_step_list: List[np.ndarray], ekf_filters: List[ExtendedKalmanFilter1D],
                           model, is_near_field: bool, Pretrained_model: bool) -> Tuple[bool, Dict]:
        """
        Process a single step in the window evaluation - data collection only.
        
        Args:
            step: Current step index
            time_series_steps: Time series data for all steps
            sources_num_per_step: Source counts for all steps
            labels_per_step_list: Labels for all steps
            ekf_filters: List of EKF filter instances
            model: Model to use for predictions
            is_near_field: Whether processing near-field or far-field
            Pretrained_model: Whether the model is the Pretrained_model or online trained model
            
        Returns:
            Tuple of (success, step_results_dict) containing only raw data
        """
        try:
            # Extract data for this step
            step_data_tensor = time_series_steps[step:step+1].to(device)  # Shape: [1, N, T]
            num_sources_this_step = sources_num_per_step[step]
            true_angles_this_step = labels_per_step_list[step][:num_sources_this_step]
            
            # Forward pass through model
            model.eval()
            with torch.no_grad():
                if not is_near_field:
                    # Model expects num_sources as int or 0-dim tensor
                    angles_pred, _, _ = model(step_data_tensor, num_sources_this_step)
                    
                    # Compare model weights properly
                    model_state = model.state_dict()
                    trained_state = self.trained_model.state_dict()
                    weights_equal = all(torch.equal(model_state[key], trained_state[key]) for key in model_state.keys())
                    true_angles_tensor = torch.tensor(true_angles_this_step, device=device).unsqueeze(0)

                    if weights_equal and not Pretrained_model:
                        logger.error("Model and trained model have the same weights - online model was not properly copied!")
                        raise RuntimeError("Online model and trained model have identical weights. This indicates the online model was not properly initialized as an independent copy. Cannot proceed with online learning.")
                    elif Pretrained_model:
                        logger.info("The evaluated model is not the online model,online model is not initialized yet or this is evaluation comparison")
                    else:
                        logger.info("Model and trained model dont have the same weights - online model is properly initialized")
                        # Debug: Compare online model with pretrained model (no loss calculation)
                        pretrained_model_angle_pred, _, _ = self.trained_model(step_data_tensor,num_sources_this_step)
                        pretrained_model_angle_pred= pretrained_model_angle_pred.view(1, -1)[:, :num_sources_this_step]
                        model_perm = self._get_optimal_permutation(pretrained_model_angle_pred.cpu().numpy().flatten(), true_angles_this_step)
                        pretrained_model_angle_pred = pretrained_model_angle_pred[:, torch.tensor(model_perm, device=device)]
                        model_perm = self._get_optimal_permutation(angles_pred.cpu().numpy().flatten(), true_angles_this_step)
                        angles_pred = angles_pred[:, torch.tensor(model_perm, device=device)]
                    # Prepare pre-EKF predictions tensor
                    # Ensure pre_ekf_angles_pred has shape [1, num_sources] for loss functions
                    if angles_pred.dim() == 3:  # [batch, channels, sources] -> [batch, sources]
                        pre_ekf_angles_pred = angles_pred.squeeze(1)[:, :num_sources_this_step]
                    elif angles_pred.dim() == 2:  # [batch, sources] or [batch, features]
                        pre_ekf_angles_pred = angles_pred.view(1, -1)[:, :num_sources_this_step]
                    else:  # [sources] -> [1, sources]
                        pre_ekf_angles_pred = angles_pred.view(1, -1)[:, :num_sources_this_step]
                    
                    # Get optimal permutation for model predictions (need numpy for permutation)
                    angles_pred_np = angles_pred.cpu().numpy().flatten()[:num_sources_this_step]
                    model_perm = self._get_optimal_permutation(angles_pred_np, true_angles_this_step)
                
                    # Apply permutation to both numpy and tensor versions
                    angles_pred_np = angles_pred_np[model_perm]
                    pre_ekf_angles_pred = pre_ekf_angles_pred[:, model_perm]
                
                    # EKF update for each source - use tensor directly
                    step_predictions = []
                    step_covariances = []
                    step_innovations = []
                    step_kalman_gains = []
                    step_kalman_gain_times_innovation = []
                    step_y_s_inv_y = []
                    step_Innovation_Covariance = []
                    
                    for i in range(num_sources_this_step):
                        # Predict and update in one step - pass tensor directly

                        predicted_angle, updated_angle, innovation, kalman_gain, kalman_gain_times_innovation, y_s_inv_y,Innovation_Covariance = ekf_filters[i].predict_and_update(
                            measurement= pre_ekf_angles_pred.flatten()[i],  # Flatten to get proper indexing
                            true_state= true_angles_this_step[i]
                        )      
                        # Store prediction, covariance and innovation
                        step_predictions.append(updated_angle)
                        step_covariances.append(ekf_filters[i].P)
                        step_innovations.append(innovation)
                        step_kalman_gains.append(kalman_gain)
                        step_kalman_gain_times_innovation.append(kalman_gain_times_innovation)
                        step_y_s_inv_y.append(y_s_inv_y)
                        step_Innovation_Covariance.append(Innovation_Covariance)
                    # Create tensor from EKF predictions
                    # Ensure ekf_angles_pred has shape [1, num_sources] for loss functions
                    ekf_angles_pred = torch.tensor(step_predictions, device=device).unsqueeze(0)  # Shape: [1, num_sources]
                    
                    step_results = {
                        'success': True,
                        'step_predictions': step_predictions,  # List of tensors
                        'step_covariances': step_covariances,  # List of tensors
                        'step_innovations': step_innovations,  # List of tensors
                        'step_kalman_gains': step_kalman_gains,  # List of tensors
                        'step_kalman_gain_times_innovation': step_kalman_gain_times_innovation,  # List of tensors
                        'step_y_s_inv_y': step_y_s_inv_y,  # List of tensors
                        'pre_ekf_angles_pred_tensor': pre_ekf_angles_pred,  # Tensor for window-level loss calculation
                        'ekf_angles_pred_tensor': ekf_angles_pred,  # Tensor for window-level loss calculation
                        'true_angles_tensor': true_angles_tensor,  # Tensor for window-level loss calculation
                        'num_sources': num_sources_this_step,
                        'step_Innovation_Covariance': step_Innovation_Covariance  # List of tensors
                    }
                    
                    return True, step_results
                else:
                    # Near-field case - not supported
                    error_msg = "Near-field processing is not supported in this project. Please use far-field models only."
                    logger.error(error_msg)
                    raise NotImplementedError(error_msg)
            
        except Exception as e:
            logger.warning(f"Error processing step {step}: {e}")
            return False, {'success': False, 'error': str(e)}

    def _calculate_metrics(self, step_results_list: List[Dict], current_window_len: int, 
                          max_sources: int, current_eta: float, is_near_field: bool, 
                          loss_config=None) -> WindowEvaluationResult:
        """
        Calculate aggregated metrics from step results.
        
        Args:
            step_results_list: List of step result dictionaries
            current_window_len: Length of the window
            max_sources: Maximum number of sources
            current_eta: Current eta value
            is_near_field: Whether processing near-field or far-field
            loss_config: Loss configuration object (optional)
            
        Returns:
            WindowEvaluationResult containing all calculated metrics
        """
        # Initialize tensors for storing results
        ekf_predictions = torch.empty((current_window_len, max_sources), dtype=torch.float64)
        ekf_covariances = torch.empty((current_window_len, max_sources), dtype=torch.float64)
        ekf_innovations = torch.empty((current_window_len, max_sources), dtype=torch.float64)
        ekf_kalman_gains = torch.empty((current_window_len, max_sources), dtype=torch.float64)
        ekf_kalman_gain_times_innovation = torch.empty((current_window_len, max_sources), dtype=torch.float64)
        ekf_y_s_inv_y = torch.empty((current_window_len, max_sources), dtype=torch.float64)
        pre_ekf_angles_pred_list = torch.empty((current_window_len, max_sources), dtype=torch.float64)
        
        # Collect all tensors for window-level loss calculation
        all_pre_ekf_preds = []
        all_ekf_preds = []
        all_true_angles = []
        all_innovation_covariances = []
        
        # Initialize accumulation variables
        num_valid_steps = 0
        total_covariance = 0.0
        total_cov_points = 0
        
        # Process each step result
        for step, step_result in enumerate(step_results_list):
            if not step_result['success']:
                continue
                
            num_sources = step_result['num_sources']
            
            # Store EKF metrics (all are tensors now)
            for i in range(num_sources):
                ekf_predictions[step, i] = step_result['step_predictions'][i].item()
                ekf_covariances[step, i] = step_result['step_covariances'][i].item()
                ekf_innovations[step, i] = step_result['step_innovations'][i].item()
                ekf_kalman_gains[step, i] = step_result['step_kalman_gains'][i].item()
                ekf_kalman_gain_times_innovation[step, i] = step_result['step_kalman_gain_times_innovation'][i].item()
                ekf_y_s_inv_y[step, i] = step_result['step_y_s_inv_y'][i].item()
            
            # Store pre-EKF predictions (tensor)
            pre_ekf_preds = step_result['pre_ekf_angles_pred_tensor'].flatten()
            for i in range(min(num_sources, len(pre_ekf_preds))):
                pre_ekf_angles_pred_list[step, i] = pre_ekf_preds[i].item()
            
            # Collect tensors for window-level loss calculation
            all_pre_ekf_preds.append(step_result['pre_ekf_angles_pred_tensor'])
            all_ekf_preds.append(step_result['ekf_angles_pred_tensor'])
            all_true_angles.append(step_result['true_angles_tensor'])
            all_innovation_covariances.append(step_result['step_Innovation_Covariance'])
            
            # Accumulate covariance (convert tensors to scalars for sum)
            total_covariance += sum(tensor.item() for tensor in step_result['step_covariances'])
            total_cov_points += num_sources
            
            num_valid_steps += 1
        
        # Calculate window-level losses using unified method
        if num_valid_steps > 0 and all_pre_ekf_preds:
            # Stack predictions across time steps
            window_pre_ekf_preds = torch.cat(all_pre_ekf_preds, dim=0)  # [window_size, num_sources]
            window_ekf_preds = torch.cat(all_ekf_preds, dim=0)          # [window_size, num_sources]
            window_true_angles = torch.cat(all_true_angles, dim=0)      # [window_size, num_sources]
            
            # Calculate ALL losses at window level
            loss_metrics = self._calculate_all_losses(
                window_pre_ekf_preds, window_ekf_preds, window_true_angles, all_innovation_covariances, loss_config
            )
            
            avg_covariance = total_covariance / total_cov_points if total_cov_points > 0 else float('nan')
        else:
            # Create default loss metrics if no valid steps
            loss_metrics = LossMetrics(
                main_loss=float('inf'),
                main_loss_db=float('inf'),
                main_loss_config="no_valid_steps",
                online_training_reference_loss=float('inf'),
                online_training_reference_loss_config="no_valid_steps",
                pre_ekf_loss=float('inf'),
                ekf_gain_rmspe=0.0,
                ekf_gain_rmape=0.0
            )
            avg_covariance = float('nan')
        
        # Calculate averaged metrics across time steps
        avg_ekf_angle_pred = []
        avg_pre_ekf_angle_pred = []
        avg_ekf_covariances = []
        avg_ekf_innovations = []
        avg_ekf_kalman_gains = []
        avg_ekf_kalman_gain_times_innovation = []
        avg_ekf_y_s_inv_y = []
        avg_step_innovation_covariances = []
        
        if num_valid_steps > 0:
            # Get the number of sources from the first valid step
            first_valid_step = next((result for result in step_results_list if result['success']), None)
            if first_valid_step:
                num_sources = first_valid_step['num_sources']
                
                # Average EKF predictions across time steps for each source
                for source_idx in range(num_sources):
                    source_predictions = []
                    for step_result in step_results_list:
                        if step_result['success'] and len(step_result['step_predictions']) > source_idx:
                            source_predictions.append(step_result['step_predictions'][source_idx].item())
                    
                    if source_predictions:
                        avg_ekf_angle_pred.append(float(sum(source_predictions) / len(source_predictions)))
                
                # Average pre-EKF predictions across time steps for each source
                for source_idx in range(num_sources):
                    source_predictions = []
                    for step_result in step_results_list:
                        if step_result['success']:
                            pre_ekf_preds = step_result['pre_ekf_angles_pred_tensor'].flatten()
                            if len(pre_ekf_preds) > source_idx:
                                source_predictions.append(pre_ekf_preds[source_idx].item())
                    
                    if source_predictions:
                        avg_pre_ekf_angle_pred.append(float(sum(source_predictions) / len(source_predictions)))
        
                # Average EKF covariances across time steps for each source
                for source_idx in range(num_sources):
                    source_covariances = []
                    for step_result in step_results_list:
                        if step_result['success'] and len(step_result['step_covariances']) > source_idx:
                            source_covariances.append(step_result['step_covariances'][source_idx].item())
                    
                    if source_covariances:
                        avg_ekf_covariances.append(float(sum(source_covariances) / len(source_covariances)))
                
                # Average EKF innovations across time steps for each source
                for source_idx in range(num_sources):
                    source_innovations = []
                    for step_result in step_results_list:
                        if step_result['success'] and len(step_result['step_innovations']) > source_idx:
                            source_innovations.append(step_result['step_innovations'][source_idx].item())
                    
                    if source_innovations:
                        avg_ekf_innovations.append(float(sum(source_innovations) / len(source_innovations)))
                
                # Average EKF Kalman gains across time steps for each source
                for source_idx in range(num_sources):
                    source_kalman_gains = []
                    for step_result in step_results_list:
                        if step_result['success'] and len(step_result['step_kalman_gains']) > source_idx:
                            source_kalman_gains.append(step_result['step_kalman_gains'][source_idx].item())
                    
                    if source_kalman_gains:
                        avg_ekf_kalman_gains.append(float(sum(source_kalman_gains) / len(source_kalman_gains)))
                
                # Average EKF Kalman gain times innovation across time steps for each source
                for source_idx in range(num_sources):
                    source_kalman_gain_times_innovation = []
                    for step_result in step_results_list:
                        if step_result['success'] and len(step_result['step_kalman_gain_times_innovation']) > source_idx:
                            source_kalman_gain_times_innovation.append(step_result['step_kalman_gain_times_innovation'][source_idx].item())
                    
                    if source_kalman_gain_times_innovation:
                        avg_ekf_kalman_gain_times_innovation.append(float(sum(source_kalman_gain_times_innovation) / len(source_kalman_gain_times_innovation)))
                
                # Average EKF y_s_inv_y across time steps for each source
                for source_idx in range(num_sources):
                    source_y_s_inv_y = []
                    for step_result in step_results_list:
                        if step_result['success'] and len(step_result['step_y_s_inv_y']) > source_idx:
                            source_y_s_inv_y.append(step_result['step_y_s_inv_y'][source_idx].item())
                    
                    if source_y_s_inv_y:
                        avg_ekf_y_s_inv_y.append(float(sum(source_y_s_inv_y) / len(source_y_s_inv_y)))
                
                # Average step innovation covariances across time steps for each source
                for source_idx in range(num_sources):
                    source_innovation_covariances = []
                    for step_result in step_results_list:
                        if step_result['success'] and 'step_Innovation_Covariance' in step_result and len(step_result['step_Innovation_Covariance']) > source_idx:
                            source_innovation_covariances.append(step_result['step_Innovation_Covariance'][source_idx].item())
                    
                    if source_innovation_covariances:
                        avg_step_innovation_covariances.append(float(sum(source_innovation_covariances) / len(source_innovation_covariances)))
        
        # Create window metrics with averaged values
        window_metrics = WindowMetrics(
            window_size=current_window_len,
            num_sources=num_sources if num_valid_steps > 0 else 0,
            avg_covariance=avg_covariance,
            eta_value=current_eta,
            is_near_field=is_near_field,
            avg_ekf_angle_pred=avg_ekf_angle_pred,
            avg_pre_ekf_angle_pred=avg_pre_ekf_angle_pred,
            avg_ekf_covariances=avg_ekf_covariances,
            avg_ekf_innovations=avg_ekf_innovations,
            avg_ekf_kalman_gains=avg_ekf_kalman_gains,
            avg_ekf_kalman_gain_times_innovation=avg_ekf_kalman_gain_times_innovation,
            avg_ekf_y_s_inv_y=avg_ekf_y_s_inv_y,
            avg_step_innovation_covariances=avg_step_innovation_covariances
        )
        
        # Create step metrics (renamed from EKF metrics)
        step_metrics = StepMetrics(
            covariances=ekf_covariances,
            innovations=ekf_innovations,
            kalman_gains=ekf_kalman_gains,
            kalman_gain_times_innovation=ekf_kalman_gain_times_innovation,
            y_s_inv_y=ekf_y_s_inv_y
        )
        
        # Create DOA metrics with predictions and true angles
        doa_metrics = DOAMetrics(
            ekf_predictions=ekf_predictions,
            pre_ekf_predictions=pre_ekf_angles_pred_list,
            true_angles=torch.cat(all_true_angles, dim=0) if all_true_angles else torch.empty((0, 0), dtype=torch.float64),
            avg_ekf_angle_pred=avg_ekf_angle_pred,
            avg_pre_ekf_angle_pred=avg_pre_ekf_angle_pred
        )
        
        return WindowEvaluationResult(
            loss_metrics=loss_metrics,
            window_metrics=window_metrics,
            step_metrics=step_metrics,
            doa_metrics=doa_metrics,
            is_valid=num_valid_steps > 0
        )

    def _evaluate_window(self, window_time_series, window_sources_num, window_labels, trajectory_idx: int = 0, window_idx: int = 0,
                         is_first_window: bool = True, last_ekf_predictions: List = None, last_ekf_covariances: List = None, model=None) -> WindowEvaluationResult:
        """
        Calculate loss on a window of trajectory data using Extended Kalman Filter.
        
        Args:
            window_time_series: Time series data for window [batch, window_size, N, T]
            window_sources_num: Source counts for window [batch, window_size]
            window_labels: Labels for window (list of tensors)
            trajectory_idx: Index of the current trajectory
            window_idx: Index of the current window within the trajectory
            is_first_window: Flag indicating if this is the first window evaluation
            last_ekf_predictions: List of last EKF predictions from previous window
            last_ekf_covariances: List of last EKF covariances from previous window
            model: Model to use for predictions (defaults to trained_model)
            
        Returns:
            WindowEvaluationResult containing all metrics and data
        """
        # Debug: Log input shapes
        logger.debug(f"_evaluate_window input shapes: "
                     f"window_time_series={window_time_series.shape if hasattr(window_time_series, 'shape') else 'not tensor'}, "
                     f"window_sources_num={len(window_sources_num)}, "
                     f"window_labels={len(window_labels)}")
        
        # Unpack arguments directly, assuming they are for a single window
        time_series_steps = window_time_series  # Already [window_size, N, T]
        sources_num_per_step = window_sources_num  # List[int]
        labels_per_step_list = window_labels  # List[np.ndarray]
        
        # Validate inputs
        current_window_len, error_message = self._validate_inputs(time_series_steps, sources_num_per_step, labels_per_step_list)
        if error_message:
            logger.error(error_message)
            return WindowEvaluationResult.create_error_result(error_message)
        
        # Use provided model or default to trained model
        if model is None:
            model = self.trained_model
            Pretrained_model = True
        else:
            Pretrained_model = False
        
        # Check if we're dealing with far-field or near-field
        is_near_field = hasattr(model, 'field_type') and model.field_type.lower() == "near"
        
        # Initialize Extended Kalman Filters
        max_sources = self.config.system_model.M
        ekf_filters = self._initialize_ekf_filters(max_sources, window_idx, 0)
        
        # Get current eta value from system model
        current_eta = self.system_model.params.eta
        
        # Get loss configuration for online learning
        loss_config = getattr(self.config.online_learning, 'loss_config', None)
        
        # Process each step in window
        step_results_list = []
        for step in range(current_window_len):
            # Initialize EKF state if this is the first step
            num_sources_this_step = sources_num_per_step[step]
            true_angles_this_step = labels_per_step_list[step][:num_sources_this_step]
            self._initialize_ekf_state(step, num_sources_this_step, true_angles_this_step, 
                                     ekf_filters, is_first_window, last_ekf_predictions, last_ekf_covariances)
            # Process single step
            success, step_result = self._process_single_step(
                step, time_series_steps, sources_num_per_step, labels_per_step_list,
                ekf_filters, model, is_near_field, Pretrained_model
            )
            
            step_results_list.append(step_result)
        
        # Calculate aggregated metrics using unified method
        result = self._calculate_metrics(step_results_list, current_window_len, max_sources, current_eta, is_near_field, loss_config)
        
        # Log window summary
        if result.is_valid:
            log_window_summary(result.loss_metrics, result.window_metrics.avg_covariance, 
                             current_eta, is_near_field, trajectory_idx, window_idx)
        
        return result

    def _fix_tensor_shape_for_loss(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Fix tensor shape for loss calculations.
        Expected: [window_size, num_sources]
        Actual: [window_size, 1, num_sources] -> squeeze middle dimension
        """
        if tensor.dim() == 3 and tensor.shape[1] == 1:
            return tensor.squeeze(1)  # Remove middle dimension
        return tensor

    def _calculate_all_losses(self, pre_ekf_preds: torch.Tensor, ekf_preds: torch.Tensor, 
                            true_angles: torch.Tensor, innovation_covariances: list = None, loss_config=None) -> 'LossMetrics':
        """
        Calculate all losses at window level.
        
        Args:
            pre_ekf_preds: Pre-EKF predictions tensor [window_size, num_sources]
            ekf_preds: EKF predictions tensor [window_size, num_sources]
            true_angles: True angles tensor [window_size, num_sources]
            innovation_covariances: List of innovation covariance tensors for each step (optional)
            loss_config: Loss configuration object (optional)
        
        Returns:
            LossMetrics object with all calculated losses
        """
        # Import loss criteria
        from DCD_MUSIC.src.metrics.rmspe_loss import RMSPELoss
        from DCD_MUSIC.src.metrics.rmape_loss import RMAPELoss
        
        rmspe_criterion = RMSPELoss().to(device)
        rmape_criterion = RMAPELoss().to(device)
        
        # Fix tensor shapes - remove extra dimension if present
        pre_ekf_preds = self._fix_tensor_shape_for_loss(pre_ekf_preds)
        ekf_preds = self._fix_tensor_shape_for_loss(ekf_preds)
        true_angles = self._fix_tensor_shape_for_loss(true_angles)
        
        # Get window size for proper averaging
        window_size = pre_ekf_preds.shape[0]
        
        # Main loss (what the system is optimized for - uses supervision + metric)
        if loss_config is None:
            raise RuntimeError("loss_config is required for _calculate_all_losses but was None")
        
        # Determine targets based on supervision mode
        if loss_config.supervision == "supervised":
            targets = true_angles
        elif loss_config.supervision == "unsupervised":  # unsupervised
            targets = pre_ekf_preds
        else:
            raise RuntimeError(f"Unknown supervision mode: {loss_config.supervision}. Must be one of: supervised, unsupervised")
        
        # Calculate main loss using supervision + metric (NOT training_loss_type)
        main_loss_config = f"{loss_config.supervision}_{loss_config.metric}"
        if loss_config.metric == "rmspe":
            main_loss = rmspe_criterion(ekf_preds, targets) / window_size  # RMSPE sums across batch, divide by window size
        elif loss_config.metric == "rmape": # rmape
            main_loss = rmape_criterion(ekf_preds, targets) / window_size  # RMAPE sums across batch, divide by window size
        else:
            raise RuntimeError(f"Unknown metric: {loss_config.metric}. Must be one of: rmspe, rmape")
        
        # Calculate main loss in dB units (20 * log10(main_loss))
        import math
        main_loss_value = main_loss.item() if hasattr(main_loss, 'item') else main_loss
        main_loss_db = 20 * math.log10(main_loss_value)  # Avoid log(0) with small epsilon
        
        # Online training reference loss (uses training_loss_type configuration)
        if not hasattr(loss_config, 'training_loss_type'):
            raise RuntimeError("loss_config.training_loss_type is required for online_training_reference_loss but was not found")
        
        training_loss_type = loss_config.training_loss_type
        online_training_reference_loss_config = training_loss_type
        
        if training_loss_type == "multimoment":
            # Multi-Moment loss: use pre-EKF as predictions, EKF as targets
            try:
                multimoment_criterion = MultiMomentInnovationConsistencyLoss(
                    alpha=getattr(loss_config, 'multimoment_alpha', 1.0),
                    beta=getattr(loss_config, 'multimoment_beta', 1.0)
                ).to(device)
                online_training_reference_loss = multimoment_criterion(
                    angles_pred=pre_ekf_preds,
                    angles=ekf_preds,
                    return_components=False
                )
                # Multi-Moment already divides by batch_size, no need for .mean()
            except Exception as e:
                raise RuntimeError(f"Failed to calculate Multi-Moment reference loss: {e}")
        elif training_loss_type == "unsupervised_rmspe":
            # Unsupervised RMSPE: EKF vs pre-EKF
            online_training_reference_loss = rmspe_criterion(ekf_preds, pre_ekf_preds) / window_size  # RMSPE sums across batch, divide by window size
        elif training_loss_type == "unsupervised_rmape":
            # Unsupervised RMAPE: EKF vs pre-EKF
            online_training_reference_loss = rmape_criterion(ekf_preds, pre_ekf_preds) / window_size  # RMAPE sums across batch, divide by window size
        elif training_loss_type == "supervised_rmspe":
            # Supervised RMSPE: EKF vs true angles
            online_training_reference_loss = rmspe_criterion(ekf_preds, true_angles) / window_size  # RMSPE sums across batch, divide by window size
        elif training_loss_type == "supervised_rmape":
            # Supervised RMAPE: EKF vs true angles
            online_training_reference_loss = rmape_criterion(ekf_preds, true_angles) / window_size  # RMAPE sums across batch, divide by window size
        else:
            # Unknown training_loss_type, terminate with error
            raise RuntimeError(f"Unknown training_loss_type: {training_loss_type}. Must be one of: multimoment, unsupervised_rmspe, unsupervised_rmape, supervised_rmspe, supervised_rmape")
        
        # Pre-EKF loss (raw model performance)
        pre_ekf_loss = rmspe_criterion(pre_ekf_preds, true_angles) / window_size  # RMSPE sums across batch, divide by window size
        
        # EKF gain losses (EKF improvement over raw predictions)
        ekf_gain_rmspe = rmspe_criterion(ekf_preds, pre_ekf_preds) / window_size  # RMSPE sums across batch, divide by window size
        ekf_gain_rmape = rmape_criterion(ekf_preds, pre_ekf_preds) / window_size  # RMAPE sums across batch, divide by window size
        
        return LossMetrics(
            main_loss=main_loss.item() if hasattr(main_loss, 'item') else main_loss,
            main_loss_db=main_loss_db,
            main_loss_config=main_loss_config,
            online_training_reference_loss=online_training_reference_loss.item() if hasattr(online_training_reference_loss, 'item') else online_training_reference_loss,
            online_training_reference_loss_config=online_training_reference_loss_config,
            pre_ekf_loss=pre_ekf_loss.item() if hasattr(pre_ekf_loss, 'item') else pre_ekf_loss,
            ekf_gain_rmspe=ekf_gain_rmspe.item() if hasattr(ekf_gain_rmspe, 'item') else ekf_gain_rmspe,
            ekf_gain_rmape=ekf_gain_rmape.item() if hasattr(ekf_gain_rmape, 'item') else ekf_gain_rmape
        )


    def _calculate_window_training_loss(self, step_results_list: List[Dict], 
                                      loss_config=None, rmspe_criterion=None, rmape_criterion=None) -> torch.Tensor:
        """
        Calculate window-level training loss based on configuration.
        
        Args:
            step_results_list: List of step result dictionaries from window
            loss_config: Loss configuration (optional)
            rmspe_criterion: RMSPE loss criterion instance
            rmape_criterion: RMAPE loss criterion instance
        
        Returns:
            Window-level loss tensor
        """
        if not step_results_list:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Collect all tensors for window-level loss calculation
        all_pre_ekf_preds = []
        all_ekf_preds = []
        all_true_angles = []
        
        for step_result in step_results_list:
            if step_result['success']:
                all_pre_ekf_preds.append(step_result['pre_ekf_angles_pred_tensor'])
                all_ekf_preds.append(step_result['ekf_angles_pred_tensor'])
                all_true_angles.append(step_result['true_angles_tensor'])
        
        if not all_pre_ekf_preds:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Stack predictions across time steps
        window_pre_ekf_preds = torch.cat(all_pre_ekf_preds, dim=0)  # [window_size, num_sources]
        window_ekf_preds = torch.cat(all_ekf_preds, dim=0)          # [window_size, num_sources]
        window_true_angles = torch.cat(all_true_angles, dim=0)      # [window_size, num_sources]
        
        # Fix tensor shapes - remove extra dimension if present
        window_pre_ekf_preds = self._fix_tensor_shape_for_loss(window_pre_ekf_preds)
        window_ekf_preds = self._fix_tensor_shape_for_loss(window_ekf_preds)
        window_true_angles = self._fix_tensor_shape_for_loss(window_true_angles)
        
        # Get window size for proper averaging
        window_size = window_pre_ekf_preds.shape[0]
        
        # Calculate window-level loss based on training_loss_type configuration
        if loss_config is not None and hasattr(loss_config, 'training_loss_type'):
            training_loss_type = loss_config.training_loss_type
            
            if training_loss_type == "multimoment":
                # Multi-Moment loss: use pre-EKF as predictions, EKF as targets
                try:
                    multimoment_criterion = MultiMomentInnovationConsistencyLoss(
                        alpha=getattr(loss_config, 'multimoment_alpha', 1.0),
                        beta=getattr(loss_config, 'multimoment_beta', 1.0)
                    ).to(device)
                    return multimoment_criterion(
                        angles_pred=window_pre_ekf_preds,
                        angles=window_ekf_preds,
                        return_components=False
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to calculate Multi-Moment training loss: {e}")
            elif training_loss_type == "unsupervised_rmspe":
                # Unsupervised RMSPE: EKF vs pre-EKF
                return rmspe_criterion(window_ekf_preds, window_pre_ekf_preds) / window_size  # RMSPE sums across batch, divide by window size
            elif training_loss_type == "unsupervised_rmape":
                # Unsupervised RMAPE: EKF vs pre-EKF
                return rmape_criterion(window_ekf_preds, window_pre_ekf_preds) / window_size  # RMAPE sums across batch, divide by window size
            elif training_loss_type == "supervised_rmspe":
                # Supervised RMSPE: EKF vs true angles
                return rmspe_criterion(window_ekf_preds, window_true_angles) / window_size  # RMSPE sums across batch, divide by window size
            elif training_loss_type == "supervised_rmape":
                # Supervised RMAPE: EKF vs true angles
                return rmape_criterion(window_ekf_preds, window_true_angles) / window_size  # RMAPE sums across batch, divide by window size
            else:
                # Unknown training_loss_type, terminate with error
                raise RuntimeError(f"Unknown training_loss_type: {training_loss_type}. Must be one of: multimoment, unsupervised_rmspe, unsupervised_rmape, supervised_rmspe, supervised_rmape")
        else:
            # Default to RMSPE loss
            return rmspe_criterion(window_ekf_preds, window_true_angles) / window_size  # RMSPE sums across batch, divide by window size



    def _get_optimal_permutation(self, predictions: np.ndarray, true_angles: np.ndarray) -> np.ndarray:
        """
        Calculate optimal permutation between predictions and true angles using RMSPE.
        
        Args:
            predictions: Array of predicted angles [num_sources]
            true_angles: Array of true angles [num_sources]
            
        Returns:
            optimal_perm: Array containing the optimal permutation indices
        """
        import torch
        from DCD_MUSIC.src.metrics.rmspe_loss import RMSPELoss
        from itertools import permutations
        
        # Convert inputs to tensors and reshape
        pred_tensor = torch.tensor(predictions, device=device).view(1, -1)
        true_tensor = torch.tensor(true_angles, device=device).view(1, -1)
        
        num_sources = pred_tensor.shape[1]
        perm = list(permutations(range(num_sources), num_sources))
        num_of_perm = len(perm)
        
        # Calculate errors for all permutations
        err_angle = (pred_tensor[:, perm] - torch.tile(true_tensor[:, None, :], (1, num_of_perm, 1)).to(torch.float32))
        err_angle += torch.pi / 2
        err_angle %= torch.pi
        err_angle -= torch.pi / 2
        rmspe_angle_all_permutations = np.sqrt(1 / num_sources) * torch.linalg.norm(err_angle, dim=-1)
        _, min_idx = torch.min(rmspe_angle_all_permutations, dim=-1)
        
        # Get optimal permutation
        optimal_perm = torch.tensor(perm, dtype=torch.long, device=device)[min_idx]
        return optimal_perm.cpu().numpy()

    def _get_optimal_permutation_tensor(self, predictions: torch.Tensor, true_angles: torch.Tensor) -> torch.Tensor:
        """
        Calculate optimal permutation between predictions and true angles using RMSPE.
        Tensor version that works directly with tensors.
        
        Args:
            predictions: Tensor of predicted angles [num_sources] or [1, num_sources]
            true_angles: Tensor of true angles [num_sources] or [1, num_sources]
            
        Returns:
            optimal_perm: Tensor containing the optimal permutation indices
        """
        from itertools import permutations
        
        # Ensure inputs are 2D tensors
        if predictions.dim() == 1:
            pred_tensor = predictions.unsqueeze(0)
        else:
            pred_tensor = predictions
            
        if true_angles.dim() == 1:
            true_tensor = true_angles.unsqueeze(0)
        else:
            true_tensor = true_angles
        
        num_sources = pred_tensor.shape[1]
        perm = list(permutations(range(num_sources), num_sources))
        num_of_perm = len(perm)
        
        # Calculate errors for all permutations
        err_angle = (pred_tensor[:, perm] - torch.tile(true_tensor[:, None, :], (1, num_of_perm, 1)).to(torch.float32))
        err_angle += torch.pi / 2
        err_angle %= torch.pi
        err_angle -= torch.pi / 2
        rmspe_angle_all_permutations = torch.sqrt(torch.tensor(1.0 / num_sources, device=predictions.device)) * torch.linalg.norm(err_angle, dim=-1)
        _, min_idx = torch.min(rmspe_angle_all_permutations, dim=-1)
        
        # Get optimal permutation
        optimal_perm = torch.tensor(perm, dtype=torch.long, device=predictions.device)[min_idx]
        return optimal_perm


    def _average_online_learning_results_across_trajectories(self, results_list):
        """
        Average results across multiple trajectories.
        
        Args:
            results_list: List of dictionaries containing results from each trajectory
        
        Returns:
            Dictionary with averaged results
        """
        import numpy as np
            
        # Initialize lists to store results from all trajectories
        all_window_losses = []
        all_window_covariances = []
        all_window_eta_values = []
        all_window_updates = []
        all_drift_detected = []
        all_model_updated = []
        all_window_count = []
        all_window_size = []
        all_stride = []
        all_loss_threshold = []
        all_ekf_predictions = []
        all_ekf_covariances = []
        all_ekf_innovations = []  # New list for innovations
        all_ekf_kalman_gains = []  # New list for Kalman gains
        all_ekf_kalman_gain_times_innovation = []  # New list for K*y
        all_ekf_y_s_inv_y = []  # New list for y*(S^-1)*y
        all_pre_ekf_losses = []
        all_window_labels = []
        all_window_pre_ekf_angles_pred = []
        
        # Online learning data
        all_online_window_losses = []
        all_online_window_covariances = []
        all_online_pre_ekf_losses = []
        all_online_ekf_predictions = []
        all_online_ekf_covariances = []
        all_online_ekf_innovations = []
        all_online_ekf_kalman_gains = []
        all_online_ekf_kalman_gain_times_innovation = []
        all_online_ekf_y_s_inv_y = []
        all_online_window_indices = []
        all_online_pre_ekf_angles_pred = []
        
        # Training data
        all_training_window_losses = []
        all_training_window_covariances = []
        all_training_pre_ekf_losses = []
        all_training_ekf_predictions = []
        all_training_ekf_covariances = []
        all_training_ekf_innovations = []
        all_training_ekf_kalman_gains = []
        all_training_ekf_kalman_gain_times_innovation = []
        all_training_ekf_y_s_inv_y = []
        all_training_window_indices = []
        all_learning_start_windows = []
        all_training_pre_ekf_angles_pred = []
        
        # Collect results from each trajectory
        for result in results_list:
            all_window_losses.append(result["window_losses"])
            all_window_covariances.append(result["window_covariances"])
            all_window_eta_values.append(result["window_eta_values"])
            all_window_updates.append(result["window_updates"])
            all_drift_detected.append(result["drift_detected_count"])
            all_model_updated.append(result["model_updated_count"])
            all_window_count.append(result["window_count"])
            all_window_size.append(result["window_size"])
            all_stride.append(result["stride"])
            all_loss_threshold.append(result["loss_threshold"])
            all_ekf_predictions.append(result["ekf_predictions"])
            all_ekf_covariances.append(result["ekf_covariances"])
            all_ekf_innovations.append(result["ekf_innovations"])  # Collect innovations
            all_ekf_kalman_gains.append(result["ekf_kalman_gains"])
            all_ekf_kalman_gain_times_innovation.append(result["ekf_kalman_gain_times_innovation"])
            all_ekf_y_s_inv_y.append(result["ekf_y_s_inv_y"])  # Collect y*(S^-1)*y
            all_pre_ekf_losses.append(result["pre_ekf_losses"])
            all_window_labels.append(result["window_labels"])
            
            # Collect pre-EKF angle predictions for static model
            if "window_pre_ekf_angles_pred" in result:
                all_window_pre_ekf_angles_pred.append(result["window_pre_ekf_angles_pred"])
            else:
                all_window_pre_ekf_angles_pred.append([])
            
            # Collect online learning results if available
            if "online_window_losses" in result and len(result["online_window_losses"]) > 0:
                all_online_window_losses.append(result["online_window_losses"])
                all_online_window_covariances.append(result["online_window_covariances"])
                all_online_pre_ekf_losses.append(result["online_pre_ekf_losses"])
                all_online_ekf_predictions.append(result["online_ekf_predictions"])
                all_online_ekf_covariances.append(result["online_ekf_covariances"])
                all_online_ekf_innovations.append(result["online_ekf_innovations"])
                all_online_ekf_kalman_gains.append(result["online_ekf_kalman_gains"])
                all_online_ekf_kalman_gain_times_innovation.append(result["online_ekf_kalman_gain_times_innovation"])
                all_online_ekf_y_s_inv_y.append(result["online_ekf_y_s_inv_y"])
                all_online_window_indices.append(result["online_window_indices"])
                
                # Collect online pre-EKF angle predictions if available
                if "online_pre_ekf_angles_pred" in result:
                    all_online_pre_ekf_angles_pred.append(result["online_pre_ekf_angles_pred"])
                else:
                    all_online_pre_ekf_angles_pred.append([])
            else:
                # Add empty lists if no online learning data
                all_online_window_losses.append([])
                all_online_window_covariances.append([])
                all_online_pre_ekf_losses.append([])
                all_online_ekf_predictions.append([])
                all_online_ekf_covariances.append([])
                all_online_ekf_innovations.append([])
                all_online_ekf_kalman_gains.append([])
                all_online_ekf_kalman_gain_times_innovation.append([])
                all_online_ekf_y_s_inv_y.append([])
                all_online_window_indices.append([])
                all_online_pre_ekf_angles_pred.append([])
            
            # Collect training data if available
            if "training_window_losses" in result and len(result["training_window_losses"]) > 0:
                all_training_window_losses.append(result["training_window_losses"])
                all_training_window_covariances.append(result["training_window_covariances"])
                all_training_pre_ekf_losses.append(result["training_pre_ekf_losses"])
                all_training_ekf_predictions.append(result["training_ekf_predictions"])
                all_training_ekf_covariances.append(result["training_ekf_covariances"])
                all_training_ekf_innovations.append(result["training_ekf_innovations"])
                all_training_ekf_kalman_gains.append(result["training_ekf_kalman_gains"])
                all_training_ekf_kalman_gain_times_innovation.append(result["training_ekf_kalman_gain_times_innovation"])
                all_training_ekf_y_s_inv_y.append(result["training_ekf_y_s_inv_y"])
                all_training_window_indices.append(result["training_window_indices"])
                all_learning_start_windows.append(result["learning_start_window"])
                
                # Collect training pre-EKF angle predictions if available
                if "training_pre_ekf_angles_pred" in result:
                    all_training_pre_ekf_angles_pred.append(result["training_pre_ekf_angles_pred"])
                else:
                    all_training_pre_ekf_angles_pred.append([])
            else:
                # Add empty lists if no training data
                all_training_window_losses.append([])
                all_training_window_covariances.append([])
                all_training_pre_ekf_losses.append([])
                all_training_ekf_predictions.append([])
                all_training_ekf_covariances.append([])
                all_training_ekf_innovations.append([])
                all_training_ekf_kalman_gains.append([])
                all_training_ekf_kalman_gain_times_innovation.append([])
                all_training_ekf_y_s_inv_y.append([])
                all_training_window_indices.append([])
                all_learning_start_windows.append(None)
                all_training_pre_ekf_angles_pred.append([])
        
        # Average numerical results
        avg_window_losses = np.mean(all_window_losses, axis=0)
        avg_window_covariances = np.mean(all_window_covariances, axis=0)
        avg_window_eta_values = all_window_eta_values[0]  # Should be same for all trajectories
        avg_window_updates = np.mean(all_window_updates, axis=0)
        avg_drift_detected = np.mean(all_drift_detected)
        avg_model_updated = np.mean(all_model_updated)
        avg_pre_ekf_losses = np.mean(all_pre_ekf_losses, axis=0)
        
        # Average innovations - handle nested structure
        avg_ekf_innovations = []
        for window_idx in range(len(all_ekf_innovations[0])):  # For each window
            window_innovations = []
            for step_idx in range(len(all_ekf_innovations[0][window_idx])):  # For each step
                step_innovations = []
                for source_idx in range(len(all_ekf_innovations[0][window_idx][step_idx])):  # For each source
                    innovations = [traj[window_idx][step_idx][source_idx] for traj in all_ekf_innovations]
                    step_innovations.append(np.mean(innovations)) # source innovation averaged over all trajectories
                window_innovations.append(step_innovations) # step innovation averaged on all trajectories
            avg_ekf_innovations.append(window_innovations) # window innovation averaged over all trajectories
        
        # Average Kalman gains - handle nested structure
        avg_ekf_kalman_gains = []
        for window_idx in range(len(all_ekf_kalman_gains[0])):  # For each window
            window_kalman_gains = []
            for step_idx in range(len(all_ekf_kalman_gains[0][window_idx])):  # For each step
                step_kalman_gains = []
                for source_idx in range(len(all_ekf_kalman_gains[0][window_idx][step_idx])):  # For each source
                    kalman_gains = [traj[window_idx][step_idx][source_idx] for traj in all_ekf_kalman_gains]
                    step_kalman_gains.append(np.mean(kalman_gains))
                window_kalman_gains.append(step_kalman_gains)
            avg_ekf_kalman_gains.append(window_kalman_gains)
        
        # Average K*y - handle nested structure
        avg_ekf_kalman_gain_times_innovation = []
        for window_idx in range(len(all_ekf_kalman_gain_times_innovation[0])):  # For each window
            window_k_times_y = []
            for step_idx in range(len(all_ekf_kalman_gain_times_innovation[0][window_idx])):  # For each step
                step_k_times_y = []
                for source_idx in range(len(all_ekf_kalman_gain_times_innovation[0][window_idx][step_idx])):  # For each source
                    k_times_y = [traj[window_idx][step_idx][source_idx] for traj in all_ekf_kalman_gain_times_innovation]
                    step_k_times_y.append(np.mean(k_times_y))
                window_k_times_y.append(step_k_times_y)
            avg_ekf_kalman_gain_times_innovation.append(window_k_times_y)
        
        # Average y*(S^-1)*y - handle nested structure
        avg_ekf_y_s_inv_y = []
        for window_idx in range(len(all_ekf_y_s_inv_y[0])):  # For each window
            window_y_s_inv_y = []
            for step_idx in range(len(all_ekf_y_s_inv_y[0][window_idx])):  # For each step
                step_y_s_inv_y = []
                for source_idx in range(len(all_ekf_y_s_inv_y[0][window_idx][step_idx])):  # For each source
                    y_s_inv_y = [traj[window_idx][step_idx][source_idx] for traj in all_ekf_y_s_inv_y]
                    step_y_s_inv_y.append(np.mean(y_s_inv_y))
                window_y_s_inv_y.append(step_y_s_inv_y)
            avg_ekf_y_s_inv_y.append(window_y_s_inv_y)
        
        # Average online learning results if available
        has_online_data = any(len(online_losses) > 0 for online_losses in all_online_window_losses)
        if has_online_data:
            # Filter out empty trajectories for online learning averaging
            valid_online_trajectories = [i for i, online_losses in enumerate(all_online_window_losses) if len(online_losses) > 0]
            
            if valid_online_trajectories:
                # Average online window losses
                valid_online_losses = [all_online_window_losses[i] for i in valid_online_trajectories]
                avg_online_window_losses = np.mean(valid_online_losses, axis=0).tolist()
                
                # Average online window covariances
                valid_online_covs = [all_online_window_covariances[i] for i in valid_online_trajectories]
                avg_online_window_covariances = np.mean(valid_online_covs, axis=0).tolist()
                
                # Average online pre-EKF losses
                valid_online_pre_ekf = [all_online_pre_ekf_losses[i] for i in valid_online_trajectories]
                avg_online_pre_ekf_losses = np.mean(valid_online_pre_ekf, axis=0).tolist()
                
                # For nested structures, take the first valid trajectory's data
                first_valid_idx = valid_online_trajectories[0]
                avg_online_ekf_predictions = all_online_ekf_predictions[first_valid_idx]
                avg_online_ekf_covariances = all_online_ekf_covariances[first_valid_idx]
                avg_online_ekf_innovations = all_online_ekf_innovations[first_valid_idx]
                avg_online_ekf_kalman_gains = all_online_ekf_kalman_gains[first_valid_idx]
                avg_online_ekf_kalman_gain_times_innovation = all_online_ekf_kalman_gain_times_innovation[first_valid_idx]
                avg_online_ekf_y_s_inv_y = all_online_ekf_y_s_inv_y[first_valid_idx]
                avg_online_window_indices = all_online_window_indices[first_valid_idx]
                avg_online_pre_ekf_angles_pred = all_online_pre_ekf_angles_pred[first_valid_idx]
            else:
                # No valid online data
                avg_online_window_losses = []
                avg_online_window_covariances = []
                avg_online_pre_ekf_losses = []
                avg_online_ekf_predictions = []
                avg_online_ekf_covariances = []
                avg_online_ekf_innovations = []
                avg_online_ekf_kalman_gains = []
                avg_online_ekf_kalman_gain_times_innovation = []
                avg_online_ekf_y_s_inv_y = []
                avg_online_window_indices = []
                avg_online_pre_ekf_angles_pred = []
        else:
            # No online data at all
            avg_online_window_losses = []
            avg_online_window_covariances = []
            avg_online_pre_ekf_losses = []
            avg_online_ekf_predictions = []
            avg_online_ekf_covariances = []
            avg_online_ekf_innovations = []
            avg_online_ekf_kalman_gains = []
            avg_online_ekf_kalman_gain_times_innovation = []
            avg_online_ekf_y_s_inv_y = []
            avg_online_window_indices = []
            avg_online_pre_ekf_angles_pred = []
        
        # Average training data if available
        has_training_data = any(len(training_losses) > 0 for training_losses in all_training_window_losses)
        if has_training_data:
            # Filter out empty trajectories for training data averaging
            valid_training_trajectories = [i for i, training_losses in enumerate(all_training_window_losses) if len(training_losses) > 0]
            
            if valid_training_trajectories:
                # Average training window losses
                valid_training_losses = [all_training_window_losses[i] for i in valid_training_trajectories]
                avg_training_window_losses = np.mean(valid_training_losses, axis=0).tolist()
                
                # Average training window covariances
                valid_training_covs = [all_training_window_covariances[i] for i in valid_training_trajectories]
                avg_training_window_covariances = np.mean(valid_training_covs, axis=0).tolist()
                
                # Average training pre-EKF losses
                valid_training_pre_ekf = [all_training_pre_ekf_losses[i] for i in valid_training_trajectories]
                avg_training_pre_ekf_losses = np.mean(valid_training_pre_ekf, axis=0).tolist()
                
                # For nested structures, take the first valid trajectory's data
                first_valid_idx = valid_training_trajectories[0]
                avg_training_ekf_predictions = all_training_ekf_predictions[first_valid_idx]
                avg_training_ekf_covariances = all_training_ekf_covariances[first_valid_idx]
                avg_training_ekf_innovations = all_training_ekf_innovations[first_valid_idx]
                avg_training_ekf_kalman_gains = all_training_ekf_kalman_gains[first_valid_idx]
                avg_training_ekf_kalman_gain_times_innovation = all_training_ekf_kalman_gain_times_innovation[first_valid_idx]
                avg_training_ekf_y_s_inv_y = all_training_ekf_y_s_inv_y[first_valid_idx]
                avg_training_window_indices = all_training_window_indices[first_valid_idx]
                avg_learning_start_window = all_learning_start_windows[first_valid_idx]
                avg_training_pre_ekf_angles_pred = all_training_pre_ekf_angles_pred[first_valid_idx]
            else:
                # No valid training data
                avg_training_window_losses = []
                avg_training_window_covariances = []
                avg_training_pre_ekf_losses = []
                avg_training_ekf_predictions = []
                avg_training_ekf_covariances = []
                avg_training_ekf_innovations = []
                avg_training_ekf_kalman_gains = []
                avg_training_ekf_kalman_gain_times_innovation = []
                avg_training_ekf_y_s_inv_y = []
                avg_training_window_indices = []
                avg_learning_start_window = None
                avg_training_pre_ekf_angles_pred = []
        else:
            # No training data at all
            avg_training_window_losses = []
            avg_training_window_covariances = []
            avg_training_pre_ekf_losses = []
            avg_training_ekf_predictions = []
            avg_training_ekf_covariances = []
            avg_training_ekf_innovations = []
            avg_training_ekf_kalman_gains = []
            avg_training_ekf_kalman_gain_times_innovation = []
            avg_training_ekf_y_s_inv_y = []
            avg_training_window_indices = []
            avg_learning_start_window = None
            avg_training_pre_ekf_angles_pred = []
        
        return {
            "window_losses": avg_window_losses.tolist(),
            "window_covariances": avg_window_covariances.tolist(),
            "window_eta_values": avg_window_eta_values,
            "window_updates": avg_window_updates.tolist(),
            "drift_detected_count": float(avg_drift_detected),
            "model_updated_count": float(avg_model_updated),
            "window_count": all_window_count[0],  # Should be same for all trajectories
            "window_size": all_window_size[0],    # Should be same for all trajectories
            "stride": all_stride[0],              # Should be same for all trajectories
            "loss_threshold": all_loss_threshold[0],  # Should be same for all trajectories
            "ekf_predictions": all_ekf_predictions[0],  # Take first trajectory's predictions
            "ekf_covariances": all_ekf_covariances[0],  # Take first trajectory's covariances
            "ekf_innovations": avg_ekf_innovations,  # Add averaged innovations
            "ekf_kalman_gains": avg_ekf_kalman_gains,  # Add averaged Kalman gains
            "ekf_kalman_gain_times_innovation": avg_ekf_kalman_gain_times_innovation,  # Add averaged K*y
            "ekf_y_s_inv_y": avg_ekf_y_s_inv_y,  # Add averaged y*(S^-1)*y
            "pre_ekf_losses": avg_pre_ekf_losses.tolist(),
            "window_labels": all_window_labels[0],  # Take first trajectory's labels
            "window_pre_ekf_angles_pred": all_window_pre_ekf_angles_pred[0] if all_window_pre_ekf_angles_pred else [],  # Take first trajectory's pre-EKF angle predictions
            "window_avg_ekf_angle_pred": results_list[0].get("window_avg_ekf_angle_pred", []) if results_list else [],  # Take first trajectory's averaged EKF angle predictions
            "window_avg_pre_ekf_angle_pred": results_list[0].get("window_avg_pre_ekf_angle_pred", []) if results_list else [],  # Take first trajectory's averaged pre-EKF angle predictions
            
            # Online learning results
            "online_window_losses": avg_online_window_losses,
            "online_window_covariances": avg_online_window_covariances,
            "online_pre_ekf_losses": avg_online_pre_ekf_losses,
            "online_ekf_predictions": avg_online_ekf_predictions,
            "online_ekf_covariances": avg_online_ekf_covariances,
            "online_ekf_innovations": avg_online_ekf_innovations,
            "online_ekf_kalman_gains": avg_online_ekf_kalman_gains,
            "online_ekf_kalman_gain_times_innovation": avg_online_ekf_kalman_gain_times_innovation,
            "online_ekf_y_s_inv_y": avg_online_ekf_y_s_inv_y,
            "online_window_indices": avg_online_window_indices,
            "online_pre_ekf_angles_pred": avg_online_pre_ekf_angles_pred,
            "online_avg_ekf_angle_pred": results_list[0].get("online_avg_ekf_angle_pred", []) if results_list else [],  # Take first trajectory's online averaged EKF angle predictions
            "online_avg_pre_ekf_angle_pred": results_list[0].get("online_avg_pre_ekf_angle_pred", []) if results_list else [],  # Take first trajectory's online averaged pre-EKF angle predictions
            
            # Training data results
            "training_window_losses": avg_training_window_losses,
            "training_window_covariances": avg_training_window_covariances,
            "training_pre_ekf_losses": avg_training_pre_ekf_losses,
            "training_ekf_predictions": avg_training_ekf_predictions,
            "training_ekf_covariances": avg_training_ekf_covariances,
            "training_ekf_innovations": avg_training_ekf_innovations,
            "training_ekf_kalman_gains": avg_training_ekf_kalman_gains,
            "training_ekf_kalman_gain_times_innovation": avg_training_ekf_kalman_gain_times_innovation,
            "training_ekf_y_s_inv_y": avg_training_ekf_y_s_inv_y,
            "training_window_indices": avg_training_window_indices,
            "learning_start_window": avg_learning_start_window,
            "training_pre_ekf_angles_pred": avg_training_pre_ekf_angles_pred,
            "training_avg_ekf_angle_pred": results_list[0].get("training_avg_ekf_angle_pred", []) if results_list else [],  # Take first trajectory's training averaged EKF angle predictions
            "training_avg_pre_ekf_angle_pred": results_list[0].get("training_avg_pre_ekf_angle_pred", []) if results_list else []  # Take first trajectory's training averaged pre-EKF angle predictions
        }


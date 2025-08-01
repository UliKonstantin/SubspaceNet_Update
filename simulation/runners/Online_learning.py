from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import logging
import numpy as np
import torch
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import time
import copy
import yaml
from itertools import permutations
import itertools

from config.schema import Config
from simulation.runners.data import TrajectoryDataHandler, create_online_learning_dataset
from simulation.runners.training import Trainer, TrainingConfig, TrajectoryTrainer, OnlineTrainer
from simulation.runners.evaluation import Evaluator
from simulation.kalman_filter import KalmanFilter1D, BatchKalmanFilter1D, BatchExtendedKalmanFilter1D
from DCD_MUSIC.src.metrics.rmspe_loss import RMSPELoss
from DCD_MUSIC.src.signal_creation import Samples
from DCD_MUSIC.src.evaluation import get_model_based_method, evaluate_model_based
from simulation.kalman_filter.extended import ExtendedKalmanFilter1D

logger = logging.getLogger(__name__)

# Device setup for evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        
        # Dual model online learning state variables
        self.drift_detected = False
        self.learning_done = False
        self.online_model = None
        self.online_training_count = 0
        self.first_eta_change = True  # Track if this is the first eta change
        
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
                
                # Plot single trajectory results
                if trajectory_result.get("status") != "error":
                    self._plot_single_trajectory_results(trajectory_result["online_learning_results"], trajectory_idx)
                
                if trajectory_result.get("status") == "error":
                    logger.error(f"Error in trajectory {trajectory_idx + 1}: {trajectory_result.get('message')}")
                    continue
                    
                all_results.append(trajectory_result["online_learning_results"])
            
            if not all_results:
                logger.error("No successful trajectory results")
                self.results["online_learning_error"] = "No successful trajectory results"
                return {"status": "error", "message": "No successful trajectory results"}
                
            # Average results across all trajectories
            averaged_results_across_trajectories = self._average_online_learning_results_across_trajectories(all_results)
            
            # Store averaged results
            self.results["online_learning"] = averaged_results_across_trajectories
            
            # Plot averaged results
            frobenius_norms = self._calculate_frobenius_norm_per_window(averaged_results_across_trajectories["ekf_covariances"])
            self._plot_online_learning_results(
                window_losses=averaged_results_across_trajectories["window_losses"],
                window_covariances=averaged_results_across_trajectories["window_covariances"],
                window_eta_values=averaged_results_across_trajectories["window_eta_values"],
                window_updates=averaged_results_across_trajectories["window_updates"],
                window_pre_ekf_losses=averaged_results_across_trajectories["pre_ekf_losses"],
                window_labels=averaged_results_across_trajectories["window_labels"],
                ekf_covariances=averaged_results_across_trajectories["ekf_covariances"],
                frobenius_norms=frobenius_norms,
                ekf_kalman_gains=averaged_results_across_trajectories["ekf_kalman_gains"],
                ekf_kalman_gain_times_innovation=averaged_results_across_trajectories["ekf_kalman_gain_times_innovation"],
                ekf_y_s_inv_y=averaged_results_across_trajectories["ekf_y_s_inv_y"]
            )
            
            logger.info(f"Online learning completed over {dataset_size} trajectories: "
                       f"{averaged_results_across_trajectories['drift_detected_count']:.1f} avg drifts detected, "
                       f"{averaged_results_across_trajectories['model_updated_count']:.1f} avg model updates")
            
            return {"status": "success", "online_learning_results": self.results.get("online_learning", {})}
            
        except Exception as e:
            logger.exception(f"Error running online learning: {e}")
            self.results["online_learning_error"] = str(e)
            return {"status": "error", "message": str(e), "exception": type(e).__name__}

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
            
            # Initialize online model as copy of trained model using PyTorch's method
            import torch
            from copy import deepcopy
            
            # Create a new model instance and copy the state dict
            try:
                # Method 1: Create new instance and load state dict (preferred)
                self.online_model = type(self.trained_model)()
                self.online_model.load_state_dict(self.trained_model.state_dict())
                self.online_model.eval()  # Set to eval mode like the trained model
                logger.info("Initialized online model as copy of trained model using state_dict")
            except Exception as e:
                logger.warning(f"State dict copy failed ({e}), trying clone approach")
                try:
                    # Method 2: Use torch.clone() for parameters
                    self.online_model = deepcopy(self.trained_model.cpu())
                    if torch.cuda.is_available() and next(self.trained_model.parameters()).is_cuda:
                        self.online_model = self.online_model.cuda()
                    logger.info("Initialized online model using CPU deepcopy then moved to GPU")
                except Exception as e2:
                    logger.error(f"All model copying methods failed: {e2}")
                    self.online_model = self.trained_model  # Use same model as fallback
                    logger.warning("Using shared model reference as fallback")
            
            # Reset dual model state for new trajectory
            self.drift_detected = False
            self.learning_done = False
            self.online_training_count = 0
            self.first_eta_change = True
            # Reset online optimizer to start fresh for new trajectory
            if hasattr(self, 'online_optimizer'):
                delattr(self, 'online_optimizer')
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
            
            # Prepare for tracking results
            window_losses = []
            window_covariances = []
            window_update_flags = []
            window_eta_values = []
            window_ekf_predictions = []
            window_ekf_covariances = []
            window_ekf_innovations = []  # New list to track innovations
            window_ekf_kalman_gains = []  # New list to track Kalman gains
            window_ekf_kalman_gain_times_innovation = []  # New list to track K*y
            window_ekf_y_s_inv_y = []  # New list to track y*(S^-1)*y
            window_pre_ekf_losses = []
            window_labels = []  # Store labels for each window
            drift_detected_count = 0
            model_updated_count = 0
            last_ekf_predictions = None  # Track last window's EKF predictions
            last_ekf_covariances = None  # Track last window's EKF covariances
            
            # Online model tracking variables
            online_window_losses = []
            online_window_covariances = []
            online_window_pre_ekf_losses = []
            online_ekf_predictions = []
            online_ekf_covariances = []
            online_ekf_innovations = []
            online_ekf_kalman_gains = []
            online_ekf_kalman_gain_times_innovation = []
            online_ekf_y_s_inv_y = []
            
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
                        
                        # Set drift detected on first eta change only
                        if self.first_eta_change:
                            self.drift_detected = True
                            self.first_eta_change = False
                            logger.info(f"Drift detected due to first eta modification at window {window_idx}")
                        
                        # The dataset holds the generator, which updates the shared self.system_model.params.eta
                        online_learning_dataloader.dataset.update_eta(new_eta)
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
                current_window_loss, avg_window_cov  , ekf_predictions, ekf_covariances, pre_ekf_loss, ekf_innovations, ekf_kalman_gains, ekf_kalman_gain_times_innovation, ekf_y_s_inv_y = self._evaluate_window(
                    time_series_single_window, 
                    sources_num_single_window_list, 
                    labels_single_window_list_of_arrays,
                    trajectory_idx,
                    window_idx,
                    is_first_window=(window_idx == 0),
                    last_ekf_predictions=last_ekf_predictions,
                    last_ekf_covariances=last_ekf_covariances
                )
               
                # Update tracking variables
                window_losses.append(current_window_loss)
                window_covariances.append(avg_window_cov)
                window_eta_values.append(self.system_model.params.eta)
                window_ekf_predictions.append(ekf_predictions)
                window_ekf_covariances.append(ekf_covariances)
                window_ekf_innovations.append(ekf_innovations)  # Store innovations
                window_ekf_kalman_gains.append(ekf_kalman_gains)
                window_ekf_kalman_gain_times_innovation.append(ekf_kalman_gain_times_innovation)
                window_ekf_y_s_inv_y.append(ekf_y_s_inv_y)  # Store y*(S^-1)*y
                window_pre_ekf_losses.append(pre_ekf_loss)
                window_labels.append(labels_single_window_list_of_arrays)
                
                # Update last predictions and covariances for next window
                last_ekf_predictions = ekf_predictions
                last_ekf_covariances = ekf_covariances
                
                logger.info(f"Window {window_idx}: Loss = {current_window_loss:.6f}, Cov = {avg_window_cov:.6f} (current eta={self.system_model.params.eta:.4f})")
                
                # Log both pre-EKF and EKF losses for comparison
                logger.info(f"Window {window_idx}: Pre-EKF Loss = {pre_ekf_loss:.6f}, EKF Loss = {current_window_loss:.6f}, Cov = {avg_window_cov:.6f} (eta={self.system_model.params.eta:.4f})")
                
                # Check if loss exceeds threshold (drift detected)
                if current_window_loss > loss_threshold:
                    logger.info(f"Drift detected in window {window_idx} (loss: {current_window_loss:.6f} > threshold: {loss_threshold:.6f})")
                    window_update_flags.append(False)
                    self.drift_detected = True
                else:
                    logger.info(f"No drift detected in window {window_idx} (loss: {current_window_loss:.6f} <= threshold: {loss_threshold:.6f})")
                    window_update_flags.append(False)
                    # Keep previous drift_detected state if no drift in current window
                
                # Dual model processing logic
                if self.drift_detected:
                    if self.learning_done:
                        # Online model finished training, evaluate it normally
                        logger.info(f"Evaluating online model (post-training) for window {window_idx}")
                        online_window_loss, online_avg_window_cov, online_ekf_preds, online_ekf_covs, online_pre_ekf_loss, online_ekf_inns, online_ekf_kgains, online_ekf_kg_times_inn, online_ekf_y_s_inv_y_val = self._evaluate_window(
                            time_series_single_window, 
                            sources_num_single_window_list, 
                            labels_single_window_list_of_arrays,
                            trajectory_idx,
                            window_idx,
                            is_first_window=(window_idx == 0),
                            last_ekf_predictions=last_ekf_predictions,
                            last_ekf_covariances=last_ekf_covariances
                        )
                        
                        # Store online model results for comparison
                        online_window_losses.append(online_window_loss)
                        online_window_covariances.append(online_avg_window_cov)
                        online_window_pre_ekf_losses.append(online_pre_ekf_loss)
                        online_ekf_predictions.append(online_ekf_preds)
                        online_ekf_covariances.append(online_ekf_covs)
                        online_ekf_innovations.append(online_ekf_inns)
                        online_ekf_kalman_gains.append(online_ekf_kgains)
                        online_ekf_kalman_gain_times_innovation.append(online_ekf_kg_times_inn)
                        online_ekf_y_s_inv_y.append(online_ekf_y_s_inv_y_val)
                        
                        logger.info(f"Online model - Window {window_idx}: Loss = {online_window_loss:.6f}, Cov = {online_avg_window_cov:.6f}")
                    else:
                        # Online model still learning/adapting
                        logger.info(f"Training online model for window {window_idx}")
                        training_window_loss, training_avg_window_cov, training_ekf_preds, training_ekf_covs, training_pre_ekf_loss, training_ekf_inns, training_ekf_kgains, training_ekf_kg_times_inn, training_ekf_y_s_inv_y_val = self._online_training_window(
                            time_series_single_window, 
                            sources_num_single_window_list, 
                            labels_single_window_list_of_arrays,
                            trajectory_idx,
                            window_idx
                        )
                        
                        # Store training results for logging (optional, could be used for analysis)
                        logger.info(f"Online training - Window {window_idx}: EKF Loss = {training_window_loss:.6f}, Cov = {training_avg_window_cov:.6f}, Pre-EKF Loss = {training_pre_ekf_loss:.6f}")
            
            # Save final model if it was updated
            if model_updated_count > 0:
                model_save_path = self._save_model_state(
                    self.trained_model,
                    model_type=f"{self.config.model.type}_online_updated"
                )
                logger.info(f"Saved final online-updated model to {model_save_path}")
            
            # Return results
            return {
                "status": "success",
                "online_learning_results": {
                    # Static model results (always available)
                    "window_losses": window_losses,
                    "window_covariances": window_covariances,
                    "window_eta_values": window_eta_values,
                    "window_updates": window_update_flags,
                    "drift_detected_count": drift_detected_count,
                    "model_updated_count": model_updated_count,
                    "window_count": len(online_learning_dataloader),
                    "window_size": online_config.window_size,
                    "stride": online_config.stride,
                    "loss_threshold": loss_threshold,
                    "ekf_predictions": window_ekf_predictions,
                    "ekf_covariances": window_ekf_covariances,
                    "ekf_innovations": window_ekf_innovations,  # Add innovations to results
                    "ekf_kalman_gains": window_ekf_kalman_gains,
                    "ekf_kalman_gain_times_innovation": window_ekf_kalman_gain_times_innovation,
                    "ekf_y_s_inv_y": window_ekf_y_s_inv_y,  # Add y*(S^-1)*y to results
                    "pre_ekf_losses": window_pre_ekf_losses,
                    "window_labels": window_labels,
                    
                    # Online model results (available when drift detected and learning complete)
                    "online_window_losses": online_window_losses,
                    "online_window_covariances": online_window_covariances,
                    "online_pre_ekf_losses": online_window_pre_ekf_losses,
                    "online_ekf_predictions": online_ekf_predictions,
                    "online_ekf_covariances": online_ekf_covariances,
                    "online_ekf_innovations": online_ekf_innovations,
                    "online_ekf_kalman_gains": online_ekf_kalman_gains,
                    "online_ekf_kalman_gain_times_innovation": online_ekf_kalman_gain_times_innovation,
                    "online_ekf_y_s_inv_y": online_ekf_y_s_inv_y,
                    
                    # Dual model state tracking
                    "drift_detected_final": self.drift_detected,
                    "learning_done_final": self.learning_done,
                    "first_eta_change_final": self.first_eta_change,
                    "online_training_count_final": self.online_training_count
                }
            }
            
        except Exception as e:
            logger.exception(f"Error during online learning: {e}")
            return {"status": "error", "message": str(e), "exception": type(e).__name__}

    def _online_training_window(self, window_time_series, window_sources_num, window_labels, trajectory_idx: int = 0, window_idx: int = 0):
        """
        Train the online model on a single window, then evaluate it like _evaluate_window.
        
        This function performs the same evaluation as _evaluate_window but adds a training step
        after the model forward pass and before the Kalman filter processing.
        
        Args:
            window_time_series: Time series data for window
            window_sources_num: Source counts for window  
            window_labels: Labels for window
            trajectory_idx: Index of the current trajectory
            window_idx: Index of the current window within the trajectory
        """
        # Increment training counter
        self.online_training_count += 1
        logger.info(f"Online training step {self.online_training_count} called for trajectory {trajectory_idx}, window {window_idx}")
        
        # Set learning done after 3 training calls
        if self.online_training_count >= 3:
            self.learning_done = True
            logger.info(f"Online model training completed after {self.online_training_count} training windows")
        
        # Imports for training and EKF
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from DCD_MUSIC.src.metrics.rmspe_loss import RMSPELoss
        import numpy as np
        
        # Debug: Log input shapes
        logger.debug(f"_online_training_window input shapes: "
                     f"window_time_series={window_time_series.shape if hasattr(window_time_series, 'shape') else 'not tensor'}, "
                     f"window_sources_num={len(window_sources_num)}, "
                     f"window_labels={len(window_labels)}")
        
        # Unpack arguments directly, assuming they are for a single window
        time_series_steps = window_time_series # Already [window_size, N, T]
        sources_num_per_step = window_sources_num # List[int]
        labels_per_step_list = window_labels    # List[np.ndarray]
        
        # Check that sizes match to avoid index errors
        current_window_len = time_series_steps.shape[0]
        
        # Sanity check the input lengths
        if len(sources_num_per_step) < current_window_len:
            logger.warning(f"Window source count list length ({len(sources_num_per_step)}) is less than time series length ({current_window_len}). Truncating window.")
            current_window_len = len(sources_num_per_step)
        
        if len(labels_per_step_list) < current_window_len:
            logger.warning(f"Window labels list length ({len(labels_per_step_list)}) is less than time series length ({current_window_len}). Truncating window.")
            current_window_len = len(labels_per_step_list)
        
        if current_window_len == 0:
            logger.error("Window has zero valid steps. Cannot train.")
            return
        
        # Check if we're dealing with far-field or near-field
        is_near_field = hasattr(self.online_model, 'field_type') and self.online_model.field_type.lower() == "near"
        
        # Use RMSPE loss for training and evaluation
        rmspe_criterion = RMSPELoss().to(device)
        
        # Set up optimizer for online training
        if not hasattr(self, 'online_optimizer'):
            self.online_optimizer = optim.Adam(self.online_model.parameters(), lr=1e-4)
        
        # Initialize Extended Kalman Filters - one for each potential source
        max_sources = self.config.system_model.M
        ekf_filters = []
        
        # Create EKF instances
        for i in range(max_sources):
            ekf_filter = ExtendedKalmanFilter1D.create_from_config(
                self.config, 
                trajectory_type=self.config.trajectory.trajectory_type
            )
            ekf_filters.append(ekf_filter)
        
        # Get current eta value from system model
        current_eta = self.system_model.params.eta
        logger.info(f"Online training: Initialized {max_sources} EKF instances for window (eta={current_eta:.4f})")
        
        # Training phase: Accumulate loss for the entire window
        self.online_model.train()  # Set to training mode
        total_training_loss = 0.0
        num_training_steps = 0
        
        # Process each step for training
        for step in range(current_window_len):
            try:
                # Extract data for this step
                step_data_tensor = time_series_steps[step:step+1].to(device)  # Shape: [1, N, T]
                num_sources_this_step = sources_num_per_step[step]
                
                # Skip if no sources
                if num_sources_this_step <= 0:
                    continue
                
                # Get ground truth labels for this step
                true_angles_this_step = labels_per_step_list[step][:num_sources_this_step]
                
                if not is_near_field:
                    # Forward pass through online model (with gradients for training)
                    angles_pred, _, _ = self.online_model(step_data_tensor, num_sources_this_step)
                    
                    # Convert predictions and true angles to proper format
                    angles_pred_tensor = angles_pred.view(1, -1)[:, :num_sources_this_step]
                    true_angles_tensor = torch.tensor(true_angles_this_step, device=device).unsqueeze(0)
                    
                    # Get optimal permutation for training
                    angles_pred_np = angles_pred.detach().cpu().numpy().flatten()[:num_sources_this_step]
                    model_perm = self._get_optimal_permutation(angles_pred_np, true_angles_this_step)
                    angles_pred_tensor = angles_pred_tensor[:, model_perm]
                    
                    # Calculate training loss
                    training_loss = rmspe_criterion(angles_pred_tensor, true_angles_tensor)
                    total_training_loss += training_loss.item()
                    num_training_steps += 1
                    
                    # Backward pass for this step
                    training_loss.backward()
                
                else:
                    # Near-field case - not supported
                    error_msg = "Near-field processing is not supported in this project. Please use far-field models only."
                    logger.error(error_msg)
                    raise NotImplementedError(error_msg)
                
            except Exception as e:
                logger.warning(f"Error during online training step {step}: {e}")
                continue
        
        # Update model parameters after processing all steps in the window
        if num_training_steps > 0:
            avg_training_loss = total_training_loss / num_training_steps
            self.online_optimizer.step()
            self.online_optimizer.zero_grad()
            logger.info(f"Online training step {self.online_training_count}: Updated model with avg loss = {avg_training_loss:.6f} over {num_training_steps} steps")
        else:
            avg_training_loss = float('inf')  # Set default value if no training steps
            logger.warning(f"No valid training steps in window {window_idx}")
        
        # Set model back to eval mode for EKF evaluation
        self.online_model.eval()
        
        # Now evaluate the trained model with EKF (same as _evaluate_window but with no gradients)
        total_loss = 0.0
        num_valid_steps_for_loss = 0
        
        # Add pre-EKF loss tracking for far-field case
        pre_ekf_total_loss = 0.0
        pre_ekf_num_valid_steps = 0
        
        # Track the EKF predictions, covariances and innovations for each step and source
        ekf_predictions = []
        ekf_covariances = []
        ekf_innovations = []  # New list to track innovations
        ekf_kalman_gains = []  # New list to track Kalman gains
        ekf_kalman_gain_times_innovation = []  # New list to track K*y
        ekf_y_s_inv_y = []  # New list to track y*(S^-1)*y
        
        # Process each step in window for EKF evaluation
        for step in range(current_window_len):
            try:
                # Extract data for this step
                step_data_tensor = time_series_steps[step:step+1].to(device)  # Shape: [1, N, T] (add batch dim for model)
                num_sources_this_step = sources_num_per_step[step]
                
                # Skip if no sources
                if num_sources_this_step <= 0:
                    ekf_predictions.append([])
                    ekf_covariances.append([])
                    ekf_innovations.append([])
                    ekf_kalman_gains.append([])
                    ekf_kalman_gain_times_innovation.append([])
                    ekf_y_s_inv_y.append([])
                    continue
                
                # Get ground truth labels for this step
                true_angles_this_step = labels_per_step_list[step][:num_sources_this_step]
                
                # Forward pass through online model (no gradients for EKF evaluation)
                with torch.no_grad():
                    if not is_near_field:
                        # Model expects num_sources as int or 0-dim tensor
                        angles_pred, _, _ = self.online_model(step_data_tensor, num_sources_this_step)
                        
                        # Convert predictions to numpy for EKF processing
                        angles_pred_np = angles_pred.cpu().numpy().flatten()[:num_sources_this_step]
                        
                        # Calculate pre-EKF loss (raw model predictions)
                        pre_ekf_angles_pred = angles_pred.view(1, -1)[:, :num_sources_this_step]
                        true_angles_tensor = torch.tensor(true_angles_this_step, device=device).unsqueeze(0)
                        
                        # Get optimal permutation for model predictions
                        model_perm = self._get_optimal_permutation(angles_pred_np, true_angles_this_step)
                        angles_pred_np = angles_pred_np[model_perm]
                        pre_ekf_angles_pred = pre_ekf_angles_pred[:, model_perm]
                        
                        # Calculate pre-EKF loss with reordered predictions
                        pre_ekf_loss = rmspe_criterion(pre_ekf_angles_pred, true_angles_tensor)
                        pre_ekf_total_loss += pre_ekf_loss.item()
                        pre_ekf_num_valid_steps += 1
                        
                        # Initialize EKF state if this is the first step
                        if step == 0:
                            # For online training, always initialize with true angles since it's training
                            for i in range(num_sources_this_step):
                                ekf_filters[i].initialize_state(true_angles_this_step[i])
                        
                        # EKF update for each source
                        step_predictions = []
                        step_covariances = []
                        step_innovations = []  # New list for this step's innovations
                        step_kalman_gains = []  # New list for this step's Kalman gains
                        step_kalman_gain_times_innovation = []  # New list for this step's K*y
                        step_y_s_inv_y = []  # New list for this step's y*(S^-1)*y
                        for i in range(num_sources_this_step):
                            # Predict and update in one step
                            predicted_angle, updated_angle, innovation, kalman_gain, kalman_gain_times_innovation, y_s_inv_y = ekf_filters[i].predict_and_update(
                                measurement=angles_pred_np.flatten()[:num_sources_this_step][i], true_state=true_angles_this_step[i]
                            )
                            
                            # Store prediction, covariance and innovation
                            step_predictions.append(updated_angle)
                            step_covariances.append(ekf_filters[i].P)
                            step_innovations.append(innovation)
                            step_kalman_gains.append(kalman_gain)
                            step_kalman_gain_times_innovation.append(kalman_gain_times_innovation)
                            step_y_s_inv_y.append(y_s_inv_y)
                        
                        # Store predictions, covariances and innovations for this step
                        ekf_predictions.append(step_predictions)
                        ekf_covariances.append(step_covariances)
                        ekf_innovations.append(step_innovations)
                        ekf_kalman_gains.append(step_kalman_gains)
                        ekf_kalman_gain_times_innovation.append(step_kalman_gain_times_innovation)
                        ekf_y_s_inv_y.append(step_y_s_inv_y)
                        
                        # Create tensor from EKF predictions for loss calculation
                        ekf_angles_pred = torch.tensor(step_predictions, device=device).unsqueeze(0)
                        true_angles_tensor = torch.tensor(true_angles_this_step, device=device).unsqueeze(0)
                        
                        # Calculate loss using EKF predictions
                        loss = rmspe_criterion(ekf_angles_pred, true_angles_tensor)
                    else:
                        # Near-field case - not supported
                        error_msg = "Near-field processing is not supported in this project. Please use far-field models only."
                        logger.error(error_msg)
                        raise NotImplementedError(error_msg)
                
                total_loss += loss.item()
                num_valid_steps_for_loss += 1
            except Exception as e:
                logger.warning(f"Error processing EKF step {step}: {e}")
                # Skip this step on error
                continue
        
        # Calculate average loss and covariance for the window
        if num_valid_steps_for_loss > 0:
            avg_loss = total_loss / num_valid_steps_for_loss
            
            # Calculate pre-EKF average loss for far-field case
            avg_pre_ekf_loss = 0.0
            if not is_near_field and pre_ekf_num_valid_steps > 0:
                avg_pre_ekf_loss = pre_ekf_total_loss / pre_ekf_num_valid_steps
            elif is_near_field:
                # Near-field case - not supported
                error_msg = "Near-field processing is not supported in this project. Please use far-field models only."
                logger.error(error_msg)
                raise NotImplementedError(error_msg)
            
            # Calculate average covariance across all sources and steps
            avg_window_cov = 0.0
            total_cov_points = 0
            
            if len(ekf_covariances) > 0 and any(len(step_covs) > 0 for step_covs in ekf_covariances):
                # Calculate overall average covariance across all sources and steps
                for step_covs in ekf_covariances:
                    if len(step_covs) > 0:
                        avg_window_cov += sum(step_covs)
                        total_cov_points += len(step_covs)
            
            # Calculate the overall average covariance across all sources and steps
            if total_cov_points > 0:
                avg_window_cov /= total_cov_points
            else:
                avg_window_cov = float('nan')  # No valid covariance points
            
            # Log window summary with columnar format
            logger.info(f"Online training window {window_idx}: "
                       f"Pre-EKF Loss = {avg_pre_ekf_loss:.6f}, "
                       f"EKF Loss = {avg_loss:.6f}, "
                       f"Avg Cov = {avg_window_cov:.6f}, "
                       f"Training Loss = {avg_training_loss:.6f}")
            
            return avg_loss, avg_window_cov, ekf_predictions, ekf_covariances, avg_pre_ekf_loss, ekf_innovations, ekf_kalman_gains, ekf_kalman_gain_times_innovation, ekf_y_s_inv_y
        else:
            logger.warning("No valid steps with sources found in the window for EKF evaluation.")
            return float('inf'), float('nan'), [], [], float('inf'), [], [], [], []  # Include pre-EKF loss in return

    def _evaluate_window(self, window_time_series, window_sources_num, window_labels, trajectory_idx: int = 0, window_idx: int = 0, 
                         is_first_window: bool = True, last_ekf_predictions: List = None, last_ekf_covariances: List = None):
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
            
        Returns:
            Tuple of (average EKF-filtered loss across window, average covariance across window, 
                     ekf_predictions, ekf_covariances, average pre-EKF loss across window, ekf_innovations)
        """
        # Imports for EKF
        import torch
        from DCD_MUSIC.src.metrics.rmspe_loss import RMSPELoss
        import numpy as np
        
        # Debug: Log input shapes
        logger.debug(f"_evaluate_window input shapes: "
                     f"window_time_series={window_time_series.shape if hasattr(window_time_series, 'shape') else 'not tensor'}, "
                     f"window_sources_num={len(window_sources_num)}, "
                     f"window_labels={len(window_labels)}")
        
        # Unpack arguments directly, assuming they are for a single window
        time_series_steps = window_time_series # Already [window_size, N, T]
        sources_num_per_step = window_sources_num # List[int]
        labels_per_step_list = window_labels    # List[np.ndarray]
        
        # Check that sizes match to avoid index errors
        current_window_len = time_series_steps.shape[0]
        
        # Sanity check the input lengths
        if len(sources_num_per_step) < current_window_len:
            logger.warning(f"Window source count list length ({len(sources_num_per_step)}) is less than time series length ({current_window_len}). Truncating window.")
            current_window_len = len(sources_num_per_step)
        
        if len(labels_per_step_list) < current_window_len:
            logger.warning(f"Window labels list length ({len(labels_per_step_list)}) is less than time series length ({current_window_len}). Truncating window.")
            current_window_len = len(labels_per_step_list)
        
        if current_window_len == 0:
            logger.error("Window has zero valid steps. Cannot evaluate.")
            return float('inf'), float('nan'), [], [], float('inf'), []  # Include pre-EKF loss in return
        
        total_loss = 0.0
        num_valid_steps_for_loss = 0
        
        # Add pre-EKF loss tracking for far-field case
        pre_ekf_total_loss = 0.0
        pre_ekf_num_valid_steps = 0
        
        # Check if we're dealing with far-field or near-field
        is_near_field = hasattr(self.trained_model, 'field_type') and self.trained_model.field_type.lower() == "near"
        
        # Use RMSPE loss for evaluation
        rmspe_criterion = RMSPELoss().to(device)
        
        # Initialize Extended Kalman Filters - one for each potential source
        # Use the system's trajectory configuration to determine the appropriate model
        max_sources = self.config.system_model.M
        ekf_filters = []
        
        # Create EKF instances
        for i in range(max_sources):
            ekf_filter = ExtendedKalmanFilter1D.create_from_config(
                self.config, 
                trajectory_type=self.config.trajectory.trajectory_type
            )
            ekf_filters.append(ekf_filter)
        
        # Get current eta value from system model
        current_eta = self.system_model.params.eta
        
        logger.info(f"Initialized {max_sources} EKF instances for window evaluation (eta={current_eta:.4f})")
        
        # Track the EKF predictions, covariances and innovations for each step and source
        ekf_predictions = []
        ekf_covariances = []
        ekf_innovations = []  # New list to track innovations
        ekf_kalman_gains = []  # New list to track Kalman gains
        ekf_kalman_gain_times_innovation = []  # New list to track K*y
        ekf_y_s_inv_y = []  # New list to track y*(S^-1)*y
        
        # Process each step in window
        for step in range(current_window_len):
            try:
                # Extract data for this step
                step_data_tensor = time_series_steps[step:step+1].to(device)  # Shape: [1, N, T] (add batch dim for model)
                num_sources_this_step = sources_num_per_step[step]
                
                # Skip if no sources
                if num_sources_this_step <= 0:
                    ekf_predictions.append([])
                    ekf_covariances.append([])
                    ekf_innovations.append([])
                    ekf_kalman_gains.append([])
                    ekf_kalman_gain_times_innovation.append([])
                    ekf_y_s_inv_y.append([])
                    continue
                
                # Get ground truth labels for this step
                true_angles_this_step = labels_per_step_list[step][:num_sources_this_step]
                
                # Forward pass through model
                self.trained_model.eval() # Ensure model is in eval mode for this evaluation part
                with torch.no_grad():
                    if not is_near_field:
                        # Model expects num_sources as int or 0-dim tensor
                        angles_pred, _, _ = self.trained_model(step_data_tensor, num_sources_this_step)
                        
                        # Convert predictions to numpy for EKF processing
                        angles_pred_np = angles_pred.cpu().numpy().flatten()[:num_sources_this_step]
                        
                        # Calculate pre-EKF loss (raw model predictions)
                        pre_ekf_angles_pred = angles_pred.view(1, -1)[:, :num_sources_this_step]
                        true_angles_tensor = torch.tensor(true_angles_this_step, device=device).unsqueeze(0)
                        
                        # Get optimal permutation for model predictions
                        model_perm = self._get_optimal_permutation(angles_pred_np, true_angles_this_step)
                        angles_pred_np = angles_pred_np[model_perm]
                        pre_ekf_angles_pred = pre_ekf_angles_pred[:, model_perm]
                        
                        # Calculate pre-EKF loss with reordered predictions
                        pre_ekf_loss = rmspe_criterion(pre_ekf_angles_pred, true_angles_tensor)
                        pre_ekf_total_loss += pre_ekf_loss.item()
                        pre_ekf_num_valid_steps += 1
                        
                        # Initialize EKF state if this is the first step
                        if step == 0:
                            if is_first_window:
                                # Initialize with true angles for first window
                                for i in range(num_sources_this_step):
                                    ekf_filters[i].initialize_state(true_angles_this_step[i])
                            else:
                                # Initialize with last predictions from previous window
                                if last_ekf_predictions and len(last_ekf_predictions[-1]) >= num_sources_this_step and \
                                   len(last_ekf_covariances) > 0 and len(last_ekf_covariances[-1]) >= num_sources_this_step:
                                    # Get the last predictions and calculate their optimal permutation
                                    last_predictions_pre_perm = np.array(last_ekf_predictions[-1])[:num_sources_this_step]
                                    last_perm = self._get_optimal_permutation(last_predictions_pre_perm, true_angles_this_step)
                                    last_predictions = last_predictions_pre_perm[last_perm]
                                    
                                    # Get the last covariances and apply the same permutation
                                    last_covariances_pre_perm = np.array(last_ekf_covariances[-1])[:num_sources_this_step]
                                    last_covariances = last_covariances_pre_perm[last_perm]
                                    
                                    for i in range(num_sources_this_step):
                                        ekf_filters[i].initialize_state(last_predictions.flatten()[:num_sources_this_step][i])
                                        ekf_filters[i].P = last_covariances.flatten()[:num_sources_this_step][i]  # Update the covariance matrix
                                else:
                                    # Fallback to true angles if no valid last predictions
                                    logger.warning("No valid last predictions or covariances available, falling back to true angles")
                                    for i in range(num_sources_this_step):
                                        ekf_filters[i].initialize_state(true_angles_this_step[i])
                        
                        # EKF update for each source
                        step_predictions = []
                        step_covariances = []
                        step_innovations = []  # New list for this step's innovations
                        step_kalman_gains = []  # New list for this step's Kalman gains
                        step_kalman_gain_times_innovation = []  # New list for this step's K*y
                        step_y_s_inv_y = []  # New list for this step's y*(S^-1)*y
                        for i in range(num_sources_this_step):
                            # Predict and update in one step
                            predicted_angle, updated_angle, innovation, kalman_gain, kalman_gain_times_innovation, y_s_inv_y = ekf_filters[i].predict_and_update(
                                measurement=angles_pred_np.flatten()[:num_sources_this_step][i], true_state=true_angles_this_step[i]
                            )
                            
                            # Store prediction, covariance and innovation
                            step_predictions.append(updated_angle)
                            step_covariances.append(ekf_filters[i].P)
                            step_innovations.append(innovation)
                            step_kalman_gains.append(kalman_gain)
                            step_kalman_gain_times_innovation.append(kalman_gain_times_innovation)
                            step_y_s_inv_y.append(y_s_inv_y)
                        
                        # Store predictions, covariances and innovations for this step
                        ekf_predictions.append(step_predictions)
                        ekf_covariances.append(step_covariances)
                        ekf_innovations.append(step_innovations)
                        ekf_kalman_gains.append(step_kalman_gains)
                        ekf_kalman_gain_times_innovation.append(step_kalman_gain_times_innovation)
                        ekf_y_s_inv_y.append(step_y_s_inv_y)
                        
                        # Create tensor from EKF predictions for loss calculation
                        ekf_angles_pred = torch.tensor(step_predictions, device=device).unsqueeze(0)
                        true_angles_tensor = torch.tensor(true_angles_this_step, device=device).unsqueeze(0)
                        
                        # Calculate loss using EKF predictions
                        loss = rmspe_criterion(ekf_angles_pred, true_angles_tensor)
                    else:
                        # Near-field case - not supported
                        error_msg = "Near-field processing is not supported in this project. Please use far-field models only."
                        logger.error(error_msg)
                        raise NotImplementedError(error_msg)
                
                total_loss += loss.item()
                num_valid_steps_for_loss += 1
            except Exception as e:
                logger.warning(f"Error processing step {step}: {e}")
                # Skip this step on error
                continue
        
        # Calculate average loss and covariance for the window
        if num_valid_steps_for_loss > 0:
            avg_loss = total_loss / num_valid_steps_for_loss
            
            # Calculate pre-EKF average loss for far-field case
            avg_pre_ekf_loss = 0.0
            if not is_near_field and pre_ekf_num_valid_steps > 0:
                avg_pre_ekf_loss = pre_ekf_total_loss / pre_ekf_num_valid_steps
            elif is_near_field:
                # Near-field case - not supported
                error_msg = "Near-field processing is not supported in this project. Please use far-field models only."
                logger.error(error_msg)
                raise NotImplementedError(error_msg)
            
            # Calculate average covariance across all sources and steps
            avg_window_cov = 0.0
            total_cov_points = 0
            
            if len(ekf_covariances) > 0 and any(len(step_covs) > 0 for step_covs in ekf_covariances):
                # Calculate overall average covariance across all sources and steps
                for step_covs in ekf_covariances:
                    if len(step_covs) > 0:
                        avg_window_cov += sum(step_covs)
                        total_cov_points += len(step_covs)
            
            # Calculate the overall average covariance across all sources and steps
            if total_cov_points > 0:
                avg_window_cov /= total_cov_points
            else:
                avg_window_cov = float('nan')  # No valid covariance points
            
            # Log window summary with columnar format
            self._log_window_summary(avg_pre_ekf_loss, avg_loss, avg_window_cov, current_eta, is_near_field, trajectory_idx, window_idx)
            
            return avg_loss, avg_window_cov, ekf_predictions, ekf_covariances, avg_pre_ekf_loss, ekf_innovations, ekf_kalman_gains, ekf_kalman_gain_times_innovation, ekf_y_s_inv_y
        else:
            logger.warning("No valid steps with sources found in the window for loss calculation.")
            return float('inf'), float('nan'), [], [], float('inf'), [], [], [], []  # Include pre-EKF loss in return

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

    def _log_window_summary(
        self,
        avg_pre_ekf_loss: float,
        avg_loss: float,
        avg_window_cov: float,
        current_eta: float,
        is_near_field: bool,
        trajectory_idx: int = 0,
        window_idx: int = 0
    ) -> None:
        """
        Log window summary results in a columnar format similar to evaluation results.
        
        Args:
            avg_pre_ekf_loss: Average pre-EKF loss for the window
            avg_loss: Average EKF loss for the window
            avg_window_cov: Average covariance for the window
            current_eta: Current eta value
            is_near_field: Whether this is near field scenario
            trajectory_idx: Index of the current trajectory
            window_idx: Index of the current window within the trajectory
        """
        print(f"\n{'WINDOW SUMMARY - WINDOW ' + str(window_idx) + ' TRAJECTORY ' + str(trajectory_idx):^100}")
        print("-"*100)
        print(f"{'Metric':<20} {'Loss Value':<20} {'Loss (degrees)':<25} {'Additional Info':<30}")
        print("-"*100)
        
        if not is_near_field:
            # Convert losses to degrees
            pre_ekf_loss_degrees = avg_pre_ekf_loss * 180 / np.pi
            ekf_loss_degrees = avg_loss * 180 / np.pi
            loss_difference_degrees = ekf_loss_degrees - pre_ekf_loss_degrees  # Negative = EKF better
            
            # Determine performance status
            ekf_improved = loss_difference_degrees < 0
            abs_difference = abs(loss_difference_degrees)
            improvement_percent = abs_difference / pre_ekf_loss_degrees * 100
            best_method = "EKF" if ekf_improved else "SubspaceNet"
            best_loss_degrees = min(pre_ekf_loss_degrees, ekf_loss_degrees)
            
            # Display individual losses
            print(f"{'SubspaceNet Loss':<20} {avg_pre_ekf_loss:<20.6f} {pre_ekf_loss_degrees:<25.6f} {f'eta: {current_eta:.4f}, w: {window_idx}, t: {trajectory_idx}':<30}")
            print(f"{'EKF Loss':<20} {avg_loss:<20.6f} {ekf_loss_degrees:<25.6f} {f'Avg Cov: {avg_window_cov:.2e}, w: {window_idx}, t: {trajectory_idx}':<30}")
            
            # Display comparison result
            if ekf_improved:
                status_icon = "✓"
                status_text = "EKF OVERCOMES SubspaceNet"
                change_text = f"↓ {abs_difference:.4f}° ({improvement_percent:.1f}% better)"
            else:
                status_icon = "✗"
                status_text = "SubspaceNet BETTER than EKF"
                change_text = f"↑ {abs_difference:.4f}° ({improvement_percent:.1f}% worse)"
            
            print(f"{'WINNER':<20} {best_method:<20} {best_loss_degrees:<25.6f} {status_icon + ' ' + status_text:<30}")
            print(f"{'Performance':<20} {change_text:<45} {'w: ' + str(window_idx) + ', t: ' + str(trajectory_idx):<30}")
            print("-" * 100)
        else:
            # Near field - only EKF loss (no SubspaceNet comparison available)
            ekf_loss_degrees = avg_loss * 180 / np.pi
            print(f"{'EKF Loss':<20} {avg_loss:<20.6f} {ekf_loss_degrees:<25.6f} {f'eta: {current_eta:.4f}, w: {window_idx}, t: {trajectory_idx}':<30}")
            print(f"{'Mode':<20} {'NEAR FIELD':<20} {'(No SubspaceNet comparison)':<25} {'w: ' + str(window_idx) + ', t: ' + str(trajectory_idx):<30}")
            print("-" * 100)

    def _save_model_state(self, model, model_type=None):
        """
        Save model state dictionary to file.
        
        Args:
            model: Model to save
            model_type: Type identifier for filename
            
        Returns:
            Path to saved model file
        """
        if model_type is None:
            model_type = "model"
        
        # Create timestamp for unique filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create model save directory
        model_save_dir = self.output_dir / "checkpoints"
        model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        model_save_path = model_save_dir / f"saved_{model_type}_{timestamp}.pt"
        
        # Save only the state_dict, not the entire model
        try:
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Model saved successfully to {model_save_path}")
            return model_save_path
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return None

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
            "window_labels": all_window_labels[0]  # Take first trajectory's labels
        }

    def _calculate_frobenius_norm_per_window(self, window_ekf_covariances):
        """Calculate Frobenius norm of covariance matrices for each window."""
        frobenius_norms = []
        for window_covs in window_ekf_covariances:
            if window_covs:
                # Flatten the nested structure: window_covs contains lists of covariances per step
                all_covs = []
                for step_covs in window_covs:
                    if step_covs and len(step_covs) > 0:
                        all_covs.extend([abs(cov) for cov in step_covs if cov is not None])
                
                if all_covs:
                    window_norm = np.mean(all_covs)
                    frobenius_norms.append(window_norm)
                else:
                    frobenius_norms.append(0.0)
            else:
                frobenius_norms.append(0.0)
        return frobenius_norms

    def _plot_single_trajectory_results(self, trajectory_results, trajectory_idx):
        """
        Plot results for a single trajectory including loss difference and innovation.
        
        Args:
            trajectory_results: Dictionary containing single trajectory results
            trajectory_idx: Index of the trajectory being plotted
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import datetime
            
            # Create timestamp and plot directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_dir = self.output_dir / "plots" / "single_trajectories"
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract data from trajectory results
            pre_ekf_losses = np.array(trajectory_results['pre_ekf_losses'])  # Shape: (30,)
            window_losses = np.array(trajectory_results['window_losses'])    # Shape: (30,) - post EKF losses
            ekf_innovations = np.array(trajectory_results['ekf_innovations']) # Shape: (30, 5, 3)
            window_eta_values = trajectory_results['window_eta_values']
            
            # Calculate loss difference: pre_ekf_loss - post_ekf_loss (positive means EKF improved)
            loss_difference = pre_ekf_losses - window_losses
            
            # Calculate average innovation magnitude per window
            # Aggregate across steps (5) and sources (3) to get one value per window (30)
            avg_innovations_per_window = []
            for window_idx in range(ekf_innovations.shape[0]):  # 30 windows
                window_innovations = []
                for step_idx in range(ekf_innovations.shape[1]):  # 5 steps
                    for source_idx in range(ekf_innovations.shape[2]):  # 3 sources
                        innovation_val = ekf_innovations[window_idx, step_idx, source_idx]
                        window_innovations.append(abs(innovation_val))  # Use absolute value
                
                if window_innovations:
                    avg_innovations_per_window.append(np.mean(window_innovations))
                else:
                    avg_innovations_per_window.append(0)
            
            avg_innovations_per_window = np.array(avg_innovations_per_window)
            
            # Find indices where eta changes
            eta_changes = []
            eta_values = []
            for i in range(1, len(window_eta_values)):
                if abs(window_eta_values[i] - window_eta_values[i-1]) > 1e-6:
                    eta_changes.append(i)
                    eta_values.append(window_eta_values[i])
            
            # Create figure with 2 subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Window indices
            window_indices = np.arange(len(window_losses))
            
            # Plot 1: Loss difference (Pre-EKF Loss - Post-EKF Loss)
            ax1.plot(window_indices, loss_difference, 'g-', marker='o', linewidth=2, markersize=6, label='Pre-EKF Loss - Post-EKF Loss')
            ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='No Improvement Line')
            
            # Add eta change markers
            for idx, eta in zip(eta_changes, eta_values):
                ax1.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                ax1.text(idx, ax1.get_ylim()[1], f'η={eta:.3f}', rotation=90, verticalalignment='top')
            
            ax1.set_xlabel('Window Index')
            ax1.set_ylabel('Loss Difference')
            ax1.set_title(f'Trajectory {trajectory_idx}: Pre-EKF Loss - Post-EKF Loss vs Window Index\n(Positive = EKF Improved, Negative = EKF Worse)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Innovation magnitude
            ax2.plot(window_indices, avg_innovations_per_window, 'b-', marker='s', linewidth=2, markersize=6, label='Average Innovation Magnitude')
            
            # Add eta change markers
            for idx, eta in zip(eta_changes, eta_values):
                ax2.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                ax2.text(idx, ax2.get_ylim()[1], f'η={eta:.3f}', rotation=90, verticalalignment='top')
            
            ax2.set_xlabel('Window Index')
            ax2.set_ylabel('Average Innovation Magnitude')
            ax2.set_title(f'Trajectory {trajectory_idx}: EKF Innovation Magnitude vs Window Index\n|Innovation| = |z_k - H x̂_k|k-1| = |measurement - prediction|')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Adjust layout and save
            plt.tight_layout()
            plot_path = plot_dir / f"single_trajectory_{trajectory_idx}_{timestamp}.png"
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Single trajectory plot saved: {plot_path}")
            
        except ImportError:
            logger.warning("matplotlib not available for single trajectory plotting")
        except Exception as e:
            logger.error(f"Error plotting single trajectory results for trajectory {trajectory_idx}: {e}")
            logger.debug("Error details:", exc_info=True)

    # Placeholder methods for plotting - these call the full implementations in core.py for now
    def _plot_online_learning_results(self, window_losses, window_covariances, window_eta_values, window_updates, window_pre_ekf_losses, window_labels, ekf_covariances, frobenius_norms, ekf_kalman_gains=None, ekf_kalman_gain_times_innovation=None, ekf_y_s_inv_y=None):
        """
        Plot online learning results including plots as a function of eta.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import datetime
            
            # Create timestamp and plot directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_dir = self.output_dir / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            def set_adjusted_ylim(ax, data, padding=0.1):
                """Helper function to set y limits excluding first point"""
                if len(data) > 1:
                    data_without_first = data[1:]
                    ymin = min(data_without_first)
                    ymax = max(data_without_first)
                    range_y = ymax - ymin
                    ax.set_ylim([ymin - range_y * padding, ymax + range_y * padding])
            
            # Find indices where eta changes
            eta_changes = []
            eta_values = []
            for i in range(1, len(window_eta_values)):
                if abs(window_eta_values[i] - window_eta_values[i-1]) > 1e-6:
                    eta_changes.append(i)
                    eta_values.append(window_eta_values[i])
            
            # Create figure with multiple subplots (4x2 layout for 8 plots)
            fig = plt.figure(figsize=(20, 20))
            
            # 1. Plot loss vs window index
            ax1 = fig.add_subplot(4, 2, 1)
            x = np.arange(len(window_losses))[1:]  # Start from second sample
            ax1.plot(x, np.array(window_losses)[1:], 'b-', marker='o', label='EKF Loss')
            ax1.plot(x, np.array(window_pre_ekf_losses)[1:], 'r-', marker='s', label='SubspaceNet Loss')
            
            # Set y-axis limit to 0.14 for loss plot only
            ax1.set_ylim([None, 0.14])
            
            # Add eta change markers (adjusted for starting from second sample)
            for idx, eta in zip(eta_changes, eta_values):
                if idx >= 1:  # Only show markers from second sample onwards
                    ax1.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                    ax1.text(idx, 0.13, f'η={eta:.3f}', rotation=90, verticalalignment='top')
            
            # Add labels and title
            ax1.set_xlabel('Window Index')
            ax1.set_ylabel('Loss')
            ax1.set_title('Loss vs Window Index (Starting from Window 1)\nRMSPE = √(1/N * Σ(θ_pred - θ_true)²)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Plot EKF improvement vs window index (reversed: SubspaceNet - EKF)
            ax2 = fig.add_subplot(4, 2, 2)
            x = np.arange(len(window_losses))[1:]  # Start from second sample
            improvement = np.array(window_pre_ekf_losses)[1:] - np.array(window_losses)[1:]  # Reversed calculation, starting from second sample
            ax2.plot(x, improvement, 'g-', marker='o')
            
            # Add eta change markers (adjusted for starting from second sample)
            for idx, eta in zip(eta_changes, eta_values):
                if idx >= 1:  # Only show markers from second sample onwards
                    ax2.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                    ax2.text(idx, ax2.get_ylim()[1], f'η={eta:.3f}', rotation=90, verticalalignment='top')
            
            # Add labels and title
            ax2.set_xlabel('Window Index')
            ax2.set_ylabel('Loss Difference')
            ax2.set_title('SubspaceNet Loss - EKF Loss vs Window Index (Starting from Window 1)\nImprovement = L_SubspaceNet - L_EKF')
            ax2.grid(True, alpha=0.3)
            
            # 3. Plot covariance vs window index
            ax3 = fig.add_subplot(4, 2, 3)
            x = np.arange(len(window_covariances))[1:]  # Start from second sample
            ax3.plot(x, np.array(window_covariances)[1:], 'g-', marker='o', label='Average Covariance')
            
            # Add eta change markers (adjusted for starting from second sample)
            for idx, eta in zip(eta_changes, eta_values):
                if idx >= 1:  # Only show markers from second sample onwards
                    ax3.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                    ax3.text(idx, max(np.array(window_covariances)[1:]), f'η={eta:.3f}', rotation=90, verticalalignment='top')
            
            # Highlight windows where model was updated (adjusted for starting from second sample)
            if window_updates:
                update_indices = [i for i, updated in enumerate(window_updates) if updated and i >= 1]
                update_covs = [window_covariances[i] for i in update_indices]
                ax3.scatter(update_indices, update_covs, color='r', s=80, marker='o', label='Model Updated')
            
            # Add labels and title
            ax3.set_xlabel('Window Index')
            ax3.set_ylabel('Average Covariance')
            ax3.set_title('Covariance vs Window Index (Starting from Window 1)\nP_k|k = (I - K_k H) P_k|k-1')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Plot average innovation magnitude vs window index
            ax4 = fig.add_subplot(4, 2, 4)
            x = np.arange(len(window_losses))[1:]  # Start from second sample
            
            # Calculate average innovation magnitude per window from the results
            avg_innovations = []
            if "ekf_innovations" in self.results.get("online_learning", {}):
                ekf_innovations = self.results["online_learning"]["ekf_innovations"]
                for window_innovations in ekf_innovations:
                    window_avg = []
                    for step_innovations in window_innovations:
                        if step_innovations:  # Check if there are any innovations in this step
                            window_avg.extend([abs(inn) for inn in step_innovations])  # Use absolute values
                    if window_avg:
                        avg_innovations.append(np.mean(window_avg))
                    else:
                        avg_innovations.append(0)
            else:
                # Fallback: create zeros if no innovation data available
                avg_innovations = [0] * len(window_losses)
            
            # Ensure we have the right number of points and start from second sample
            if len(avg_innovations) != len(window_losses):
                avg_innovations = avg_innovations[:len(window_losses)] if len(avg_innovations) > len(window_losses) else avg_innovations + [0] * (len(window_losses) - len(avg_innovations))
            
            ax4.plot(x, np.array(avg_innovations)[1:], 'b-', marker='o', linewidth=2, markersize=6, label='Average Innovation Magnitude')
            
            # Add eta change markers (adjusted for starting from second sample)
            for idx, eta in zip(eta_changes, eta_values):
                if idx >= 1:  # Only show markers from second sample onwards
                    ax4.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                    ax4.text(idx, ax4.get_ylim()[1], f'η={eta:.3f}', rotation=90, verticalalignment='top')
            
            # Add labels and title
            ax4.set_xlabel('Window Index')
            ax4.set_ylabel('Average Innovation')
            ax4.set_title('|EKF Innovation|  vs Window Index (Starting from Window 1)\nInnovation = z_k - H x̂_k|k-1 = measurement - prediction')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # 5. Plot average Kalman gain vs window index
            if ekf_kalman_gains is not None:
                ax5 = fig.add_subplot(4, 2, 5)
                # Calculate average Kalman gain per window
                avg_kalman_gains = []
                for window_gains in ekf_kalman_gains:
                    window_avg = []
                    for step_gains in window_gains:
                        if step_gains:  # Check if there are any gains in this step
                            window_avg.extend(step_gains)
                    if window_avg:
                        avg_kalman_gains.append(np.mean(window_avg))
                    else:
                        avg_kalman_gains.append(0)
                
                x = np.arange(len(avg_kalman_gains))[1:]  # Start from second sample
                ax5.plot(x, np.array(avg_kalman_gains)[1:], 'purple', marker='d', label='Average Kalman Gain')
                
                # Add eta change markers (adjusted for starting from second sample)
                for idx, eta in zip(eta_changes, eta_values):
                    if idx >= 1:  # Only show markers from second sample onwards
                        ax5.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                        ax5.text(idx, ax5.get_ylim()[1], f'η={eta:.3f}', rotation=90, verticalalignment='top')
            
                # Add labels and title
                ax5.set_xlabel('Window Index')
                ax5.set_ylabel('Average Kalman Gain')
                ax5.set_title('Average Kalman Gain vs Window Index (Starting from Window 1)\nK_k = P_k|k-1 H^T (H P_k|k-1 H^T + R)^-1')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
            
            # 6. Plot average K*y vs window index
            if ekf_kalman_gain_times_innovation is not None:
                ax6 = fig.add_subplot(4, 2, 6)
                # Calculate average K*y per window
                avg_k_times_y = []
                for window_k_times_y in ekf_kalman_gain_times_innovation:
                    window_avg = []
                    for step_k_times_y in window_k_times_y:
                        if step_k_times_y:  # Check if there are any values in this step
                            window_avg.extend(step_k_times_y)
                    if window_avg:
                        avg_k_times_y.append(np.mean(window_avg))
                    else:
                        avg_k_times_y.append(0)
                
                x = np.arange(len(avg_k_times_y))[1:]  # Start from second sample
                ax6.plot(x, np.array(avg_k_times_y)[1:], 'orange', marker='v', label='Average K*Innovation')
                
                # Add eta change markers (adjusted for starting from second sample)
                for idx, eta in zip(eta_changes, eta_values):
                    if idx >= 1:  # Only show markers from second sample onwards
                        ax6.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                        ax6.text(idx, ax6.get_ylim()[1], f'η={eta:.3f}', rotation=90, verticalalignment='top')
            
                # Add labels and title
                ax6.set_xlabel('Window Index')
                ax6.set_ylabel('Average K*Innovation')
                ax6.set_title('Average Kalman Gain × Innovation vs Window Index (Starting from Window 1)\nK_k × ν_k = K_k × (z_k - H x̂_k|k-1)')
                ax6.legend()
                ax6.grid(True, alpha=0.3)
            
            # 7. Plot average y*(S^-1)*y vs window index
            if ekf_y_s_inv_y is not None:
                ax7 = fig.add_subplot(4, 2, 7)
                # Calculate average y*(S^-1)*y per window
                avg_y_s_inv_y = []
                for window_y_s_inv_y in ekf_y_s_inv_y:
                    window_avg = []
                    for step_y_s_inv_y in window_y_s_inv_y:
                        if step_y_s_inv_y:  # Check if there are any values in this step
                            window_avg.extend(step_y_s_inv_y)
                    if window_avg:
                        avg_y_s_inv_y.append(np.mean(window_avg))
                    else:
                        avg_y_s_inv_y.append(0)
                
                x = np.arange(len(avg_y_s_inv_y))[1:]  # Start from second sample
                ax7.plot(x, np.array(avg_y_s_inv_y)[1:], 'red', marker='^', label='Average y*(S^-1)*y')
                
                # Add eta change markers (adjusted for starting from second sample)
                for idx, eta in zip(eta_changes, eta_values):
                    if idx >= 1:  # Only show markers from second sample onwards
                        ax7.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                        ax7.text(idx, ax7.get_ylim()[1], f'η={eta:.3f}', rotation=90, verticalalignment='top')
                
                # Add labels and title
                ax7.set_xlabel('Window Index')
                ax7.set_ylabel('Average y*(S^-1)*y')
                ax7.set_title('Average Innovation Covariance Metric vs Window Index (Starting from Window 1)\ny*(S^-1)*y = ν^T S^-1 ν')
                ax7.legend()
                ax7.grid(True, alpha=0.3)
            
            # Adjust layout and save
            plt.tight_layout()
            plot_path = plot_dir / f"online_learning_results_{timestamp}.png"
            plt.savefig(plot_path)
            plt.close()
            
            # Plot Frobenius norms with adjusted y-axis
            self._plot_frobenius_norms(frobenius_norms, window_eta_values, plot_dir, timestamp, exclude_first=True)
            
            # Plot online learning trajectory
            self._plot_online_learning_trajectory(window_labels, plot_dir, timestamp)
            
        except ImportError:
            logger.warning("matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting online learning results: {e}")
            logger.debug("Error details:", exc_info=True)
        
    def _plot_online_learning_trajectory(self, window_labels, plot_dir, timestamp):
        """
        Plot the full trajectory across all windows for online learning.
        
        Args:
            window_labels: List of labels for each window, where each window contains a list of numpy arrays
            plot_dir: Directory to save the plots
            timestamp: Timestamp for the plot filename
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Flatten all labels across all windows to get the complete trajectory
            all_angles = []
            all_distances = []
            window_indices = []  # Track which window each step belongs to
            
            for window_idx, window_label_list in enumerate(window_labels):
                for step_idx, step_labels in enumerate(window_label_list):
                    # step_labels is a numpy array containing [angle1, angle2, ..., angleM] (in radians)
                    # For far-field, we assume distances are constant (e.g., 30m)
                    # For near-field, distances would be included in the labels
                    
                    # Convert radians to degrees for plotting
                    angles_deg = step_labels * (180.0 / np.pi)
                    all_angles.append(angles_deg)
                    
                    # Use the same range increment mechanism as in generate_trajectories
                    # Start at 20m and increment by 1m each step
                    global_step = len(all_angles) - 1  # Current global step index
                    base_distance = 20.0 + global_step * 1.0  # Increment by 1m each step
                    
                    num_sources = len(step_labels)
                    distances = np.full(num_sources, base_distance)
                    all_distances.append(distances)
                    
                    # Track window index
                    window_indices.append(window_idx)
            
            if not all_angles:
                logger.warning("No trajectory data available for plotting")
                return
            
            # Find the maximum number of sources across all steps
            max_sources = max(len(angles) for angles in all_angles)
            total_steps = len(all_angles)
            
            # Pad arrays to have consistent dimensions
            padded_angles = np.full((total_steps, max_sources), np.nan)
            padded_distances = np.full((total_steps, max_sources), np.nan)
            
            for step_idx, (angles, distances) in enumerate(zip(all_angles, all_distances)):
                num_sources = len(angles)
                padded_angles[step_idx, :num_sources] = angles
                padded_distances[step_idx, :num_sources] = distances
            
            # Create the trajectory plot
            plt.figure(figsize=(12, 10))
            
            # Plot each source trajectory
            for s in range(max_sources):
                # Get valid data for this source (some steps might have fewer sources)
                valid_mask = ~np.isnan(padded_angles[:, s])
                if np.any(valid_mask):
                    angles_rad = padded_angles[valid_mask, s] * (np.pi / 180.0)  # Convert back to radians for plotting
                    distances = padded_distances[valid_mask, s]
                    
                    # Convert from polar to Cartesian coordinates
                    x = distances * np.cos(angles_rad)
                    y = distances * np.sin(angles_rad)
                    
                    # Plot trajectory
                    plt.plot(x, y, '-o', markersize=4, label=f'Source {s+1}')
                    
                    # Mark start and end points
                    if len(x) > 0:
                        plt.plot(x[0], y[0], 'go', markersize=8)  # Green for start
                        plt.plot(x[-1], y[-1], 'ro', markersize=8)  # Red for end
            
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
            plt.title(f'Online Learning Full Trajectory (T={total_steps}, Sources={max_sources}, Windows={len(window_labels)})')
            plt.legend()
            
            # Add window boundary markers if there are multiple windows
            if len(window_labels) > 1:
                # Find window boundaries
                window_boundaries = []
                current_window = window_indices[0]
                for i, window_idx in enumerate(window_indices):
                    if window_idx != current_window:
                        window_boundaries.append(i)
                        current_window = window_idx
                
                # Add vertical lines for window boundaries
                for boundary_idx in window_boundaries:
                    if boundary_idx < len(all_angles):
                        # Get the position at the boundary
                        angles_at_boundary = all_angles[boundary_idx]
                        distances_at_boundary = all_distances[boundary_idx]
                        
                        # Plot boundary markers for each source
                        for s in range(len(angles_at_boundary)):
                            if not np.isnan(angles_at_boundary[s]):
                                angle_rad = angles_at_boundary[s] * (np.pi / 180.0)
                                distance = distances_at_boundary[s]
                                x = distance * np.cos(angle_rad)
                                y = distance * np.sin(angle_rad)
                                plt.plot(x, y, 'ks', markersize=10, markerfacecolor='none', markeredgewidth=2)
            
            # Save the plot
            plot_path = plot_dir / f"online_learning_trajectory_{timestamp}.png"
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Online learning trajectory plot saved to {plot_dir}:")
            logger.info(f"  - Trajectory plot: {plot_path.name}")
            
        except ImportError:
            logger.warning("matplotlib not available for plotting online learning trajectory")
        except Exception as e:
            logger.error(f"Error plotting online learning trajectory: {e}")
        
    def _plot_frobenius_norms(self, frobenius_norms, window_eta_values, plot_dir, timestamp, exclude_first=False):
        """
        Plot Frobenius norms of covariance matrices per window.
        
        Args:
            frobenius_norms: List of Frobenius norms per window
            window_eta_values: List of eta values per window
            plot_dir: Directory to save the plots
            timestamp: Timestamp for the plot filename
            exclude_first: Whether to exclude the first point from the plot
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            def set_adjusted_ylim(ax, data, padding=0.1):
                """Helper function to set y limits excluding first point"""
                if len(data) > 1:
                    data_without_first = data[1:]
                    ymin = min(data_without_first)
                    ymax = max(data_without_first)
                    range_y = ymax - ymin
                    ax.set_ylim([ymin - range_y * padding, ymax + range_y * padding])
            
            # Find indices where eta changes
            eta_changes = []
            eta_values = []
            for i in range(1, len(window_eta_values)):
                if abs(window_eta_values[i] - window_eta_values[i-1]) > 1e-6:
                    eta_changes.append(i)
                    eta_values.append(window_eta_values[i])
            
            # Create figure for Frobenius norms
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot Frobenius norm vs window index
            x = np.arange(len(frobenius_norms))
            ax1.plot(x, frobenius_norms, 'b-', marker='o', linewidth=2, markersize=6, label='Frobenius Norm')
            
            # Add eta change markers
            for idx, eta in zip(eta_changes, eta_values):
                ax1.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
                ax1.text(idx, max(frobenius_norms[1:]), f'η={eta:.3f}', rotation=90, verticalalignment='top')
            
            # Set y-axis limits excluding first point if requested
            if exclude_first:
                set_adjusted_ylim(ax1, frobenius_norms)
            
            # Add labels and title
            ax1.set_xlabel('Window Index')
            ax1.set_ylabel('Frobenius Norm')
            ax1.set_title('Covariance Matrix Frobenius Norm vs Window Index' + 
                         (' (Excluding Initial Value)' if exclude_first else ''))
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot Frobenius norm vs eta
            ax2.plot(window_eta_values, frobenius_norms, 'b-', marker='o', linewidth=2, markersize=6, label='Frobenius Norm')
            
            # Add eta change markers
            for eta in eta_values:
                ax2.scatter([eta], [frobenius_norms[window_eta_values.index(eta)]], color='red', marker='*', s=200, zorder=5)
                ax2.text(eta, max(frobenius_norms[1:]), f'η={eta:.3f}', rotation=90, verticalalignment='top')
            
            # Set y-axis limits excluding first point if requested
            if exclude_first:
                set_adjusted_ylim(ax2, frobenius_norms)
            
            # Add labels and title
            ax2.set_xlabel('Eta Value')
            ax2.set_ylabel('Frobenius Norm')
            ax2.set_title('Covariance Matrix Frobenius Norm vs Eta' + 
                         (' (Excluding Initial Value)' if exclude_first else ''))
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Adjust layout and save
            plt.tight_layout()
            plot_path = plot_dir / f"frobenius_norms_{timestamp}.png"
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Frobenius norm plots saved to {plot_path}")
            
        except ImportError:
            logger.warning("matplotlib not available for plotting Frobenius norms")
        except Exception as e:
            logger.error(f"Error plotting Frobenius norms: {e}")
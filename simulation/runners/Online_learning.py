from ast import Not
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
from utils.plotting import plot_single_trajectory_results, plot_online_learning_results, plot_online_learning_trajectory
from utils.utils import log_window_summary, save_model_state
from simulation.kalman_filter import KalmanFilter1D, BatchKalmanFilter1D, BatchExtendedKalmanFilter1D
from DCD_MUSIC.src.metrics.rmspe_loss import RMSPELoss
from DCD_MUSIC.src.metrics.rmape_loss import RMAPELoss
from DCD_MUSIC.src.signal_creation import Samples
from DCD_MUSIC.src.evaluation import get_model_based_method, evaluate_model_based
from simulation.kalman_filter.extended import ExtendedKalmanFilter1D
from simulation.losses import KalmanInnovationLoss

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
        self.learning_start_window = None  # Track when learning started
        self.training_window_indices = []  # Track which windows were used for training
        
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
                    plot_single_trajectory_results(trajectory_result["online_learning_results"], trajectory_idx, self.output_dir)
                
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
            

            plot_online_learning_results(self.output_dir,
                window_losses=averaged_results_across_trajectories["window_losses"],
                window_covariances=averaged_results_across_trajectories["window_covariances"],
                window_eta_values=averaged_results_across_trajectories["window_eta_values"],
                window_updates=averaged_results_across_trajectories["window_updates"],
                window_pre_ekf_losses=averaged_results_across_trajectories["pre_ekf_losses"],
                window_labels=averaged_results_across_trajectories["window_labels"],
                ekf_covariances=averaged_results_across_trajectories["ekf_covariances"],
                ekf_kalman_gains=averaged_results_across_trajectories["ekf_kalman_gains"],
                ekf_kalman_gain_times_innovation=averaged_results_across_trajectories["ekf_kalman_gain_times_innovation"],
                ekf_y_s_inv_y=averaged_results_across_trajectories["ekf_y_s_inv_y"],
                # Online learning data
                online_window_losses=averaged_results_across_trajectories.get("online_window_losses", []),
                online_window_covariances=averaged_results_across_trajectories.get("online_window_covariances", []),
                online_pre_ekf_losses=averaged_results_across_trajectories.get("online_pre_ekf_losses", []),
                online_ekf_innovations=averaged_results_across_trajectories.get("online_ekf_innovations", []),
                online_ekf_kalman_gains=averaged_results_across_trajectories.get("online_ekf_kalman_gains", []),
                online_ekf_kalman_gain_times_innovation=averaged_results_across_trajectories.get("online_ekf_kalman_gain_times_innovation", []),
                online_ekf_y_s_inv_y=averaged_results_across_trajectories.get("online_ekf_y_s_inv_y", []),
                online_window_indices=averaged_results_across_trajectories.get("online_window_indices", []),
                # Training data
                training_window_losses=averaged_results_across_trajectories.get("training_window_losses", []),
                training_window_covariances=averaged_results_across_trajectories.get("training_window_covariances", []),
                training_pre_ekf_losses=averaged_results_across_trajectories.get("training_pre_ekf_losses", []),
                training_ekf_innovations=averaged_results_across_trajectories.get("training_ekf_innovations", []),
                training_ekf_kalman_gains=averaged_results_across_trajectories.get("training_ekf_kalman_gains", []),
                training_ekf_kalman_gain_times_innovation=averaged_results_across_trajectories.get("training_ekf_kalman_gain_times_innovation", []),
                training_ekf_y_s_inv_y=averaged_results_across_trajectories.get("training_ekf_y_s_inv_y", []),
                training_window_indices=averaged_results_across_trajectories.get("training_window_indices", []),
                learning_start_window=averaged_results_across_trajectories.get("learning_start_window", None),
                window_delta_losses=all_results[0].get("window_delta_losses", []) if all_results else [],
                window_delta_rmape_losses=all_results[0].get("window_delta_rmape_losses", []) if all_results else [],
                online_delta_losses=all_results[0].get("online_delta_losses", []) if all_results else [],
                online_delta_rmape_losses=all_results[0].get("online_delta_rmape_losses", []) if all_results else [],
                training_delta_losses=all_results[0].get("training_delta_losses", []) if all_results else [],
                training_delta_rmape_losses=all_results[0].get("training_delta_rmape_losses", []) if all_results else [],
                # Pre-EKF angle predictions
                window_pre_ekf_angles_pred=averaged_results_across_trajectories.get("window_pre_ekf_angles_pred", []),
                online_pre_ekf_angles_pred=averaged_results_across_trajectories.get("online_pre_ekf_angles_pred", []),
                training_pre_ekf_angles_pred=averaged_results_across_trajectories.get("training_pre_ekf_angles_pred", []),
                # EKF predictions
                window_ekf_predictions=averaged_results_across_trajectories.get("ekf_predictions", []),
                online_ekf_predictions=averaged_results_across_trajectories.get("online_ekf_predictions", []),
                training_ekf_predictions=averaged_results_across_trajectories.get("training_ekf_predictions", [])
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
            window_delta_losses = []  # Track delta losses for static model
            window_delta_rmape_losses = []  # Track delta RMAPE losses for static model
            window_pre_ekf_angles_pred = []  # Track pre-EKF angle predictions for static model
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
            online_window_indices = []  # Track which windows online model was evaluated on
            online_delta_losses = []  # Track delta losses for online model
            online_delta_rmape_losses = []  # Track delta RMAPE losses for online model
            online_pre_ekf_angles_pred = []  # Track pre-EKF angle predictions for online model
            
            # Training data tracking variables
            training_window_losses = []
            training_window_covariances = []
            training_window_pre_ekf_losses = []
            training_ekf_predictions = []
            training_ekf_covariances = []
            training_ekf_innovations = []
            training_ekf_kalman_gains = []
            training_ekf_kalman_gain_times_innovation = []
            training_ekf_y_s_inv_y = []
            training_window_indices = []
            training_delta_losses = []  # Track delta losses for training model
            training_delta_rmape_losses = []  # Track delta RMAPE losses for training model
            training_pre_ekf_angles_pred = []  # Track pre-EKF angle predictions for training model
            
            # EKF state tracking for online learning
            online_last_ekf_predictions = None
            online_last_ekf_covariances = None
            
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
                        
                        # Initialize online EKF state with static model's current state
                        online_last_ekf_predictions = last_ekf_predictions
                        online_last_ekf_covariances = last_ekf_covariances
                        logger.info(f"Initialized online EKF state with static model's state at window {window_idx}")
                        
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
                current_window_loss, avg_window_cov  , ekf_predictions, ekf_covariances, pre_ekf_loss, ekf_innovations, ekf_kalman_gains, ekf_kalman_gain_times_innovation, ekf_y_s_inv_y, delta_loss, pre_ekf_angles_pred, delta_rmape_loss = self._evaluate_window(
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
                window_delta_losses.append(delta_loss)  # Store delta loss
                window_delta_rmape_losses.append(delta_rmape_loss)  # Store delta RMAPE loss
                window_pre_ekf_angles_pred.append(pre_ekf_angles_pred)  # Store pre-EKF angle predictions
                
                # Update last predictions and covariances for next window
                last_ekf_predictions = ekf_predictions
                last_ekf_covariances = ekf_covariances
                
                logger.info(f"Window {window_idx}: Loss = {current_window_loss:.6f}, Cov = {avg_window_cov:.6f} (current eta={self.system_model.params.eta:.4f})")
                
                # Log both pre-EKF and EKF losses for comparison
                logger.info(f"Window {window_idx}: Pre-EKF Loss = {pre_ekf_loss:.6f}, EKF Loss = {current_window_loss:.6f}, Cov = {avg_window_cov:.6f} (eta={self.system_model.params.eta:.4f})")
                
                # Check if loss exceeds threshold (drift detected)
                if current_window_loss > loss_threshold:
                    logger.info(f"Drift detected in window {window_idx} (loss: {current_window_loss:.6f} > threshold: {loss_threshold:.6f})")
                    # window_update_flags.append(False)
                    # self.drift_detected = True
                    # Track when learning started
                    # if self.learning_start_window is None:
                    #     self.learning_start_window = window_idx
                    #     logger.info(f"Online learning started at window {window_idx}")
                else:
                    logger.info(f"No drift detected in window {window_idx} (loss: {current_window_loss:.6f} <= threshold: {loss_threshold:.6f})")
                    window_update_flags.append(False)
                    # Keep previous drift_detected state if no drift in current window
                
                # Dual model processing logic
                if self.drift_detected:
                    if self.learning_done:
                        # Online model finished training, evaluate it normally
                        logger.info(f"Evaluating online model (post-training) for window {window_idx}")
                        online_window_loss, online_avg_window_cov, online_ekf_preds, online_ekf_covs, online_pre_ekf_loss, online_ekf_inns, online_ekf_kgains, online_ekf_kg_times_inn, online_ekf_y_s_inv_y_val, online_delta_loss, online_pre_ekf_angles_pred, online_delta_rmape_loss = self._evaluate_window(
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
                        online_window_indices.append(window_idx)  # Track which window this evaluation was for
                        online_delta_losses.append(online_delta_loss)  # Store online delta loss
                        online_delta_rmape_losses.append(online_delta_rmape_loss)  # Store online delta RMAPE loss
                        online_pre_ekf_angles_pred.append(online_pre_ekf_angles_pred)  # Store online pre-EKF angle predictions
                        
                        logger.info(f"Online model - Window {window_idx}: Loss = {online_window_loss:.6f}, Cov = {online_avg_window_cov:.6f}")
                        
                        # Update online EKF state for next window
                        online_last_ekf_predictions = online_ekf_preds
                        online_last_ekf_covariances = online_ekf_covs
                    else:
                        # Online model still learning/adapting
                        logger.info(f"Training online model for window {window_idx}")
                        training_window_loss, training_avg_window_cov, training_ekf_preds, training_ekf_covs, training_pre_ekf_loss, training_ekf_inns, training_ekf_kgains, training_ekf_kg_times_inn, training_ekf_y_s_inv_y_val, training_delta_loss, training_pre_ekf_angles_pred, training_delta_rmape_loss = self._online_training_window(
                            time_series_single_window, 
                            sources_num_single_window_list, 
                            labels_single_window_list_of_arrays,
                            trajectory_idx,
                            window_idx,
                            is_first_window=(window_idx == 0),
                            last_ekf_predictions=online_last_ekf_predictions,
                            last_ekf_covariances=online_last_ekf_covariances
                        )
                        
                        # Store training results for analysis and plotting
                        training_window_losses.append(training_window_loss)
                        training_window_covariances.append(training_avg_window_cov)
                        training_window_pre_ekf_losses.append(training_pre_ekf_loss)
                        training_ekf_predictions.append(training_ekf_preds)
                        training_ekf_covariances.append(training_ekf_covs)
                        training_ekf_innovations.append(training_ekf_inns)
                        training_ekf_kalman_gains.append(training_ekf_kgains)
                        training_ekf_kalman_gain_times_innovation.append(training_ekf_kg_times_inn)
                        training_ekf_y_s_inv_y.append(training_ekf_y_s_inv_y_val)
                        training_window_indices.append(window_idx)
                        training_delta_losses.append(training_delta_loss)  # Store training delta loss
                        training_delta_rmape_losses.append(training_delta_rmape_loss)  # Store training delta RMAPE loss
                        training_pre_ekf_angles_pred.append(training_pre_ekf_angles_pred)  # Store training pre-EKF angle predictions
                        
                        logger.info(f"Online training - Window {window_idx}: EKF Loss = {training_window_loss:.6f}, Cov = {training_avg_window_cov:.6f}, Pre-EKF Loss = {training_pre_ekf_loss:.6f}")
                        
                        # Update online EKF state for next window
                        online_last_ekf_predictions = training_ekf_preds
                        online_last_ekf_covariances = training_ekf_covs
            
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
                    "window_delta_losses": window_delta_losses,
                    "window_delta_rmape_losses": window_delta_rmape_losses,
                    "window_pre_ekf_angles_pred": window_pre_ekf_angles_pred,
                    
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
                    "online_window_indices": online_window_indices,
                    "online_delta_losses": online_delta_losses,
                    "online_delta_rmape_losses": online_delta_rmape_losses,
                    "online_pre_ekf_angles_pred": online_pre_ekf_angles_pred,
                    
                    # Training data results (available during learning process)
                    "training_window_losses": training_window_losses,
                    "training_window_covariances": training_window_covariances,
                    "training_pre_ekf_losses": training_window_pre_ekf_losses,
                    "training_ekf_predictions": training_ekf_predictions,
                    "training_ekf_covariances": training_ekf_covariances,
                    "training_ekf_innovations": training_ekf_innovations,
                    "training_ekf_kalman_gains": training_ekf_kalman_gains,
                    "training_ekf_kalman_gain_times_innovation": training_ekf_kalman_gain_times_innovation,
                    "training_ekf_y_s_inv_y": training_ekf_y_s_inv_y,
                    "training_window_indices": training_window_indices,
                    "training_delta_losses": training_delta_losses,
                    "training_delta_rmape_losses": training_delta_rmape_losses,
                    "training_pre_ekf_angles_pred": training_pre_ekf_angles_pred,
                    
                    # Learning state tracking
                    "learning_start_window": self.learning_start_window,
                    "drift_detected_final": self.drift_detected,
                    "learning_done_final": self.learning_done,
                    "first_eta_change_final": self.first_eta_change,
                    "online_training_count_final": self.online_training_count
                }
            }
            
        except Exception as e:
            logger.exception(f"Error during online learning: {e}")
            return {"status": "error", "message": str(e), "exception": type(e).__name__}

    def _online_training_window(self, window_time_series, window_sources_num, window_labels, trajectory_idx: int = 0, window_idx: int = 0, 
                               is_first_window: bool = True, last_ekf_predictions: List = None, last_ekf_covariances: List = None):
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
        if self.online_training_count >= 7:
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
        
        # Use RMSPE loss for training and evaluation (keeping for comparison)
        rmspe_criterion = RMSPELoss().to(device)
        
        # Use Kalman Innovation Loss for online training
        kalman_criterion = KalmanInnovationLoss(reduction='mean')
        
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
                trajectory_type=self.config.trajectory.trajectory_type,
                device=device
            )
            ekf_filters.append(ekf_filter)
        
        # Get current eta value from system model
        current_eta = self.system_model.params.eta
        logger.info(f"Online training: Initialized {max_sources} EKF instances for window (eta={current_eta:.4f})")
        
        # Training phase: Run 3 gradient descent steps per window
        self.online_model.train()  # Set to training mode
        total_training_loss = 0.0
        num_training_steps = 0
        
        # Number of gradient descent steps per window
        num_gd_steps = 3
        
        for gd_step in range(num_gd_steps):
            step_training_loss = 0.0
            step_count = 0
            
            # Process each step for training in this gradient descent iteration
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
                        angles_pred_np = angles_pred_np[model_perm].flatten()
                        
                        # ============ NEW: Kalman Loss Calculation ============
                        # Initialize EKF for this training step if first step
                        if step == 0 and gd_step == 0:
                            # Initialize training EKF filters (separate from evaluation EKF)
                            self.training_ekf_filters = []
                            for i in range(num_sources_this_step):
                                training_ekf = ExtendedKalmanFilter1D.create_from_config(
                                    self.config, 
                                    trajectory_type=self.config.trajectory.trajectory_type,
                                    device=device
                                )
                                # Initialize with true angles for stable training
                                training_ekf.initialize_state(true_angles_this_step[i])
                                self.training_ekf_filters.append(training_ekf)
                        
                        # Run EKF step to get kalman gains and innovations for loss
                        # ============ NEW: Kalman Loss Calculation ============
                        step_kalman_gains = []
                        step_innovations = []
                        
                        # Apply EKF to each source prediction using tensor inputs to preserve gradients
                        for i in range(num_sources_this_step):
                            if i < len(self.training_ekf_filters) and i < angles_pred_tensor.size(2):
                                ekf_filter = self.training_ekf_filters[i]
                                
                                # Use tensor measurement to preserve gradients (shape: [batch, seq, sources])
                                tensor_measurement = angles_pred_tensor[0, 0, i]
                                
                                # EKF predict and update with tensor measurement
                                _, _, innovation, kalman_gain, _, _ = ekf_filter.predict_and_update(
                                    measurement=tensor_measurement, 
                                    true_state=true_angles_this_step[i]
                                )
                                
                                # Verify tensors maintain gradients - fail hard if not
                                if not isinstance(kalman_gain, torch.Tensor):
                                    raise RuntimeError(f"EKF kalman_gain is not a tensor: {type(kalman_gain)}. EKF must return tensors for gradient computation.")
                                if not isinstance(innovation, torch.Tensor):
                                    raise RuntimeError(f"EKF innovation is not a tensor: {type(innovation)}. EKF must return tensors for gradient computation.")
                                if not innovation.requires_grad:
                                    raise RuntimeError(f"EKF innovation tensor does not require gradients. This breaks the computation graph.")
                                
                                step_kalman_gains.append(kalman_gain)
                                step_innovations.append(innovation)
                            else:
                                # No fallback - fail hard if EKF not available or index out of bounds
                                if i >= len(self.training_ekf_filters):
                                    raise RuntimeError(f"EKF filter index {i} out of bounds. Expected {len(self.training_ekf_filters)} training EKF filters but got {num_sources_this_step} sources.")
                                if i >= angles_pred_tensor.size(2):
                                    raise RuntimeError(f"Source index {i} out of bounds for tensor shape {angles_pred_tensor.shape}. Cannot access source {i} from {angles_pred_tensor.size(2)} sources.")
                                raise RuntimeError(f"EKF filter {i} not available but should be. This indicates a serious configuration error.")
                        
                        # Calculate both Kalman and RMSPE losses for comparison
                        # Only Kalman loss will be used for backward pass
                        kalman_loss = kalman_criterion(step_kalman_gains, step_innovations)
                        rmspe_loss = rmspe_criterion(angles_pred_tensor, true_angles_tensor)
                        
                        # Use Kalman loss for training (instead of RMSPE)
                        step_training_loss += kalman_loss.item()
                        step_count += 1
                        
                        # Backward pass using Kalman loss - fail if gradients don't work
                        try:
                            kalman_loss.backward()
                        except RuntimeError as e:
                            if "modified by an inplace operation" in str(e) or "backward through the graph a second time" in str(e):
                                raise RuntimeError(f"Tensor gradient computation failed in training step {step}, GD iteration {gd_step}. "
                                                 f"This indicates the EKF or model operations are not properly tensor-compatible. "
                                                 f"Original error: {e}")
                            else:
                                raise
                        
                        # Log comparison between RMSPE and Kalman loss for all steps
                        kalman_loss_val = kalman_loss.item() if hasattr(kalman_loss, 'item') else float(kalman_loss)
                        rmspe_loss_val = rmspe_loss.item() if hasattr(rmspe_loss, 'item') else float(rmspe_loss)
                        logger.info(f"Training step {step}, GD {gd_step}: RMSPE = {rmspe_loss_val:.6f}, Kalman = {kalman_loss_val:.6f}")
                    
                    else:
                        # Near-field case - not supported
                        error_msg = "Near-field processing is not supported in this project. Please use far-field models only."
                        logger.error(error_msg)
                        raise NotImplementedError(error_msg)
                    
                except Exception as e:
                    logger.warning(f"Error during online training step {step} in GD iteration {gd_step}: {e}")
                    continue
            
            # Update model parameters after processing all steps in this gradient descent iteration
            if step_count > 0:
                avg_step_loss = step_training_loss / step_count
                self.online_optimizer.step()
                self.online_optimizer.zero_grad()
                total_training_loss += step_training_loss
                num_training_steps += step_count
                logger.info(f"Online training GD step {gd_step + 1}/{num_gd_steps}: Updated model with avg loss = {avg_step_loss:.6f} over {step_count} steps")
            else:
                logger.warning(f"No valid training steps in GD iteration {gd_step} for window {window_idx}")
        
        # Calculate overall average training loss
        if num_training_steps > 0:
            avg_training_loss = total_training_loss / num_training_steps
            logger.info(f"Online training step {self.online_training_count}: Completed {num_gd_steps} GD iterations with overall avg loss = {avg_training_loss:.6f} over {num_training_steps} total steps")
        else:
            avg_training_loss = float('inf')  # Set default value if no training steps
            logger.warning(f"No valid training steps in window {window_idx}")
        
        # Set model back to eval mode for EKF evaluation
        self.online_model.eval()
        
        # Clean up training EKF filters to free memory
        if hasattr(self, 'training_ekf_filters'):
            delattr(self, 'training_ekf_filters')
            logger.debug(f"Cleaned up training EKF filters after window {window_idx}")
        
        # Now evaluate the trained model with EKF (same as _evaluate_window but with no gradients)
        total_loss = 0.0
        num_valid_steps_for_loss = 0
        
        # Add pre-EKF loss tracking for far-field case
        pre_ekf_total_loss = 0.0
        pre_ekf_num_valid_steps = 0
        total_delta_loss = 0.0
        total_delta_rmape_loss = 0.0
        
        # Track the EKF predictions, covariances and innovations for each step and source
        ekf_predictions = []
        ekf_covariances = []
        ekf_innovations = []  # New list to track innovations
        ekf_kalman_gains = []  # New list to track Kalman gains
        ekf_kalman_gain_times_innovation = []  # New list to track K*y
        ekf_y_s_inv_y = []  # New list to track y*(S^-1)*y
        pre_ekf_angles_pred_list = []  # New list to track pre-EKF angle predictions
        
        # Track subspace_kalman_delta (prediction differences) for each step
        step_delta_losses = []
        
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
                    pre_ekf_angles_pred_list.append([])
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
                        
                        # Store pre-EKF predictions for this step
                        pre_ekf_angles_pred_list.append(pre_ekf_angles_pred.cpu().numpy())
                        
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
                        
                        # Calculate subspace_kalman_delta (difference between predictions)
                        step_delta_predictions = rmspe_criterion(pre_ekf_angles_pred, ekf_angles_pred)
                        
                        # Also calculate RMAPE between EKF and pre-EKF predictions
                        rmape_criterion = RMAPELoss().to(device)
                        step_delta_predictions_rmape = rmape_criterion(pre_ekf_angles_pred, ekf_angles_pred)
                        
                        step_delta_losses.append(step_delta_predictions)
                        total_delta_loss += step_delta_predictions
                        total_delta_rmape_loss += step_delta_predictions_rmape
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
            avg_delta_rmape_loss = 0.0
            if not is_near_field and pre_ekf_num_valid_steps > 0:
                avg_pre_ekf_loss = pre_ekf_total_loss / pre_ekf_num_valid_steps
                avg_delta_loss = total_delta_loss / pre_ekf_num_valid_steps
                avg_delta_rmape_loss = total_delta_rmape_loss / pre_ekf_num_valid_steps
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
            
            # Note: avg_delta_loss and avg_delta_rmape_loss already calculated above
            
            # Log window summary with columnar format
            logger.info(f"Online training window {window_idx}: "
                       f"Pre-EKF Loss = {avg_pre_ekf_loss:.6f}, "
                       f"EKF Loss = {avg_loss:.6f}, "
                       f"Avg Cov = {avg_window_cov:.6f}, "
                       f"Training Loss = {avg_training_loss:.6f}, "
                       f"Delta Loss = {avg_delta_loss:.6f}, "
                       f"Delta RMAPE = {avg_delta_rmape_loss:.6f}")
            
            return avg_loss, avg_window_cov, ekf_predictions, ekf_covariances, avg_pre_ekf_loss, ekf_innovations, ekf_kalman_gains, ekf_kalman_gain_times_innovation, ekf_y_s_inv_y, avg_delta_loss, pre_ekf_angles_pred_list, avg_delta_rmape_loss
        else:
            logger.warning("No valid steps with sources found in the window for EKF evaluation.")
            return float('inf'), float('nan'), [], [], float('inf'), [], [], [], [], 0.0, [], 0.0  # Include pre-EKF loss, delta loss, pre_ekf_angles_pred, and delta rmape loss in return

    def _evaluate_window(self, window_time_series, window_sources_num, window_labels, trajectory_idx: int = 0, window_idx: int = 0, 
                         is_first_window: bool = True, last_ekf_predictions: List = None, last_ekf_covariances: List = None, model=None):
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
            return float('inf'), float('nan'), [], [], float('inf'), [], [], [], [], 0.0, [], 0.0  # Include pre-EKF loss, delta loss, pre_ekf_angles_pred, and delta rmape loss in return
        
        total_loss = 0.0
        num_valid_steps_for_loss = 0
        
        # Add pre-EKF loss tracking for far-field case
        pre_ekf_total_loss = 0.0
        pre_ekf_num_valid_steps = 0
        total_delta_loss=0.0
        total_delta_rmape_loss=0.0
        # Use provided model or default to trained model
        if model is None:
            model = self.trained_model
            model_is_trained = True
        else:
            model_is_trained = False
        # Check if we're dealing with far-field or near-field
        is_near_field = hasattr(model, 'field_type') and model.field_type.lower() == "near"
        
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
                trajectory_type=self.config.trajectory.trajectory_type,
                device=device
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
        pre_ekf_angles_pred_list = []  # New list to track pre-EKF angle predictions
        
        # Track subspace_kalman_delta (prediction differences) for each step
        step_delta_losses = []
        
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
                    pre_ekf_angles_pred_list.append([])
                    continue
                
                # Get ground truth labels for this step
                true_angles_this_step = labels_per_step_list[step][:num_sources_this_step]
                
                # Forward pass through model
                model.eval() # Ensure model is in eval mode for this evaluation part
                with torch.no_grad():
                    if not is_near_field:
                        # Model expects num_sources as int or 0-dim tensor
                        angles_pred, _, _ = model(step_data_tensor, num_sources_this_step)
                        angles_old, _, _ = self.trained_model(step_data_tensor, num_sources_this_step)
                        
                        # Compare model weights properly
                        model_state = model.state_dict()
                        trained_state = self.trained_model.state_dict()
                        weights_equal = all(torch.equal(model_state[key], trained_state[key]) for key in model_state.keys())
                        if weights_equal and not model_is_trained:
                            logger.error("Model and trained model have the same weights - online model was not properly copied!")
                            raise RuntimeError("Online model and trained model have identical weights. This indicates the online model was not properly initialized as an independent copy. Cannot proceed with online learning.")
                        elif model_is_trained:
                            logger.info("online model is not initialized yet")
                        else:
                            logger.info("Model and trained model have the same weights - online model is properly initialized")
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
                        
                        # Store pre-EKF predictions for this step
                        pre_ekf_angles_pred_list.append(pre_ekf_angles_pred.cpu().numpy())
                        
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
                        
                        # Calculate subspace_kalman_delta (difference between predictions)
                        step_delta_predictions = rmspe_criterion(ekf_angles_pred, pre_ekf_angles_pred)
                        
                        # Also calculate RMAPE between EKF and pre-EKF predictions
                        rmape_criterion = RMAPELoss().to(device)
                        step_delta_predictions_rmape = rmape_criterion(ekf_angles_pred, pre_ekf_angles_pred)
                        
                        step_delta_losses.append(step_delta_predictions)
                        total_delta_loss+=step_delta_predictions
                        total_delta_rmape_loss+=step_delta_predictions_rmape
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
            avg_delta_rmape_loss = 0.0
            if not is_near_field and pre_ekf_num_valid_steps > 0:
                avg_pre_ekf_loss = pre_ekf_total_loss / pre_ekf_num_valid_steps
                avg_delta_loss = total_delta_loss / pre_ekf_num_valid_steps
                avg_delta_rmape_loss = total_delta_rmape_loss / pre_ekf_num_valid_steps
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
            
            # Calculate average subspace_kalman_delta (prediction differences) for the window
            avg_delta_loss = 0.0
            if len(step_delta_losses) > 0:
                avg_delta_loss = sum(step_delta_losses) / len(step_delta_losses)
            
            # Log window summary with columnar format
            log_window_summary(avg_pre_ekf_loss, avg_loss, avg_window_cov, current_eta, is_near_field, trajectory_idx, window_idx)
            
            return avg_loss, avg_window_cov, ekf_predictions, ekf_covariances, avg_pre_ekf_loss, ekf_innovations, ekf_kalman_gains, ekf_kalman_gain_times_innovation, ekf_y_s_inv_y, avg_delta_loss, pre_ekf_angles_pred_list, avg_delta_rmape_loss
        else:
            logger.warning("No valid steps with sources found in the window for loss calculation.")
            return float('inf'), float('nan'), [], [], float('inf'), [], [], [], [], 0.0, [], 0.0  # Include pre-EKF loss, delta loss, pre_ekf_angles_pred, and delta rmape loss in return

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
            "training_pre_ekf_angles_pred": avg_training_pre_ekf_angles_pred
        }


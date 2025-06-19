"""
Evaluation components for simulations.

This module handles model and method evaluation.
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from DCD_MUSIC.src.metrics.rmspe_loss import RMSPELoss
from DCD_MUSIC.src.evaluation import get_model_based_method, evaluate_model_based
from simulation.kalman_filter import KalmanFilter1D, BatchKalmanFilter1D
from simulation.kalman_filter.extended import ExtendedKalmanFilter1D

logger = logging.getLogger(__name__)

# Set device for evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Evaluator:
    """
    Handles evaluation for simulations.
    
    Responsible for:
    - Evaluating model performance
    - Comparing with baseline methods
    - Collecting metrics
    """
    def __init__(self, config, model=None, system_model=None, output_dir=None):
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration object
            model: The model to evaluate (can be set later)
            system_model: The system model for baseline methods
            output_dir: Directory for saving results
        """
        self.config = config
        self.model = model
        self.system_model = system_model
        self.output_dir = output_dir or Path("experiments/results")
        self.results = {}
        
        # Check if near-field mode is active
        if model is not None:
            self.is_near_field = hasattr(model, 'field_type') and model.field_type.lower() == "near"
        else:
            self.is_near_field = False
    
    def set_model(self, model):
        """Set the model to evaluate."""
        self.model = model
        # Update near-field flag
        self.is_near_field = hasattr(model, 'field_type') and model.field_type.lower() == "near"
        
    def set_system_model(self, system_model):
        """Set the system model for baseline methods."""
        self.system_model = system_model
    
    def evaluate(self, test_dataloader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate the model with the given test dataloader.
        
        Args:
            test_dataloader: DataLoader with test data
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Starting evaluation pipeline")
        
        # Validate that we have a model to evaluate
        if self.model is None:
            logger.error("No model available for evaluation")
            return {"status": "error", "message": "No model available for evaluation"}
        
        # Ensure model is in evaluation mode
        self.model.eval()
        
        try:
            logger.info("Evaluating model(s) on test data")
            
            if self.is_near_field:
                error_msg = "Near-field option is not available in the current evaluation pipeline"
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}
            
            # Get Kalman filter parameters from config
            kf_Q, kf_R, kf_P0 = KalmanFilter1D.from_config(self.config)
            
            # Instantiate RMSPELoss criterion
            rmspe_criterion = RMSPELoss().to(device)
            
            # Result containers
            dnn_trajectory_results = []  # Store DNN+KF results
            
            # Overall metrics
            dnn_total_loss = 0.0
            dnn_total_samples = 0
            classic_methods_losses = {}  # Dictionary to store losses for each classic method
            
            # Get list of classic subspace methods to evaluate
            classic_methods = []
            if hasattr(self.config.simulation, 'subspace_methods') and self.config.simulation.subspace_methods:
                classic_methods = self.config.simulation.subspace_methods
                logger.info(f"Will evaluate classic subspace methods: {classic_methods}")
                
                # Initialize loss accumulators for each method
                for method in classic_methods:
                    classic_methods_losses[method] = {"total_loss": 0.0, "total_samples": 0}
            
            # Trajectory-level evaluation with both DNN+KF and classic methods
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(tqdm(test_dataloader, desc="Evaluating trajectories")):
                    # Get batch data
                    trajectories, sources_num, labels = batch_data
                    batch_size, trajectory_length = trajectories.shape[0], trajectories.shape[1]
                    
                    # Initialize storage for batch results
                    batch_dnn_preds = [[] for _ in range(batch_size)]
                    batch_dnn_kf_preds = [[] for _ in range(batch_size)]
                    
                    # Find maximum number of sources in batch for efficient batch processing
                    max_sources = torch.max(sources_num).item()
                    
                    # Initialize the batch Kalman filter
                    batch_kf = BatchKalmanFilter1D.from_config(
                        self.config,
                        batch_size=batch_size,
                        max_sources=max_sources,
                        device=device
                    )
                    
                    # Initialize states directly from first step ground truth
                    # Each trajectory might have different number of sources
                    batch_kf.initialize_states(labels[:, 0, :max_sources], sources_num[:, 0])
                    
                    # Process each time step in the trajectory sequentially (required for Kalman filtering)
                    for step in range(trajectory_length):
                        # Get data for this time step across all trajectories
                        step_data = trajectories[:, step].to(device)
                        step_sources = sources_num[:, step].to(device)
                        
                        # Create source mask for this step
                        step_mask = torch.arange(max_sources, device=device).expand(batch_size, -1) < step_sources[:, None]
                        
                        # Evaluate DNN model and update Kalman filters for this time step
                        model_preds, kf_preds, step_loss = self._evaluate_dnn_model_kf_step_batch(
                            step_data, step_sources, labels[:, step, :max_sources], step_mask, batch_kf, 
                            rmspe_criterion
                        )
                        
                        # Accumulate loss and store predictions
                        dnn_total_loss += step_loss
                        dnn_total_samples += batch_size
                        
                        # Store predictions for this time step
                        for i in range(batch_size):
                            batch_dnn_preds[i].append(model_preds[i])
                            batch_dnn_kf_preds[i].append(kf_preds[i])
                        
                        # Evaluate classic methods if enabled
                        if classic_methods and self.system_model is not None:
                            classic_results = self._evaluate_classic_methods_step_batch(
                                step_data, step_sources, labels[:, step, :max_sources],
                                rmspe_criterion, classic_methods
                            )
                            
                            # Update classic method losses
                            for method, results in classic_results.items():
                                classic_methods_losses[method]["total_loss"] += results["total_loss"]
                                classic_methods_losses[method]["total_samples"] += results["count"]
                    
                    # Store complete trajectory results
                    for i in range(batch_size):
                        num_sources_array = sources_num[i].cpu().numpy()
                        gt_trajectory = np.array([
                            labels[i, t, :num_sources_array[t]].cpu().numpy()
                            for t in range(trajectory_length)
                        ])
                        
                        dnn_trajectory_results.append({
                            'model_predictions': batch_dnn_preds[i],
                            'kf_predictions': batch_dnn_kf_preds[i],
                            'ground_truth': gt_trajectory,
                            'sources': num_sources_array
                        })
                
                # Log and store evaluation results
                self._log_evaluation_results(
                    dnn_total_loss, 
                    dnn_total_samples,
                    classic_methods_losses,
                    dnn_trajectory_results,
                    None  # No classic trajectory results to log
                )
                
                # Store DNN results
                self.results["dnn_trajectory_results"] = dnn_trajectory_results
                self.results["status"] = "success"
                
                # Save results to file if output_dir is specified
                if self.output_dir:
                    self._save_results_to_file()
                
            return self.results
                    
        except Exception as e:
            logger.exception(f"Error during evaluation: {e}")
            return {"status": "error", "message": str(e)}
    
    def _evaluate_dnn_model_kf_step_batch(
        self, 
        step_data: torch.Tensor, 
        step_sources: torch.Tensor,
        step_angles: torch.Tensor,
        step_mask: torch.Tensor,
        batch_kf: BatchKalmanFilter1D,
        rmspe_criterion: RMSPELoss
    ) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
        """
        Evaluate the DNN model for a batch of steps and apply Kalman filtering.
        
        Args:
            step_data: Batch of trajectory step data for current time step [batch_size, T, N]
            step_sources: Number of sources for each trajectory at current step [batch_size]
            step_angles: Ground truth angles tensor [batch_size, max_sources]
            step_mask: Mask indicating valid angles for each trajectory at current step
            batch_kf: BatchKalmanFilter1D instance for efficient batch processing
            rmspe_criterion: Loss criterion for evaluation
            
        Returns:
            Tuple containing:
            - List of model predictions for each trajectory
            - List of KF predictions for each trajectory
            - Total loss for this batch at this step
        """
        batch_size = step_data.shape[0]
        max_sources = batch_kf.max_sources
        
        # Collect model predictions per trajectory
        batch_angles_pred_list = []
        total_loss = 0.0

        # Loop through batch because the model (or underlying method) doesn't support batch source counts
        for i in range(batch_size):
            # Extract data for a single trajectory
            single_step_data = step_data[i].unsqueeze(0) # Add batch dim back
            single_step_sources = step_sources[i] # This should be a scalar tensor
            num_sources_item = single_step_sources.item()

            # Skip if no sources
            if num_sources_item <= 0:
                batch_angles_pred_list.append(torch.empty((0,), device=device))
                continue

            # Model forward pass for single trajectory
            if self.is_near_field:
                # Near-field case (currently not supported)
                angles_pred_single, _, _ = self.model(single_step_data, single_step_sources)
            else:
                # Far-field case
                angles_pred_single, _, _ = self.model(single_step_data, single_step_sources)

            # Ensure prediction tensor has correct shape [1, num_sources]
            angles_pred_single = angles_pred_single.view(1, -1)[:, :num_sources_item]
            batch_angles_pred_list.append(angles_pred_single.squeeze(0))

            # Calculate loss for this trajectory
            with torch.no_grad():
                truth = step_angles[i, :num_sources_item].unsqueeze(0)
                loss = rmspe_criterion(angles_pred_single, truth)
                total_loss += loss.item()

        # Combine predictions into a batch tensor for KF update
        # Need padding if source counts vary
        padded_angles_pred = torch.zeros((batch_size, max_sources), device=device)
        for i, pred in enumerate(batch_angles_pred_list):
            num_sources = pred.shape[0]
            if num_sources > 0:
                padded_angles_pred[i, :num_sources] = pred

        # Apply the Kalman filter to the predicted angles (using the padded batch tensor)
        # First, get predictions (before update) for output
        kf_predictions_before_update = batch_kf.predict().cpu().numpy()

        # Then update with new measurements (model predictions)
        batch_kf.update(padded_angles_pred, step_mask)

        # Convert predictions to list of numpy arrays with proper dimensions
        model_predictions_list = []
        kf_predictions_list = []

        for i in range(batch_size):
            num_sources = step_sources[i].item()
            if num_sources > 0:
                model_predictions_list.append(padded_angles_pred[i, :num_sources].cpu().numpy())
                kf_predictions_list.append(kf_predictions_before_update[i, :num_sources])
            else:
                model_predictions_list.append(np.array([]))
                kf_predictions_list.append(np.array([]))

        return model_predictions_list, kf_predictions_list, total_loss
    
    def _evaluate_classic_methods_step_batch(
        self,
        step_data: torch.Tensor,
        step_sources: torch.Tensor,
        step_angles: torch.Tensor,
        rmspe_criterion: RMSPELoss,
        methods: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate classic subspace methods for a batch of trajectories at one time step.
        """
        batch_size = step_data.shape[0]
        results = {}
        
        for method in methods:
            results[method] = {"total_loss": 0.0, "count": 0}
            try:
                # Get system model parameters
                system_model_params = self.system_model.params
                # Prepare batch_data in the expected format for test_step
                mask = torch.ones_like(step_angles)
                batch_data = (
                    step_data,      # [batch_size, N, T] or [batch_size, ...]
                    step_sources,   # [batch_size]
                    step_angles,    # [batch_size, max_sources]
                    mask            # [batch_size, max_sources]
                )
                classic_method = get_model_based_method(method, system_model_params)
                if classic_method is not None:
                    # Send the entire batch
                    loss, accuracy, num_samples = classic_method.test_step(batch_data,  0)
                    #logger.info(f"Classic method {method}: batch_size={batch_size}, num_samples={num_samples}")
                    results[method]["total_loss"] += loss
                    results[method]["count"] += num_samples
                else:
                    logger.warning(f"Failed to initialize classic method: {method}")
            except Exception as method_error:
                logger.error(f"Error with method {method}: {method_error}")
                import traceback
                logger.error(traceback.format_exc())
        return results
    
    def _log_evaluation_results(
        self,
        dnn_total_loss: float,
        dnn_total_samples: int,
        classic_methods_losses: Dict[str, Dict[str, float]],
        dnn_trajectory_results: List[Dict[str, Any]],
        classic_trajectory_results: List[Dict[str, Any]]
    ) -> None:
        """
        Log the evaluation results for both DNN and classic methods.
        
        Args:
            dnn_total_loss: Accumulated loss for DNN model
            dnn_total_samples: Total samples evaluated for DNN
            classic_methods_losses: Dictionary of losses for classic methods
            dnn_trajectory_results: List of trajectory results for DNN
            classic_trajectory_results: List of trajectory results for classic methods
        """
        # Calculate average loss for DNN model
        dnn_avg_loss = dnn_total_loss / max(dnn_total_samples, 1)
        
        # Log DNN results
        logger.info(f"DNN Model - Average loss: {dnn_avg_loss:.6f}")
        
        # Calculate and log average losses for classic methods
        classic_methods_avg_losses = {}
        for method, loss_data in classic_methods_losses.items():
            if loss_data["total_samples"] > 0:
                avg_loss = loss_data["total_loss"] / loss_data["total_samples"]
                classic_methods_avg_losses[method] = avg_loss
                logger.info(f"Classic Method {method} - Average loss: {avg_loss:.6f}")
        
        # Store metrics in results
        self.results["dnn_test_loss"] = dnn_avg_loss
        self.results["classic_methods_test_losses"] = classic_methods_avg_losses
        
        # Compare DNN vs classic methods if both are available
        if classic_methods_avg_losses:
            logger.info("Comparative Results:")
            for method, avg_loss in classic_methods_avg_losses.items():
                diff = dnn_avg_loss - avg_loss
                relative_diff = diff / avg_loss * 100 if avg_loss != 0 else float('inf')
                
                if diff < 0:
                    logger.info(f"DNN outperforms {method} by {abs(diff):.6f} ({abs(relative_diff):.2f}%)")
                else:
                    logger.info(f"{method} outperforms DNN by {diff:.6f} ({relative_diff:.2f}%)")
        
        # Log total number of trajectories evaluated
        logger.info(f"Evaluated {len(dnn_trajectory_results)} trajectories with DNN model")
        if classic_trajectory_results:
            logger.info(f"Evaluated {len(classic_trajectory_results)} trajectories with classic methods")
            
        # Add print statements for detailed evaluation results
        print("\n" + "="*80)
        print(f"{'EVALUATION RESULTS':^80}")
        print("="*80)
        
        # Print DNN results
        print(f"\n{'DNN MODEL WITH KALMAN FILTER':^80}")
        print("-"*80)
        print(f"Average Loss: {dnn_avg_loss:.6f}")
        print(f"Total Samples: {dnn_total_samples}")
        print(f"Trajectories Evaluated: {len(dnn_trajectory_results)}")
        
        # Print classic methods results if available
        if classic_methods_avg_losses:
            print(f"\n{'CLASSIC SUBSPACE METHODS':^80}")
            print("-"*80)
            print(f"{'Method':<20} {'Average Loss':<20} {'Comparison with DNN':<30}")
            print("-"*80)
            
            for method, avg_loss in classic_methods_avg_losses.items():
                diff = dnn_avg_loss - avg_loss
                relative_diff = diff / avg_loss * 100 if avg_loss != 0 else float('inf')
                
                if diff < 0:
                    comparison = f"DNN better by {abs(diff):.6f} ({abs(relative_diff):.2f}%)"
                elif diff > 0:
                    comparison = f"DNN worse by {diff:.6f} ({relative_diff:.2f}%)"
                else:
                    comparison = "Equal performance"
                    
                print(f"{method:<20} {avg_loss:<20.6f} {comparison:<30}")
        
        print("\n" + "="*80) 
    
    def _save_results_to_file(self) -> None:
        """Save evaluation results to a file."""
        import json
        import datetime
        
        # Create timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"evaluation_results_{timestamp}.json"
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert any non-serializable objects to strings
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            else:
                return str(obj)
                
        # Filter out non-serializable nested objects
        serializable_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                serializable_results[key] = {k: convert_to_serializable(v) for k, v in value.items()}
            elif isinstance(value, list):
                # Handle lists of dictionaries
                if value and isinstance(value[0], dict):
                    serializable_list = []
                    for item in value:
                        serializable_item = {k: convert_to_serializable(v) for k, v in item.items()}
                        serializable_list.append(serializable_item)
                    serializable_results[key] = serializable_list
                else:
                    serializable_results[key] = [convert_to_serializable(v) for v in value]
            else:
                serializable_results[key] = convert_to_serializable(value)
        
        # Write to file
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {results_file}") 
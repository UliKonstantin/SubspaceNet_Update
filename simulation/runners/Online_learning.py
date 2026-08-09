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
from simulation.runners.sandbox import glrt_changepoint_detection, plot_results
from utils import drift_gates



logger = logging.getLogger(__name__)

# Device setup for evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _compute_loss_from_type(
    loss_type: str,
    ekf_preds: torch.Tensor,
    pre_ekf_preds: torch.Tensor,
    true_angles: torch.Tensor,
    window_size: int,
    rmspe_criterion,
    rmape_criterion,
    loss_config=None,
):
    """Compute scalar window loss from a typed loss config string."""
    if loss_type == "multimoment":
        if loss_config is None:
            raise RuntimeError("loss_config required for multimoment loss")
        multimoment_criterion = MultiMomentInnovationConsistencyLoss(
            alpha=getattr(loss_config, "multimoment_alpha", 1.0),
            beta=getattr(loss_config, "multimoment_beta", 1.0),
        ).to(device)
        return multimoment_criterion(
            angles_pred=pre_ekf_preds,
            angles=ekf_preds,
            return_components=False,
        )
    if loss_type == "unsupervised_rmspe":
        return rmspe_criterion(ekf_preds, pre_ekf_preds) / window_size
    if loss_type == "unsupervised_rmape":
        return rmape_criterion(ekf_preds, pre_ekf_preds) / window_size
    if loss_type == "supervised_rmspe":
        return rmspe_criterion(ekf_preds, true_angles) / window_size
    if loss_type == "supervised_rmape":
        return rmape_criterion(ekf_preds, true_angles) / window_size
    raise RuntimeError(
        f"Unknown loss type: {loss_type}. Must be one of: "
        "multimoment, unsupervised_rmspe, unsupervised_rmape, supervised_rmspe, supervised_rmape"
    )


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
    reference_metric_loss: float       # Eval-only tracking metric (reference_metric config)
    reference_metric_loss_db: float    # dB scale (20 * log10)
    reference_metric_config: str       # e.g. "supervised_rmspe"
    adaptation_loss: float             # MSIE / adaptation objective (adaptation_loss config)
    adaptation_loss_config: str        # e.g. "unsupervised_rmspe"
    pre_ekf_loss: float                # Raw DNN vs GT
    ekf_gain_rmape: float              # Diagnostic RMAPE(EKF, pre-EKF)


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
    
    @classmethod
    def create_error_result(cls, error_message: str) -> 'WindowEvaluationResult':
        """Create an error result with default values."""
        return cls(
            loss_metrics=LossMetrics(
                reference_metric_loss=float('inf'),
                reference_metric_loss_db=float('inf'),
                reference_metric_config="error",
                adaptation_loss=float('inf'),
                adaptation_loss_config="error",
                pre_ekf_loss=float('inf'),
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
        # Dual model online learning state variables
        self.drift_detected = False
        self.learning_done = False
        self.online_model = None
        self.online_training_count = 0
        self.first_eta_change = True  # Track if this is the first eta change
        self.learning_start_window = None  # Track when learning started (set by GLRT detection)
        self.training_window_indices = []  # Track which windows were used for training
        
        # Get time_to_learn from configuration (delay after GLRT detection)
        online_config = self.config.online_learning
        self.time_to_learn = getattr(online_config, 'time_to_learn', None)
        if self.time_to_learn is None:
            logger.warning("time_to_learn not specified in config, drift detection will trigger immediately after GLRT threshold")
        else:
            logger.info(f"Drift detection will be delayed by {self.time_to_learn} windows after GLRT threshold is exceeded")
        
        self.glrt_drift_detection_window = None  # Action gate: window where z-score first exceeded threshold
        self.glrt_history = []  # Rolling log-GLR samples (g-stream for Scope B)
        self.drift_history_max_size = getattr(online_config, 'drift_history_max_size', None)
        self.drift_z_threshold = getattr(online_config, 'drift_z_threshold', 2.5)
        self.drift_warmup_windows = getattr(online_config, 'drift_warmup_windows', 7)
        self.drift_guard_samples = getattr(online_config, 'drift_guard_samples', 3)
        self.use_adaptive_learning_rate = getattr(online_config, 'use_adaptive_learning_rate', False)
        self.adaptive_lr_min = getattr(online_config, 'adaptive_lr_min', 0.0005)
        self.adaptive_lr_max = getattr(online_config, 'adaptive_lr_max', 0.0356)
        self.adaptive_lr_k_sigmoid = getattr(online_config, 'adaptive_lr_k_sigmoid', 0.7336)
        self.adaptive_lr_dG0 = getattr(online_config, 'adaptive_lr_dG0', 69.2599)
        self.num_gd_steps = getattr(online_config, 'num_gd_steps', 3)

        first_g = drift_gates.first_g_window(self.drift_warmup_windows)
        first_z = drift_gates.first_z_window(
            self.drift_warmup_windows, self.drift_guard_samples
        )
        logger.info(
            "Drift detection: scope_a_warmup=%s, scope_b_baseline_min=%s, guard_samples=%s, "
            "z_threshold=%s (first g at window %s, first z at window %s), time_to_learn=%s",
            self.drift_warmup_windows,
            drift_gates.SCOPE_B_BASELINE_MIN_SAMPLES,
            self.drift_guard_samples,
            self.drift_z_threshold,
            first_g,
            first_z,
            self.time_to_learn,
        )
    
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
            all_drift_detection_dicts = []
            
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
                # Collect drift detection dicts from this trajectory
                if "online_learning_results" in trajectory_result and "drift_detection_dicts" in trajectory_result["online_learning_results"]:
                    all_drift_detection_dicts.extend(trajectory_result["online_learning_results"]["drift_detection_dicts"])
            
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
                reference_metric_config = first_window_result.loss_metrics.reference_metric_config
                adaptation_loss_config = first_window_result.loss_metrics.adaptation_loss_config
            else:
                reference_metric_config = "unknown"
                adaptation_loss_config = "unknown"
            
            # Get training and eta change info from results
            training_start_window = None
            training_end_window = None
            drift_detection_window = None
            eta_change_windows = []
            if all_results and len(all_results) > 0:
                first_result = all_results[0]["online_learning_results"]
                training_start_window = first_result.get("training_start_window")
                training_end_window = first_result.get("training_end_window")
                eta_change_windows = first_result.get("eta_change_windows", [])
                drift_detection_window = first_result.get("drift_detection_window")
            
            # ALSO call the new direct averaged plotting function
            if averaged_results_across_trajectories.get("status") == "success":
                from utils.plotting import plot_averaged_online_learning_results
                
                averaged_data = averaged_results_across_trajectories["averaged_results"]
                plot_averaged_online_learning_results(
                    self.output_dir,
                    averaged_data["averaged_pretrained_trajectory"],
                    averaged_data["averaged_online_trajectory"],
                    reference_metric_config,
                    adaptation_loss_config,
                    training_start_window=training_start_window,
                    training_end_window=training_end_window,
                    drift_detection_window=drift_detection_window,
                    eta_change_windows=eta_change_windows,
                    averaged_supervised_metrics=averaged_data.get("averaged_supervised_trajectory"),
                )
            
            # GLRT drift detection averaged plotting (using results from averaging function)
            glrt_results = averaged_results_across_trajectories.get("averaged_results", {}).get("glrt_results", {})
            
            glrt_window_offset = self.drift_warmup_windows
            eta_markers = eta_change_windows if eta_change_windows else None
            gate_milestones = drift_gates.drift_detection_milestones(
                self.drift_warmup_windows, self.drift_guard_samples
            )

            # Plot reference loss GLRT results
            if "adaptation_loss" in glrt_results:
                ref_data = glrt_results["adaptation_loss"]
                avg_ref_losses = ref_data["avg_losses"]
                min_segment_size = ref_data["min_segment_size"]
                plot_offset = ref_data.get("window_index_offset", glrt_window_offset)
                
                if len(avg_ref_losses) >= 2 * min_segment_size + 1:
                    try:
                        ref_changepoint, ref_log_glr, ref_all_log_glr, ref_candidate_points = glrt_changepoint_detection(
                            avg_ref_losses, min_segment_size=min_segment_size
                        )
                        ref_fig_loss, ref_fig_glrt = plot_results(
                            avg_ref_losses, ref_changepoint, ref_all_log_glr, ref_candidate_points,
                            window_index_offset=plot_offset, event_windows=eta_markers,
                            gate_milestones=gate_milestones,
                        )
                        title = f'GLRT Drift Detection - Adaptation Loss (Averaged across {ref_data["trajectory_count"]} trajectories)'
                        if ref_data["avg_changepoint_window"] is not None:
                            title += f'\nAvg Changepoint Window: {ref_data["avg_changepoint_window"]:.2f} ± {ref_data["std_changepoint_window"]:.2f}, Avg Log-GLR: {ref_data["avg_likelihood"]:.4f} ± {ref_data["std_likelihood"]:.4f}'
                        ref_fig_loss.suptitle(title + ' - Loss', fontsize=14)
                        ref_fig_glrt.suptitle(title + ' - GLRT Statistics', fontsize=14)
                        ref_fig_loss.subplots_adjust(top=0.88)  # Adjust spacing after suptitle
                        ref_fig_glrt.subplots_adjust(top=0.88)  # Adjust spacing after suptitle
                        ref_loss_plot_path = self.output_dir / "glrt_adaptation_loss_averaged_loss.png"
                        ref_glrt_plot_path = self.output_dir / "glrt_adaptation_loss_averaged_glrt.png"
                        ref_fig_loss.savefig(ref_loss_plot_path, dpi=150, bbox_inches='tight')
                        ref_fig_glrt.savefig(ref_glrt_plot_path, dpi=150, bbox_inches='tight')
                        plt.close(ref_fig_loss)
                        plt.close(ref_fig_glrt)
                        logger.info(f"Saved averaged GLRT reference loss plots to {ref_loss_plot_path} and {ref_glrt_plot_path}")
                        logger.info(f"Adaptation Loss GLRT: Avg Changepoint = {ref_data['avg_changepoint_window']:.2f} ± {ref_data['std_changepoint_window']:.2f}, "
                                   f"Avg Log-GLR = {ref_data['avg_likelihood']:.4f} ± {ref_data['std_likelihood']:.4f}")
                    except Exception as e:
                        logger.warning(f"Failed to plot averaged GLRT reference loss results: {e}")
            
            # Plot main loss GLRT results
            if "reference_metric" in glrt_results:
                main_data = glrt_results["reference_metric"]
                avg_main_losses = main_data["avg_losses"]
                min_segment_size = main_data["min_segment_size"]
                plot_offset = main_data.get("window_index_offset", glrt_window_offset)
                
                if len(avg_main_losses) >= 2 * min_segment_size + 1:
                    try:
                        main_changepoint, main_log_glr, main_all_log_glr, main_candidate_points = glrt_changepoint_detection(
                            avg_main_losses, min_segment_size=min_segment_size
                        )
                        main_fig_loss, main_fig_glrt = plot_results(
                            avg_main_losses, main_changepoint, main_all_log_glr, main_candidate_points,
                            window_index_offset=plot_offset, event_windows=eta_markers,
                            gate_milestones=gate_milestones,
                        )
                        title = f'GLRT Drift Detection - Reference Metric (Averaged across {main_data["trajectory_count"]} trajectories)'
                        if main_data["avg_changepoint_window"] is not None:
                            title += f'\nAvg Changepoint Window: {main_data["avg_changepoint_window"]:.2f} ± {main_data["std_changepoint_window"]:.2f}, Avg Log-GLR: {main_data["avg_likelihood"]:.4f} ± {main_data["std_likelihood"]:.4f}'
                        main_fig_loss.suptitle(title + ' - Loss', fontsize=14)
                        main_fig_glrt.suptitle(title + ' - GLRT Statistics', fontsize=14)
                        main_fig_loss.subplots_adjust(top=0.88)  # Adjust spacing after suptitle
                        main_fig_glrt.subplots_adjust(top=0.88)  # Adjust spacing after suptitle
                        main_loss_plot_path = self.output_dir / "glrt_reference_metric_averaged_loss.png"
                        main_glrt_plot_path = self.output_dir / "glrt_reference_metric_averaged_glrt.png"
                        main_fig_loss.savefig(main_loss_plot_path, dpi=150, bbox_inches='tight')
                        main_fig_glrt.savefig(main_glrt_plot_path, dpi=150, bbox_inches='tight')
                        plt.close(main_fig_loss)
                        plt.close(main_fig_glrt)
                        logger.info(f"Saved averaged GLRT main loss plots to {main_loss_plot_path} and {main_glrt_plot_path}")
                        logger.info(f"Reference Metric GLRT: Avg Changepoint = {main_data['avg_changepoint_window']:.2f} ± {main_data['std_changepoint_window']:.2f}, "
                                   f"Avg Log-GLR = {main_data['avg_likelihood']:.4f} ± {main_data['std_likelihood']:.4f}")
                    except Exception as e:
                        logger.warning(f"Failed to plot averaged GLRT main loss results: {e}")

            if getattr(self.config.online_learning, "plot_trajectory", False):
                plot_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                stride = self.config.online_learning.stride
                for traj_idx, traj_result in enumerate(pretrained_trajectory_results):
                    suffix = f"_traj{traj_idx}" if len(pretrained_trajectory_results) > 1 else ""
                    plot_online_learning_trajectory(
                        traj_result.window_labels,
                        self.output_dir,
                        f"{plot_ts}{suffix}",
                        window_indices=traj_result.window_indices,
                        stride=stride,
                    )
            
            logger.info(f"Online learning completed over {dataset_size} trajectories")
            
            # Prepare return results
            return_results = {
                "status": "success", 
                "online_learning_results": {
                    "pretrained_trajectory_results": pretrained_trajectory_results,
                    "online_trajectory_results": online_trajectory_results,
                    "dataset_size": dataset_size,
                    "training_start_window": training_start_window,
                    "training_end_window": training_end_window,
                    "eta_change_windows": eta_change_windows,
                    "drift_detection_window": drift_detection_window,
                },
                "drift_detection_dicts": all_drift_detection_dicts
            }
            
            # Add averaged results if available
            if averaged_results_across_trajectories.get("status") == "success":
                return_results["averaged_results"] = averaged_results_across_trajectories["averaged_results"]
                # Also add GLRT results at top level for easy access
                if "glrt_results" in averaged_results_across_trajectories["averaged_results"]:
                    return_results["glrt_results"] = averaged_results_across_trajectories["averaged_results"]["glrt_results"]
                    # Extract z-score and learning rate from GLRT results for easier access
                    if "adaptation_loss" in return_results["glrt_results"]:
                        main_glrt = return_results["glrt_results"]["adaptation_loss"]
                        return_results["avg_glrt_z_score"] = main_glrt.get("avg_z_score")
                        return_results["std_glrt_z_score"] = main_glrt.get("std_z_score")
                        return_results["avg_learning_rate_at_detection"] = main_glrt.get("avg_learning_rate")
                        return_results["std_learning_rate_at_detection"] = main_glrt.get("std_learning_rate")
                        return_results["avg_actual_learning_rate"] = main_glrt.get("avg_actual_learning_rate")
                        return_results["std_actual_learning_rate"] = main_glrt.get("std_actual_learning_rate")
            
            return return_results
            
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
            
            import torch
            from config.factory import create_model
            
            # Initialize online model as copy of trained model
            self.online_model = create_model(self.config, self.system_model)
            self.online_model.load_state_dict(self.trained_model.state_dict())
            self.online_model.eval()
            
            # Initialize supervised model as copy of trained model
            self.supervised_trained_model = create_model(self.config, self.system_model)
            self.supervised_trained_model.load_state_dict(self.trained_model.state_dict())
            self.supervised_trained_model.train()
            
            # Reset dual model state for new trajectory
            self.drift_detected = False
            self.learning_done = False
            self.online_training_count = 0
            self.first_eta_change = True
            self.adaptive_lr_fixed = None  # Fixed LR from first training window (adaptive mode)
            self.actual_lr_per_training_window = []  # Track actual LR used per window for heatmap
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
            self._trajectory_dc_offsets = getattr(
                online_learning_dataset.generator, "sine_accel_dc_offsets", None
            )
            
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
            
            # Initialize trajectory results structure
            trajectory_results = TrajectoryResults()
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
            
            # Track drift detection metrics
            drift_detection_dicts = []
            
            # GLRT drift detection tracking variables
            glrt_adaptation_loss_changepoint_window = None
            glrt_adaptation_loss_likelihood = None
            glrt_adaptation_loss_all_log_glr = None
            glrt_adaptation_loss_candidate_points = None
            glrt_adaptation_losses = None
            glrt_reference_metric_changepoint_window = None
            glrt_reference_metric_likelihood = None
            glrt_reference_metric_all_log_glr = None
            glrt_reference_metric_candidate_points = None
            glrt_reference_metric_losses = None
            current_glrt_likelihood = None  # Current GLRT likelihood for adaptive learning rate (persists across windows)
            current_glrt_z_score = None  # Z-score normalized GLRT for adaptive learning rate
            current_glrt_baseline_mean = None  # Baseline mean of GLRT history for adaptive learning rate (persists across windows)
            
            # Reset GLRT drift detection window for new trajectory
            self.glrt_drift_detection_window = None
            # Reset tracking variables for drift detection metrics
            self.glrt_z_score_at_detection = None  # Z-score at the moment drift was detected
            self.learning_rate_at_detection = None  # Learning rate calculated at detection time
            # Note: GLRT history is kept across trajectories for better baseline estimation
            # Uncomment the line below to reset per trajectory if needed:
            # self.glrt_history = []
            
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
                        # The dataset holds the generator, which updates the shared self.system_model.params.eta
                        online_learning_dataloader.dataset.update_eta(new_eta)
                        if self.first_eta_change:
                            self.first_eta_change = False
                            logger.info(f"First eta modification at window {window_idx}")
                            # Set drift detected on first eta change only
                    
                    
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
                
                # --- GLRT drift detection (adaptation_loss = MSIE trigger stream) ---
                if window_idx >= self.drift_warmup_windows:
                    min_segment_size = drift_gates.glrt_min_segment_size()
                    post_warmup_results = trajectory_results.window_results[
                        self.drift_warmup_windows :
                    ]
                    adaptation_losses = [
                        wr.loss_metrics.adaptation_loss for wr in post_warmup_results
                    ]
                    reference_metric_losses = [
                        wr.loss_metrics.reference_metric_loss for wr in post_warmup_results
                    ]

                    if drift_gates.has_enough_losses_for_changepoint_glr(
                        len(adaptation_losses), min_segment_size
                    ):
                        adapt_changepoint, adapt_log_glr, adapt_all_log_glr, adapt_candidate_points = glrt_changepoint_detection(
                            adaptation_losses, min_segment_size=min_segment_size
                        )
                        glrt_adaptation_loss_changepoint_window = adapt_changepoint + self.drift_warmup_windows
                        glrt_adaptation_loss_likelihood = adapt_log_glr
                        glrt_adaptation_loss_all_log_glr = adapt_all_log_glr
                        glrt_adaptation_loss_candidate_points = adapt_candidate_points
                        glrt_adaptation_losses = adaptation_losses
                        logger.info(
                            f"GLRT (adaptation_loss) - Window {window_idx}: "
                            f"Changepoint at absolute window {glrt_adaptation_loss_changepoint_window} "
                            f"(post-warmup idx {adapt_changepoint}), Log-GLR: {adapt_log_glr:.4f}"
                        )

                        ref_changepoint, ref_log_glr, ref_all_log_glr, ref_candidate_points = glrt_changepoint_detection(
                            reference_metric_losses, min_segment_size=min_segment_size
                        )
                        glrt_reference_metric_changepoint_window = ref_changepoint + self.drift_warmup_windows
                        glrt_reference_metric_likelihood = ref_log_glr
                        glrt_reference_metric_all_log_glr = ref_all_log_glr
                        glrt_reference_metric_candidate_points = ref_candidate_points
                        glrt_reference_metric_losses = reference_metric_losses
                        current_glrt_likelihood = adapt_log_glr

                        self.glrt_history.append(adapt_log_glr)
                        if (
                            self.drift_history_max_size is not None
                            and len(self.glrt_history) > self.drift_history_max_size
                        ):
                            self.glrt_history = self.glrt_history[-self.drift_history_max_size:]

                        current_glrt_z_score = None
                        history_len = len(self.glrt_history)
                        if drift_gates.can_compute_drift_z_score(
                            history_len,
                            self.drift_guard_samples,
                        ):
                            current_glrt_z_score, baseline_mean, baseline_std = drift_gates.compute_drift_z_score(
                                adapt_log_glr,
                                self.glrt_history,
                                self.drift_guard_samples,
                            )
                            current_glrt_baseline_mean = baseline_mean

                            logger.info(
                                f"GLRT (adaptation_loss) - Window {window_idx}: "
                                f"Changepoint at absolute window {glrt_adaptation_loss_changepoint_window} "
                                f"(post-warmup idx {adapt_changepoint}), Log-GLR: {adapt_log_glr:.4f}, "
                                f"Z-score: {current_glrt_z_score:.4f} "
                                f"(baseline: {baseline_mean:.4f} ± {baseline_std:.4f}, "
                                f"guard_samples={self.drift_guard_samples})"
                            )

                            if (
                                current_glrt_z_score is not None
                                and current_glrt_z_score > self.drift_z_threshold
                                and self.glrt_drift_detection_window is None
                                and not self.drift_detected
                            ):
                                self.glrt_drift_detection_window = window_idx
                                self.glrt_z_score_at_detection = current_glrt_z_score

                                import math
                                base_lr = getattr(self.config.online_learning, "learning_rate", 1e-3)
                                if self.use_adaptive_learning_rate:
                                    G = adapt_log_glr if adapt_log_glr is not None else self.adaptive_lr_dG0
                                    dG = G - baseline_mean if baseline_mean is not None else 0.0
                                    log_lr_min = math.log10(self.adaptive_lr_min)
                                    log_lr_max = math.log10(self.adaptive_lr_max)
                                    log_lr = log_lr_min + (log_lr_max - log_lr_min) / (
                                        1.0 + math.exp(-self.adaptive_lr_k_sigmoid * (dG - self.adaptive_lr_dG0))
                                    )
                                    self.learning_rate_at_detection = 10 ** log_lr
                                else:
                                    self.learning_rate_at_detection = base_lr

                                time_to_learn = self.time_to_learn if self.time_to_learn is not None else 0
                                trigger_window = window_idx + time_to_learn
                                logger.info(
                                    f"Drift detected at window {window_idx} "
                                    f"(z={current_glrt_z_score:.4f} > {self.drift_z_threshold}). "
                                    f"Training starts at window {trigger_window} "
                                    f"(time_to_learn={time_to_learn}). "
                                    f"LR at detection: {self.learning_rate_at_detection:.6f}"
                                )

                                drift_detection_dicts.append({
                                    "eta": self.system_model.params.eta,
                                    "window_idx": window_idx,
                                    "baseline_mean": baseline_mean,
                                    "adaptation_log_glr": adapt_log_glr,
                                    "baseline_std": baseline_std,
                                    "current_glrt_z_score": current_glrt_z_score,
                                    "learning_rate_at_detection": self.learning_rate_at_detection,
                                })
                        else:
                            logger.info(
                                f"Drift baseline warmup - Window {window_idx}: "
                                f"collecting g-history ({history_len - self.drift_guard_samples if history_len > self.drift_guard_samples else 0}/"
                                f"{drift_gates.SCOPE_B_BASELINE_MIN_SAMPLES} baseline samples), "
                                f"z-score from window "
                                f"{drift_gates.first_z_window(self.drift_warmup_windows, self.drift_guard_samples)}"
                            )
                    else:
                        first_g = drift_gates.first_g_window(self.drift_warmup_windows)
                        logger.info(
                            f"Drift Scope A warmup - Window {window_idx}: "
                            f"need {2 * min_segment_size + 1} post-warmup losses "
                            f"(first g at window {first_g})"
                        )
                
                # Check if we've reached the window to trigger drift detection (detection_window + time_to_learn)
                # This check runs every window, not just when GLRT is calculated
                if self.glrt_drift_detection_window is not None and not self.drift_detected:
                    time_to_learn = self.time_to_learn if self.time_to_learn is not None else 0
                    trigger_window = self.glrt_drift_detection_window + time_to_learn
                    if window_idx >= trigger_window:
                        self.drift_detected = True
                        logger.info(f"Drift detection triggered at window {window_idx} (GLRT detected at window {self.glrt_drift_detection_window} + {time_to_learn} windows delay)")
                        # Initialize online EKF state with static model's current state
                        online_last_ekf_predictions = last_ekf_predictions
                        online_last_ekf_covariances = last_ekf_covariances
                        logger.info(f"Initialized online EKF state with static model's state at window {window_idx}")
                
                # Update last predictions and covariances for next window
                last_ekf_predictions = window_result.doa_metrics.ekf_predictions
                last_ekf_covariances = window_result.step_metrics.covariances
                
                logger.info(f"Window {window_idx}: Reference Metric = {window_result.loss_metrics.reference_metric_loss:.6f} ({window_result.loss_metrics.reference_metric_config}), Cov = {window_result.window_metrics.avg_covariance:.6f} (current eta={self.system_model.params.eta:.4f})")
                
                # Log all loss metrics for comparison
                logger.info(f"Window {window_idx}: Pre-EKF Loss = {window_result.loss_metrics.pre_ekf_loss:.6f}, Reference Metric = {window_result.loss_metrics.reference_metric_loss:.6f} ({window_result.loss_metrics.reference_metric_config}), Training Ref Loss = {window_result.loss_metrics.adaptation_loss:.6f} ({window_result.loss_metrics.adaptation_loss_config}), Cov = {window_result.window_metrics.avg_covariance:.6f} (eta={self.system_model.params.eta:.4f})")
                
                # Store trained model results for comparison
                trained_subspacenet_loss = window_result.loss_metrics.pre_ekf_loss
                trained_ekf_loss = window_result.loss_metrics.reference_metric_loss
                
                # Dual model processing logic
                if self.drift_detected:
                    if self.learning_done:
                        # Online model finished training, evaluate it normally
                        online_window_result, online_last_ekf_predictions, online_last_ekf_covariances = self._evaluate_and_record(
                            self.online_model, online_trajectory_results,
                            time_series_single_window, sources_num_single_window_list, labels_single_window_list_of_arrays,
                            trajectory_idx, window_idx, online_last_ekf_predictions, online_last_ekf_covariances,
                            model_label="Online model"
                        )
                        
                        log_online_learning_window_summary(
                            subspacenet_loss=trained_subspacenet_loss,
                            ekf_loss=trained_ekf_loss,
                            online_ekf_loss=online_window_result.loss_metrics.reference_metric_loss,
                            current_eta=self.system_model.params.eta,
                            is_near_field=hasattr(self.trained_model, 'field_type') and self.trained_model.field_type.lower() == "near",
                            trajectory_idx=trajectory_idx,
                            window_idx=window_idx,
                            is_learning=False
                        )
                        
                        # Evaluate supervised trained model (post-training)
                        _, supervised_last_ekf_predictions, supervised_last_ekf_covariances = self._evaluate_and_record(
                            self.supervised_trained_model, supervised_trajectory_results,
                            time_series_single_window, sources_num_single_window_list, labels_single_window_list_of_arrays,
                            trajectory_idx, window_idx, supervised_last_ekf_predictions, supervised_last_ekf_covariances,
                            model_label="Supervised trained model"
                        )
                    else:
                        # Online model still learning/adapting
                        logger.info(f"Training online model for window {window_idx}")
                        
                        # Track training start on first training call
                        if training_start_window is None:
                            training_start_window = window_idx
                            self.learning_start_window = window_idx
                            logger.info(f"Training started at window {window_idx}")
                        
                        # Call the refactored _online_training_window method with adaptive learning rate
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
                        
                        logger.info(f"Online training - Window {window_idx}: Reference Metric = {training_result.loss_metrics.reference_metric_loss:.6f} ({training_result.loss_metrics.reference_metric_config}), Cov = {training_result.window_metrics.avg_covariance:.6f}, Pre-EKF Loss = {training_result.loss_metrics.pre_ekf_loss:.6f}")
                        
                        # Log online learning window summary (learning phase)
                        log_online_learning_window_summary(
                            subspacenet_loss=trained_subspacenet_loss,
                            ekf_loss=trained_ekf_loss,
                            online_ekf_loss=training_result.loss_metrics.reference_metric_loss,
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
                        supervised_loss_config.adaptation_loss = self.config.online_learning.loss_config.supervised_offline_loss
                        
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
                            loss_config_override=supervised_loss_config,
                            increment_adaptation_count=False,
                        )
                        
                        # Add supervised training result to supervised trajectory results
                        supervised_trajectory_results.add_window_result(window_idx, supervised_training_result, self.system_model.params.eta, labels_single_window_list_of_arrays)
                        
                        logger.info(f"Supervised training - Window {window_idx}: Reference Metric = {supervised_training_result.loss_metrics.reference_metric_loss:.6f} ({supervised_training_result.loss_metrics.reference_metric_config}), Cov = {supervised_training_result.window_metrics.avg_covariance:.6f}, Pre-EKF Loss = {supervised_training_result.loss_metrics.pre_ekf_loss:.6f}")
                        
                        # Update supervised EKF state for next window
                        supervised_last_ekf_predictions = supervised_training_result.doa_metrics.ekf_predictions
                        supervised_last_ekf_covariances = supervised_training_result.step_metrics.covariances
            
            # Save final model if online training was performed
            if self.online_training_count > 0:
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
                    "window_count": len(online_learning_dataloader),
                    "window_size": online_config.window_size,
                    "stride": online_config.stride,
                    "learning_start_window": self.learning_start_window,
                    
                    # Learning state tracking
                    "drift_detected_final": self.drift_detected,
                    "learning_done_final": self.learning_done,
                    "first_eta_change_final": self.first_eta_change,
                    "online_training_count_final": self.online_training_count,
                    
                    # Training and eta change tracking
                    "eta_change_windows": eta_change_windows,
                    "drift_detection_window": self.glrt_drift_detection_window,
                    "training_start_window": training_start_window,
                    "training_end_window": training_end_window,
                    
                    # GLRT drift detection results
                    "glrt_adaptation_loss_changepoint_window": glrt_adaptation_loss_changepoint_window,
                    "glrt_adaptation_loss_likelihood": glrt_adaptation_loss_likelihood,
                    "glrt_adaptation_losses": glrt_adaptation_losses,
                    "glrt_reference_metric_changepoint_window": glrt_reference_metric_changepoint_window,
                    "glrt_reference_metric_likelihood": glrt_reference_metric_likelihood,
                    "glrt_reference_metric_losses": glrt_reference_metric_losses,
                    "glrt_loss_window_offset": self.drift_warmup_windows,
                    # GLRT z-score and learning rate at detection time
                    "glrt_z_score_at_detection": self.glrt_z_score_at_detection if hasattr(self, 'glrt_z_score_at_detection') else None,
                    "learning_rate_at_detection": self.learning_rate_at_detection if hasattr(self, 'learning_rate_at_detection') else None,
                    # Actual per-window LR used during training (for heatmap - matches what optimizer used)
                    "actual_lr_per_training_window": list(self.actual_lr_per_training_window) if hasattr(self, 'actual_lr_per_training_window') and self.actual_lr_per_training_window else [],
                    "drift_detection_dicts": drift_detection_dicts
                }
            }
            
        except Exception as e:
            logger.exception(f"Error during online learning: {e}")
            return {"status": "error", "message": str(e), "exception": type(e).__name__}

    def _online_training_window(self, window_time_series, window_sources_num, window_labels, trajectory_idx: int = 0, window_idx: int = 0, 
                               is_first_window: bool = True, last_ekf_predictions: Optional[torch.Tensor] = None, 
                               last_ekf_covariances: Optional[torch.Tensor] = None, model=None, loss_config_override=None,
                               increment_adaptation_count: bool = True) -> WindowEvaluationResult:
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
        # Increment online adaptation counter (supervised shadow model must not count)
        online_config = self.config.online_learning
        adaptation_window_count = getattr(online_config, "adaptation_window_count", 5)
        if increment_adaptation_count:
            self.online_training_count += 1
            logger.info(f"Online training step {self.online_training_count} called for trajectory {trajectory_idx}, window {window_idx}")
        else:
            logger.info(f"Supervised shadow training for trajectory {trajectory_idx}, window {window_idx}")
        
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
        
        # Get base learning rate from config or use default
        base_lr = getattr(self.config.online_learning, 'learning_rate', 1e-3)
        
        # Adaptive LR: use the value computed at detection time (clean pre-drift baseline)
        # rather than recomputing here where the baseline is contaminated by post-drift GLRT values.
        if self.use_adaptive_learning_rate:
            if self.learning_rate_at_detection is not None:
                adaptive_lr = self.learning_rate_at_detection
                logger.debug(f"Using adaptive LR from detection time: {adaptive_lr:.6f}")
            else:
                adaptive_lr = base_lr
                logger.warning(f"Adaptive LR requested but no detection-time LR available, using base_lr={base_lr:.6f}")
        else:
            adaptive_lr = base_lr
            logger.debug(f"Using fixed learning rate: {adaptive_lr:.6f} (adaptive LR disabled)")
        
        # Track actual LR used (for heatmap) - only when training online model to avoid duplicates
        if training_model is self.online_model and hasattr(self, 'actual_lr_per_training_window'):
            self.actual_lr_per_training_window.append(adaptive_lr)
        
        # Set up optimizer for training with adaptive learning rate
        if training_model is self.online_model:
            # Use online optimizer for online model
            if not hasattr(self, 'online_optimizer'):
                self.online_optimizer = optim.Adam(self.online_model.parameters(), lr=adaptive_lr)
            else:
                # Update learning rate for existing optimizer
                for param_group in self.online_optimizer.param_groups:
                    param_group['lr'] = adaptive_lr
            optimizer = self.online_optimizer
        elif training_model is self.supervised_trained_model:
            # Use supervised optimizer for supervised model
            if not hasattr(self, 'supervised_optimizer'):
                self.supervised_optimizer = optim.Adam(self.supervised_trained_model.parameters(), lr=adaptive_lr)
            else:
                # Update learning rate for existing optimizer
                for param_group in self.supervised_optimizer.param_groups:
                    param_group['lr'] = adaptive_lr
            optimizer = self.supervised_optimizer
        else:
            # Fallback: create a temporary optimizer
            optimizer = optim.Adam(training_model.parameters(), lr=adaptive_lr)
        
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
        
        num_gd_steps = self.num_gd_steps
        
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
                        # Forward pass with gradients + shape normalization + permutation alignment
                        angles_pred_tensor, true_angles_tensor = self._forward_and_align(
                            training_model, step_data_tensor, num_sources_this_step, true_angles_this_step, require_grad=True
                        )
                        
                        # ============ EKF Processing for Training ============
                        # Initialize training EKF filters (separate from evaluation EKF)
                        self.training_ekf_filters = self._initialize_ekf_filters(num_sources_this_step, window_idx, step)
                        
                        # Use the _initialize_ekf_state method to properly initialize state and covariance
                        # This ensures we use the last predictions and covariances from the previous window
                        self._initialize_ekf_state(
                            step=0,
                            num_sources_this_step=num_sources_this_step,
                            true_angles_this_step=true_angles_this_step,
                            ekf_filters=self.training_ekf_filters,
                            is_first_window=False,
                            last_ekf_predictions=last_ekf_predictions,
                            last_ekf_covariances=last_ekf_covariances,
                            window_idx=window_idx,
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
            
            self._initialize_ekf_state(
                step,
                num_sources_this_step,
                true_angles_this_step,
                ekf_filters,
                is_first_window,
                last_ekf_predictions,
                last_ekf_covariances,
                window_idx=window_idx,
            )
            
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
                       f"Reference Metric = {result.loss_metrics.reference_metric_loss:.6f} ({result.loss_metrics.reference_metric_config}), "
                       f"Avg Cov = {result.window_metrics.avg_covariance:.6f}, "
                       f"Training Loss = {avg_training_loss:.6f}")
        
        if increment_adaptation_count and self.online_training_count >= adaptation_window_count:
            self.learning_done = True
            logger.info(
                f"Online model training completed after {self.online_training_count} training windows "
                f"(adaptation_window_count={adaptation_window_count})"
            )
        
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

    def _evaluate_and_record(self, model, trajectory_results: 'TrajectoryResults',
                            time_series, sources_num, labels, trajectory_idx: int, window_idx: int,
                            last_ekf_predictions, last_ekf_covariances, model_label: str):
        """
        Evaluate a model on a window, record to trajectory results, and return updated EKF state.

        Returns:
            Tuple of (window_result, updated_ekf_predictions, updated_ekf_covariances)
        """
        window_result = self._evaluate_window(
            time_series, sources_num, labels,
            trajectory_idx, window_idx,
            is_first_window=(window_idx == 0),
            last_ekf_predictions=last_ekf_predictions,
            last_ekf_covariances=last_ekf_covariances,
            model=model
        )
        trajectory_results.add_window_result(window_idx, window_result, self.system_model.params.eta, labels)
        logger.info(f"{model_label} - Window {window_idx}: Reference Metric = {window_result.loss_metrics.reference_metric_loss:.6f} ({window_result.loss_metrics.reference_metric_config}), Cov = {window_result.window_metrics.avg_covariance:.6f}")
        return window_result, window_result.doa_metrics.ekf_predictions, window_result.step_metrics.covariances

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

    def _forward_and_align(self, model, step_data_tensor, num_sources: int, true_angles, require_grad: bool = False):
        """
        Run model forward pass and align predictions via optimal permutation.

        Args:
            model: The neural network model
            step_data_tensor: Input tensor [1, N, T]
            num_sources: Number of sources
            true_angles: Ground truth angles for permutation alignment
            require_grad: If True, run with gradients; if False, wrap in torch.no_grad()

        Returns:
            Tuple of (angles_pred_tensor, true_angles_tensor) both shaped [1, num_sources],
            with permutation applied.
        """
        import torch

        if require_grad:
            angles_pred, _, _ = model(step_data_tensor, num_sources)
        else:
            with torch.no_grad():
                angles_pred, _, _ = model(step_data_tensor, num_sources)

        # Shape normalization to [1, num_sources]
        if angles_pred.dim() == 3:
            angles_pred_tensor = angles_pred.squeeze(1)[:, :num_sources]
        elif angles_pred.dim() == 2:
            angles_pred_tensor = angles_pred.view(1, -1)[:, :num_sources]
        else:
            angles_pred_tensor = angles_pred.view(1, -1)[:, :num_sources]

        true_angles_tensor = torch.tensor(true_angles, device=device).unsqueeze(0)

        # Optimal permutation alignment
        angles_pred_np = angles_pred.detach().cpu().numpy().flatten()[:num_sources]
        model_perm = self._get_optimal_permutation(angles_pred_np, true_angles)
        angles_pred_tensor = angles_pred_tensor[:, model_perm]

        return angles_pred_tensor, true_angles_tensor

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
        # Trajectory step index for this window/step (matches sliding-window start in the dataset).
        stride = getattr(self.config.online_learning, "stride", self.config.online_learning.window_size)
        initial_time = window_idx * stride + step_idx
        
        ekf_filters = []
        
        # Create EKF filters for each source with source-specific parameters
        for i in range(max_sources):
            # Create EKF filter with source index i - the filter will use source-specific parameters
            ekf_filter = ExtendedKalmanFilter1D.create_from_config(
                self.config, 
                trajectory_type=self.config.trajectory.trajectory_type,
                device=device,
                source_idx=i,
                initial_time=initial_time,
                dc_offsets=getattr(self, "_trajectory_dc_offsets", None),
            )
            ekf_filters.append(ekf_filter)
        
        current_eta = self.system_model.params.eta
        logger.info(f"Initialized {max_sources} EKF instances for window {window_idx}, step {step_idx} (initial_time={initial_time}, eta={current_eta:.4f})")
        
        return ekf_filters

    def _initialize_ekf_state(
        self,
        step: int,
        num_sources_this_step: int,
        true_angles_this_step: np.ndarray,
        ekf_filters: List[ExtendedKalmanFilter1D],
        is_first_window: bool,
        last_ekf_predictions: Optional[List],
        last_ekf_covariances: Optional[List],
        window_idx: int = 0,
    ) -> None:
        """
        Initialize EKF state for the current step.

        Handoff (window_idx > 0, step == 0): restore x, P, and motion-model time from the
        previous window's posterior. Skip re-filtering that overlapping step (see
        _process_single_step) since stride < window_size revisits the same trajectory index.
        """
        stride = self.config.online_learning.stride
        global_step = window_idx * stride + step
        self._handoff_reuse_step0 = False

        if step != 0:
            return

        if is_first_window:
            for i in range(num_sources_this_step):
                ekf_filters[i].initialize_state(true_angles_this_step[i])
                if hasattr(ekf_filters[i].state_model, "reset_time"):
                    ekf_filters[i].state_model.reset_time(global_step)
            return

        if (
            last_ekf_predictions is not None
            and last_ekf_covariances is not None
            and last_ekf_predictions.shape[0] > 0
            and last_ekf_covariances.shape[0] > 0
            and last_ekf_predictions.shape[1] >= num_sources_this_step
            and last_ekf_covariances.shape[1] >= num_sources_this_step
        ):
            last_predictions_pre_perm = last_ekf_predictions[-1, :num_sources_this_step]
            true_angles_tensor = torch.tensor(true_angles_this_step, device=last_predictions_pre_perm.device)
            last_perm = self._get_optimal_permutation_tensor(last_predictions_pre_perm, true_angles_tensor)
            last_predictions = last_predictions_pre_perm[last_perm]
            last_covariances_pre_perm = last_ekf_covariances[-1, :num_sources_this_step]
            last_covariances = last_covariances_pre_perm[last_perm]

            next_predict_t = global_step + 1
            for i in range(num_sources_this_step):
                ekf_filters[i].restore_handoff_state(
                    last_predictions.flatten()[i].item(),
                    last_covariances.flatten()[i].item(),
                    next_predict_t,
                )
            self._handoff_reuse_step0 = True
        else:
            logger.warning("No valid last predictions or covariances available, falling back to true angles")
            for i in range(num_sources_this_step):
                ekf_filters[i].initialize_state(true_angles_this_step.flatten()[i].flatten())
                if hasattr(ekf_filters[i].state_model, "reset_time"):
                    ekf_filters[i].state_model.reset_time(global_step)

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
                    # Weight comparison check (debug/safety)
                    model_state = model.state_dict()
                    trained_state = self.trained_model.state_dict()
                    weights_equal = all(torch.equal(model_state[key], trained_state[key]) for key in model_state.keys())

                    if weights_equal and not Pretrained_model:
                        logger.error("Model and trained model have the same weights - online model was not properly copied!")
                        raise RuntimeError("Online model and trained model have identical weights. This indicates the online model was not properly initialized as an independent copy. Cannot proceed with online learning.")
                    elif Pretrained_model:
                        logger.info("The evaluated model is not the online model,online model is not initialized yet or this is evaluation comparison")
                    else:
                        logger.info("Model and trained model dont have the same weights - online model is properly initialized")

                    # Forward pass + shape normalization + permutation alignment
                    pre_ekf_angles_pred, true_angles_tensor = self._forward_and_align(
                        model, step_data_tensor, num_sources_this_step, true_angles_this_step, require_grad=False
                    )
                
                    # EKF update for each source - use tensor directly
                    step_predictions = []
                    step_covariances = []
                    step_innovations = []
                    step_kalman_gains = []
                    step_kalman_gain_times_innovation = []
                    step_y_s_inv_y = []
                    step_Innovation_Covariance = []
                    
                    for i in range(num_sources_this_step):
                        measurement = pre_ekf_angles_pred.flatten()[i]

                        if getattr(self, "_handoff_reuse_step0", False) and step == 0:
                            # Posterior already at this trajectory step — do not re-predict/update.
                            ekf_filter = ekf_filters[i]
                            updated_angle = ekf_filter.x
                            innovation = measurement - updated_angle
                            kalman_gain = ekf_filter.P / (ekf_filter.P + ekf_filter.R)
                            kalman_gain_times_innovation = kalman_gain * innovation
                            y_s_inv_y = innovation * (innovation / (ekf_filter.P + ekf_filter.R))
                            Innovation_Covariance = ekf_filter.P + ekf_filter.R
                            predicted_angle = updated_angle
                        else:
                            predicted_angle, updated_angle, innovation, kalman_gain, kalman_gain_times_innovation, y_s_inv_y, Innovation_Covariance = ekf_filters[i].predict_and_update(
                                measurement=measurement,
                                true_state=true_angles_this_step[i],
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
                reference_metric_loss=float('inf'),
                reference_metric_loss_db=float('inf'),
                reference_metric_config="no_valid_steps",
                adaptation_loss=float('inf'),
                adaptation_loss_config="no_valid_steps",
                pre_ekf_loss=float('inf'),
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
        self._handoff_reuse_step0 = False
        for step in range(current_window_len):
            # Initialize EKF state if this is the first step
            num_sources_this_step = sources_num_per_step[step]
            true_angles_this_step = labels_per_step_list[step][:num_sources_this_step]
            self._initialize_ekf_state(
                step,
                num_sources_this_step,
                true_angles_this_step,
                ekf_filters,
                is_first_window,
                last_ekf_predictions,
                last_ekf_covariances,
                window_idx=window_idx,
            )
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
        
        if loss_config is None:
            raise RuntimeError("loss_config is required for _calculate_all_losses but was None")

        reference_metric_config = loss_config.reference_metric
        reference_metric_loss = _compute_loss_from_type(
            reference_metric_config, ekf_preds, pre_ekf_preds, true_angles,
            window_size, rmspe_criterion, rmape_criterion, loss_config,
        )

        import math
        reference_metric_loss_value = (
            reference_metric_loss.item() if hasattr(reference_metric_loss, "item") else reference_metric_loss
        )
        reference_metric_loss_db = 20 * math.log10(max(reference_metric_loss_value, 1e-12))

        adaptation_loss_config = loss_config.adaptation_loss
        adaptation_loss = _compute_loss_from_type(
            adaptation_loss_config, ekf_preds, pre_ekf_preds, true_angles,
            window_size, rmspe_criterion, rmape_criterion, loss_config,
        )

        pre_ekf_loss = rmspe_criterion(pre_ekf_preds, true_angles) / window_size
        ekf_gain_rmape = rmape_criterion(ekf_preds, pre_ekf_preds) / window_size

        return LossMetrics(
            reference_metric_loss=reference_metric_loss_value,
            reference_metric_loss_db=reference_metric_loss_db,
            reference_metric_config=reference_metric_config,
            adaptation_loss=adaptation_loss.item() if hasattr(adaptation_loss, "item") else adaptation_loss,
            adaptation_loss_config=adaptation_loss_config,
            pre_ekf_loss=pre_ekf_loss.item() if hasattr(pre_ekf_loss, "item") else pre_ekf_loss,
            ekf_gain_rmape=ekf_gain_rmape.item() if hasattr(ekf_gain_rmape, "item") else ekf_gain_rmape,
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
        
        window_size = window_pre_ekf_preds.shape[0]

        if loss_config is not None and hasattr(loss_config, "adaptation_loss"):
            loss_type = loss_config.adaptation_loss
            return _compute_loss_from_type(
                loss_type, window_ekf_preds, window_pre_ekf_preds, window_true_angles,
                window_size, rmspe_criterion, rmape_criterion, loss_config,
            )
        return rmspe_criterion(window_ekf_preds, window_pre_ekf_preds) / window_size



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



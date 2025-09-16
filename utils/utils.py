import numpy as np
import datetime
import torch
import logging
from pathlib import Path


def log_window_summary(
    loss_metrics,
    avg_window_cov: float,
    current_eta: float,
    is_near_field: bool,
    trajectory_idx: int = 0,
    window_idx: int = 0
) -> None:
    """
    Log window summary results in a columnar format similar to evaluation results.
    
    Args:
        loss_metrics: LossMetrics object containing all loss information
        avg_window_cov: Average covariance for the window
        current_eta: Current eta value
        is_near_field: Whether this is near field scenario
        trajectory_idx: Index of the current trajectory
        window_idx: Index of the current window within the trajectory
    """
    print(f"\n{'Online Mode; Vs Pretrained Model SUMMARY - WINDOW ' + str(window_idx) + ' TRAJECTORY ' + str(trajectory_idx):^100}")
    print("-"*100)
    print(f"{'Metric':<25} {'Loss Value':<20} {'Loss (degrees)':<25} {'Config':<15} {'Additional Info':<15}")
    print("-"*100)
    
    if not is_near_field:
        # Convert losses to degrees
        pre_ekf_loss_degrees = loss_metrics.pre_ekf_loss * 180 / np.pi
        main_loss_degrees = loss_metrics.main_loss * 180 / np.pi
        ekf_gain_rmspe_degrees = loss_metrics.ekf_gain_rmspe * 180 / np.pi
        ekf_gain_rmape_degrees = loss_metrics.ekf_gain_rmape * 180 / np.pi
        
        # Display all loss metrics
        print(f"{'Supervised Loss':<25} {loss_metrics.main_loss:<20.6f} {main_loss_degrees:<25.6f} {loss_metrics.main_loss_config:<15} {f'w: {window_idx}':<15}")
        print(f"{'Unsupervised loss':<25} {loss_metrics.online_training_reference_loss:<20.6f} {loss_metrics.online_training_reference_loss * 180 / np.pi:<25.6f} {loss_metrics.online_training_reference_loss_config:<15} {f't: {trajectory_idx}':<15}")
        print(f"{'EKF Gain (RMSPE)':<25} {loss_metrics.ekf_gain_rmspe:<20.6f} {ekf_gain_rmspe_degrees:<25.6f} {'N/A':<15} {f'Cov: {avg_window_cov:.2e}':<15}")
        print(f"{'EKF Gain (RMAPE)':<25} {loss_metrics.ekf_gain_rmape:<20.6f} {ekf_gain_rmape_degrees:<25.6f} {'N/A':<15} {'':<15}")
        
        # Display EKF improvement analysis
        if ekf_gain_rmspe_degrees < 0:
            improvement_text = f"EKF improves by {abs(ekf_gain_rmspe_degrees):.4f}°"
            status_icon = "✓"
        else:
            improvement_text = f"EKF degrades by {ekf_gain_rmspe_degrees:.4f}°"
            status_icon = "✗"
        
        print(f"{'EKF Performance':<25} {improvement_text:<45} {status_icon:<15} {'':<15}")
        print("-" * 100)
    else:
        # Near field - display available metrics
        main_loss_degrees = loss_metrics.main_loss * 180 / np.pi
        print(f"{'Supervised Loss':<25} {loss_metrics.main_loss:<20.6f} {main_loss_degrees:<25.6f} {loss_metrics.main_loss_config:<15} {f'eta: {current_eta:.4f}':<15}")
        print(f"{'Unsupervised Loss':<25} {loss_metrics.online_training_reference_loss:<20.6f} {loss_metrics.online_training_reference_loss * 180 / np.pi:<25.6f} {loss_metrics.online_training_reference_loss_config:<15} {f'w: {window_idx}':<15}")
        print(f"{'Mode':<25} {'NEAR FIELD':<20} {'(No SubspaceNet comparison)':<25} {'N/A':<15} {f't: {trajectory_idx}':<15}")
        print("-" * 100)


def save_model_state(model, output_dir, model_type=None):
    """
    Save model state dictionary to file.
    
    Args:
        model: Model to save
        output_dir: Output directory for saving model
        model_type: Type identifier for filename
        
    Returns:
        Path to saved model file
    """
    logger = logging.getLogger(__name__)
    
    if model_type is None:
        model_type = "model"
    
    # Create timestamp for unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create model save directory
    model_save_dir = Path(output_dir) / "checkpoints"
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


def log_online_learning_window_summary(
    subspacenet_loss: float,
    ekf_loss: float,
    online_ekf_loss: float,
    current_eta: float,
    is_near_field: bool,
    trajectory_idx: int = 0,
    window_idx: int = 0,
    is_learning: bool = False
) -> None:
    """
    Log online learning window summary results comparing SubspaceNet, EKF, and online learning EKF losses.
    
    Args:
        subspacenet_loss: SubspaceNet loss for the window
        ekf_loss: EKF loss for the window (trained model)
        online_ekf_loss: Online learning EKF loss for the window
        current_eta: Current eta value
        is_near_field: Whether this is near field scenario
        trajectory_idx: Index of the current trajectory
        window_idx: Index of the current window within the trajectory
        is_learning: Whether this is during learning phase (True) or post-learning evaluation (False)
    """
    print(f"\n{'ONLINE LEARNING WINDOW SUMMARY - WINDOW ' + str(window_idx) + ' TRAJECTORY ' + str(trajectory_idx):^100}")
    print("-"*100)
    print(f"{'Metric':<20} {'Loss Value':<20} {'Loss (degrees)':<25} {'Additional Info':<30}")
    print("-"*100)
    
    if not is_near_field:
        # Convert losses to degrees
        subspacenet_loss_degrees = subspacenet_loss * 180 / np.pi
        ekf_loss_degrees = ekf_loss * 180 / np.pi
        online_ekf_loss_degrees = online_ekf_loss * 180 / np.pi
        
        # Determine best method
        losses = [subspacenet_loss_degrees, ekf_loss_degrees, online_ekf_loss_degrees]
        methods = ["SubspaceNet", "Pre-trained Model", "Online Model"]
        best_idx = losses.index(min(losses))
        best_method = methods[best_idx]
        best_loss_degrees = min(losses)
        
        # Calculate improvements
        online_vs_subspacenet = online_ekf_loss_degrees - subspacenet_loss_degrees
        online_vs_ekf = online_ekf_loss_degrees - ekf_loss_degrees
        
        # Display individual losses
        #print(f"{'SubspaceNet Loss':<20} {subspacenet_loss:<20.6f} {subspacenet_loss_degrees:<25.6f} {f'eta: {current_eta:.4f}, w: {window_idx}, t: {trajectory_idx}':<30}")
        print(f"{'pre-trained Model Supervised Loss ':<20} {ekf_loss:<20.6f} {ekf_loss_degrees:<25.6f} {f'w: {window_idx}, t: {trajectory_idx}':<30}")
        
        # Display online learning status
        if is_learning:
            status_text = "LEARNING PHASE"
            print(f"{'Online Model Supervised Loss':<20} {online_ekf_loss:<20.6f} {online_ekf_loss_degrees:<25.6f} {f'{status_text}, w: {window_idx}, t: {trajectory_idx}':<30}")
        else:
            status_text = "POST-LEARNING"
            print(f"{'Online Model Supervised Loss':<20} {online_ekf_loss:<20.6f} {online_ekf_loss_degrees:<25.6f} {f'{status_text}, w: {window_idx}, t: {trajectory_idx}':<30}")
        
        # Display comparison results
        print(f"{'WINNER':<20} {best_method:<20} {best_loss_degrees:<25.6f} {'w: ' + str(window_idx) + ', t: ' + str(trajectory_idx):<30}")
        
        # Show online vs others comparison
        if online_vs_subspacenet < 0:
            online_vs_subspacenet_text = f"Online Model better than SubspaceNet by {abs(online_vs_subspacenet):.4f}°"
        else:
            online_vs_subspacenet_text = f"Online Model worse than SubspaceNet by {online_vs_subspacenet:.4f}°"
            
        if online_vs_ekf < 0:
            online_vs_ekf_text = f"Online better than Pretrained by {abs(online_vs_ekf):.4f}°"
        else:
            online_vs_ekf_text = f"Online worse than Pretrained by {online_vs_ekf:.4f}°"
        
        print(f"{'Online Model vs SubspaceNet':<20} {online_vs_subspacenet_text:<45} {'w: ' + str(window_idx) + ', t: ' + str(trajectory_idx):<30}")
        print(f"{'Online Model vs Pretrained':<20} {online_vs_ekf_text:<45} {'w: ' + str(window_idx) + ', t: ' + str(trajectory_idx):<30}")
        print("-" * 100)
    else:
        # Near field - only EKF losses (no SubspaceNet comparison available)
        ekf_loss_degrees = ekf_loss * 180 / np.pi
        online_ekf_loss_degrees = online_ekf_loss * 180 / np.pi
        
        # Determine best method for near field
        if online_ekf_loss_degrees < ekf_loss_degrees:
            best_method = "Online Model"
            best_loss_degrees = online_ekf_loss_degrees
            improvement = ekf_loss_degrees - online_ekf_loss_degrees
            status_text = f"Online better by {improvement:.4f}°"
        else:
            best_method = "Pretrained Model"
            best_loss_degrees = ekf_loss_degrees
            degradation = online_ekf_loss_degrees - ekf_loss_degrees
            status_text = f"Online Model worse by {degradation:.4f}°"
        
        print(f"{'Pretrained model':<20} {ekf_loss:<20.6f} {ekf_loss_degrees:<25.6f} {f'eta: {current_eta:.4f}, w: {window_idx}, t: {trajectory_idx}':<30}")
        
        if is_learning:
            status_text_full = "LEARNING PHASE"
        else:
            status_text_full = "POST-LEARNING"
            
        print(f"{'Online Model':<20} {online_ekf_loss:<20.6f} {online_ekf_loss_degrees:<25.6f} {f'{status_text_full}, w: {window_idx}, t: {trajectory_idx}':<30}")
        print(f"{'WINNER':<20} {best_method:<20} {best_loss_degrees:<25.6f} {status_text + ', w: ' + str(window_idx) + ', t: ' + str(trajectory_idx):<30}")
        print(f"{'Mode':<20} {'NEAR FIELD':<20} {'(No SubspaceNet comparison)':<25} {'w: ' + str(window_idx) + ', t: ' + str(trajectory_idx):<30}")
        print("-" * 100)


def average_online_learning_results_across_trajectories(results_list: list) -> dict:
    """
    Average online learning results across multiple trajectories.
    
    This method takes a list of trajectory results and computes averaged metrics
    across all trajectories for more robust analysis.
    
    Args:
        results_list: List of dictionaries containing results from each trajectory.
                     Each dictionary should have the structure returned by 
                     _run_single_trajectory_online_learning()
    
    Returns:
        Dictionary with averaged results containing:
        - averaged_pretrained_trajectory: Averaged metrics from pretrained model
        - averaged_online_trajectory: Averaged metrics from online model  
        - summary_statistics: Overall statistics across trajectories
        - trajectory_count: Number of trajectories averaged
    """
    import numpy as np
    from dataclasses import dataclass
    from typing import List, Optional
    
    logger = logging.getLogger(__name__)
    
    if not results_list:
        logger.warning("Empty results list provided for averaging")
        return {"status": "error", "message": "No results to average"}
    
    logger.info(f"Averaging results across {len(results_list)} trajectories")
    
    # Extract trajectory results from each result
    pretrained_trajectories = []
    online_trajectories = []
    metadata_list = []
    
    for result in results_list:
        if result.get("status") != "success":
            logger.warning(f"Skipping failed trajectory result: {result.get('message', 'Unknown error')}")
            continue
            
        ol_results = result["online_learning_results"]
        pretrained_trajectories.append(ol_results["pretrained_model_trajectory_results"])
        online_trajectories.append(ol_results["online_model_trajectory_results"])
        
        # Extract metadata
        metadata = {
            "drift_detected_count": ol_results.get("drift_detected_count", 0),
            "model_updated_count": ol_results.get("model_updated_count", 0),
            "window_count": ol_results.get("window_count", 0),
            "window_size": ol_results.get("window_size", 0),
            "stride": ol_results.get("stride", 0),
            "loss_threshold": ol_results.get("loss_threshold", 0.0),
        }
        metadata_list.append(metadata)
    
    if not pretrained_trajectories:
        logger.error("No valid trajectory results found for averaging")
        return {"status": "error", "message": "No valid trajectory results"}
    
    # Average pretrained model trajectories
    averaged_pretrained = _average_trajectory_results(pretrained_trajectories, "pretrained")
    
    # Average online model trajectories  
    averaged_online = _average_trajectory_results(online_trajectories, "online")
    
    # Calculate summary statistics
    summary_stats = _calculate_trajectory_summary_statistics(metadata_list)
    
    logger.info(f"Successfully averaged results from {len(pretrained_trajectories)} trajectories")
    
    return {
        "status": "success",
        "averaged_results": {
            "averaged_pretrained_trajectory": averaged_pretrained,
            "averaged_online_trajectory": averaged_online,
            "summary_statistics": summary_stats,
            "trajectory_count": len(pretrained_trajectories)
        }
    }


def _average_trajectory_results(trajectory_list: list, model_type: str) -> dict:
    """
    Average trajectory results for a specific model type (pretrained or online).
    
    Args:
        trajectory_list: List of TrajectoryResults objects
        model_type: Type of model ("pretrained" or "online")
        
    Returns:
        Dictionary with averaged trajectory metrics
    """
    import numpy as np
    
    logger = logging.getLogger(__name__)
    
    if not trajectory_list:
        return {}
    
    # Get the number of windows from the first trajectory
    num_windows = len(trajectory_list[0].window_results)
    
    # Initialize averaged metrics (focus on meaningful metrics, not angle predictions)
    averaged_metrics = {
        "window_indices": [],
        "window_eta_values": [],
        "main_losses": [],
        "main_losses_db": [],
        "training_reference_losses": [],
        "avg_covariances": [],
        "ekf_gain_rmspe": [],
        "ekf_gain_rmape": [],
        "avg_innovations": [],
        "avg_kalman_gains": [],
        "avg_kalman_gain_times_innovation": [],
        "avg_y_s_inv_y": [],
    }
    
    # Average across all windows
    for window_idx in range(num_windows):
        # Collect metrics from all trajectories for this window
        window_main_losses = []
        window_main_losses_db = []
        window_training_ref_losses = []
        window_covariances = []
        window_ekf_gains_rmspe = []
        window_ekf_gains_rmape = []
        window_eta_values = []
        window_indices = []
        window_innovations = []
        window_kalman_gains = []
        window_kalman_gain_times_innovation = []
        window_y_s_inv_y = []
        
        valid_trajectories = 0
        
        for traj in trajectory_list:
            if window_idx < len(traj.window_results):
                window_result = traj.window_results[window_idx]
                
                if window_result.is_valid:
                    # Loss metrics
                    window_main_losses.append(window_result.loss_metrics.main_loss)
                    window_main_losses_db.append(window_result.loss_metrics.main_loss_db)
                    window_training_ref_losses.append(window_result.loss_metrics.online_training_reference_loss)
                    
                    # Window metrics
                    window_covariances.append(window_result.window_metrics.avg_covariance)
                    
                    # EKF gains
                    window_ekf_gains_rmspe.append(window_result.loss_metrics.ekf_gain_rmspe)
                    window_ekf_gains_rmape.append(window_result.loss_metrics.ekf_gain_rmape)
                    
                    # Eta values and indices
                    window_eta_values.append(traj.window_eta_values[window_idx])
                    window_indices.append(traj.window_indices[window_idx])
                    
                    # EKF metrics averages (focus on performance metrics, not angle predictions)
                    if window_result.window_metrics.avg_ekf_innovations is not None:
                        window_innovations.append(np.mean(window_result.window_metrics.avg_ekf_innovations))
                    if window_result.window_metrics.avg_ekf_kalman_gains is not None:
                        window_kalman_gains.append(np.mean(window_result.window_metrics.avg_ekf_kalman_gains))
                    if window_result.window_metrics.avg_ekf_kalman_gain_times_innovation is not None:
                        window_kalman_gain_times_innovation.append(np.mean(window_result.window_metrics.avg_ekf_kalman_gain_times_innovation))
                    if window_result.window_metrics.avg_ekf_y_s_inv_y is not None:
                        window_y_s_inv_y.append(np.mean(window_result.window_metrics.avg_ekf_y_s_inv_y))
                    
                    valid_trajectories += 1
        
        # Calculate averages for this window
        if valid_trajectories > 0:
            averaged_metrics["window_indices"].append(int(np.mean(window_indices)) if window_indices else window_idx)
            averaged_metrics["window_eta_values"].append(np.mean(window_eta_values) if window_eta_values else 0.0)
            averaged_metrics["main_losses"].append(np.mean(window_main_losses) if window_main_losses else 0.0)
            averaged_metrics["main_losses_db"].append(np.mean(window_main_losses_db) if window_main_losses_db else 0.0)
            averaged_metrics["training_reference_losses"].append(np.mean(window_training_ref_losses) if window_training_ref_losses else 0.0)
            averaged_metrics["avg_covariances"].append(np.mean(window_covariances) if window_covariances else 0.0)
            averaged_metrics["ekf_gain_rmspe"].append(np.mean(window_ekf_gains_rmspe) if window_ekf_gains_rmspe else 0.0)
            averaged_metrics["ekf_gain_rmape"].append(np.mean(window_ekf_gains_rmape) if window_ekf_gains_rmape else 0.0)
            
            # Average step-level performance metrics
            averaged_metrics["avg_innovations"].append(np.mean(window_innovations) if window_innovations else 0.0)
            averaged_metrics["avg_kalman_gains"].append(np.mean(window_kalman_gains) if window_kalman_gains else 0.0)
            averaged_metrics["avg_kalman_gain_times_innovation"].append(np.mean(window_kalman_gain_times_innovation) if window_kalman_gain_times_innovation else 0.0)
            averaged_metrics["avg_y_s_inv_y"].append(np.mean(window_y_s_inv_y) if window_y_s_inv_y else 0.0)
        else:
            logger.warning(f"No valid trajectories found for {model_type} model at window {window_idx}")
    
    logger.info(f"Averaged {model_type} model results across {len(trajectory_list)} trajectories, {num_windows} windows")
    
    return averaged_metrics


def _calculate_trajectory_summary_statistics(metadata_list: list) -> dict:
    """
    Calculate summary statistics across all trajectories.
    
    Args:
        metadata_list: List of metadata dictionaries from each trajectory
        
    Returns:
        Dictionary with summary statistics
    """
    import numpy as np
    
    if not metadata_list:
        return {}
    
    # Extract statistics
    drift_counts = [meta["drift_detected_count"] for meta in metadata_list]
    model_update_counts = [meta["model_updated_count"] for meta in metadata_list]
    window_counts = [meta["window_count"] for meta in metadata_list]
    loss_thresholds = [meta["loss_threshold"] for meta in metadata_list]
    
    summary = {
        "total_trajectories": len(metadata_list),
        "avg_drift_detected": np.mean(drift_counts),
        "std_drift_detected": np.std(drift_counts),
        "avg_model_updates": np.mean(model_update_counts),
        "std_model_updates": np.std(model_update_counts),
        "avg_window_count": np.mean(window_counts),
        "avg_loss_threshold": np.mean(loss_thresholds),
        "total_drift_detected": sum(drift_counts),
        "total_model_updates": sum(model_update_counts),
    }
    
    return summary

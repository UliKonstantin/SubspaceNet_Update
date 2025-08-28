import numpy as np
import datetime
import torch
import logging
from pathlib import Path


def log_window_summary(
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
        methods = ["SubspaceNet", "EKF (Trained)", "EKF (Online)"]
        best_idx = losses.index(min(losses))
        best_method = methods[best_idx]
        best_loss_degrees = min(losses)
        
        # Calculate improvements
        online_vs_subspacenet = online_ekf_loss_degrees - subspacenet_loss_degrees
        online_vs_ekf = online_ekf_loss_degrees - ekf_loss_degrees
        
        # Display individual losses
        print(f"{'SubspaceNet Loss':<20} {subspacenet_loss:<20.6f} {subspacenet_loss_degrees:<25.6f} {f'eta: {current_eta:.4f}, w: {window_idx}, t: {trajectory_idx}':<30}")
        print(f"{'EKF Loss (Trained)':<20} {ekf_loss:<20.6f} {ekf_loss_degrees:<25.6f} {f'w: {window_idx}, t: {trajectory_idx}':<30}")
        
        # Display online learning status
        if is_learning:
            status_text = "LEARNING PHASE"
            print(f"{'EKF Loss (Online)':<20} {online_ekf_loss:<20.6f} {online_ekf_loss_degrees:<25.6f} {f'{status_text}, w: {window_idx}, t: {trajectory_idx}':<30}")
        else:
            status_text = "POST-LEARNING"
            print(f"{'EKF Loss (Online)':<20} {online_ekf_loss:<20.6f} {online_ekf_loss_degrees:<25.6f} {f'{status_text}, w: {window_idx}, t: {trajectory_idx}':<30}")
        
        # Display comparison results
        print(f"{'WINNER':<20} {best_method:<20} {best_loss_degrees:<25.6f} {'w: ' + str(window_idx) + ', t: ' + str(trajectory_idx):<30}")
        
        # Show online vs others comparison
        if online_vs_subspacenet < 0:
            online_vs_subspacenet_text = f"Online better than SubspaceNet by {abs(online_vs_subspacenet):.4f}°"
        else:
            online_vs_subspacenet_text = f"Online worse than SubspaceNet by {online_vs_subspacenet:.4f}°"
            
        if online_vs_ekf < 0:
            online_vs_ekf_text = f"Online better than EKF by {abs(online_vs_ekf):.4f}°"
        else:
            online_vs_ekf_text = f"Online worse than EKF by {online_vs_ekf:.4f}°"
        
        print(f"{'Online vs SubspaceNet':<20} {online_vs_subspacenet_text:<45} {'w: ' + str(window_idx) + ', t: ' + str(trajectory_idx):<30}")
        print(f"{'Online vs EKF (Trained)':<20} {online_vs_ekf_text:<45} {'w: ' + str(window_idx) + ', t: ' + str(trajectory_idx):<30}")
        print("-" * 100)
    else:
        # Near field - only EKF losses (no SubspaceNet comparison available)
        ekf_loss_degrees = ekf_loss * 180 / np.pi
        online_ekf_loss_degrees = online_ekf_loss * 180 / np.pi
        
        # Determine best method for near field
        if online_ekf_loss_degrees < ekf_loss_degrees:
            best_method = "EKF (Online)"
            best_loss_degrees = online_ekf_loss_degrees
            improvement = ekf_loss_degrees - online_ekf_loss_degrees
            status_text = f"Online better by {improvement:.4f}°"
        else:
            best_method = "EKF (Trained)"
            best_loss_degrees = ekf_loss_degrees
            degradation = online_ekf_loss_degrees - ekf_loss_degrees
            status_text = f"Online worse by {degradation:.4f}°"
        
        print(f"{'EKF Loss (Trained)':<20} {ekf_loss:<20.6f} {ekf_loss_degrees:<25.6f} {f'eta: {current_eta:.4f}, w: {window_idx}, t: {trajectory_idx}':<30}")
        
        if is_learning:
            status_text_full = "LEARNING PHASE"
        else:
            status_text_full = "POST-LEARNING"
            
        print(f"{'EKF Loss (Online)':<20} {online_ekf_loss:<20.6f} {online_ekf_loss_degrees:<25.6f} {f'{status_text_full}, w: {window_idx}, t: {trajectory_idx}':<30}")
        print(f"{'WINNER':<20} {best_method:<20} {best_loss_degrees:<25.6f} {status_text + ', w: ' + str(window_idx) + ', t: ' + str(trajectory_idx):<30}")
        print(f"{'Mode':<20} {'NEAR FIELD':<20} {'(No SubspaceNet comparison)':<25} {'w: ' + str(window_idx) + ', t: ' + str(trajectory_idx):<30}")
        print("-" * 100)

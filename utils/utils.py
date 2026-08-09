import numpy as np
import datetime
import torch
import logging
from pathlib import Path
from typing import Optional

from utils.drift_gates import GLRT_MIN_SEGMENT_SIZE


def log_window_summary(
    loss_metrics,
    avg_window_cov: float,
    current_eta: float,
    is_near_field: bool,
    trajectory_idx: int = 0,
    window_idx: int = 0
) -> None:
    """Log window summary results in a columnar format."""
    print(f"\n{'Online Mode; Vs Pretrained Model SUMMARY - WINDOW ' + str(window_idx) + ' TRAJECTORY ' + str(trajectory_idx):^100}")
    print("-"*100)
    print(f"{'Metric':<25} {'Loss Value':<20} {'Loss (degrees)':<25} {'Config':<15} {'Additional Info':<15}")
    print("-"*100)

    if not is_near_field:
        ref_deg = loss_metrics.reference_metric_loss * 180 / np.pi
        adapt_deg = loss_metrics.adaptation_loss * 180 / np.pi
        rmape_deg = loss_metrics.ekf_gain_rmape * 180 / np.pi

        print(f"{'Reference metric':<25} {loss_metrics.reference_metric_loss:<20.6f} {ref_deg:<25.6f} {loss_metrics.reference_metric_config:<15} {f'w: {window_idx}':<15}")
        print(f"{'Adaptation loss':<25} {loss_metrics.adaptation_loss:<20.6f} {adapt_deg:<25.6f} {loss_metrics.adaptation_loss_config:<15} {f't: {trajectory_idx}':<15}")
        print(f"{'EKF gain (RMAPE)':<25} {loss_metrics.ekf_gain_rmape:<20.6f} {rmape_deg:<25.6f} {'N/A':<15} {f'Cov: {avg_window_cov:.2e}':<15}")
        print("-" * 100)
    else:
        ref_deg = loss_metrics.reference_metric_loss * 180 / np.pi
        print(f"{'Reference metric':<25} {loss_metrics.reference_metric_loss:<20.6f} {ref_deg:<25.6f} {loss_metrics.reference_metric_config:<15} {f'eta: {current_eta:.4f}':<15}")
        print(f"{'Adaptation loss':<25} {loss_metrics.adaptation_loss:<20.6f} {loss_metrics.adaptation_loss * 180 / np.pi:<25.6f} {loss_metrics.adaptation_loss_config:<15} {f'w: {window_idx}':<15}")
        print(f"{'Mode':<25} {'NEAR FIELD':<20} {'(No SubspaceNet comparison)':<25} {'N/A':<15} {f't: {trajectory_idx}':<15}")
        print("-" * 100)


def save_model_state(model, output_dir, model_type=None):
    """Save model state dictionary to file."""
    logger = logging.getLogger(__name__)

    if model_type is None:
        model_type = "model"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_type}_{timestamp}.pt"
    model_path = Path(output_dir) / model_filename

    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    return model_path


def log_online_learning_window_summary(
    subspacenet_loss,
    ekf_loss,
    online_ekf_loss,
    current_eta,
    is_near_field,
    trajectory_idx=0,
    window_idx=0,
    is_learning=False
):
    """Log comparison between pretrained and online model for a window."""
    mode = "LEARNING" if is_learning else "EVAL"
    print(f"\n{'ONLINE LEARNING ' + mode + ' - Window ' + str(window_idx) + ' Trajectory ' + str(trajectory_idx):^80}")
    print(f"Pretrained SubspaceNet Loss: {subspacenet_loss:.6f}")
    print(f"Pretrained EKF (reference): {ekf_loss:.6f}")
    print(f"Online EKF (reference):      {online_ekf_loss:.6f}")
    print(f"Current eta: {current_eta:.4f}")


def average_online_learning_results_across_trajectories(results_list: list) -> dict:
    """Average online learning results across multiple trajectories."""
    import numpy as np

    logger = logging.getLogger(__name__)

    if not results_list:
        return {"status": "error", "message": "No results to average"}

    pretrained_trajectories = []
    online_trajectories = []
    supervised_trajectories = []
    metadata_list = []

    for result in results_list:
        if result.get("status") != "success":
            continue
        ol_results = result.get("online_learning_results", {})
        if "pretrained_model_trajectory_results" in ol_results:
            pretrained_trajectories.append(ol_results["pretrained_model_trajectory_results"])
        if "online_model_trajectory_results" in ol_results:
            online_trajectories.append(ol_results["online_model_trajectory_results"])
        if "supervised_model_trajectory_results" in ol_results:
            supervised_trajectories.append(ol_results["supervised_model_trajectory_results"])
        metadata_list.append({
            "training_start_window": ol_results.get("training_start_window"),
            "training_end_window": ol_results.get("training_end_window"),
            "eta_change_windows": ol_results.get("eta_change_windows", []),
        })

    if not pretrained_trajectories:
        return {"status": "error", "message": "No valid trajectory results found"}

    averaged_pretrained = _average_trajectory_results(pretrained_trajectories, "pretrained")
    averaged_online = _average_trajectory_results(online_trajectories, "online") if online_trajectories else {}
    averaged_supervised = _average_trajectory_results(supervised_trajectories, "supervised") if supervised_trajectories else None

    summary_stats = _calculate_trajectory_summary_statistics(metadata_list)
    glrt_results = _average_glrt_results(results_list)

    logger.info(f"Successfully averaged results from {len(pretrained_trajectories)} trajectories")

    result_dict = {
        "status": "success",
        "averaged_results": {
            "averaged_pretrained_trajectory": averaged_pretrained,
            "averaged_online_trajectory": averaged_online,
            "summary_statistics": summary_stats,
            "trajectory_count": len(pretrained_trajectories),
        },
    }

    if averaged_supervised is not None:
        result_dict["averaged_results"]["averaged_supervised_trajectory"] = averaged_supervised

    if glrt_results:
        result_dict["averaged_results"]["glrt_results"] = glrt_results

    return result_dict


def _average_trajectory_results(trajectory_list: list, model_type: str) -> dict:
    """Average trajectory results for a specific model type."""
    import numpy as np

    if not trajectory_list:
        return {}

    num_windows = len(trajectory_list[0].window_results)

    averaged_metrics = {
        "window_indices": [],
        "window_eta_values": [],
        "reference_metric_losses": [],
        "reference_metric_losses_db": [],
        "pre_ekf_losses": [],
        "adaptation_losses": [],
        "avg_covariances": [],
        "ekf_gain_rmape": [],
        "avg_innovations": [],
        "avg_kalman_gains": [],
        "avg_kalman_gain_times_innovation": [],
        "avg_y_s_inv_y": [],
    }

    for window_idx in range(num_windows):
        window_ref_losses = []
        window_ref_losses_db = []
        window_pre_ekf_losses = []
        window_adaptation_losses = []
        window_covariances = []
        window_ekf_gains_rmape = []
        window_eta_values = []
        actual_window_indices = []
        window_innovations = []
        window_kalman_gains = []
        window_kalman_gain_times_innovation = []
        window_y_s_inv_y = []

        for traj in trajectory_list:
            if window_idx < len(traj.window_results):
                window_result = traj.window_results[window_idx]
                if window_result.is_valid:
                    actual_idx = (
                        traj.window_indices[window_idx]
                        if window_idx < len(traj.window_indices)
                        else window_idx
                    )
                    window_ref_losses.append(window_result.loss_metrics.reference_metric_loss)
                    window_ref_losses_db.append(window_result.loss_metrics.reference_metric_loss_db)
                    window_pre_ekf_losses.append(window_result.loss_metrics.pre_ekf_loss)
                    window_adaptation_losses.append(window_result.loss_metrics.adaptation_loss)
                    window_covariances.append(window_result.window_metrics.avg_covariance)
                    window_ekf_gains_rmape.append(window_result.loss_metrics.ekf_gain_rmape)
                    window_eta_values.append(window_result.window_metrics.eta_value)
                    actual_window_indices.append(actual_idx)

                    if window_result.window_metrics.avg_ekf_innovations:
                        window_innovations.append(np.mean(window_result.window_metrics.avg_ekf_innovations))
                    if window_result.window_metrics.avg_ekf_kalman_gains:
                        window_kalman_gains.append(np.mean(window_result.window_metrics.avg_ekf_kalman_gains))
                    if window_result.window_metrics.avg_ekf_kalman_gain_times_innovation:
                        window_kalman_gain_times_innovation.append(
                            np.mean(window_result.window_metrics.avg_ekf_kalman_gain_times_innovation)
                        )
                    if window_result.window_metrics.avg_ekf_y_s_inv_y:
                        window_y_s_inv_y.append(np.mean(window_result.window_metrics.avg_ekf_y_s_inv_y))

        if actual_window_indices:
            averaged_metrics["window_indices"].append(int(np.mean(actual_window_indices)))
            averaged_metrics["window_eta_values"].append(np.mean(window_eta_values) if window_eta_values else 0.0)
            averaged_metrics["reference_metric_losses"].append(np.mean(window_ref_losses) if window_ref_losses else 0.0)
            averaged_metrics["reference_metric_losses_db"].append(np.mean(window_ref_losses_db) if window_ref_losses_db else 0.0)
            averaged_metrics["pre_ekf_losses"].append(np.mean(window_pre_ekf_losses) if window_pre_ekf_losses else 0.0)
            averaged_metrics["adaptation_losses"].append(np.mean(window_adaptation_losses) if window_adaptation_losses else 0.0)
            averaged_metrics["avg_covariances"].append(np.mean(window_covariances) if window_covariances else 0.0)
            averaged_metrics["ekf_gain_rmape"].append(np.mean(window_ekf_gains_rmape) if window_ekf_gains_rmape else 0.0)
            averaged_metrics["avg_innovations"].append(np.mean(window_innovations) if window_innovations else 0.0)
            averaged_metrics["avg_kalman_gains"].append(np.mean(window_kalman_gains) if window_kalman_gains else 0.0)
            averaged_metrics["avg_kalman_gain_times_innovation"].append(
                np.mean(window_kalman_gain_times_innovation) if window_kalman_gain_times_innovation else 0.0
            )
            averaged_metrics["avg_y_s_inv_y"].append(np.mean(window_y_s_inv_y) if window_y_s_inv_y else 0.0)

    return averaged_metrics


def mean_reference_loss_after_training(
    window_indices: list,
    reference_metric_losses: list,
    training_end_window=None,
    training_start_window=None,
    *,
    fallback_last_n: int = 10,
) -> Optional[float]:
    """Average reference-metric loss on post-training evaluation windows."""
    import numpy as np

    if not reference_metric_losses:
        return None

    post_learning_losses = []
    if window_indices and len(window_indices) == len(reference_metric_losses):
        if training_end_window is not None:
            post_learning_losses = [
                loss for w, loss in zip(window_indices, reference_metric_losses) if w > training_end_window
            ]
        elif training_start_window is not None:
            post_learning_losses = [
                loss for w, loss in zip(window_indices, reference_metric_losses) if w > training_start_window
            ]
    elif training_end_window is not None:
        # Legacy fallback: only safe when indices are 0..N-1 aligned with absolute windows
        if training_end_window + 1 < len(reference_metric_losses):
            post_learning_losses = reference_metric_losses[training_end_window + 1 :]
    elif training_start_window is not None and training_start_window + 1 < len(reference_metric_losses):
        post_learning_losses = reference_metric_losses[training_start_window + 1 :]

    if not post_learning_losses and fallback_last_n:
        n = min(fallback_last_n, len(reference_metric_losses))
        post_learning_losses = reference_metric_losses[-n:]

    if not post_learning_losses:
        return None

    return float(np.mean(post_learning_losses))


def _calculate_trajectory_summary_statistics(metadata_list: list) -> dict:
    """Calculate summary statistics across trajectory metadata."""
    import numpy as np

    if not metadata_list:
        return {}

    training_starts = [m["training_start_window"] for m in metadata_list if m.get("training_start_window") is not None]
    training_ends = [m["training_end_window"] for m in metadata_list if m.get("training_end_window") is not None]
    window_counts = [len(m.get("eta_change_windows", [])) for m in metadata_list]

    return {
        "avg_training_start_window": float(np.mean(training_starts)) if training_starts else None,
        "std_training_start_window": float(np.std(training_starts)) if training_starts else None,
        "avg_training_end_window": float(np.mean(training_ends)) if training_ends else None,
        "std_training_end_window": float(np.std(training_ends)) if training_ends else None,
        "avg_window_count": np.mean(window_counts),
    }


def _average_glrt_results(results_list: list) -> dict:
    """Average GLRT drift detection results across trajectories."""
    import numpy as np

    if not results_list:
        return {}

    min_segment_size = GLRT_MIN_SEGMENT_SIZE
    adaptation_sequences = []
    reference_metric_sequences = []
    adaptation_changepoint_windows = []
    reference_metric_changepoint_windows = []
    adaptation_likelihoods = []
    reference_metric_likelihoods = []
    z_scores = []
    learning_rates = []
    actual_lrs = []
    window_index_offset = None

    for result in results_list:
        if result.get("status") != "success":
            continue
        online_results = result.get("online_learning_results", {})
        if window_index_offset is None:
            window_index_offset = online_results.get("glrt_loss_window_offset", 0)

        if online_results.get("glrt_adaptation_losses") is not None:
            adaptation_sequences.append(online_results["glrt_adaptation_losses"])
            if online_results.get("glrt_adaptation_loss_changepoint_window") is not None:
                adaptation_changepoint_windows.append(online_results["glrt_adaptation_loss_changepoint_window"])
            if online_results.get("glrt_adaptation_loss_likelihood") is not None:
                adaptation_likelihoods.append(online_results["glrt_adaptation_loss_likelihood"])

        if online_results.get("glrt_reference_metric_losses") is not None:
            reference_metric_sequences.append(online_results["glrt_reference_metric_losses"])
            if online_results.get("glrt_reference_metric_changepoint_window") is not None:
                reference_metric_changepoint_windows.append(online_results["glrt_reference_metric_changepoint_window"])
            if online_results.get("glrt_reference_metric_likelihood") is not None:
                reference_metric_likelihoods.append(online_results["glrt_reference_metric_likelihood"])

        if online_results.get("glrt_z_score_at_detection") is not None:
            z_scores.append(online_results["glrt_z_score_at_detection"])
        if online_results.get("learning_rate_at_detection") is not None:
            learning_rates.append(online_results["learning_rate_at_detection"])
        if online_results.get("actual_lr_per_training_window"):
            actual_lrs.extend(online_results["actual_lr_per_training_window"])

    glrt_results = {}

    if adaptation_sequences:
        min_length = min(len(losses) for losses in adaptation_sequences)
        avg_adaptation_losses = [
            np.mean([losses[i] for losses in adaptation_sequences]) for i in range(min_length)
        ]
        glrt_results["adaptation_loss"] = {
            "avg_losses": avg_adaptation_losses,
            "window_index_offset": window_index_offset if window_index_offset is not None else 0,
            "avg_changepoint_window": float(np.mean(adaptation_changepoint_windows)) if adaptation_changepoint_windows else None,
            "std_changepoint_window": float(np.std(adaptation_changepoint_windows)) if adaptation_changepoint_windows else None,
            "avg_likelihood": float(np.mean(adaptation_likelihoods)) if adaptation_likelihoods else None,
            "std_likelihood": float(np.std(adaptation_likelihoods)) if adaptation_likelihoods else None,
            "avg_z_score": float(np.mean(z_scores)) if z_scores else None,
            "std_z_score": float(np.std(z_scores)) if z_scores else None,
            "avg_learning_rate": float(np.mean(learning_rates)) if learning_rates else None,
            "std_learning_rate": float(np.std(learning_rates)) if learning_rates else None,
            "avg_actual_learning_rate": float(np.mean(actual_lrs)) if actual_lrs else None,
            "std_actual_learning_rate": float(np.std(actual_lrs)) if actual_lrs else None,
            "trajectory_count": len(adaptation_sequences),
            "min_segment_size": min_segment_size,
            "individual_changepoint_windows": adaptation_changepoint_windows,
            "individual_likelihoods": adaptation_likelihoods,
            "individual_z_scores": z_scores,
            "individual_learning_rates": learning_rates,
        }

    if reference_metric_sequences:
        min_length = min(len(losses) for losses in reference_metric_sequences)
        avg_reference_metric_losses = [
            np.mean([losses[i] for losses in reference_metric_sequences]) for i in range(min_length)
        ]
        glrt_results["reference_metric"] = {
            "avg_losses": avg_reference_metric_losses,
            "window_index_offset": window_index_offset if window_index_offset is not None else 0,
            "avg_changepoint_window": float(np.mean(reference_metric_changepoint_windows)) if reference_metric_changepoint_windows else None,
            "std_changepoint_window": float(np.std(reference_metric_changepoint_windows)) if reference_metric_changepoint_windows else None,
            "avg_likelihood": float(np.mean(reference_metric_likelihoods)) if reference_metric_likelihoods else None,
            "std_likelihood": float(np.std(reference_metric_likelihoods)) if reference_metric_likelihoods else None,
            "trajectory_count": len(reference_metric_sequences),
            "min_segment_size": min_segment_size,
            "individual_changepoint_windows": reference_metric_changepoint_windows,
            "individual_likelihoods": reference_metric_likelihoods,
        }

    return glrt_results

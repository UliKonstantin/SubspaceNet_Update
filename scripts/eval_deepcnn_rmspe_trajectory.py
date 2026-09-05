#!/usr/bin/env python3
"""
Per-step DeepCNN RMSPE on the OL sine-accel trajectory (eta=0).

Reports pre-EKF angle error in radians and degrees, peak-decoder stats,
and optional static i.i.d. test-set RMSPE for the same checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import scipy.signal
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))


def _count_spectrum_peaks(spectrum: np.ndarray) -> int:
    peaks, _ = scipy.signal.find_peaks(
        spectrum.flatten(), prominence=0.05, height=0.01
    )
    return int(len(peaks))


def _step_rmspe_rad(pred: torch.Tensor, truth: torch.Tensor) -> float:
    from DCD_MUSIC.src.metrics.rmspe_loss import RMSPELoss
    from DCD_MUSIC.src.utils import device

    crit = RMSPELoss().to(device)
    with torch.no_grad():
        loss = crit(pred.view(1, -1), truth.view(1, -1))
    return float(loss.item())


def eval_ol_trajectory(config_path: Path, model_path: Path, out_dir: Path) -> dict:
    from config_handler import setup_configuration
    from DCD_MUSIC.src.utils import device
    from simulation.core import Simulation

    overrides = [
        "simulation.evaluate_model=false",
        "simulation.load_model=true",
        f"simulation.model_path={model_path}",
        "online_learning.enabled=false",
        "trajectory.enabled=true",
        "system_model.eta=0.0",
        "dataset.samples_size=1",
    ]
    config, components, _ = setup_configuration(str(config_path), str(out_dir), overrides)
    sim = Simulation(config, components, out_dir)
    sim._run_data_pipeline(scenario="evaluation")
    ok, msg = sim._load_and_apply_weights(model_path, device)
    if not ok:
        raise RuntimeError(msg)

    model = sim.trained_model
    model.eval()

    batch = next(iter(sim.test_dataloader))
    trajectories, sources_num, labels = batch
    traj_len = trajectories.shape[1]
    m = int(config.system_model.M)

    step_rmspe_rad = []
    step_rmspe_deg = []
    peaks_found = []
    used_random_pad = []

    from models.deep_cnn_adapter import get_k_peaks_topk_pad, snapshots_to_covariance_channels

    with torch.no_grad():
        for step in range(traj_len):
            x = trajectories[0, step].unsqueeze(0).to(device)
            num_sources = int(sources_num[0, step].item())
            truth = torch.tensor(
                labels[0, step, :num_sources], device=device, dtype=torch.float32
            )

            cov = snapshots_to_covariance_channels(x)
            spectrum = model.cnn(cov)[0].detach().cpu().numpy()
            n_peaks = _count_spectrum_peaks(spectrum)
            peaks_found.append(n_peaks)
            used_random_pad.append(n_peaks < num_sources)

            angles_pred, _, _ = model(x, num_sources)
            angles_pred = angles_pred.view(-1)[:num_sources]
            rmspe = _step_rmspe_rad(angles_pred, truth)
            step_rmspe_rad.append(rmspe)
            step_rmspe_deg.append(rmspe * 180.0 / math.pi)

    arr_deg = np.array(step_rmspe_deg)
    summary = {
        "checkpoint": str(model_path),
        "trajectory": config.trajectory.trajectory_type,
        "eta": float(config.system_model.eta),
        "N": int(config.system_model.N),
        "M": m,
        "snr": float(config.system_model.snr),
        "peak_method": getattr(config.model.params, "peak_method", "peaks"),
        "steps": traj_len,
        "pre_ekf_rmspe_rad_mean": float(np.mean(step_rmspe_rad)),
        "pre_ekf_rmspe_deg_mean": float(np.mean(arr_deg)),
        "pre_ekf_rmspe_deg_median": float(np.median(arr_deg)),
        "pre_ekf_rmspe_deg_std": float(np.std(arr_deg)),
        "pre_ekf_rmspe_deg_p95": float(np.percentile(arr_deg, 95)),
        "pre_ekf_rmspe_deg_max": float(np.max(arr_deg)),
        "steps_rmspe_deg_gt_5": int(np.sum(arr_deg > 5.0)),
        "steps_rmspe_deg_gt_10": int(np.sum(arr_deg > 10.0)),
        "steps_peaks_lt_M": int(np.sum(np.array(peaks_found) < m)),
        "steps_topk_pad_used": int(np.sum(used_random_pad)),
        "peak_decoder": "get_k_peaks_topk_pad (adapter, deterministic)",
        "per_step_rmspe_deg": step_rmspe_deg,
        "per_step_peaks_found": peaks_found,
    }
    return summary


def eval_static_test(config_path: Path, model_path: Path, out_dir: Path, samples: int) -> dict:
    from config_handler import setup_configuration
    from simulation.core import Simulation

    overrides = [
        "simulation.evaluate_model=true",
        "simulation.load_model=true",
        f"simulation.model_path={model_path}",
        "trajectory.enabled=false",
        "system_model.eta=0.0",
        f"dataset.samples_size={samples}",
        "dataset.test_validation_train_split=1.0,0.0,0.0",
        "training.batch_size=64",
    ]
    config, components, _ = setup_configuration(str(config_path), str(out_dir / "static"), overrides)
    sim = Simulation(config, components, out_dir / "static")
    result = sim.run_evaluation()
    if result.get("status") != "success":
        return {"error": result.get("message", "static eval failed")}
    eval_res = result.get("evaluation_results") or sim.results
    dnn = eval_res.get("dnn_test_loss")
    ekf = eval_res.get("ekf_test_loss")
    if dnn is None or ekf is None:
        return {"error": eval_res.get("evaluation_error", "missing dnn_test_loss")}
    return {
        "samples": samples,
        "dnn_rmspe_rad_mean": float(dnn),
        "dnn_rmspe_deg_mean": float(dnn) * 180.0 / math.pi,
        "ekf_rmspe_rad_mean": float(ekf),
        "ekf_rmspe_deg_mean": float(ekf) * 180.0 / math.pi,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepCNN RMSPE trajectory diagnostic")
    parser.add_argument(
        "-c",
        default="configs/Used_for_paper/paper_deepcnn_rmspe_eval.yaml",
        help="Base config (OL trajectory params)",
    )
    parser.add_argument(
        "-m",
        default="experiments/results/basemodels_for_journal_paper/deepcnn/snr_10.0/checkpoints/final_DeepCNN_20260820_173256.pt",
        help="Checkpoint path",
    )
    parser.add_argument(
        "-o",
        default="experiments/results/smoke_runs/journal_paper/deepcnn_rmspe_eval",
        help="Output directory",
    )
    parser.add_argument(
        "--static-samples",
        type=int,
        default=512,
        help="i.i.d. test samples for static RMSPE eval (0 to skip)",
    )
    args = parser.parse_args()

    out_dir = Path(args.o)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / args.c
    model_path = ROOT / args.m

    print("=== OL trajectory (eta=0, pre-EKF RMSPE vs GT) ===")
    traj_summary = eval_ol_trajectory(config_path, model_path, out_dir)
    print(json.dumps({k: v for k, v in traj_summary.items() if k != "per_step_rmspe_deg"}, indent=2))
    print(
        f"Pre-EKF RMSPE: mean={traj_summary['pre_ekf_rmspe_deg_mean']:.3f}° "
        f"median={traj_summary['pre_ekf_rmspe_deg_median']:.3f}° "
        f"max={traj_summary['pre_ekf_rmspe_deg_max']:.3f}°"
    )
    print(
        f"Spikes: {traj_summary['steps_rmspe_deg_gt_5']} steps >5°, "
        f"{traj_summary['steps_rmspe_deg_gt_10']} steps >10°; "
        f"find_peaks<M on {traj_summary['steps_peaks_lt_M']} steps "
        f"(top-k pad used on {traj_summary['steps_topk_pad_used']})"
    )

    payload = {"trajectory": traj_summary}
    if args.static_samples > 0:
        print("\n=== Static i.i.d. test set (RMSPE, same ckpt) ===")
        static_summary = eval_static_test(config_path, model_path, out_dir, args.static_samples)
        print(json.dumps(static_summary, indent=2))
        payload["static_iid"] = static_summary

    out_json = out_dir / "deepcnn_rmspe_eval_summary.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()

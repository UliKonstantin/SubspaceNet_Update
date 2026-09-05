"""Export journal paper benchmark metrics JSON from OL results."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("SubspaceNet.benchmark_export")

BENCHMARK_FILENAME = "benchmark_metrics.json"
PAIR_FILENAME = "benchmark_pair.json"


def _mean_or_none(values) -> Optional[float]:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def _arm_summary(traj: Dict[str, Any]) -> Dict[str, Any]:
    if not traj:
        return {}
    return {
        "reference_metric_losses": [float(x) for x in traj.get("reference_metric_losses", [])],
        "adaptation_losses": [float(x) for x in traj.get("adaptation_losses", [])],
        "pre_ekf_losses": [float(x) for x in traj.get("pre_ekf_losses", [])],
        "window_indices": [int(x) for x in traj.get("window_indices", [])],
        "mean_reference_metric": _mean_or_none(traj.get("reference_metric_losses")),
        "mean_adaptation_loss": _mean_or_none(traj.get("adaptation_losses")),
        "mean_pre_ekf_loss": _mean_or_none(traj.get("pre_ekf_losses")),
    }


def extract_benchmark_payload(result: Dict[str, Any], config) -> Optional[Dict[str, Any]]:
    """Extract benchmark arms from a single OL result dict."""
    if not isinstance(result, dict) or result.get("status") != "success":
        return None

    averaged = result.get("averaged_results", {})
    arms = {
        "no_adapt": _arm_summary(averaged.get("averaged_pretrained_trajectory", {})),
        "unsupervised_ours": _arm_summary(averaged.get("averaged_online_trajectory", {})),
        "supervised_genie": _arm_summary(averaged.get("averaged_supervised_trajectory", {})),
    }

    glrt = averaged.get("glrt_results") or result.get("glrt_results")
    payload = {
        "model_type": getattr(config.model, "type", "unknown"),
        "seed": getattr(config.simulation, "seed", None),
        "benchmark_arms": arms,
        "trajectory_count": averaged.get("trajectory_count"),
        "training_start_window": result.get("online_learning_results", {}).get("training_start_window"),
        "training_end_window": result.get("online_learning_results", {}).get("training_end_window"),
        "drift_detection_window": result.get("online_learning_results", {}).get("drift_detection_window"),
    }
    if glrt:
        payload["glrt"] = glrt
    return payload


def _is_scenario_map(result: Dict[str, Any]) -> bool:
    if not isinstance(result, dict) or "status" in result:
        return False
    return any(isinstance(v, dict) for v in result.values())


def export_benchmark_metrics(result: Dict[str, Any], config, output_dir: Path) -> Optional[Path]:
    """Write benchmark_metrics.json for a single OL run or 1D sweep."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if _is_scenario_map(result):
        scenarios = {}
        for key, entry in result.items():
            if not isinstance(entry, dict):
                continue
            if "lr_sweep_results" in entry:
                scenarios[str(key)] = {
                    lr_key: extract_benchmark_payload(lr_data.get("result", {}), config)
                    for lr_key, lr_data in entry["lr_sweep_results"].items()
                    if isinstance(lr_data, dict)
                }
            else:
                scenarios[str(key)] = extract_benchmark_payload(entry, config)
        payload = {
            "format": "sweep",
            "model_type": getattr(config.model, "type", "unknown"),
            "seed": getattr(config.simulation, "seed", None),
            "scenarios": scenarios,
        }
    else:
        payload = extract_benchmark_payload(result, config)
        if payload is None:
            logger.debug("Skipping benchmark export: result is not a successful OL run")
            return None
        payload["format"] = "single"

    out_path = output_dir / BENCHMARK_FILENAME
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Wrote benchmark metrics to %s", out_path)
    return out_path


def merge_benchmark_pair(
    subspacenet_dir: Path,
    deepcnn_dir: Path,
    output_dir: Path,
    seed: Optional[int] = None,
) -> Path:
    """Merge SubspaceNet + DeepCNN benchmark_metrics.json into benchmark_pair.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _load(path: Path) -> Dict[str, Any]:
        with (path / BENCHMARK_FILENAME).open(encoding="utf-8") as handle:
            return json.load(handle)

    merged = {
        "seed": seed,
        "subspacenet": _load(Path(subspacenet_dir)),
        "deepcnn": _load(Path(deepcnn_dir)),
    }
    out_path = output_dir / PAIR_FILENAME
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2)
    logger.info("Wrote paired benchmark metrics to %s", out_path)
    return out_path

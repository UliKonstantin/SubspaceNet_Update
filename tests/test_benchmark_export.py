"""Tests for PF-10 benchmark metrics export."""

import json
from pathlib import Path

from utils.benchmark_export import export_benchmark_metrics, merge_benchmark_pair


def _sample_result():
    return {
        "status": "success",
        "online_learning_results": {
            "training_start_window": 10,
            "training_end_window": 15,
            "drift_detection_window": 12,
        },
        "averaged_results": {
            "trajectory_count": 1,
            "averaged_pretrained_trajectory": {
                "reference_metric_losses": [0.5, 0.4],
                "adaptation_losses": [0.3, 0.2],
            },
            "averaged_online_trajectory": {
                "reference_metric_losses": [0.45, 0.35],
                "adaptation_losses": [0.25, 0.15],
            },
            "averaged_supervised_trajectory": {
                "reference_metric_losses": [0.2, 0.1],
                "adaptation_losses": [0.15, 0.05],
            },
        },
    }


class _Cfg:
    class model:
        type = "SubspaceNet"

    class simulation:
        seed = 42


def test_export_benchmark_metrics_single(tmp_path: Path):
    path = export_benchmark_metrics(_sample_result(), _Cfg(), tmp_path)
    assert path is not None and path.exists()
    payload = json.loads(path.read_text())
    assert payload["format"] == "single"
    assert payload["model_type"] == "SubspaceNet"
    assert "no_adapt" in payload["benchmark_arms"]
    assert payload["benchmark_arms"]["no_adapt"]["mean_reference_metric"] == 0.45


def test_merge_benchmark_pair(tmp_path: Path):
    sn_dir = tmp_path / "sn"
    cnn_dir = tmp_path / "cnn"
    sn_dir.mkdir()
    cnn_dir.mkdir()
    export_benchmark_metrics(_sample_result(), _Cfg(), sn_dir)

    class CnnCfg:
        class model:
            type = "DeepCNN"

        class simulation:
            seed = 42

    export_benchmark_metrics(_sample_result(), CnnCfg(), cnn_dir)
    pair_path = merge_benchmark_pair(sn_dir, cnn_dir, tmp_path, seed=42)
    payload = json.loads(pair_path.read_text())
    assert payload["subspacenet"]["model_type"] == "SubspaceNet"
    assert payload["deepcnn"]["model_type"] == "DeepCNN"

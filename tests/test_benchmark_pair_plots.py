"""Tests for SubspaceNet vs DeepCNN benchmark comparison plots."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from utils.benchmark_export import export_benchmark_metrics, merge_benchmark_pair
from utils.plotting.benchmark_pair import (
    _aligned_pretrained_vs_online,
    infer_scenario_axis,
    load_benchmark_pair,
    plot_average_rmspe_vs_axis,
    plot_benchmark_pair_comparisons,
    plot_side_by_side_window_rmspe,
)


def _sweep_result(ref_losses, *, start=0, online_scale=0.9, train_start=1):
    online_losses = [x * online_scale for x in ref_losses[train_start:]]
    return {
        "status": "success",
        "online_learning_results": {
            "training_start_window": train_start,
            "training_end_window": train_start + 1,
        },
        "averaged_results": {
            "trajectory_count": 1,
            "averaged_pretrained_trajectory": {
                "reference_metric_losses": ref_losses,
                "window_indices": list(range(len(ref_losses))),
            },
            "averaged_online_trajectory": {
                "reference_metric_losses": online_losses,
                "window_indices": list(range(train_start, train_start + len(online_losses))),
            },
            "averaged_supervised_trajectory": {
                "reference_metric_losses": [x * 0.8 for x in ref_losses[train_start:]],
                "window_indices": list(range(train_start, train_start + len(ref_losses[train_start:]))),
            },
        },
    }


class _SnCfg:
    class model:
        type = "SubspaceNet"

    class simulation:
        seed = 42


class _CnnCfg:
    class model:
        type = "DeepCNN"

    class simulation:
        seed = 42


def _make_pair(tmp_path: Path):
    sn_dir = tmp_path / "subspacenet"
    cnn_dir = tmp_path / "deepcnn"
    sn_dir.mkdir()
    cnn_dir.mkdir()

    sn_sweep = {
        "6.0": _sweep_result([0.05, 0.04, 0.03]),
        "9.0": _sweep_result([0.04, 0.03, 0.02]),
        "18.0": _sweep_result([0.03, 0.02, 0.01]),
    }
    cnn_sweep = {
        "6.0": _sweep_result([0.06, 0.05, 0.04]),
        "9.0": _sweep_result([0.05, 0.04, 0.03]),
        "18.0": _sweep_result([0.04, 0.03, 0.02]),
    }
    export_benchmark_metrics(sn_sweep, _SnCfg(), sn_dir)
    export_benchmark_metrics(cnn_sweep, _CnnCfg(), cnn_dir)
    return merge_benchmark_pair(sn_dir, cnn_dir, tmp_path, seed=42)


def test_infer_scenario_axis_n():
    assert infer_scenario_axis([6.0, 9.0, 18.0]) == "n"


def test_load_benchmark_pair_from_dir(tmp_path: Path):
    pair_path = _make_pair(tmp_path)
    loaded = load_benchmark_pair(tmp_path)
    assert loaded["subspacenet"]["model_type"] == "SubspaceNet"
    assert pair_path.exists()


def test_plot_benchmark_pair_comparisons(tmp_path: Path):
    _make_pair(tmp_path)
    out_dir = tmp_path / "plots"
    paths = plot_benchmark_pair_comparisons(
        tmp_path,
        out_dir,
        scenario=9.0,
        axis="n",
    )
    assert paths["side_by_side_window_rmspe"].exists()
    assert paths["average_rmspe_vs_axis"].exists()


def test_aligned_pretrained_vs_online(tmp_path: Path):
    pair_path = _make_pair(tmp_path)
    import json

    pair = json.loads(pair_path.read_text())
    x, pre, online = _aligned_pretrained_vs_online(pair["subspacenet"], "9.0")
    assert len(x) == len(pre) == len(online)
    assert int(x[0]) == 1


def test_plot_side_by_side_and_sweep(tmp_path: Path):
    pair_path = _make_pair(tmp_path)
    import json

    pair = json.loads(pair_path.read_text())
    plot_side_by_side_window_rmspe(
        pair,
        tmp_path / "side.png",
        scenario=9.0,
    )
    plot_average_rmspe_vs_axis(
        pair,
        tmp_path / "sweep.png",
        axis="n",
    )
    assert (tmp_path / "side.png").exists()
    assert (tmp_path / "sweep.png").exists()

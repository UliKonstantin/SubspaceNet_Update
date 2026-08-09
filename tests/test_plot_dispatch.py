"""Tests for unified plot dispatch."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cli.types import Goal, RunRequest, SweepAxis, SweepType
from utils.plot_dispatch import (
    PlotContext,
    dispatch_one_d_sweep_iteration_plots,
    dispatch_plots,
    resolve_plot_jobs,
    should_save_plots,
    _lr_sweep_output_subdir,
)


def _mock_sim(save_plots: bool = True):
    sim = MagicMock()
    sim.config.simulation.save_plots = save_plots
    sim.results = {}
    return sim


class TestPlotDispatch:
    def test_should_save_plots_respects_config(self):
        ctx = PlotContext({}, RunRequest(goal=Goal.TRAIN, config_path=Path("x.yaml")), Path("."), _mock_sim(False), Goal.TRAIN, SweepType.NONE)
        assert should_save_plots(ctx) is False
        ctx.sim.config.simulation.save_plots = True
        assert should_save_plots(ctx) is True

    def test_resolve_full_goal_multiple_jobs(self):
        request = RunRequest(goal=Goal.FULL, config_path=Path("x.yaml"))
        sim = _mock_sim()
        sim.results = {"training_curves": {}, "dnn_trajectory_results": [{}]}
        result = {"trained_model": True, "online_learning_results": {}, "averaged_results": {}}
        jobs = resolve_plot_jobs(request, result, sim)
        assert (Goal.TRAIN, SweepType.NONE) in jobs
        assert (Goal.EVALUATE, SweepType.NONE) in jobs
        assert (Goal.ONLINE_LEARNING, SweepType.NONE) in jobs

    def test_dispatch_skipped_when_save_plots_false(self, tmp_path):
        request = RunRequest(goal=Goal.TRAIN, config_path=Path("x.yaml"))
        sim = _mock_sim(save_plots=False)
        sim.results = {"training_curves": {"train_losses": [1.0], "valid_losses": [0.5], "train_accuracies": [0.1], "valid_accuracies": [0.2]}}
        with patch("utils.plotting.plot_training_curves") as mock_plot:
            dispatch_plots({"status": "success"}, request, tmp_path, sim)
        mock_plot.assert_not_called()

    def test_dispatch_train_calls_plot_training_curves(self, tmp_path):
        request = RunRequest(goal=Goal.TRAIN, config_path=Path("x.yaml"))
        sim = _mock_sim(save_plots=True)
        metrics = {
            "train_losses": [1.0],
            "valid_losses": [0.5],
            "train_accuracies": [0.1],
            "valid_accuracies": [0.2],
        }
        sim.results = {"training_curves": metrics}
        with patch("utils.plotting.plot_training_curves") as mock_plot:
            dispatch_plots({"status": "success", "trained_model": True}, request, tmp_path, sim)
        mock_plot.assert_called_once_with(metrics, tmp_path)

    def test_dispatch_evaluate_none(self, tmp_path):
        request = RunRequest(goal=Goal.EVALUATE, config_path=Path("x.yaml"))
        sim = _mock_sim(save_plots=True)
        sim.results = {"dnn_trajectory_results": [{"model_predictions": []}]}
        with patch("utils.plotting.plot_eval_dnn_ekf_loss_vs_time") as mock_plot:
            dispatch_plots({"status": "success"}, request, tmp_path, sim)
        mock_plot.assert_called_once()

    def test_dispatch_online_learning_none(self, tmp_path):
        request = RunRequest(goal=Goal.ONLINE_LEARNING, config_path=Path("x.yaml"))
        sim = _mock_sim(save_plots=True)
        result = {
            "status": "success",
            "averaged_results": {
                "averaged_pretrained_trajectory": {"window_indices": [0], "reference_metric_losses": [1.0], "adaptation_losses": [1.0], "window_eta_values": [0.0], "pre_ekf_losses": [1.0]},
                "averaged_online_trajectory": {"window_indices": [0], "reference_metric_losses": [1.0], "adaptation_losses": [1.0], "window_eta_values": [0.0], "pre_ekf_losses": [1.0]},
            },
            "online_learning_results": {},
        }
        with patch("utils.plotting.plot_single_online_learning_run") as mock_plot:
            dispatch_plots(result, request, tmp_path, sim)
        mock_plot.assert_called_once()

    def test_lr_sweep_subdir_paths(self):
        assert _lr_sweep_output_subdir("eta", 0.5, 0, {"lr_row_id": 0, "lr_value": 0.001}) == "eta_0.5/lr_run0_0.001"
        assert _lr_sweep_output_subdir("eta", 1.0, "adaptive", {}) == "eta_1.0/lr_adaptive"

    def test_dispatch_one_d_sweep_iteration_plots_lr_sweep(self, tmp_path):
        ol_result = {
            "status": "success",
            "averaged_results": {"averaged_pretrained_trajectory": {}, "averaged_online_trajectory": {}},
            "online_learning_results": {},
        }
        scenario_results = {
            0.5: {
                "lr_sweep_results": {
                    0: {"lr_row_id": 0, "lr_value": 0.001, "result": ol_result},
                    "adaptive": {"result": ol_result},
                }
            }
        }
        config = MagicMock()
        with patch("utils.plotting.plot_single_online_learning_run") as mock_plot:
            dispatch_one_d_sweep_iteration_plots(scenario_results, tmp_path, config, "eta")
        assert mock_plot.call_count == 2
        mock_plot.assert_any_call(ol_result, tmp_path / "eta_0.5/lr_run0_0.001", config)
        mock_plot.assert_any_call(ol_result, tmp_path / "eta_0.5/lr_adaptive", config)

    def test_dispatch_eta_sweep_calls_per_iteration_plots(self, tmp_path):
        request = RunRequest(
            goal=Goal.ONLINE_LEARNING,
            config_path=Path("x.yaml"),
            sweep=SweepType.ONE_D,
            sweep_axis=SweepAxis.ETA,
            lr_sweep=True,
        )
        sim = _mock_sim(save_plots=True)
        sim.results = {"lr_sweep_heatmap_data": {"eta_values": [], "lr_values": [], "lr_types": [], "lr_row_ids": [], "avg_losses": []}}
        scenario_results = {
            0.5: {"lr_sweep_results": {0: {"lr_row_id": 0, "lr_value": 0.001, "result": {"status": "success", "averaged_results": {}}}}}
        }
        with patch("utils.plotting.plot_eta_scenario_comparison"), patch(
            "utils.plotting.plot_performance_improvement_table_eta"
        ), patch("utils.plotting.plot_scenario_results"), patch(
            "utils.plotting.plot_lr_sweep_heatmap"
        ), patch("utils.lr_analysis.postprocess_lr_sweep_analysis"), patch(
            "utils.plot_dispatch.dispatch_one_d_sweep_iteration_plots"
        ) as mock_iter, patch(
            "simulation.drift.drift_metrics.plot_drift_detection_metrics_in_output_dir"
        ) as mock_drift:
            dispatch_plots(scenario_results, request, tmp_path, sim)
        mock_iter.assert_called_once_with(scenario_results, tmp_path, sim.config, "eta")
        mock_drift.assert_called_once_with(tmp_path)

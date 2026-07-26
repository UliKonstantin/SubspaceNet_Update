"""Tests for resolve_run_request."""
import sys
from pathlib import Path

import pytest

WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

CONFIGS_DIR = Path(__file__).parent / "configs"

from cli.resolver import infer_goal, resolve_run_request
from cli.types import Goal, RoutingError, SweepAxis, SweepType


class TestInferGoal:
    def test_train_from_yaml(self):
        from config.loader import load_config

        config = load_config(CONFIGS_DIR / "test_training.yaml")
        assert infer_goal(config) == Goal.TRAIN

    def test_online_learning_from_yaml(self):
        from config.loader import load_config

        config = load_config(CONFIGS_DIR / "test_online_learning_single.yaml")
        assert infer_goal(config) == Goal.ONLINE_LEARNING


class TestResolveRunRequest:
    def test_train_single_run(self):
        req = resolve_run_request(str(CONFIGS_DIR / "test_training.yaml"))
        assert req.goal == Goal.TRAIN
        assert req.sweep == SweepType.NONE

    def test_snr_sweep_from_scenario_config(self):
        req = resolve_run_request(str(CONFIGS_DIR / "test_training_snr_scenario.yaml"))
        assert req.goal == Goal.TRAIN
        assert req.sweep == SweepType.ONE_D
        assert req.sweep_axis == SweepAxis.SNR
        assert req.sweep_values == [5, 10]

    def test_cli_values_override_yaml(self):
        req = resolve_run_request(
            str(CONFIGS_DIR / "test_training_snr_scenario.yaml"),
            values=[1.0, 2.0],
        )
        assert req.sweep_values == [1.0, 2.0]

    def test_cli_goal_override(self):
        req = resolve_run_request(
            str(CONFIGS_DIR / "test_training.yaml"),
            goal="full",
        )
        assert req.goal == Goal.FULL

    def test_retrain_cli_overrides_yaml(self):
        req = resolve_run_request(
            str(CONFIGS_DIR / "test_training_snr_scenario.yaml"),
            retrain_per_sweep=False,
        )
        assert req.retrain_per_sweep is False

    def test_retrain_from_yaml_when_cli_unset(self):
        req = resolve_run_request(str(CONFIGS_DIR / "test_training_snr_scenario.yaml"))
        assert req.retrain_per_sweep is True

    def test_retrain_false_without_scenario_config(self):
        req = resolve_run_request(str(CONFIGS_DIR / "test_training.yaml"))
        assert req.retrain_per_sweep is False
        assert req.retrain_per_sweep_cli is None

    def test_online_learning_infers_trajectory(self):
        req = resolve_run_request(str(CONFIGS_DIR / "test_online_learning_single.yaml"))
        assert req.goal == Goal.ONLINE_LEARNING
        assert req.trajectory_cli is False
        assert req.trajectory_enabled is True
        assert req.model_path is not None

    def test_eta_sweep_axis_and_values(self):
        req = resolve_run_request(str(CONFIGS_DIR / "test_online_learning_eta_sweep.yaml"))
        assert req.sweep == SweepType.ONE_D
        assert req.sweep_axis == SweepAxis.ETA
        assert req.sweep_values == [0.5, 1.0]
        assert req.model_path is not None

    def test_model_path_from_scenario_config(self):
        req = resolve_run_request(str(CONFIGS_DIR / "test_online_learning_eta_sweep.yaml"))
        assert req.model_path.name.startswith("final_SubspaceNet")

    def test_unknown_goal_raises(self):
        with pytest.raises(RoutingError, match="Unknown --goal"):
            resolve_run_request(str(CONFIGS_DIR / "test_training.yaml"), goal="bogus")

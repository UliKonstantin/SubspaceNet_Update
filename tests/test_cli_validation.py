"""Tests for RunRequest validation."""
import sys
from pathlib import Path

import pytest

WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from cli.types import Goal, Grid4DParams, RoutingError, RunRequest, SweepAxis, SweepType
from cli.validation import validate


def _base_request(**kwargs) -> RunRequest:
    defaults = dict(
        goal=Goal.TRAIN,
        config_path=Path("configs/default_config.yaml"),
    )
    defaults.update(kwargs)
    return RunRequest(**defaults)


class TestValidationErrors:
    def test_evaluate_requires_model(self):
        req = _base_request(goal=Goal.EVALUATE)
        with pytest.raises(RoutingError, match="--model"):
            validate(req)

    def test_online_learning_requires_model(self):
        req = _base_request(goal=Goal.ONLINE_LEARNING, trajectory=True)
        with pytest.raises(RoutingError, match="--model"):
            validate(req)

    def test_online_learning_requires_trajectory(self):
        req = _base_request(
            goal=Goal.ONLINE_LEARNING,
            model_path=Path("model.pt"),
            trajectory=False,
        )
        with pytest.raises(RoutingError, match="trajectory"):
            validate(req)

    def test_full_pipeline_no_sweep(self):
        req = _base_request(
            goal=Goal.FULL,
            sweep=SweepType.ONE_D,
            sweep_axis=SweepAxis.SNR,
            sweep_values=[0, 10],
        )
        with pytest.raises(RoutingError, match="does not support parameter sweeps"):
            validate(req)

    def test_kalman_2d_evaluate_only(self):
        req = _base_request(
            goal=Goal.TRAIN,
            sweep=SweepType.KALMAN_2D,
            model_path=Path("model.pt"),
        )
        with pytest.raises(RoutingError, match="2D kalman"):
            validate(req)

    def test_4d_grid_online_only(self):
        req = _base_request(
            goal=Goal.EVALUATE,
            sweep=SweepType.GRID_4D,
            model_path=Path("model.pt"),
            trajectory=True,
            grid_params=Grid4DParams(),
        )
        with pytest.raises(RoutingError, match="4D grid"):
            validate(req)

    def test_lr_sweep_requires_eta(self):
        req = _base_request(
            goal=Goal.ONLINE_LEARNING,
            model_path=Path("model.pt"),
            trajectory=True,
            sweep=SweepType.ONE_D,
            sweep_axis=SweepAxis.SNR,
            sweep_values=[0, 10],
            lr_sweep=True,
        )
        with pytest.raises(RoutingError, match="--axis eta"):
            validate(req)

    def test_1d_sweep_requires_axis(self):
        req = _base_request(sweep=SweepType.ONE_D, sweep_values=[0, 10])
        with pytest.raises(RoutingError, match="requires --axis"):
            validate(req)

    def test_1d_sweep_requires_values(self):
        req = _base_request(
            sweep=SweepType.ONE_D,
            sweep_axis=SweepAxis.SNR,
            sweep_values=None,
        )
        with pytest.raises(RoutingError, match="requires values"):
            validate(req)


class TestValidationPasses:
    def test_train_snr_sweep(self):
        req = _base_request(
            sweep=SweepType.ONE_D,
            sweep_axis=SweepAxis.SNR,
            sweep_values=[5, 10],
        )
        validate(req)

    def test_evaluate_kalman_2d(self):
        req = _base_request(
            goal=Goal.EVALUATE,
            model_path=Path("model.pt"),
            sweep=SweepType.KALMAN_2D,
        )
        validate(req)

    def test_online_learning_4d_grid(self):
        req = _base_request(
            goal=Goal.ONLINE_LEARNING,
            model_path=Path("model.pt"),
            trajectory=True,
            sweep=SweepType.GRID_4D,
            grid_params=Grid4DParams(eta_values=[0.01]),
        )
        validate(req)

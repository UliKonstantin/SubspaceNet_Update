"""Verify spacing-scale calibration drift propagates through OL sample generation."""

import numpy as np
import torch

from config.loader import load_config
from config.factory import _create_system_model_params
from simulation.calibration_drift import reset_calibration
from simulation.runners.data import create_online_learning_dataset
from models.deep_cnn_adapter import snapshots_to_covariance_channels


def _minimal_ol_config():
    cfg = load_config("configs/Used_for_paper/paper_T1_antenna_sweep.yaml")
    cfg.online_learning.trajectory_length = 200
    cfg.online_learning.window_size = 5
    cfg.online_learning.stride = 3
    cfg.online_learning.drift_type = "spacing_scale"
    cfg.online_learning.eta_update_interval_windows = 40
    cfg.online_learning.spacing_scale_increment = 0.03
    cfg.online_learning.max_spacing_scale = 1.03
    cfg.simulation.seed = 123
    return cfg


def test_spacing_scale_update_changes_snapshots():
    cfg = _minimal_ol_config()
    params = _create_system_model_params(cfg)
    reset_calibration(params)

    dataset = create_online_learning_dataset(
        system_model_params=params,
        config=cfg,
        window_size=cfg.online_learning.window_size,
        stride=cfg.online_learning.stride,
    )

    torch.manual_seed(123)
    np.random.seed(123)
    pre_window, _, _ = dataset[31]

    torch.manual_seed(123)
    np.random.seed(123)
    dataset.update_spacing_scale(1.03, invalidate_from_step=31 * cfg.online_learning.stride)
    post_window, _, _ = dataset[31]

    pre_cov = snapshots_to_covariance_channels(pre_window[0:1]).detach().cpu().numpy()
    post_cov = snapshots_to_covariance_channels(post_window[0:1]).detach().cpu().numpy()
    delta = float(np.linalg.norm(post_cov - pre_cov))

    assert delta > 1e-6, f"Expected snapshots to change after spacing-scale drift, delta={delta}"
    assert abs(params.spacing_scale - 1.03) < 1e-9
    assert params.eta == 0.0


def test_spacing_scale_update_truncates_stale_cache():
    cfg = _minimal_ol_config()
    params = _create_system_model_params(cfg)
    dataset = create_online_learning_dataset(
        system_model_params=params,
        config=cfg,
        window_size=cfg.online_learning.window_size,
        stride=cfg.online_learning.stride,
    )

    _ = dataset[31]
    cached_before = len(dataset.generator._step_cache)
    dataset.update_spacing_scale(1.03, invalidate_from_step=31 * cfg.online_learning.stride)
    cached_after = len(dataset.generator._step_cache)

    assert cached_after <= 31 * cfg.online_learning.stride
    assert cached_after <= cached_before


def test_steering_scale_applied_in_system_model():
    from DCD_MUSIC.src.system_model import SystemModel, SystemModelParams

    base = SystemModelParams()
    base.N = 8
    base.M = 3
    base.wavelength = 0.06
    base.spacing_scale = 1.0
    ref = SystemModel(base, nominal=True).steering_vec_far_field(
        torch.tensor([10.0]), nominal=True
    )

    scaled_params = SystemModelParams()
    scaled_params.N = 8
    scaled_params.M = 3
    scaled_params.wavelength = 0.06
    scaled_params.spacing_scale = 1.05
    scaled = SystemModel(scaled_params, nominal=True).steering_vec_far_field(
        torch.tensor([10.0]), nominal=True
    )
    assert not torch.allclose(scaled, ref)

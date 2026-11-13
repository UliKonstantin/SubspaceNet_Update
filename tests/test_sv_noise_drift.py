"""SV-noise calibration drift is independent of position η."""

import numpy as np
import torch

from config.factory import _create_system_model_params
from config.loader import load_config
from models.deep_cnn_adapter import snapshots_to_covariance_channels
from simulation.calibration_drift import reset_calibration
from simulation.runners.data import create_online_learning_dataset


def _minimal_ol_config():
    cfg = load_config("configs/Used_for_paper/paper_T1_antenna_sweep.yaml")
    cfg.online_learning.trajectory_length = 200
    cfg.online_learning.window_size = 5
    cfg.online_learning.stride = 3
    cfg.online_learning.drift_type = "sv_noise_var"
    cfg.online_learning.eta_update_interval_windows = 40
    cfg.online_learning.sv_noise_increment = 0.7
    cfg.online_learning.max_sv_noise_var = 0.7
    cfg.simulation.seed = 123
    return cfg


def test_sv_noise_update_changes_snapshots_without_touching_eta():
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
    dataset.update_sv_noise_var(0.7, invalidate_from_step=31 * cfg.online_learning.stride)
    post_window, _, _ = dataset[31]

    pre_cov = snapshots_to_covariance_channels(pre_window[0:1]).detach().cpu().numpy()
    post_cov = snapshots_to_covariance_channels(post_window[0:1]).detach().cpu().numpy()
    delta = float(np.linalg.norm(post_cov - pre_cov))

    assert delta > 1e-6, f"Expected snapshots to change after sv_noise drift, delta={delta}"
    assert abs(params.sv_noise_var - 0.7) < 1e-9
    assert params.eta == 0.0
    assert getattr(params, "pitch_eta", 0.0) == 0.0


def test_sv_noise_update_truncates_stale_cache():
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
    dataset.update_sv_noise_var(0.7, invalidate_from_step=31 * cfg.online_learning.stride)
    cached_after = len(dataset.generator._step_cache)

    assert cached_after <= 31 * cfg.online_learning.stride
    assert cached_after <= cached_before

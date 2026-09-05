"""Verify online-learning eta updates change generated snapshots."""

import numpy as np
import torch

from config.loader import load_config
from config.factory import _create_system_model_params
from simulation.runners.data import create_online_learning_dataset
from models.deep_cnn_adapter import snapshots_to_covariance_channels


def _minimal_ol_config():
    cfg = load_config("configs/Used_for_paper/paper_T1_antenna_sweep_deepcnn.yaml")
    cfg.online_learning.trajectory_length = 200
    cfg.online_learning.window_size = 5
    cfg.online_learning.stride = 3
    cfg.online_learning.eta_update_interval_windows = 32
    cfg.online_learning.eta_increment = 0.9
    cfg.online_learning.max_eta = 0.9
    cfg.simulation.seed = 123
    return cfg


def test_eta_update_changes_deepcnn_input():
    cfg = _minimal_ol_config()
    params = _create_system_model_params(cfg)
    params.eta = 0.0

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
    dataset.update_eta(0.9, invalidate_from_step=31 * cfg.online_learning.stride)
    post_window, _, _ = dataset[31]

    pre_cov = snapshots_to_covariance_channels(pre_window[0:1]).detach().cpu().numpy()
    post_cov = snapshots_to_covariance_channels(post_window[0:1]).detach().cpu().numpy()
    delta = float(np.linalg.norm(post_cov - pre_cov))

    assert delta > 1e-6, f"Expected DeepCNN covariance input to change after eta shift, delta={delta}"


def test_eta_update_truncates_stale_cache():
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
    dataset.update_eta(0.9, invalidate_from_step=31 * cfg.online_learning.stride)
    cached_after = len(dataset.generator._step_cache)

    assert cached_after <= 31 * cfg.online_learning.stride
    assert cached_after <= cached_before

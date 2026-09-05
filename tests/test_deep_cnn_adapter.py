"""Tests for DeepCNN v2 wiring and PF-09 training/checkpoint loading."""

import torch

from config.factory import create_model, create_system_model
from config.loader import load_config
from models.deep_cnn_adapter import DeepCNNDoAAdapter, snapshots_to_covariance_channels


def test_snapshots_to_covariance_channels_shape():
    x = torch.randn(2, 9, 64, dtype=torch.complex64)
    cov = snapshots_to_covariance_channels(x)
    assert cov.shape == (2, 9, 9, 3)
    assert cov.dtype == torch.float32


def test_deepcnn_factory_and_forward():
    config = load_config("configs/Used_for_paper/paper_deepcnn_smoke.yaml")
    system_model = create_system_model(config)
    model = create_model(config, system_model)
    assert isinstance(model, DeepCNNDoAAdapter)

    x = torch.randn(1, 9, 200, dtype=torch.complex64)
    model.eval()
    angles, _, _ = model(x, num_sources=3)
    assert angles.shape == (1, 3)
    assert angles.abs().max() <= torch.pi


def test_deepcnn_soft_topk_backprop():
    config = load_config("configs/Used_for_paper/paper_deepcnn_smoke.yaml")
    system_model = create_system_model(config)
    model = create_model(config, system_model)
    model.train()

    x = torch.randn(1, 9, 200, dtype=torch.complex64)
    angles, _, _ = model(x, num_sources=3)
    loss = angles.sum()
    loss.backward()

    grad_norms = [
        p.grad.detach().norm().item()
        for p in model.cnn.parameters()
        if p.grad is not None
    ]
    assert grad_norms
    assert any(norm > 0 for norm in grad_norms)


def test_deepcnn_binary_grid_uses_radian_labels():
    config = load_config("configs/Used_for_paper/paper_deepcnn_smoke.yaml")
    system_model = create_system_model(config)
    model = create_model(config, system_model)

    angles_deg = torch.tensor([[10.0, -20.0, 30.0]])
    angles_rad = angles_deg * (torch.pi / 180.0)
    targets_from_rad = model._angles_to_binary_grid(angles_rad, num_sources=3)
    targets_from_deg = model._angles_to_binary_grid(angles_deg, num_sources=3)
    assert torch.equal(targets_from_rad, targets_from_deg)


def test_deepcnn_training_step():
    config = load_config("configs/Used_for_paper/paper_deepcnn_smoke.yaml")
    system_model = create_system_model(config)
    model = create_model(config, system_model)

    x = torch.randn(4, 9, 200, dtype=torch.complex64)
    sources = torch.tensor([3, 3, 3, 3])
    angles = (torch.randn(4, 3) * 30) * (torch.pi / 180.0)
    loss, acc, reg = model.training_step((x, sources, angles), 0)
    assert loss.ndim == 0
    assert reg is None
    assert 0.0 <= acc <= 1.0
    loss.backward()


def test_deepcnn_checkpoint_roundtrip():
    config = load_config("configs/Used_for_paper/paper_deepcnn_smoke.yaml")
    system_model = create_system_model(config)
    model = create_model(config, system_model)
    clone = create_model(config, system_model)

    payload = {"model_state_dict": model.state_dict()}
    clone.load_state_dict(payload["model_state_dict"])

    model.eval()
    clone.eval()
    x = torch.randn(1, 9, 200, dtype=torch.complex64)
    angles_a, _, _ = model(x, num_sources=3)
    angles_b, _, _ = clone(x, num_sources=3)
    assert torch.allclose(angles_a, angles_b)


def test_deepcnn_loads_raw_backbone_state_dict():
    config = load_config("configs/Used_for_paper/paper_deepcnn_smoke.yaml")
    system_model = create_system_model(config)
    model = create_model(config, system_model)
    clone = create_model(config, system_model)

    clone.load_state_dict(model.cnn.state_dict())

    model.eval()
    clone.eval()
    x = torch.randn(1, 9, 200, dtype=torch.complex64)
    angles_a, _, _ = model(x, num_sources=3)
    angles_b, _, _ = clone(x, num_sources=3)
    assert torch.allclose(angles_a, angles_b)

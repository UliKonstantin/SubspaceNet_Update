"""IQ-ResNet regression backbone: shapes, tanh range, factory, yaml."""

import math

import torch
import torch.nn as nn

from config.factory import create_model, create_system_model
from config.loader import load_config
from models.iq_resnet import IQResNet, snapshots_to_iq
from simulation.runners.Online_learning import apply_deepcnn_default_losses


def test_snapshots_to_iq_shape():
    x = torch.randn(2, 9, 200, dtype=torch.complex64)
    iq = snapshots_to_iq(x)
    assert iq.shape == (2, 1, 18, 200)
    assert iq.dtype == torch.float32
    assert torch.allclose(iq[:, 0, :9], x.real)
    assert torch.allclose(iq[:, 0, 9:], x.imag)


def test_stem_output_height_and_channels():
    model = IQResNet(n_sensors=9, n_sources=3, doa_range_deg=60.0)
    x = torch.randn(2, 9, 200, dtype=torch.complex64)
    iq = snapshots_to_iq(x)
    stem = model.backbone.stem_relu(model.backbone.stem_bn(model.backbone.stem(iq)))
    assert stem.shape[1] == 64
    assert stem.shape[2] == 1
    assert stem.shape[3] == 200


def test_tanh_range_before_scale():
    model = IQResNet(n_sensors=9, n_sources=3, doa_range_deg=60.0)
    model.eval()
    x = torch.randn(4, 9, 200, dtype=torch.complex64)
    tanh_out = model.encode_tanh(x)
    assert tanh_out.shape == (4, 3)
    assert float(tanh_out.abs().max()) <= 1.0 + 1e-6


def test_forward_radians_within_fov():
    doa_range = 60.0
    model = IQResNet(n_sensors=9, n_sources=3, doa_range_deg=doa_range)
    model.eval()
    x = torch.randn(2, 9, 200, dtype=torch.complex64)
    angles, aux, cov = model(x, num_sources=3)
    assert aux is None and cov is None
    assert angles.shape == (2, 3)
    limit = (doa_range / 2.0) * (math.pi / 180.0)
    assert float(angles.abs().max()) <= limit + 1e-5


def test_training_step_backprops():
    model = IQResNet(n_sensors=9, n_sources=3, doa_range_deg=60.0)
    model.train()
    x = torch.randn(4, 9, 200, dtype=torch.complex64)
    sources = torch.tensor([3, 3, 3, 3])
    angles = (torch.randn(4, 3) * 30) * (math.pi / 180.0)
    loss, acc, reg = model.training_step((x, sources, angles), 0)
    assert loss.ndim == 0
    assert reg is None
    assert 0.0 <= acc <= 1.0
    loss.backward()
    grad_norms = [p.grad.detach().norm().item() for p in model.parameters() if p.grad is not None]
    assert grad_norms
    assert any(n > 0 for n in grad_norms)


def test_online_adaptation_mode_freezes_backbone_bn():
    model = IQResNet(n_sensors=9, n_sources=3, doa_range_deg=60.0)
    model.set_online_adaptation_mode(True)
    assert model.training
    assert not model.backbone.training
    assert not model.backbone.stem_bn.training


def test_residual_head_is_identity_at_init():
    torch.manual_seed(0)
    model = IQResNet(n_sensors=9, n_sources=3, doa_range_deg=60.0)
    model.eval()
    x = torch.randn(2, 9, 200, dtype=torch.complex64)
    iq = snapshots_to_iq(x)
    with torch.no_grad():
        feats = model.backbone(iq)
        base = torch.tanh(model.fc(feats))
        nn_out = model.encode_tanh(x)
    assert torch.allclose(nn_out, base, atol=1e-6)
    nn.init.ones_(model.adapt_down.weight)
    with torch.no_grad():
        still = model.encode_tanh(x)
    assert torch.allclose(still, base, atol=1e-6)


def test_ol_mode_trains_sn_budget_residual_head_only():
    model = IQResNet(n_sensors=9, n_sources=3, doa_range_deg=60.0)
    model.set_online_adaptation_mode(True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    expected = 512 * 80 + 80 + 80 * 3 + 3
    assert n_train == expected
    assert all(p is q for p, q in zip(trainable, model.online_adaptation_parameters()))
    for p in model.backbone.parameters():
        assert not p.requires_grad
    for p in model.fc.parameters():
        assert not p.requires_grad

    x = torch.randn(2, 9, 200, dtype=torch.complex64)
    angles, _, _ = model(x, 3)
    angles.sum().backward()
    adapter_grad = sum(p.grad.detach().abs().sum().item() for p in trainable if p.grad is not None)
    assert adapter_grad > 0
    assert all(p.grad is None for p in model.backbone.parameters())
    assert all(p.grad is None for p in model.fc.parameters())


def test_old_checkpoint_keeps_zero_adapter():
    src = IQResNet(n_sensors=9, n_sources=3, doa_range_deg=60.0)
    old = {
        k: v.clone()
        for k, v in src.state_dict().items()
        if not k.startswith(("adapt_down.", "adapt_up."))
    }
    dst = IQResNet(n_sensors=9, n_sources=3, doa_range_deg=60.0)
    nn.init.ones_(dst.adapt_up.weight)
    dst.load_state_dict(old, strict=True)
    assert torch.count_nonzero(dst.adapt_up.weight) == 0
    assert torch.count_nonzero(dst.adapt_up.bias) == 0


def test_factory_iqresnet_loads():
    config = load_config("configs/Used_for_paper/paper_iqresnet_training.yaml")
    system_model = create_system_model(config)
    model = create_model(config, system_model)
    assert isinstance(model, IQResNet)
    assert model.N == 9
    assert model.M == 3
    assert model.checkpoint_name == "IQResNet"
    x = torch.randn(1, 9, 200, dtype=torch.complex64)
    model.eval()
    angles, _, _ = model(x, num_sources=3)
    assert angles.shape == (1, 3)


def test_iqresnet_yamls_load():
    train_cfg = load_config("configs/Used_for_paper/paper_iqresnet_training.yaml")
    assert train_cfg.model.type == "IQResNet"
    assert train_cfg.training.optimizer == "Adam"
    assert train_cfg.training.scheduler == "StepLR"
    assert train_cfg.training.step_size == 10
    assert abs(train_cfg.training.gamma - 0.5) < 1e-12
    assert train_cfg.training.batch_size == 64
    assert train_cfg.training.epochs == 40

    cfg = load_config("configs/Used_for_paper/paper_T1_iqresnet.yaml")
    ol = cfg.online_learning
    assert cfg.model.type == "IQResNet"
    assert ol.drift_detector == "loss_cusum"
    assert cfg.system_model.N == 9
    assert cfg.scenario_config.values == [9]
    assert abs(ol.eta_increment - 1.30) < 1e-12
    assert ol.dataset_size == 1
    assert ol.loss_config.adaptation_loss == "unsupervised_rmspe"
    apply_deepcnn_default_losses(cfg)
    assert cfg.online_learning.loss_config.adaptation_loss == "unsupervised_rmspe"


def test_thin_variant_is_sn_budget_total_params():
    fat = IQResNet(n_sensors=9, n_sources=3, doa_range_deg=60.0)
    thin = IQResNet(n_sensors=9, n_sources=3, doa_range_deg=60.0, variant="thin")
    n_fat = sum(p.numel() for p in fat.parameters())
    n_thin = sum(p.numel() for p in thin.parameters())
    n_adapt = sum(p.numel() for p in thin.online_adaptation_parameters())
    assert thin.checkpoint_name == "IQThinNet"
    assert thin.feat_dim == 80
    assert n_thin < n_fat / 10
    assert 35_000 <= n_thin <= 50_000
    assert n_adapt == n_thin


def test_thin_ol_trains_all_weights():
    model = IQResNet(n_sensors=9, n_sources=3, doa_range_deg=60.0, variant="thin")
    model.set_online_adaptation_mode(True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_all = sum(p.numel() for p in model.parameters())
    assert n_train == n_all
    assert all(p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.fc.parameters())
    x = torch.randn(2, 9, 200, dtype=torch.complex64)
    angles, _, _ = model(x, 3)
    angles.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.parameters())

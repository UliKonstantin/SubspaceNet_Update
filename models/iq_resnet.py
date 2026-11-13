"""IQ-ResNet regression: raw I/Q snapshots → M continuous DoAs (tanh)."""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from DCD_MUSIC.src.utils import device

_DEG2RAD = math.pi / 180.0


def snapshots_to_iq(x: torch.Tensor) -> torch.Tensor:
    """
    Stack real then imag along the sensor axis.

    Args:
        x: Snapshots [B, N, T] (complex or real).

    Returns:
        Tensor [B, 1, 2N, T].
    """
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if not torch.is_complex(x):
        x = x.to(torch.complex64)
    stacked = torch.cat([x.real, x.imag], dim=1)
    return stacked.unsqueeze(1).to(dtype=torch.float32)


class ResidualBlock(nn.Module):
    """ResNet block over the time axis: 1×3 convs, stride-(1,2) downsample."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), stride=(1, 2), padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.relu(self.conv1(x))
        main = self.conv2(main)
        return self.relu(main + self.shortcut(x))


class IQResNetBackbone(nn.Module):
    """Stem + pool + 4 residual blocks + GAP → 512-d vector."""

    def __init__(self, n_sensors: int):
        super().__init__()
        self.n_sensors = n_sensors
        array_dim = 2 * n_sensors
        self.stem = nn.Conv2d(1, 64, kernel_size=(array_dim, 5), stride=(array_dim, 1), padding=(0, 2))
        self.stem_bn = nn.BatchNorm2d(64)
        self.stem_relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.block64 = ResidualBlock(64, 64)
        self.block128 = ResidualBlock(64, 128)
        self.block256 = ResidualBlock(128, 256)
        self.block512 = ResidualBlock(256, 512)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        x = self.stem_relu(self.stem_bn(self.stem(iq)))
        x = self.pool(x)
        x = self.block64(x)
        x = self.block128(x)
        x = self.block256(x)
        x = self.block512(x)
        return self.gap(x).flatten(1)


class IQThinBackbone(nn.Module):
    """Stem + 2 residual blocks + GAP → 80-d. Whole net ≈ 42k with residual head."""

    feat_dim = 80

    def __init__(self, n_sensors: int):
        super().__init__()
        self.n_sensors = n_sensors
        array_dim = 2 * n_sensors
        self.stem = nn.Conv2d(1, 16, kernel_size=(array_dim, 5), stride=(array_dim, 1), padding=(0, 2))
        self.stem_bn = nn.BatchNorm2d(16)
        self.stem_relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.block32 = ResidualBlock(16, 32)
        self.block80 = ResidualBlock(32, 80)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        x = self.stem_relu(self.stem_bn(self.stem(iq)))
        x = self.pool(x)
        x = self.block32(x)
        x = self.block80(x)
        return self.gap(x).flatten(1)


# SN-sized residual DoA head on GAP features (512→80→M ≈ 41k). Zero-init
# ``adapt_up`` ⇒ identity at t=0 so old ckpts keep their pretrained mapping.
OL_ADAPT_HIDDEN = 80
_ADAPTER_KEY_PREFIXES = ("adapt_down.", "adapt_up.")


class IQResNet(nn.Module):
    """
    Regression IQ-ResNet for v2 training / OL.

    ``forward(snapshots, num_sources)`` returns radians in ±doa_range/2, same
    contract as SubspaceNet ESPRIT.
    """

    checkpoint_name = "IQResNet"
    field_type = "far"

    def __init__(
        self,
        n_sensors: int,
        n_sources: int,
        doa_range_deg: float = 60.0,
        n_snapshots: Optional[int] = None,
        ol_adapt_hidden: int = OL_ADAPT_HIDDEN,
        variant: str = "small",
    ):
        super().__init__()
        if n_sensors < 1:
            raise ValueError(f"IQResNet requires N >= 1, got {n_sensors}")
        if n_sources < 1:
            raise ValueError(f"IQResNet requires M >= 1, got {n_sources}")
        if ol_adapt_hidden < 1:
            raise ValueError(f"ol_adapt_hidden must be >= 1, got {ol_adapt_hidden}")
        self.N = n_sensors
        self.M = n_sources
        self.T = n_snapshots
        self.variant = str(variant or "small")
        self.doa_range_deg = float(doa_range_deg)
        self.doa_half_deg = self.doa_range_deg / 2.0
        self.scale_rad = self.doa_half_deg * _DEG2RAD
        self.ol_adapt_hidden = int(ol_adapt_hidden)
        if self.variant == "thin":
            self.backbone = IQThinBackbone(n_sensors)
            self.feat_dim = IQThinBackbone.feat_dim
            self.checkpoint_name = "IQThinNet"
        else:
            self.backbone = IQResNetBackbone(n_sensors)
            self.feat_dim = 512
            self.checkpoint_name = "IQResNet"
        self.fc = nn.Linear(self.feat_dim, n_sources)
        self.adapt_down = nn.Linear(self.feat_dim, self.ol_adapt_hidden)
        self.adapt_up = nn.Linear(self.ol_adapt_hidden, n_sources)
        nn.init.zeros_(self.adapt_up.weight)
        nn.init.zeros_(self.adapt_up.bias)
        for p in list(self.adapt_down.parameters()) + list(self.adapt_up.parameters()):
            p.requires_grad = False
        self._rmspe_criterion = None

    def online_adaptation_parameters(self):
        """OL trainable set: residual head on fat IQ, every weight on thin."""
        if self.variant == "thin":
            return list(self.parameters())
        return list(self.adapt_down.parameters()) + list(self.adapt_up.parameters())

    def load_state_dict(self, state_dict, strict: bool = True):
        """Old ckpts have no adapter keys; keep zero-init identity head."""
        if not isinstance(state_dict, dict):
            return super().load_state_dict(state_dict, strict=strict)
        own = self.state_dict()
        missing = [k for k in own if k not in state_dict]
        adapter_only_missing = missing and all(
            k.startswith(_ADAPTER_KEY_PREFIXES) for k in missing
        )
        if adapter_only_missing:
            nn.init.zeros_(self.adapt_up.weight)
            nn.init.zeros_(self.adapt_up.bias)
            own = self.state_dict()
            merged = dict(state_dict)
            for k in missing:
                merged[k] = own[k]
            return super().load_state_dict(merged, strict=strict)
        return super().load_state_dict(state_dict, strict=strict)

    def set_online_adaptation_mode(self, enabled: bool) -> None:
        """OL GD: thin = all weights (SN-style). Fat IQ = residual head only."""
        if enabled:
            self.train()
            if self.variant == "thin":
                for p in self.parameters():
                    p.requires_grad = True
                return
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.fc.parameters():
                p.requires_grad = False
            for p in self.adapt_down.parameters():
                p.requires_grad = True
            for p in self.adapt_up.parameters():
                p.requires_grad = True
            self.adapt_down.train()
            self.adapt_up.train()
        else:
            self.eval()
            if self.variant == "thin":
                for p in self.parameters():
                    p.requires_grad = False
                return
            for p in self.adapt_down.parameters():
                p.requires_grad = False
            for p in self.adapt_up.parameters():
                p.requires_grad = False

    def _get_rmspe_criterion(self):
        if self._rmspe_criterion is None:
            from DCD_MUSIC.src.metrics.rmspe_loss import RMSPELoss

            self._rmspe_criterion = RMSPELoss().to(device)
        return self._rmspe_criterion

    def _angles_to_radians(self, angles: torch.Tensor) -> torch.Tensor:
        if angles.abs().max() <= math.pi + 0.01:
            return angles
        return angles * _DEG2RAD

    def encode_tanh(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone + FC + residual head + tanh in [-1, 1], shape [B, M]."""
        iq = snapshots_to_iq(x)
        feats = self.backbone(iq)
        logits = self.fc(feats) + self.adapt_up(F.relu(self.adapt_down(feats)))
        return torch.tanh(logits)

    def _decode_radians(self, tanh_out: torch.Tensor) -> torch.Tensor:
        return tanh_out * self.scale_rad

    def _permute_to_gt(self, pred_rad: torch.Tensor, true_rad: torch.Tensor) -> torch.Tensor:
        _, best_perm = self._get_rmspe_criterion()(pred_rad, true_rad, return_best_perm=True)
        return torch.gather(pred_rad, 1, best_perm.to(device=pred_rad.device))

    def _regression_loss(
        self, pred_rad: torch.Tensor, true_rad: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        aligned = self._permute_to_gt(pred_rad, true_rad)
        pred_tanh = aligned / self.scale_rad
        true_tanh = (true_rad / self.scale_rad).clamp(-1.0, 1.0)
        mse = F.mse_loss(pred_tanh, true_tanh)
        # Trainer logs loss/B * 180/π as degrees (SN RMSPE-sum convention).
        # Backprop stays L_MSE; the detached RMSPE-sum is only the logged value.
        with torch.no_grad():
            rmspe_sum = self._get_rmspe_criterion()(pred_rad, true_rad)
        err_deg = (rmspe_sum.item() / max(pred_rad.shape[0], 1)) * (180.0 / math.pi)
        loss = rmspe_sum.detach() + (mse - mse.detach())
        return loss, float(err_deg)

    def _prepare_batch(self, batch):
        x, sources_num, angles = batch
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if (sources_num != sources_num[0]).any():
            raise ValueError("IQResNet expects a constant source count within each batch")
        num_sources = int(sources_num[0].item())
        if num_sources != self.M:
            raise ValueError(f"IQResNet was built for M={self.M}, got num_sources={num_sources}")
        return x.to(device), num_sources, angles.to(device)

    def training_step(self, batch, batch_idx):
        x, num_sources, angles = self._prepare_batch(batch)
        tanh_out = self.encode_tanh(x)
        pred_rad = self._decode_radians(tanh_out)
        true_rad = self._angles_to_radians(angles[:, :num_sources])
        loss, err_deg = self._regression_loss(pred_rad, true_rad)
        acc = max(0.0, 1.0 - err_deg / max(self.doa_half_deg, 1e-6))
        return loss, acc, None

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, num_sources, angles = self._prepare_batch(batch)
            tanh_out = self.encode_tanh(x)
            pred_rad = self._decode_radians(tanh_out)
            true_rad = self._angles_to_radians(angles[:, :num_sources])
            loss, err_deg = self._regression_loss(pred_rad, true_rad)
            acc = max(0.0, 1.0 - err_deg / max(self.doa_half_deg, 1e-6))
        return loss, acc

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def forward(
        self,
        x: torch.Tensor,
        num_sources: Optional[Union[int, torch.Tensor]] = None,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, None, None]:
        tanh_out = self.encode_tanh(x)
        angles_rad = self._decode_radians(tanh_out)
        if num_sources is not None:
            if isinstance(num_sources, torch.Tensor):
                num_sources = int(num_sources.flatten()[0].item())
            angles_rad = angles_rad[:, : int(num_sources)]
        return angles_rad, None, None

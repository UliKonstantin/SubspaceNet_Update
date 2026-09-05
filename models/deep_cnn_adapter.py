"""DeepCNN adapter: snapshots -> covariance channels -> spectrum -> DoA angles."""

from __future__ import annotations

import math
from typing import Literal, Optional, Tuple, Union

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F

from DCD_MUSIC.src.utils import device

_DEG2RAD = math.pi / 180.0


def get_k_peaks_topk_pad(
    grid_size: int, k: int, prediction: torch.Tensor
) -> torch.Tensor:
    """
    scipy peak picking with deterministic top-k padding (no random angles).

    Same find_peaks thresholds as DCD_MUSIC ``get_k_peaks``; when fewer than ``k``
    peaks are found, remaining DOAs come from highest spectrum bins not yet used.
    """
    angles_grid = torch.linspace(
        -90, 90, grid_size, device=prediction.device, dtype=prediction.dtype
    )
    flat = prediction.detach().flatten()
    peaks, peaks_data = scipy.signal.find_peaks(
        flat.cpu().numpy(), prominence=0.05, height=0.01
    )
    if len(peaks):
        peak_order = peaks[np.argsort(peaks_data["peak_heights"])[::-1]]
    else:
        peak_order = np.array([], dtype=int)

    _, top_idx = torch.topk(flat, min(k, flat.numel()))
    top_list = top_idx.detach().cpu().tolist()

    chosen: list[int] = []
    for idx in peak_order.tolist():
        if len(chosen) >= k:
            break
        if idx not in chosen:
            chosen.append(int(idx))
    for idx in top_list:
        if len(chosen) >= k:
            break
        if idx not in chosen:
            chosen.append(int(idx))
    while len(chosen) < k and top_list:
        for idx in top_list:
            if len(chosen) >= k:
                break
            chosen.append(int(idx))

    idx_tensor = torch.tensor(chosen[:k], device=prediction.device, dtype=torch.long)
    return angles_grid[idx_tensor]


def snapshots_to_covariance_channels(x: torch.Tensor) -> torch.Tensor:
    """
    Build Papageorgiou-style 3-channel covariance input for DeepCNN.

    Args:
        x: Snapshot tensor [B, N, T] (complex or real).

    Returns:
        Tensor [B, N, N, 3] with channels (real, imag, phase).
    """
    if x.dim() == 2:
        x = x.unsqueeze(0)

    if not torch.is_complex(x):
        x = x.to(torch.complex64)

    x_centered = x - x.mean(dim=-1, keepdim=True)
    rx = torch.einsum("bnt,bmt->bnm", x_centered, x_centered.conj()) / x.shape[-1]

    real = rx.real
    imag = rx.imag
    phase = torch.angle(rx)
    return torch.stack([real, imag, phase], dim=-1)


class DeepCNNDoAAdapter(nn.Module):
    """
    Wrap DeepCNN so v2 training/OL can call model(snapshots, num_sources) -> (angles, _, _).

    Base training uses BCE on a degree-spaced binary grid (Papageorgiou-style).
    OL / eval ``forward()`` returns radians via hard peak picking (eval) or soft top-k (grad path).
    """

    checkpoint_name = "DeepCNN"
    field_type = "far"

    def __init__(
        self,
        cnn: nn.Module,
        grid_size: int,
        peak_method: Literal["peaks", "topk"] = "peaks",
        soft_peak_temperature: float = 0.5,
    ):
        super().__init__()
        self.cnn = cnn
        self.grid_size = grid_size
        self.peak_method = peak_method
        self.soft_peak_temperature = soft_peak_temperature
        self.N = getattr(cnn, "N", None)

    @property
    def angle_grid(self) -> torch.Tensor:
        return torch.linspace(-90, 90, self.grid_size, device=device)

    def _labels_to_degrees(self, angles: torch.Tensor) -> torch.Tensor:
        """v2 dataset labels are radians; the CNN grid is defined in degrees."""
        if angles.abs().max() <= math.pi + 0.01:
            return angles * (180.0 / math.pi)
        return angles

    def _angles_to_binary_grid(self, angles: torch.Tensor, num_sources: int) -> torch.Tensor:
        grid = self.angle_grid.to(device=angles.device, dtype=angles.dtype)
        angles_deg = self._labels_to_degrees(angles)
        batch_size = angles.shape[0]
        targets = torch.zeros(batch_size, self.grid_size, device=angles.device, dtype=angles.dtype)
        for batch_idx in range(batch_size):
            for source_idx in range(num_sources):
                angle = angles_deg[batch_idx, source_idx]
                grid_idx = torch.argmin(torch.abs(grid - angle))
                targets[batch_idx, grid_idx] = 1.0
        return targets

    def _grid_accuracy(self, spectrum: torch.Tensor, targets: torch.Tensor) -> float:
        predicted = (spectrum >= 0.5).float()
        active = targets.sum(dim=1).clamp(min=1.0)
        hits = (predicted * targets).sum(dim=1)
        return float((hits / active).mean().item())

    def _soft_k_angles_deg(self, prediction: torch.Tensor, k: int) -> torch.Tensor:
        """
        Differentiable top-k peak angles (degrees) via straight-through softmax weights.

        Forward values match hard grid lookup; backward flows through softmax(top-k vals).
        """
        grid = self.angle_grid.to(device=prediction.device, dtype=prediction.dtype)
        vals, idx = torch.topk(prediction.flatten(), k)
        weights = F.softmax(vals / self.soft_peak_temperature, dim=0)
        hard = grid[idx]
        soft = weights * grid[idx]
        return hard.detach() - soft.detach() + soft

    def _hard_k_angles_deg(self, prediction: torch.Tensor, k: int) -> torch.Tensor:
        from DCD_MUSIC.src.utils import get_k_angles

        if self.peak_method == "topk":
            return get_k_angles(self.grid_size, k, prediction).to(
                dtype=torch.float32, device=prediction.device
            )
        return get_k_peaks_topk_pad(self.grid_size, k, prediction).to(
            dtype=torch.float32, device=prediction.device
        )

    def _spectrum_to_angles_deg(self, spectrum: torch.Tensor, num_sources: int) -> torch.Tensor:
        k = max(int(num_sources), 1)
        use_soft_topk = torch.is_grad_enabled() and spectrum.requires_grad
        angles = []
        for batch_idx in range(spectrum.shape[0]):
            pred = spectrum[batch_idx]
            if use_soft_topk:
                batch_angles = self._soft_k_angles_deg(pred, k)
            else:
                batch_angles = self._hard_k_angles_deg(pred, k)
            angles.append(batch_angles.to(dtype=torch.float32, device=spectrum.device))
        return torch.stack(angles, dim=0)

    def _prepare_batch(self, batch):
        x, sources_num, angles = batch
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if (sources_num != sources_num[0]).any():
            raise ValueError("DeepCNNDoAAdapter expects a constant source count within each batch")
        num_sources = int(sources_num[0].item())
        return x.to(device), num_sources, angles.to(device)

    def training_step(self, batch, batch_idx):
        x, num_sources, angles = self._prepare_batch(batch)
        cov_input = snapshots_to_covariance_channels(x)
        spectrum = self.cnn(cov_input)
        targets = self._angles_to_binary_grid(angles[:, :num_sources], num_sources)
        loss = F.binary_cross_entropy(spectrum, targets)
        acc = self._grid_accuracy(spectrum.detach(), targets)
        return loss, acc, None

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, num_sources, angles = self._prepare_batch(batch)
            cov_input = snapshots_to_covariance_channels(x)
            spectrum = self.cnn(cov_input)
            targets = self._angles_to_binary_grid(angles[:, :num_sources], num_sources)
            loss = F.binary_cross_entropy(spectrum, targets)
            acc = self._grid_accuracy(spectrum, targets)
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
        cov_input = snapshots_to_covariance_channels(x)
        spectrum = self.cnn(cov_input)
        if num_sources is None:
            raise ValueError("DeepCNNDoAAdapter requires num_sources for peak extraction")
        if isinstance(num_sources, torch.Tensor):
            num_sources = int(num_sources.flatten()[0].item())
        angles_deg = self._spectrum_to_angles_deg(spectrum, int(num_sources))
        angles_rad = angles_deg * _DEG2RAD
        return angles_rad, None, None

    def load_state_dict(self, state_dict, strict: bool = True):
        if not state_dict:
            return super().load_state_dict(state_dict, strict=strict)

        if any(key.startswith("cnn.") for key in state_dict):
            return super().load_state_dict(state_dict, strict=strict)

        return self.cnn.load_state_dict(state_dict, strict=strict)

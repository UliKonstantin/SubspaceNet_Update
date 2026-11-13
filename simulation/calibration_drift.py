"""Shared calibration-drift helpers for online learning and sample generation."""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from simulation.runners.data import OnlineLearningTrajectoryGenerator

logger = logging.getLogger("SubspaceNet.calibration_drift")

INITIAL_ETA = 0.0
INITIAL_PITCH_ETA = 0.0
INITIAL_SPACING_SCALE = 1.0
INITIAL_SV_NOISE_VAR = 0.0
INITIAL_BORESIGHT_OFFSET_DEG = 0.0


def gradual_eta_at_window(
    window_idx: int,
    *,
    start_window: int,
    ramp_windows: int,
    target_eta: float,
) -> Optional[float]:
    """η to apply *before* this window during a linear ramp, else None (hold / not started)."""
    if ramp_windows <= 0 or start_window <= 0:
        return None
    if window_idx < start_window or window_idx >= start_window + ramp_windows:
        return None
    step = window_idx - start_window + 1
    return float(target_eta) * step / float(ramp_windows)


def _clear_location_direction(model) -> None:
    if model is not None and hasattr(model, "_location_direction"):
        model._location_direction = None


def _ensure_location_direction(model):
    u = getattr(model, "_location_direction", None)
    n = model.params.N
    if u is not None and getattr(u, "numel", lambda: 0)() == n:
        return u
    u = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=n))
    model._location_direction = u
    return u


def _apply_location_and_pitch(model, *, reuse_direction: bool = False) -> None:
    n = model.params.N
    model.eta = model._SystemModel__set_eta()
    model.pitch_eta = model._SystemModel__set_pitch_eta()
    if hasattr(model, "_SystemModel__set_bias"):
        model.bias_phys = model._SystemModel__set_bias()
    if float(getattr(model.params, "pitch_eta", 0.0) or 0.0) == 0:
        model.pitch_noise = torch.zeros(n)
    else:
        model.pitch_noise = model.get_pitch_noise(True)

    if hasattr(model, "get_bias_noise"):
        model.bias_noise = model.get_bias_noise(True)
    if hasattr(model, "get_sv_geometry_noise"):
        model.sv_geometry_noise = model.get_sv_geometry_noise(True)

    if model.params.eta == 0:
        model.location_noise = torch.zeros(n)
        return
    if reuse_direction:
        u = _ensure_location_direction(model)
        model.location_noise = model.eta * u
        return
    model.location_noise = model.get_distance_noise(True)
    if float(model.eta) != 0:
        model._location_direction = model.location_noise / model.eta


def _redraw_location_and_pitch(model) -> None:
    _apply_location_and_pitch(model, reuse_direction=False)


def _sync_system_model_geometry(system_model) -> None:
    """Refresh derived geometry fields on a SystemModel after param changes."""
    _redraw_location_and_pitch(system_model)


def _sync_samples_model_geometry(samples_model) -> None:
    _redraw_location_and_pitch(samples_model)


def _invalidate_step_cache(
    generator: "OnlineLearningTrajectoryGenerator",
    invalidate_from_step: Optional[int],
    label: str,
    old_value: float,
    new_value: float,
) -> None:
    if invalidate_from_step is None:
        return
    keep = max(0, int(invalidate_from_step))
    if keep < len(generator._step_cache):
        generator._step_cache = generator._step_cache[:keep]
    generator.current_step_in_session = len(generator._step_cache)
    logger.info(
        "Truncated step cache to %s steps for %s %.4f -> %.4f",
        keep,
        label,
        old_value,
        new_value,
    )


def reset_calibration(system_model_params, *models) -> None:
    """Reset calibration params to nominal at the start of each OL trajectory."""
    system_model_params.eta = INITIAL_ETA
    system_model_params.pitch_eta = INITIAL_PITCH_ETA
    system_model_params.spacing_scale = INITIAL_SPACING_SCALE
    system_model_params.sv_noise_var = INITIAL_SV_NOISE_VAR
    system_model_params.boresight_offset_deg = INITIAL_BORESIGHT_OFFSET_DEG
    for model in models:
        if model is None:
            continue
        _clear_location_direction(model)
        _sync_system_model_geometry(model)


def apply_position_eta_update(
    generator: "OnlineLearningTrajectoryGenerator",
    new_eta: float,
    *,
    invalidate_from_step: Optional[int] = None,
    system_model=None,
    reuse_direction: bool = False,
) -> None:
    """Apply a position-calibration (η) update. Does not touch sv_noise_var.

    reuse_direction=True scales a frozen u_n ~ Unif[-1,1] (gradual ramp).
    False redraws δ_n ~ Unif[-η_phys, η_phys] (T1 step).
    """
    params = generator.system_model_params
    old_eta = params.eta
    params.eta = new_eta
    _apply_location_and_pitch(generator.samples_model, reuse_direction=reuse_direction)
    if system_model is not None:
        if reuse_direction:
            u = getattr(generator.samples_model, "_location_direction", None)
            if u is not None:
                system_model._location_direction = u.clone()
        _apply_location_and_pitch(system_model, reuse_direction=reuse_direction)

    _invalidate_step_cache(generator, invalidate_from_step, "eta", old_eta, new_eta)
    logger.info(
        "Position eta updated from %.4f to %.4f (sv_noise_var=%.4f, spacing_scale=%.4f, reuse_direction=%s).",
        old_eta,
        params.eta,
        params.sv_noise_var,
        params.spacing_scale,
        reuse_direction,
    )


def apply_pitch_eta_update(
    generator: "OnlineLearningTrajectoryGenerator",
    new_pitch_eta: float,
    *,
    invalidate_from_step: Optional[int] = None,
    system_model=None,
) -> None:
    """Apply index-scaled pitch-η jump only. Does not touch eta / sv_noise_var / spacing_scale."""
    params = generator.system_model_params
    old = float(getattr(params, "pitch_eta", INITIAL_PITCH_ETA) or 0.0)
    params.pitch_eta = float(new_pitch_eta)
    _sync_samples_model_geometry(generator.samples_model)
    if system_model is not None:
        _sync_system_model_geometry(system_model)

    _invalidate_step_cache(generator, invalidate_from_step, "pitch_eta", old, params.pitch_eta)
    logger.info(
        "Pitch eta updated from %.4f to %.4f (eta=%.4f, sv_noise_var=%.4f).",
        old,
        params.pitch_eta,
        params.eta,
        params.sv_noise_var,
    )


def apply_sv_noise_update(
    generator: "OnlineLearningTrajectoryGenerator",
    new_sv_noise_var: float,
    *,
    invalidate_from_step: Optional[int] = None,
    system_model=None,
) -> None:
    """Apply additive steering-vector noise variance jump only. Does not touch η."""
    params = generator.system_model_params
    old_var = float(getattr(params, "sv_noise_var", INITIAL_SV_NOISE_VAR) or 0.0)
    params.sv_noise_var = float(new_sv_noise_var)
    _sync_samples_model_geometry(generator.samples_model)
    if system_model is not None:
        _sync_system_model_geometry(system_model)

    _invalidate_step_cache(
        generator, invalidate_from_step, "sv_noise_var", old_var, params.sv_noise_var
    )
    logger.info(
        "SV noise variance updated from %.4f to %.4f (eta=%.4f).",
        old_var,
        params.sv_noise_var,
        params.eta,
    )


def apply_spacing_scale_update(
    generator: "OnlineLearningTrajectoryGenerator",
    new_spacing_scale: float,
    *,
    invalidate_from_step: Optional[int] = None,
    system_model=None,
) -> None:
    """Apply a global element-spacing scale jump; regenerate cached steps."""
    params = generator.system_model_params
    old_scale = getattr(params, "spacing_scale", INITIAL_SPACING_SCALE)
    params.spacing_scale = float(new_spacing_scale)
    _sync_samples_model_geometry(generator.samples_model)
    if system_model is not None:
        _sync_system_model_geometry(system_model)

    _invalidate_step_cache(
        generator, invalidate_from_step, "spacing_scale", old_scale, params.spacing_scale
    )
    logger.info(
        "Spacing scale updated from %.4f to %.4f (eta=%.4f).",
        old_scale,
        params.spacing_scale,
        params.eta,
    )


def apply_boresight_offset_update(
    generator: "OnlineLearningTrajectoryGenerator",
    new_offset_deg: float,
    *,
    invalidate_from_step: Optional[int] = None,
    system_model=None,
) -> None:
    """Jump the common angle bias. Snapshots use θ−Δ; cached labels stay θ."""
    params = generator.system_model_params
    old_offset = float(getattr(params, "boresight_offset_deg", INITIAL_BORESIGHT_OFFSET_DEG) or 0.0)
    params.boresight_offset_deg = float(new_offset_deg)
    if system_model is not None and system_model.params is not params:
        system_model.params.boresight_offset_deg = float(new_offset_deg)

    _invalidate_step_cache(
        generator,
        invalidate_from_step,
        "boresight_offset_deg",
        old_offset,
        params.boresight_offset_deg,
    )
    logger.info(
        "Boresight offset updated from %.4f deg to %.4f deg.",
        old_offset,
        params.boresight_offset_deg,
    )

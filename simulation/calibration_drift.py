"""Shared calibration-drift helpers for online learning and sample generation."""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from simulation.runners.data import OnlineLearningTrajectoryGenerator

logger = logging.getLogger("SubspaceNet.calibration_drift")

INITIAL_ETA = 0.0
INITIAL_SPACING_SCALE = 1.0
INITIAL_SV_NOISE_VAR = 0.0


def _sync_system_model_geometry(system_model) -> None:
    """Refresh derived geometry fields on a SystemModel after param changes."""
    system_model.eta = system_model._SystemModel__set_eta()
    n = system_model.params.N
    if system_model.params.eta == 0:
        system_model.location_noise = torch.zeros(n)
    else:
        system_model.location_noise = system_model.get_distance_noise(True)


def _sync_samples_model_geometry(samples_model) -> None:
    samples_model.eta = samples_model._SystemModel__set_eta()
    n = samples_model.params.N
    if samples_model.params.eta == 0:
        samples_model.location_noise = torch.zeros(n)
    else:
        samples_model.location_noise = samples_model.get_distance_noise(True)


def reset_calibration(system_model_params, system_model=None) -> None:
    """Reset calibration params to nominal at the start of each OL trajectory."""
    system_model_params.eta = INITIAL_ETA
    system_model_params.spacing_scale = INITIAL_SPACING_SCALE
    system_model_params.sv_noise_var = INITIAL_SV_NOISE_VAR
    if system_model is not None:
        _sync_system_model_geometry(system_model)


def apply_position_eta_update(
    generator: "OnlineLearningTrajectoryGenerator",
    new_eta: float,
    *,
    invalidate_from_step: Optional[int] = None,
    system_model=None,
) -> None:
    """Apply a position-calibration (η) jump; regenerate cached steps from invalidate_from_step."""
    params = generator.system_model_params
    old_eta = params.eta
    params.eta = new_eta
    params.sv_noise_var = new_eta
    _sync_samples_model_geometry(generator.samples_model)
    if system_model is not None:
        _sync_system_model_geometry(system_model)

    if invalidate_from_step is not None:
        keep = max(0, int(invalidate_from_step))
        if keep < len(generator._step_cache):
            generator._step_cache = generator._step_cache[:keep]
        generator.current_step_in_session = len(generator._step_cache)
        logger.info(
            "Truncated step cache to %s steps for eta %.4f -> %.4f",
            keep,
            old_eta,
            new_eta,
        )

    logger.info(
        "Position eta updated from %.4f to %.4f (spacing_scale=%.4f).",
        old_eta,
        params.eta,
        params.spacing_scale,
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

    if invalidate_from_step is not None:
        keep = max(0, int(invalidate_from_step))
        if keep < len(generator._step_cache):
            generator._step_cache = generator._step_cache[:keep]
        generator.current_step_in_session = len(generator._step_cache)
        logger.info(
            "Truncated step cache to %s steps for spacing_scale %.4f -> %.4f",
            keep,
            old_scale,
            params.spacing_scale,
        )

    logger.info(
        "Spacing scale updated from %.4f to %.4f (eta=%.4f).",
        old_scale,
        params.spacing_scale,
        params.eta,
    )

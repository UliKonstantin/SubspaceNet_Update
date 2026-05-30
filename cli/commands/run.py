"""Unified run command for CLI v2."""
import logging
import sys
from typing import List, Optional

import click

from cli.help_text import RUN_HELP, RUN_SHORT_HELP
from cli.options import (
    axis_option,
    config_option,
    goal_option,
    grid_eta_option,
    grid_kf_measurement_noise_option,
    grid_kf_process_noise_option,
    grid_process_noise_option,
    lr_sweep_option,
    model_option,
    output_option,
    override_option,
    retrain_option,
    sweep_option,
    trajectory_option,
    values_option,
)
from cli.resolver import resolve_run_request
from cli.runner import run as run_simulation
from cli.types import RoutingError

logger = logging.getLogger("SubspaceNet.cli")


@click.command("run", help=RUN_HELP, short_help=RUN_SHORT_HELP)
@config_option
@output_option
@override_option
@goal_option
@model_option
@trajectory_option
@sweep_option
@axis_option
@values_option
@lr_sweep_option
@retrain_option
@grid_eta_option
@grid_process_noise_option
@grid_kf_process_noise_option
@grid_kf_measurement_noise_option
def run_command(
    config: str,
    output: Optional[str],
    override: List[str],
    goal: Optional[str],
    model: Optional[str],
    trajectory: bool,
    sweep: Optional[str],
    axis: Optional[str],
    values: List[float],
    lr_sweep: bool,
    retrain_per_sweep: Optional[bool],
    eta_values: List[float],
    process_noise_values: List[float],
    kf_process_noise_values: List[float],
    kf_measurement_noise_values: List[float],
):
    """Run an experiment with explicit goal and sweep routing."""
    try:
        request = resolve_run_request(
            config_path=config,
            output_dir=output,
            overrides=override,
            goal=goal,
            model=model,
            trajectory=trajectory,
            sweep=sweep,
            axis=axis,
            values=list(values) if values else None,
            lr_sweep=lr_sweep,
            retrain_per_sweep=retrain_per_sweep,
            eta_values=list(eta_values) if eta_values else None,
            process_noise_values=list(process_noise_values) if process_noise_values else None,
            kf_process_noise_values=list(kf_process_noise_values) if kf_process_noise_values else None,
            kf_measurement_noise_values=(
                list(kf_measurement_noise_values) if kf_measurement_noise_values else None
            ),
        )
        run_simulation(request)
    except RoutingError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        logger.error("Run failed: %s", exc, exc_info=True)
        sys.exit(1)

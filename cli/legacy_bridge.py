"""Bridge legacy main.py commands to CLI v2 postprocess dispatch."""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

from cli.postprocess import postprocess
from cli.types import Goal, Grid4DParams, RunRequest, SweepAxis, SweepType

logger = logging.getLogger("SubspaceNet.cli")

DEPRECATION_MESSAGE = (
    "main.py is deprecated — use `python3 main_v2.py run` instead. "
    "Run `python3 main_v2.py run --help` for the v2 command surface."
)


def warn_deprecated(command: str) -> None:
    warnings.warn(f"{command}: {DEPRECATION_MESSAGE}", DeprecationWarning, stacklevel=3)
    logger.warning("%s: %s", command, DEPRECATION_MESSAGE)


def sweep_axis_from_scenario(scenario: str) -> Optional[SweepAxis]:
    try:
        return SweepAxis.from_string(scenario.lower())
    except ValueError:
        return None


def legacy_postprocess(
    sim,
    result: dict,
    *,
    config_path: str,
    goal: Goal,
    sweep: SweepType = SweepType.NONE,
    sweep_axis: Optional[SweepAxis] = None,
    lr_sweep: bool = False,
    grid_params: Optional[Grid4DParams] = None,
) -> None:
    """Run unified plot dispatch for a legacy main.py execution path."""
    request = RunRequest(
        goal=goal,
        config_path=Path(config_path),
        output_dir=Path(sim.output_dir),
        sweep=sweep,
        sweep_axis=sweep_axis,
        lr_sweep=lr_sweep,
        grid_params=grid_params,
    )
    postprocess(result, request, Path(sim.output_dir), sim)

"""1D parameter sweep execution."""
import logging
from typing import Dict

from simulation.core import Simulation

from cli.types import Goal, RunRequest

logger = logging.getLogger("SubspaceNet.cli")


def run_one_d_sweep(sim: Simulation, request: RunRequest) -> Dict:
    if not request.sweep_values:
        raise ValueError(
            f"Sweep axis {request.sweep_axis.value} requires values via -v or YAML "
            "(scenario_config.values / evaluation.sweep_values)"
        )

    axis = request.sweep_axis.value
    logger.info("Running 1D sweep on %s with values %s", axis, request.sweep_values)
    full_mode = request.goal == Goal.FULL
    return sim.run_scenario(
        axis,
        list(request.sweep_values),
        full_mode=full_mode,
        goal=request.goal.value,
    )

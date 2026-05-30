"""Single entry point for CLI v2 execution."""
import logging
import sys
from typing import Dict

from config_handler import setup_configuration
from simulation.core import Simulation
from utils.logging_utils import setup_logging_from_config

from cli.postprocess import postprocess
from cli.resolver import build_overrides
from cli.sweeps.grid_4d import run_4d_grid_sweep
from cli.sweeps.kalman_2d import run_kalman_2d_sweep
from cli.sweeps.one_d import run_one_d_sweep
from cli.types import Goal, RunRequest, SweepType
from cli.validation import validate

logger = logging.getLogger("SubspaceNet.cli")


def _run_single(sim: Simulation, request: RunRequest) -> Dict:
    if request.goal == Goal.TRAIN:
        return sim.run_training()
    if request.goal == Goal.EVALUATE:
        return sim.run_evaluation()
    if request.goal == Goal.ONLINE_LEARNING:
        return sim.execute_online_learning()
    return sim.run()


def _dispatch_sweep(sim: Simulation, request: RunRequest) -> Dict:
    if request.sweep == SweepType.ONE_D:
        return run_one_d_sweep(sim, request)
    if request.sweep == SweepType.KALMAN_2D:
        return run_kalman_2d_sweep(sim, request)
    if request.sweep == SweepType.GRID_4D:
        return run_4d_grid_sweep(sim, request)
    raise ValueError(f"Unsupported sweep type: {request.sweep}")


def run(request: RunRequest) -> Dict:
    """Validate, configure, execute, and postprocess a simulation run."""
    validate(request)

    goal_overrides = build_overrides(request)
    all_overrides = goal_overrides + list(request.overrides)

    config_obj, components, output_dir = setup_configuration(
        str(request.config_path),
        str(request.output_dir) if request.output_dir else None,
        all_overrides,
    )
    setup_logging_from_config(config_obj.logging, output_dir)

    sim = Simulation(config_obj, components, output_dir)
    logger.info("Running goal=%s sweep=%s", request.goal.value, request.sweep.value)

    if request.is_sweep:
        result = _dispatch_sweep(sim, request)
    else:
        result = _run_single(sim, request)

    if isinstance(result, dict) and result.get("status") == "error":
        logger.error("Run failed: %s", result.get("message"))
        sys.exit(1)

    postprocess(result, request, output_dir, sim)
    logger.info("Run completed successfully")
    return result

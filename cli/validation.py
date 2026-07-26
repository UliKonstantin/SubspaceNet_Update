"""Validate RunRequest against the CLI decision tree."""
import logging

from cli.types import Goal, RoutingError, RunRequest, SweepType

logger = logging.getLogger("SubspaceNet.cli")


def validate(request: RunRequest) -> None:
    """Raise RoutingError if the request violates decision-tree rules."""
    if request.goal in (Goal.EVALUATE, Goal.ONLINE_LEARNING) and not request.model_path:
        raise RoutingError(
            f"--model (or simulation.model_path in YAML) is required for goal={request.goal.value}"
        )

    if request.goal == Goal.ONLINE_LEARNING and not request.trajectory_enabled:
        raise RoutingError(
            "Online learning requires trajectory data (--trajectory or trajectory.enabled=true in YAML)"
        )

    if request.goal == Goal.FULL and request.is_sweep:
        raise RoutingError("Full pipeline (goal=full) does not support parameter sweeps")

    if request.sweep == SweepType.KALMAN_2D and request.goal != Goal.EVALUATE:
        raise RoutingError("2D kalman_noise sweep is only valid for goal=evaluate")

    if request.sweep == SweepType.GRID_4D and request.goal != Goal.ONLINE_LEARNING:
        raise RoutingError("4D grid sweep is only valid for goal=online_learning")

    if request.lr_sweep:
        if request.goal != Goal.ONLINE_LEARNING:
            raise RoutingError("--lr-sweep is only valid for goal=online_learning")
        if request.sweep != SweepType.ONE_D or request.sweep_axis is None:
            raise RoutingError("--lr-sweep requires --sweep 1d with --axis eta")
        if request.sweep_axis.value != "eta":
            raise RoutingError("--lr-sweep is only valid when --axis eta")

    if request.retrain_per_sweep is False and request.goal == Goal.TRAIN and request.is_sweep:
        logger.warning(
            "retrain_per_sweep=false during training sweep is usually invalid unless comparing init only"
        )

    if request.sweep == SweepType.ONE_D and request.sweep_axis is None:
        raise RoutingError("--sweep 1d requires --axis (snr, m, t, eta, trajectory_length)")

    if request.sweep == SweepType.ONE_D and not request.sweep_values:
        raise RoutingError(
            "--sweep 1d requires values via -v or YAML "
            "(scenario_config.values / evaluation.sweep_values / defaults)"
        )

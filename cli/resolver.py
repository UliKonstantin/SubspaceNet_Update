"""Resolve CLI arguments and YAML config into a RunRequest."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from config.loader import load_config
from config.schema import Config

from cli.constants import (
    DEFAULT_GRID_ETA,
    DEFAULT_GRID_KF_NOISE,
    DEFAULT_GRID_PROCESS_NOISE,
    DEFAULT_SWEEP_VALUES,
    GOAL_OVERRIDES,
)
from cli.types import Goal, Grid4DParams, RoutingError, RunRequest, SweepAxis, SweepType

logger = logging.getLogger("SubspaceNet.cli")


def infer_goal(config: Config) -> Optional[Goal]:
    sim = config.simulation
    ol = config.online_learning
    if ol.enabled and not sim.train_model:
        return Goal.ONLINE_LEARNING
    if sim.evaluate_model and not sim.train_model and not ol.enabled:
        return Goal.EVALUATE
    if sim.train_model:
        return Goal.TRAIN
    if sim.load_model and not sim.train_model:
        return Goal.ONLINE_LEARNING
    return None


def _resolve_goal(cli_goal: Optional[str], config: Config) -> Goal:
    if cli_goal:
        try:
            return Goal(cli_goal.lower())
        except ValueError as exc:
            raise RoutingError(
                f"Unknown --goal {cli_goal!r}. Use train, evaluate, online_learning, or full."
            ) from exc
    inferred = infer_goal(config)
    if inferred is None:
        raise RoutingError("Specify --goal or set simulation.* / online_learning.enabled in YAML")
    logger.info("Inferred goal=%s from YAML", inferred.value)
    return inferred


def _resolve_sweep(cli_sweep: Optional[str], config: Config, goal: Goal) -> SweepType:
    if cli_sweep:
        try:
            return SweepType(cli_sweep.lower())
        except ValueError as exc:
            raise RoutingError(
                f"Unknown --sweep {cli_sweep!r}. Use none, 1d, 2d_kalman, or 4d_grid."
            ) from exc

    if goal == Goal.FULL:
        return SweepType.NONE

    if config.scenario_config and config.scenario_config.type:
        axis = config.scenario_config.type.lower()
        if axis == "4d_grid":
            return SweepType.GRID_4D
        if axis == "kalman_noise":
            return SweepType.KALMAN_2D if goal == Goal.EVALUATE else SweepType.ONE_D
        return SweepType.ONE_D

    if goal == Goal.EVALUATE and config.evaluation.sweep_parameter:
        return SweepType.ONE_D

    return SweepType.NONE


def _resolve_axis(cli_axis: Optional[str], config: Config, sweep: SweepType) -> Optional[SweepAxis]:
    if sweep != SweepType.ONE_D:
        return None

    if cli_axis:
        return SweepAxis.from_string(cli_axis)

    if config.scenario_config and config.scenario_config.type:
        stype = config.scenario_config.type.lower()
        if stype not in ("4d_grid", "kalman_noise"):
            return SweepAxis.from_string(stype)

    if config.evaluation.sweep_parameter:
        return SweepAxis.from_string(config.evaluation.sweep_parameter)

    return None


def _resolve_sweep_values(
    cli_values: List[float],
    config: Config,
    axis: Optional[SweepAxis],
    goal: Goal,
) -> Optional[List[float]]:
    if cli_values:
        return list(cli_values)

    if config.scenario_config and config.scenario_config.values:
        return list(config.scenario_config.values)

    if goal == Goal.EVALUATE and config.evaluation.sweep_values:
        return list(config.evaluation.sweep_values)

    if axis and axis.value in DEFAULT_SWEEP_VALUES:
        return DEFAULT_SWEEP_VALUES[axis.value].copy()

    return None


def build_overrides(request: RunRequest) -> List[str]:
    """Build dot-path overrides from goal and request fields."""
    overrides = list(GOAL_OVERRIDES.get(request.goal, []))

    if request.model_path:
        overrides.append(f"simulation.model_path={request.model_path}")

    if request.trajectory:
        overrides.append("trajectory.enabled=true")

    if request.lr_sweep:
        overrides.append("online_learning.enable_lr_sweep=true")

    if request.goal == Goal.TRAIN and request.is_sweep:
        overrides.append(
            f"scenario_config.retrain_model={'true' if request.retrain_per_sweep else 'false'}"
        )

    return overrides


def resolve_run_request(
    config_path: str,
    output_dir: Optional[str] = None,
    overrides: Optional[List[str]] = None,
    goal: Optional[str] = None,
    model: Optional[str] = None,
    trajectory: bool = False,
    sweep: Optional[str] = None,
    axis: Optional[str] = None,
    values: Optional[List[float]] = None,
    lr_sweep: bool = False,
    retrain_per_sweep: Optional[bool] = None,
    eta_values: Optional[List[float]] = None,
    process_noise_values: Optional[List[float]] = None,
    kf_process_noise_values: Optional[List[float]] = None,
    kf_measurement_noise_values: Optional[List[float]] = None,
) -> RunRequest:
    """Load YAML and merge with CLI into a fully resolved RunRequest (unvalidated)."""
    config = load_config(config_path)
    resolved_goal = _resolve_goal(goal, config)
    resolved_sweep = _resolve_sweep(sweep, config, resolved_goal)
    resolved_axis = _resolve_axis(axis, config, resolved_sweep)

    if not trajectory and config.trajectory.enabled:
        trajectory = True

    model_path = Path(model) if model else None
    if not model_path and config.simulation.model_path:
        model_path = Path(config.simulation.model_path)
    if (
        not model_path
        and config.scenario_config
        and config.scenario_config.model_paths
    ):
        model_path = Path(config.scenario_config.model_paths[0])

    resolved_lr_sweep = lr_sweep or (
        resolved_goal == Goal.ONLINE_LEARNING
        and config.online_learning.enable_lr_sweep
        and resolved_sweep == SweepType.ONE_D
        and resolved_axis == SweepAxis.ETA
    )

    if retrain_per_sweep is not None:
        retrain = retrain_per_sweep
    elif config.scenario_config:
        retrain = config.scenario_config.retrain_model
    else:
        retrain = True

    sweep_values = _resolve_sweep_values(values or [], config, resolved_axis, resolved_goal)

    grid_params = None
    if resolved_sweep == SweepType.GRID_4D:
        grid_params = Grid4DParams(
            eta_values=list(eta_values) if eta_values else DEFAULT_GRID_ETA.copy(),
            process_noise_values=(
                list(process_noise_values) if process_noise_values else DEFAULT_GRID_PROCESS_NOISE.copy()
            ),
            kf_process_noise_values=(
                list(kf_process_noise_values) if kf_process_noise_values else DEFAULT_GRID_KF_NOISE.copy()
            ),
            kf_measurement_noise_values=(
                list(kf_measurement_noise_values)
                if kf_measurement_noise_values
                else DEFAULT_GRID_KF_NOISE.copy()
            ),
        )

    return RunRequest(
        goal=resolved_goal,
        config_path=Path(config_path),
        output_dir=Path(output_dir) if output_dir else None,
        overrides=list(overrides or []),
        model_path=model_path,
        trajectory=trajectory,
        sweep=resolved_sweep,
        sweep_axis=resolved_axis,
        sweep_values=sweep_values,
        lr_sweep=resolved_lr_sweep,
        retrain_per_sweep=retrain,
        grid_params=grid_params,
    )

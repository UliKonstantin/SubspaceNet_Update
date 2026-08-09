"""Unified plot dispatch registry for CLI v2."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from cli.types import Goal, RunRequest, SweepType

logger = logging.getLogger("SubspaceNet.cli")

PlotKey = Tuple[Goal, SweepType]
PlotHandler = Callable[["PlotContext"], None]


@dataclass
class PlotContext:
    result: dict
    request: RunRequest
    output_dir: Path
    sim: object
    goal: Goal
    sweep: SweepType


def should_save_plots(ctx: PlotContext) -> bool:
    return bool(getattr(ctx.sim.config.simulation, "save_plots", False))


def _gated(handler: PlotHandler) -> PlotHandler:
    def wrapped(ctx: PlotContext) -> None:
        if not should_save_plots(ctx):
            logger.debug(
                "Skipping plots (simulation.save_plots=false) for %s/%s",
                ctx.goal.value,
                ctx.sweep.value,
            )
            return
        handler(ctx)

    return wrapped


def _plot_iteration_if_ol(result: dict, output_dir: Path, config) -> None:
    if not result or result.get("status") != "success" or not result.get("averaged_results"):
        return
    from utils.plotting import plot_single_online_learning_run

    plot_single_online_learning_run(result, output_dir, config)


def _lr_sweep_output_subdir(scenario_type: str, scenario_value, lr_key, lr_data: dict) -> str:
    if lr_key == "adaptive":
        return f"{scenario_type}_{scenario_value}/lr_adaptive"
    lr_value = lr_data.get("lr_value", lr_key)
    lr_idx = lr_data.get("lr_row_id", lr_key)
    return f"{scenario_type}_{scenario_value}/lr_run{lr_idx}_{lr_value}"


def dispatch_one_d_sweep_iteration_plots(
    scenario_results: dict,
    output_dir: Path,
    config,
    scenario_type: str,
) -> None:
    """Plot single-run OL diagnostics into each sweep iteration subdir."""
    axis = scenario_type.lower()
    for scenario_value, entry in scenario_results.items():
        if not isinstance(entry, dict):
            continue
        if "lr_sweep_results" in entry:
            for lr_key, lr_data in entry["lr_sweep_results"].items():
                if not isinstance(lr_data, dict):
                    continue
                subdir = _lr_sweep_output_subdir(axis, scenario_value, lr_key, lr_data)
                try:
                    _plot_iteration_if_ol(
                        lr_data.get("result"),
                        output_dir / subdir,
                        config,
                    )
                except Exception as exc:
                    logger.warning("Per-iteration OL plots failed for %s: %s", subdir, exc)
        elif entry.get("averaged_results"):
            subdir = f"{axis}_{scenario_value}"
            try:
                _plot_iteration_if_ol(entry, output_dir / subdir, config)
            except Exception as exc:
                logger.warning("Per-iteration OL plots failed for %s: %s", subdir, exc)


def _handle_train_none(ctx: PlotContext) -> None:
    metrics = ctx.sim.results.get("training_curves")
    if not metrics:
        logger.debug("No training_curves in sim.results; skipping training plots")
        return
    from utils.plotting import plot_training_curves

    plot_training_curves(metrics, ctx.output_dir)


def _handle_evaluate_none(ctx: PlotContext) -> None:
    traj_results = ctx.sim.results.get("dnn_trajectory_results")
    if not traj_results:
        return
    from utils.plotting import plot_eval_dnn_ekf_loss_vs_time

    plot_eval_dnn_ekf_loss_vs_time(traj_results, ctx.output_dir)


def _handle_evaluate_kalman_2d(ctx: PlotContext) -> None:
    from utils.plotting import plot_2d_kalman_noise_sweep

    plot_2d_kalman_noise_sweep(ctx.result, ctx.output_dir)


def _handle_evaluate_one_d(ctx: PlotContext) -> None:
    if not ctx.request.sweep_axis:
        return
    from utils.plotting import plot_loss_vs_scenario

    plot_loss_vs_scenario(ctx.result, ctx.request.sweep_axis.value, ctx.output_dir)


def _handle_online_learning_none(ctx: PlotContext) -> None:
    from utils.plotting import plot_single_online_learning_run

    plot_single_online_learning_run(ctx.result, ctx.output_dir, ctx.sim.config)


def _handle_online_learning_grid_4d(ctx: PlotContext) -> None:
    from utils.plotting import plot_eta_comparison_4d_grid

    plot_eta_comparison_4d_grid(ctx.result, ctx.output_dir)


def _handle_online_learning_one_d_snr(ctx: PlotContext) -> None:
    from utils.plotting import plot_performance_improvement_table, plot_scenario_results

    plot_scenario_results(ctx.result, ctx.output_dir)
    plot_performance_improvement_table(ctx.result, ctx.output_dir)
    dispatch_one_d_sweep_iteration_plots(ctx.result, ctx.output_dir, ctx.sim.config, "snr")


def _handle_online_learning_one_d_eta(ctx: PlotContext) -> None:
    from utils.plotting import (
        plot_eta_scenario_comparison,
        plot_performance_improvement_table_eta,
        plot_lr_sweep_heatmap,
        plot_scenario_results,
    )

    plot_eta_scenario_comparison(ctx.result, ctx.output_dir)
    plot_performance_improvement_table_eta(ctx.result, ctx.output_dir)
    plot_scenario_results(ctx.result, ctx.output_dir, scenario_type="eta")
    if ctx.request.lr_sweep and "lr_sweep_heatmap_data" in ctx.sim.results:
        plot_lr_sweep_heatmap(ctx.sim.results["lr_sweep_heatmap_data"], ctx.output_dir)
        from utils.lr_analysis import postprocess_lr_sweep_analysis

        postprocess_lr_sweep_analysis(ctx.output_dir, ctx.sim.results["lr_sweep_heatmap_data"])

    dispatch_one_d_sweep_iteration_plots(ctx.result, ctx.output_dir, ctx.sim.config, "eta")


def _handle_online_learning_one_d(ctx: PlotContext) -> None:
    axis = ctx.request.sweep_axis.value if ctx.request.sweep_axis else None
    if axis == "snr":
        _handle_online_learning_one_d_snr(ctx)
    elif axis == "eta":
        _handle_online_learning_one_d_eta(ctx)


PLOT_REGISTRY: Dict[PlotKey, PlotHandler] = {
    (Goal.TRAIN, SweepType.NONE): _gated(_handle_train_none),
    (Goal.EVALUATE, SweepType.NONE): _gated(_handle_evaluate_none),
    (Goal.EVALUATE, SweepType.KALMAN_2D): _gated(_handle_evaluate_kalman_2d),
    (Goal.EVALUATE, SweepType.ONE_D): _gated(_handle_evaluate_one_d),
    (Goal.ONLINE_LEARNING, SweepType.NONE): _gated(_handle_online_learning_none),
    (Goal.ONLINE_LEARNING, SweepType.GRID_4D): _gated(_handle_online_learning_grid_4d),
    (Goal.ONLINE_LEARNING, SweepType.ONE_D): _gated(_handle_online_learning_one_d),
}


def resolve_plot_jobs(request: RunRequest, result: dict, sim) -> List[PlotKey]:
    """Return (goal, sweep) pairs to dispatch. FULL runs multiple single-phase jobs."""
    if request.goal != Goal.FULL:
        return [(request.goal, request.sweep)]

    jobs: List[PlotKey] = []
    if result.get("trained_model") or sim.results.get("training_curves") is not None:
        jobs.append((Goal.TRAIN, SweepType.NONE))
    if sim.results.get("dnn_trajectory_results"):
        jobs.append((Goal.EVALUATE, SweepType.NONE))
    if "online_learning_results" in result or "averaged_results" in result:
        jobs.append((Goal.ONLINE_LEARNING, SweepType.NONE))
    return jobs


def dispatch_plots(result: dict, request: RunRequest, output_dir: Path, sim) -> None:
    """Run registered plot handlers for the completed run. Failures are logged, not raised."""
    for goal, sweep in resolve_plot_jobs(request, result, sim):
        handler = PLOT_REGISTRY.get((goal, sweep))
        if handler is None:
            logger.debug("No plot handler registered for %s/%s", goal.value, sweep.value)
            continue
        ctx = PlotContext(
            result=result,
            request=request,
            output_dir=output_dir,
            sim=sim,
            goal=goal,
            sweep=sweep,
        )
        try:
            handler(ctx)
        except Exception as exc:
            logger.warning(
                "Plot dispatch failed for %s/%s: %s",
                goal.value,
                sweep.value,
                exc,
            )

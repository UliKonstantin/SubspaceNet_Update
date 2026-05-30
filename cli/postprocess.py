"""Post-run plot dispatch for CLI v2."""
import logging
from pathlib import Path
from typing import Dict, Optional

from cli.types import Goal, RunRequest, SweepType

logger = logging.getLogger("SubspaceNet.cli")


def postprocess(result: Dict, request: RunRequest, output_dir: Path, sim) -> None:
    """Generate plots based on goal and sweep type. Failures are logged, not raised."""
    try:
        if request.goal == Goal.EVALUATE and request.sweep == SweepType.KALMAN_2D:
            from utils.plotting import plot_2d_kalman_noise_sweep
            plot_2d_kalman_noise_sweep(result, output_dir)
            return

        if request.goal == Goal.EVALUATE and request.sweep == SweepType.ONE_D and request.sweep_axis:
            from utils.plotting import plot_loss_vs_scenario
            plot_loss_vs_scenario(result, request.sweep_axis.value, output_dir)
            return

        if request.goal == Goal.ONLINE_LEARNING and request.sweep == SweepType.GRID_4D:
            from utils.plotting import plot_eta_comparison_4d_grid
            plot_eta_comparison_4d_grid(result, output_dir)
            return

        if request.goal == Goal.ONLINE_LEARNING and request.sweep == SweepType.ONE_D and request.sweep_axis:
            axis = request.sweep_axis.value
            if axis == "snr":
                from utils.plotting import plot_scenario_results, plot_performance_improvement_table
                plot_scenario_results(result, output_dir)
                plot_performance_improvement_table(result, output_dir)
            elif axis == "eta":
                from utils.plotting import (
                    plot_eta_scenario_comparison,
                    plot_performance_improvement_table_eta,
                    plot_scenario_results,
                    plot_lr_sweep_heatmap,
                )
                plot_eta_scenario_comparison(result, output_dir)
                plot_performance_improvement_table_eta(result, output_dir)
                plot_scenario_results(result, output_dir, scenario_type="eta")
                if request.lr_sweep and "lr_sweep_heatmap_data" in sim.results:
                    plot_lr_sweep_heatmap(sim.results["lr_sweep_heatmap_data"], output_dir)
    except Exception as exc:
        logger.warning("Post-processing plots failed: %s", exc)

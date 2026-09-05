"""Post-run plot dispatch for CLI v2."""
from pathlib import Path
from typing import Dict

from cli.types import Goal, RunRequest
from utils.benchmark_export import export_benchmark_metrics
from utils.plot_dispatch import dispatch_plots


def postprocess(result: Dict, request: RunRequest, output_dir: Path, sim) -> None:
    """Generate plots and benchmark JSON based on goal and sweep type."""
    dispatch_plots(result, request, output_dir, sim)
    if request.goal in (Goal.ONLINE_LEARNING, Goal.FULL) and isinstance(result, dict):
        try:
            export_benchmark_metrics(result, sim.config, output_dir)
        except Exception as exc:
            import logging

            logging.getLogger("SubspaceNet.cli").warning(
                "Benchmark metrics export failed: %s", exc
            )

"""Post-run plot dispatch for CLI v2."""
from pathlib import Path
from typing import Dict

from cli.types import RunRequest
from utils.plot_dispatch import dispatch_plots


def postprocess(result: Dict, request: RunRequest, output_dir: Path, sim) -> None:
    """Generate plots based on goal and sweep type. Failures are logged, not raised."""
    dispatch_plots(result, request, output_dir, sim)

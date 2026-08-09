"""Trajectory figure helpers — invoked only via plot dispatch, not from data/runner paths."""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def plot_online_learning_trajectories_from_result(result: dict, output_dir: Path, config) -> List[Path]:
    """
    Plot DOA trajectories for each pretrained trajectory in an OL run result.

    Gated by ``online_learning.plot_trajectory``; called from ``plot_dispatch`` only.
    """
    from utils.plotting.online_learning import plot_online_learning_trajectory

    if not getattr(getattr(config, "online_learning", None), "plot_trajectory", False):
        return []

    ol_results = result.get("online_learning_results", {})
    pretrained_trajectory_results = ol_results.get("pretrained_trajectory_results", [])
    if not pretrained_trajectory_results:
        logger.debug("plot_trajectory enabled but no pretrained_trajectory_results in result")
        return []

    output_dir = Path(output_dir)
    plot_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stride = config.online_learning.stride
    saved: List[Path] = []

    for traj_idx, traj_result in enumerate(pretrained_trajectory_results):
        suffix = f"_traj{traj_idx}" if len(pretrained_trajectory_results) > 1 else ""
        path = plot_online_learning_trajectory(
            traj_result.window_labels,
            output_dir,
            f"{plot_ts}{suffix}",
            window_indices=traj_result.window_indices,
            stride=stride,
        )
        if path is not None:
            saved.append(Path(path))

    if saved:
        logger.info("Trajectory plots saved to %s (%d file(s))", output_dir, len(saved))
    return saved

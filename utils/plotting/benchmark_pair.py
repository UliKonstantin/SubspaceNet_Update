"""Cross-model benchmark plots: SubspaceNet vs DeepCNN from benchmark_pair.json."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.plotting.style import FIG_WIDE, PLOT_COLORS, apply_paper_plot_style, save_figure, style_axes
from utils.plotting.sweeps import SCENARIO_AXIS_LABELS

logger = logging.getLogger("SubspaceNet.plotting.benchmark_pair")

BENCHMARK_ARMS = ("no_adapt", "unsupervised_ours", "supervised_genie")
ARM_LABELS = {
    "no_adapt": "Pretrained (no adapt)",
    "unsupervised_ours": "Online adapted",
    "supervised_genie": "Supervised genie",
}
MODEL_LABELS = {
    "subspacenet": "SubspaceNet",
    "deepcnn": "DeepCNN",
}
PRETRAINED_LABELS = {
    "subspacenet": "SubspaceNet",
    "deepcnn": "DeepCNN",
}
ONLINE_LABELS = {
    "subspacenet": "Online adapted SubspaceNet",
    "deepcnn": "Online adapted DeepCNN",
}
MODEL_MARKERS = {
    "subspacenet": "o",
    "deepcnn": "D",
}


def load_benchmark_pair(path: Path) -> Dict[str, Any]:
    """Load benchmark_pair.json or merge from sibling benchmark_metrics.json files."""
    path = Path(path)
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    pair_path = path / "benchmark_pair.json"
    if pair_path.exists():
        with pair_path.open(encoding="utf-8") as handle:
            return json.load(handle)

    sn_dir = path / "subspacenet"
    cnn_dir = path / "deepcnn"
    if (sn_dir / "benchmark_metrics.json").exists() and (cnn_dir / "benchmark_metrics.json").exists():
        from utils.benchmark_export import merge_benchmark_pair

        merge_benchmark_pair(sn_dir, cnn_dir, path)
        with pair_path.open(encoding="utf-8") as handle:
            return json.load(handle)

    raise FileNotFoundError(f"No benchmark_pair.json under {path}")


def _scenario_keys(model_payload: Dict[str, Any]) -> List[float]:
    scenarios = model_payload.get("scenarios") or {}
    keys = []
    for key in scenarios:
        if scenarios[key] is None:
            continue
        try:
            keys.append(float(key))
        except (TypeError, ValueError):
            continue
    return sorted(keys)


def _get_arm_data(
    model_payload: Dict[str, Any],
    scenario_key: str,
    arm: str,
) -> Dict[str, Any]:
    scenario = (model_payload.get("scenarios") or {}).get(scenario_key)
    if not scenario:
        raise KeyError(f"Scenario {scenario_key!r} missing for {model_payload.get('model_type')}")
    arms = scenario.get("benchmark_arms") or {}
    if arm not in arms:
        raise KeyError(f"Arm {arm!r} missing in scenario {scenario_key}")
    return arms[arm]


def _get_arm_losses_rad(
    model_payload: Dict[str, Any],
    scenario_key: str,
    arm: str,
) -> List[float]:
    data = _get_arm_data(model_payload, scenario_key, arm)
    losses = data.get("reference_metric_losses") or []
    if not losses:
        raise ValueError(f"No reference_metric_losses for {scenario_key}/{arm}")
    return [float(x) for x in losses]


def _aligned_pretrained_vs_online(
    model_payload: Dict[str, Any],
    scenario_key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (window_indices, pretrained_rmspe_deg, online_rmspe_deg) from adaptation start.

    Pretrained and online curves share the same absolute window indices so they are comparable.
    """
    scenario = (model_payload.get("scenarios") or {}).get(scenario_key)
    if not scenario:
        raise KeyError(f"Scenario {scenario_key!r} missing")

    start_window = scenario.get("training_start_window")
    if start_window is None:
        raise ValueError(f"No training_start_window for scenario {scenario_key}")

    pre = _get_arm_data(model_payload, scenario_key, "no_adapt")
    online = _get_arm_data(model_payload, scenario_key, "unsupervised_ours")
    pre_losses = pre.get("reference_metric_losses") or []
    online_losses = online.get("reference_metric_losses") or []
    if not online_losses:
        raise ValueError(f"No online losses for scenario {scenario_key}")

    pre_indices = pre.get("window_indices") or list(range(len(pre_losses)))
    online_indices = online.get("window_indices") or list(
        range(start_window, start_window + len(online_losses))
    )

    pre_map = {
        int(idx): float(loss)
        for idx, loss in zip(pre_indices, pre_losses)
    }
    online_map = {
        int(idx): float(loss)
        for idx, loss in zip(online_indices, online_losses)
    }

    shared_windows = sorted(
        idx for idx in online_map if idx >= int(start_window) and idx in pre_map
    )
    if not shared_windows:
        raise ValueError(f"No overlapping windows after adaptation start for {scenario_key}")

    x = np.asarray(shared_windows, dtype=int)
    pre_deg = np.degrees([pre_map[i] for i in shared_windows])
    online_deg = np.degrees([online_map[i] for i in shared_windows])
    return x, pre_deg, online_deg


def _mean_rmspe_deg(
    losses_rad: List[float],
    *,
    tail_windows: Optional[int] = None,
) -> float:
    arr = np.asarray(losses_rad, dtype=float)
    if tail_windows is not None and tail_windows > 0:
        arr = arr[-tail_windows:]
    return float(np.degrees(np.mean(arr)))


def _format_scenario_key(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value)}.0"
    return str(float(value))


def infer_scenario_axis(scenario_values: List[float]) -> str:
    """Best-effort axis id from sorted scenario values (override via CLI when ambiguous)."""
    if not scenario_values:
        return "snr"
    if all(v in (2.0, 3.0, 4.0, 5.0) for v in scenario_values) and len(scenario_values) <= 4:
        return "m"
    if all(v in (6.0, 9.0, 18.0) for v in scenario_values) and len(scenario_values) <= 3:
        return "n"
    if all(-20 <= v <= 30 for v in scenario_values):
        return "snr"
    return "snr"


def plot_side_by_side_window_rmspe(
    pair: Dict[str, Any],
    output_path: Path,
    *,
    scenario: Optional[float] = None,
    arm: str = "unsupervised_ours",
    also_plot_no_adapt: bool = True,
) -> Path:
    """
    Two-panel supervised RMSPE (degrees) vs window index.

    Each panel compares frozen pretrained vs online-adapted for that architecture,
    aligned on absolute window indices from ``training_start_window`` onward.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    del arm, also_plot_no_adapt

    apply_paper_plot_style()
    sn_payload = pair["subspacenet"]
    scenario_values = _scenario_keys(sn_payload)
    if not scenario_values:
        raise ValueError("No successful scenarios in SubspaceNet benchmark payload")

    if scenario is None:
        scenario = scenario_values[len(scenario_values) // 2]
    scenario_key = _format_scenario_key(scenario)

    fig, axes = plt.subplots(1, 2, figsize=FIG_WIDE, sharey=True)
    ymax = 0.0
    plotted = 0

    for ax, model_key in zip(axes, ("subspacenet", "deepcnn")):
        payload = pair[model_key]
        try:
            x, pre_deg, online_deg = _aligned_pretrained_vs_online(payload, scenario_key)
        except (KeyError, ValueError) as exc:
            logger.warning("%s: %s", model_key, exc)
            ax.text(
                0.5,
                0.5,
                "Online adaptation did not run",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color="#666666",
            )
            style_axes(
                ax,
                xlabel="Window index",
                ylabel="Supervised RMSPE (deg)",
                title=MODEL_LABELS[model_key],
            )
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
            ax.tick_params(axis="y", labelleft=True, labelright=False)
            continue

        ymax = max(ymax, float(np.nanmax(pre_deg)), float(np.nanmax(online_deg)))
        plotted += 1

        ax.plot(
            x,
            pre_deg,
            color=PLOT_COLORS["pretrained"],
            marker=MODEL_MARKERS[model_key],
            markersize=4,
            linewidth=1.4,
            linestyle="--",
            label=PRETRAINED_LABELS[model_key],
        )
        ax.plot(
            x,
            online_deg,
            color=PLOT_COLORS["online"],
            marker=MODEL_MARKERS[model_key],
            markersize=4,
            linewidth=1.4,
            label=ONLINE_LABELS[model_key],
        )

        style_axes(
            ax,
            xlabel="Window index",
            ylabel="Supervised RMSPE (deg)",
            title=MODEL_LABELS[model_key],
        )
        ax.legend(loc="best", fontsize=9)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
        ax.tick_params(axis="y", labelleft=True, labelright=False)

    if plotted:
        y_upper = ymax * 1.08 if ymax > 0 else 1.0
        for ax in axes:
            ax.set_ylim(0.0, y_upper)

    axis_label = infer_scenario_axis(scenario_values)
    axis_name = SCENARIO_AXIS_LABELS.get(axis_label, axis_label)
    fig.suptitle(
        f"Supervised RMSPE after adaptation start — "
        f"{axis_name.split('(')[0].strip()} = {scenario:g}",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    return save_figure(fig, output_path)


def plot_average_rmspe_vs_axis(
    pair: Dict[str, Any],
    output_path: Path,
    *,
    arm: str = "unsupervised_ours",
    axis: Optional[str] = None,
    tail_windows: Optional[int] = None,
    log_y: bool = False,
) -> Path:
    """Single plot: mean supervised RMSPE (deg) vs sweep axis for both models."""
    import matplotlib.pyplot as plt

    apply_paper_plot_style()
    sn_payload = pair["subspacenet"]
    scenario_values = _scenario_keys(sn_payload)
    if not scenario_values:
        raise ValueError("No successful scenarios in benchmark payload")

    axis_id = (axis or infer_scenario_axis(scenario_values)).lower()
    xlabel = SCENARIO_AXIS_LABELS.get(axis_id, axis_id)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for model_key in ("subspacenet", "deepcnn"):
        ys = []
        xs = []
        for val in scenario_values:
            key = _format_scenario_key(val)
            try:
                losses = _get_arm_losses_rad(pair[model_key], key, arm)
            except (KeyError, ValueError):
                logger.warning("Skipping %s scenario %s (missing data)", model_key, key)
                continue
            xs.append(val)
            ys.append(_mean_rmspe_deg(losses, tail_windows=tail_windows))

        if not xs:
            continue

        ax.plot(
            xs,
            ys,
            marker=MODEL_MARKERS[model_key],
            linestyle="-",
            linewidth=1.8,
            markersize=7,
            color=PLOT_COLORS["dnn"] if model_key == "deepcnn" else PLOT_COLORS["esprit"],
            label=MODEL_LABELS[model_key],
        )

    if log_y:
        ax.set_yscale("log")

    style_axes(
        ax,
        xlabel=xlabel,
        ylabel="Mean supervised RMSPE (deg)",
        title=f"Average tracking error vs {xlabel.split('(')[0].strip()}",
    )
    ax.legend(loc="best")
    fig.tight_layout()
    return save_figure(fig, output_path)


def plot_benchmark_pair_comparisons(
    pair_path: Path,
    output_dir: Path,
    *,
    scenario: Optional[float] = None,
    arm: str = "unsupervised_ours",
    axis: Optional[str] = None,
    tail_windows: Optional[int] = None,
) -> Dict[str, Path]:
    """Generate side-by-side window RMSPE and sweep-average RMSPE comparison plots."""
    pair = load_benchmark_pair(pair_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "side_by_side_window_rmspe": plot_side_by_side_window_rmspe(
            pair,
            output_dir / "pair_side_by_side_window_rmspe.png",
            scenario=scenario,
            arm=arm,
        ),
        "average_rmspe_vs_axis": plot_average_rmspe_vs_axis(
            pair,
            output_dir / "pair_average_rmspe_vs_axis.png",
            arm=arm,
            axis=axis,
            tail_windows=tail_windows,
        ),
    }
    return paths

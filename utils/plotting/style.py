"""Shared matplotlib styling for plotting modules."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]

# Figure sizes (inches) — readable at ~7" double-column width
FIG_SINGLE = (7.0, 4.5)
FIG_WIDE = (10.5, 4.2)
FIG_DOUBLE = (10.5, 8.0)
FIG_HEATMAP = (11.0, 7.0)
FIG_TABLE = (8.0, 10.0)

PLOT_COLORS = {
    "pretrained": "#3A76AF",
    "online": "#E07B39",
    "supervised": "#2CA02C",
    "ekf": "#C0392B",
    "dnn": "#3A76AF",
    "esprit": "#27AE60",
    "adaptive": "#E07B39",
    "static": "#5B7DB1",
    "glrt": "#27AE60",
    "event": "#F39C12",
    "changepoint": "#C0392B",
    "gain": "#6A1B9A",
}

ETA_XLABEL = r"Calibration error $\eta$"
SNR_XLABEL = "SNR (dB)"
WINDOW_XLABEL = "Window index"
RMSPE_LABEL = "RMSPE (supervised)"
MSIE_LABEL = "MSIE (unsupervised)"
RMSPE_DB_LABEL = "Mean post-learning RMSPE (dB)"

LABEL_NO_ADAPTATION_SN = "No adaptation SubspaceNet"
LABEL_ADAPTED_SN = "Adapted SubspaceNet"
LABEL_GENIE = "Genie (Supervised)"
LABEL_NO_ADAPTATION_DCNN = "No adaptation DeepCNN"
LABEL_ADAPTED_DCNN = "Adapted DeepCNN"


def _is_deepcnn(model_type: Optional[str]) -> bool:
    if not model_type:
        return False
    normalized = str(model_type).strip().lower().replace("-", "").replace("_", "")
    return "deepcnn" in normalized


def label_no_adaptation(model_type: Optional[str] = None) -> str:
    """Legend label for the frozen pretrained (no OL) arm."""
    return LABEL_NO_ADAPTATION_DCNN if _is_deepcnn(model_type) else LABEL_NO_ADAPTATION_SN


def label_adapted(model_type: Optional[str] = None) -> str:
    """Legend label for the online-adapted model arm."""
    return LABEL_ADAPTED_DCNN if _is_deepcnn(model_type) else LABEL_ADAPTED_SN


def label_genie() -> str:
    return LABEL_GENIE


def benchmark_arm_label(arm: str, model_type: Optional[str] = None) -> str:
    """Map benchmark JSON arm keys to publication plot labels."""
    if arm == "no_adapt":
        return label_no_adaptation(model_type)
    if arm == "unsupervised_ours":
        return label_adapted(model_type)
    if arm == "supervised_genie":
        return label_genie()
    return arm


def model_display_label(model_type: Optional[str]) -> str:
    """Human-readable model name for plot labels."""
    if not model_type:
        return "Model"
    normalized = str(model_type).strip().lower().replace("-", "").replace("_", "")
    if "deepcnn" in normalized:
        return "DeepCNN"
    if "subspacenet" in normalized:
        return "SubspaceNet"
    return str(model_type)


def apply_paper_plot_style() -> None:
    """Default rcParams for publication-quality figures across the plot dispatch pipeline."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.linewidth": 1.0,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.6,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#CCCCCC",
        }
    )


def apply_lr_analysis_plot_style() -> None:
    """Paper-style rcParams for LR optimality plots (slightly larger type)."""
    apply_paper_plot_style()
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "axes.linewidth": 1.1,
        }
    )


def style_axes(ax, *, xlabel: str, ylabel: str, title: str) -> None:
    """Apply consistent axis labels and title."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", pad=8)


def legend_outside(ax, *, ncol: int = 1, fontsize: Optional[float] = None) -> None:
    """Place legend below axes to reduce in-plot overlap."""
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=ncol,
        frameon=True,
        fontsize=fontsize,
    )


def save_figure(
    fig,
    path: PathLike,
    *,
    dpi: int = 300,
    bbox_inches: str = "tight",
    pad_inches: float = 0.1,
) -> Path:
    """Save and close a figure with consistent export settings."""
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    plt.close(fig)
    return out


def save_current_figure(path: PathLike, *, dpi: int = 300, bbox_inches: str = "tight") -> Path:
    """Save and close the current pyplot figure."""
    import matplotlib.pyplot as plt

    fig = plt.gcf()
    return save_figure(fig, path, dpi=dpi, bbox_inches=bbox_inches)


def _apply_lr_analysis_plot_style() -> None:
    """Backward-compatible alias."""
    apply_lr_analysis_plot_style()

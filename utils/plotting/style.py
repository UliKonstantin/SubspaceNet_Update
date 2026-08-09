"""Shared matplotlib styling for plotting modules."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]


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
    """Paper-style matplotlib rcParams for LR optimality plots (slightly larger type)."""
    apply_paper_plot_style()
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
        }
    )


def save_figure(fig, path: PathLike, *, dpi: int = 300, bbox_inches: str = "tight") -> Path:
    """Save and close a figure with consistent export settings."""
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)
    return out


def _apply_lr_analysis_plot_style() -> None:
    """Backward-compatible alias."""
    apply_lr_analysis_plot_style()

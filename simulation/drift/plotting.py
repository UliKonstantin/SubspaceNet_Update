import matplotlib.pyplot as plt
import numpy as np

from utils.plotting.style import (
    PLOT_COLORS,
    WINDOW_XLABEL,
    apply_paper_plot_style,
    style_axes,
)


def plot_results(
    losses,
    changepoint,
    all_log_glr,
    candidate_points,
    window_index_offset=0,
    event_windows=None,
    gate_milestones=None,
):
    """
    Visualize the loss time series and GLRT statistics.

    ``losses`` and ``changepoint`` are in post-warmup index space; ``window_index_offset``
    (typically ``drift_warmup_windows``) shifts the x-axis to absolute window indices.
    """
    apply_paper_plot_style()
    x = np.arange(len(losses)) + window_index_offset
    abs_changepoint = changepoint + window_index_offset
    abs_candidates = [p + window_index_offset for p in candidate_points]

    gate_styles = {
        "scope_a_loss_start": ("gray", "Scope A start"),
        "first_g": ("purple", "First GLRT sample"),
        "first_z": ("green", "Z-threshold armed"),
    }

    def _add_gates(ax):
        if not gate_milestones:
            return
        for key, (color, label) in gate_styles.items():
            if key in gate_milestones:
                ax.axvline(
                    x=gate_milestones[key], color=color, linestyle=":", linewidth=1.5,
                    alpha=0.85, label=f"{label} (w={gate_milestones[key]})",
                )

    fig_loss, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(x, losses, color=PLOT_COLORS["pretrained"], linewidth=1.8, label="Adaptation loss")
    ax1.axvline(
        x=abs_changepoint, color=PLOT_COLORS["changepoint"], linestyle="--", linewidth=2,
        label=f"GLRT changepoint (w={abs_changepoint})",
    )
    if event_windows:
        labeled = False
        for w in event_windows:
            ax1.axvline(
                x=w, color=PLOT_COLORS["event"], linestyle=":", linewidth=1.5, alpha=0.85,
                label=r"Distribution change ($\eta$)" if not labeled else None,
            )
            labeled = True
    _add_gates(ax1)
    style_axes(
        ax1,
        xlabel=WINDOW_XLABEL,
        ylabel="RMSPE loss",
        title="Loss trajectory with GLRT changepoint",
    )
    ax1.legend(loc="best", fontsize=9)
    fig_loss.tight_layout()

    fig_glrt, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(abs_candidates, all_log_glr, color=PLOT_COLORS["glrt"], linewidth=1.8, label="log-GLR")
    ax2.axvline(
        x=abs_changepoint, color=PLOT_COLORS["changepoint"], linestyle="--", linewidth=2,
        label=f"Max log-GLR (w={abs_changepoint})",
    )
    if event_windows:
        labeled = False
        for w in event_windows:
            ax2.axvline(
                x=w, color=PLOT_COLORS["event"], linestyle=":", linewidth=1.5, alpha=0.85,
                label=r"Distribution change ($\eta$)" if not labeled else None,
            )
            labeled = True
    _add_gates(ax2)
    style_axes(
        ax2,
        xlabel="Candidate changepoint (window)",
        ylabel="Log generalized likelihood ratio",
        title="GLRT statistic vs candidate changepoint",
    )
    ax2.legend(loc="best", fontsize=9)
    fig_glrt.tight_layout()

    return fig_loss, fig_glrt

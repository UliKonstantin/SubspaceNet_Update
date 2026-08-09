import matplotlib.pyplot as plt
import numpy as np


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

    ``gate_milestones`` — optional dict from ``drift_gates.drift_detection_milestones``;
    draws vertical lines for Scope A start, first g, and first z (live trigger arm).
    """
    x = np.arange(len(losses)) + window_index_offset
    abs_changepoint = changepoint + window_index_offset
    abs_candidates = [p + window_index_offset for p in candidate_points]

    gate_styles = {
        "scope_a_loss_start": ("gray", "Scope A loss start"),
        "first_g": ("purple", "first g"),
        "first_z": ("green", "first z (live trigger armed)"),
    }

    def _add_gates(ax):
        if not gate_milestones:
            return
        for key, (color, label) in gate_styles.items():
            if key in gate_milestones:
                ax.axvline(
                    x=gate_milestones[key],
                    color=color,
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.85,
                    label=f"{label} (w={gate_milestones[key]})",
                )

    fig_loss = plt.figure(figsize=(14, 5))
    ax1 = fig_loss.add_subplot(111)
    ax1.plot(x, losses, "b-", linewidth=1.5, label="RMSPE Loss")
    ax1.axvline(
        x=abs_changepoint,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Detected Change Point (window={abs_changepoint})",
    )
    if event_windows:
        labeled = False
        for w in event_windows:
            ax1.axvline(
                x=w,
                color="orange",
                linestyle=":",
                linewidth=1.5,
                alpha=0.8,
                label="Distribution Change (η)" if not labeled else None,
            )
            labeled = True
    _add_gates(ax1)
    ax1.set_xlabel("Window Index", fontsize=12)
    ax1.set_ylabel("RMSPE Loss", fontsize=12)
    ax1.set_title("Model Loss Over Time with Detected Change Point", fontsize=14)
    ax1.legend(fontsize=9, loc="best")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.88])

    fig_glrt = plt.figure(figsize=(14, 5))
    ax2 = fig_glrt.add_subplot(111)
    ax2.plot(abs_candidates, all_log_glr, "g-", linewidth=1.5)
    ax2.axvline(
        x=abs_changepoint,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Maximum log-GLR (window={abs_changepoint})",
    )
    if event_windows:
        labeled = False
        for w in event_windows:
            ax2.axvline(
                x=w,
                color="orange",
                linestyle=":",
                linewidth=1.5,
                alpha=0.8,
                label="Distribution Change (η)" if not labeled else None,
            )
            labeled = True
    _add_gates(ax2)
    ax2.set_xlabel("Candidate Change Point (Window Index)", fontsize=12)
    ax2.set_ylabel("Log Generalized Likelihood Ratio", fontsize=12)
    ax2.set_title("GLRT Statistics Across All Candidate Change Points", fontsize=14)
    ax2.legend(fontsize=9, loc="best")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.88])

    return fig_loss, fig_glrt

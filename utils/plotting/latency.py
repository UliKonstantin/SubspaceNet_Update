"""T6 latency histograms: L_det / L_learn (seconds), L_tau (windows)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simulation.latency_metrics import summarize_latency_records

logger = logging.getLogger("SubspaceNet.plotting.latency")


def plot_latency_histograms(records: list[dict], output_dir: Path) -> None:
    if not records:
        logger.info("No latency records; skip T6 histograms")
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_latency_records(records)
    (output_dir / "latency_metrics.json").write_text(json.dumps({"records": records, "summary": summary}, indent=2))

    specs = [
        (
            "L_det_s",
            "Detection delay after the calibration drift",
            "latency_L_det_hist.png",
            "Time from drift until the system identifies it (s)",
            "seconds",
        ),
        (
            "L_learn_s",
            "Time to finish unsupervised adaptation",
            "latency_L_learn_hist.png",
            "Time from drift until unsupervised learning ends (s)",
            "seconds",
        ),
        (
            "L_tau",
            "GLRT changepoint lag vs the true drift",
            "latency_L_tau_hist.png",
            "Lag (windows; negative = early)",
            "windows",
        ),
    ]
    for key, title, fname, xlabel, unit in specs:
        vals = summary.get(key) or []
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        if vals:
            arr = np.asarray(vals, dtype=float)
            if unit == "windows":
                from matplotlib.ticker import MaxNLocator

                arr_i = np.rint(arr).astype(int)
                lo, hi = int(arr_i.min()), int(arr_i.max())
                pad = 4
                edges = np.arange(lo - pad, hi + pad + 2) - 0.5
                ax.hist(arr_i, bins=edges, color="#4c78a8", edgecolor="white")
                ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
                mean = float(np.mean(arr_i))
                mean_label = f"mean = {mean:.0f} windows"
            else:
                ax.hist(arr, bins=min(12, max(4, len(arr))), color="#4c78a8", edgecolor="white")
                mean = float(np.mean(arr))
                mean_label = f"mean = {mean:.3g} s"
            ax.axvline(mean, color="#e45756", linestyle="--", linewidth=1.4, label=mean_label)
            ax.legend(frameon=False)
        else:
            ax.text(0.5, 0.5, "no fired trajs", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(output_dir / fname, dpi=140)
        plt.close(fig)
        logger.info("Wrote %s", output_dir / fname)
    plot_identification_vs_true_drift(records, output_dir)
    plot_detection_confusion_matrix(records, output_dir)
    plot_example_z_score_trajs(records, output_dir)
    plot_example_cusum_trajs(records, output_dir)


def plot_identification_vs_true_drift(records: list[dict], output_dir: Path) -> None:
    """One point per traj: true drift window vs when the system identified it."""
    output_dir = Path(output_dir)
    idx = np.arange(len(records))
    w_true = np.array([r.get("w_true") if r.get("w_true") is not None else np.nan for r in records], dtype=float)
    w_star = np.array([r.get("w_star") if r.get("w_star") is not None else np.nan for r in records], dtype=float)
    tau = np.array([r.get("tau_star") if r.get("tau_star") is not None else np.nan for r in records], dtype=float)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.scatter(idx, w_true, marker="s", s=28, color="#111111", label="True drift", zorder=3)
    ax.scatter(idx, w_star, marker="o", s=28, color="#e45756", label="System identified", zorder=4)
    ax.scatter(idx, tau, marker="x", s=28, color="#54a24b", label="GLRT split", zorder=5)
    ax.set_title("When drift happens vs when the system flags it")
    ax.set_xlabel("Trajectory")
    ax.set_ylabel("Window index")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    dest = output_dir / "latency_identification_vs_true.png"
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    logger.info("Wrote %s", dest)


def plot_detection_confusion_matrix(records: list[dict], output_dir: Path) -> None:
    from simulation.latency_metrics import confusion_from_records

    cm = confusion_from_records(records)
    mat = np.array([[cm["tp"], cm["fn"]], [cm["fp"], cm["tn"]]], dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1], ["Identified", "Not identified"])
    ax.set_yticks([0, 1], ["Drift present", "No drift"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=14, color="#111")
    ax.set_title("Detection confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    dest = Path(output_dir) / "latency_confusion_matrix.png"
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    logger.info("Wrote %s", dest)


def plot_example_z_score_trajs(records: list[dict], output_dir: Path) -> None:
    """A few single-traj z-score traces: early flag, late flag, no-drift."""
    if not any(r.get("z_windows") for r in records):
        return
    picks = []
    early = next(
        (
            r for r in records
            if r.get("z_windows") and r.get("w_true") is not None and r.get("w_star") is not None
            and r["w_star"] < r["w_true"]
        ),
        None,
    )
    late = next(
        (
            r for r in records
            if r.get("z_windows") and r.get("w_true") is not None and r.get("w_star") is not None
            and r["w_star"] > r["w_true"]
        ),
        None,
    )
    nodrift = next(
        (r for r in records if r.get("z_windows") and r.get("w_true") is None),
        None,
    )
    if early:
        picks.append(("Identified before the true drift", early))
    if late:
        picks.append(("Identified after the true drift", late))
    if nodrift:
        title = "No drift" + (" — false alarm" if nodrift.get("w_star") is not None else " — correctly quiet")
        picks.append((title, nodrift))
    if not picks:
        return
    fig, axes = plt.subplots(len(picks), 1, figsize=(7.6, 3.1 * len(picks)), sharex=True)
    if len(picks) == 1:
        axes = [axes]
    for ax, (title, rec) in zip(axes, picks):
        ax.plot(rec["z_windows"], rec["z_scores"], color="#4c78a8", linewidth=1.6)
        ax.axhline(2.5, color="#f58518", linestyle="--", linewidth=1.2, label="z threshold")
        if rec.get("w_true") is not None:
            ax.axvline(rec["w_true"], color="#111111", linestyle=":", linewidth=1.6, label="True drift")
        if rec.get("w_star") is not None:
            ax.axvline(rec["w_star"], color="#e45756", linestyle="-", linewidth=1.4, label="Identified")
        if rec.get("tau_star") is not None:
            ax.axvline(rec["tau_star"], color="#54a24b", linestyle="-.", linewidth=1.4, label="GLRT split")
        ax.set_title(title)
        ax.set_ylabel("z-score")
        ax.legend(frameon=False, fontsize=8, loc="best")
    axes[-1].set_xlabel("Window index")
    fig.tight_layout()
    dest = Path(output_dir) / "latency_example_z_scores.png"
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    logger.info("Wrote %s", dest)


def plot_example_cusum_trajs(records: list[dict], output_dir: Path) -> None:
    """Unsupervised L and CUSUM S for early / late / no-drift examples."""
    if not any(r.get("cusum_windows") for r in records):
        return
    picks = []
    early = next(
        (
            r for r in records
            if r.get("cusum_windows") and r.get("w_true") is not None and r.get("w_star") is not None
            and r["w_star"] < r["w_true"]
        ),
        None,
    )
    late = next(
        (
            r for r in records
            if r.get("cusum_windows") and r.get("w_true") is not None and r.get("w_star") is not None
            and r["w_star"] >= r["w_true"]
        ),
        None,
    )
    nodrift = next(
        (r for r in records if r.get("cusum_windows") and r.get("w_true") is None),
        None,
    )
    if early:
        picks.append(("Identified before the true drift", early))
    if late:
        picks.append(("Identified at/after the true drift", late))
    if nodrift:
        title = "No drift" + (" — false alarm" if nodrift.get("w_star") is not None else " — correctly quiet")
        picks.append((title, nodrift))
    if not picks:
        return
    fig, axes = plt.subplots(len(picks), 1, figsize=(7.6, 3.1 * len(picks)), sharex=True)
    if len(picks) == 1:
        axes = [axes]
    for ax, (title, rec) in zip(axes, picks):
        from utils.plotting.style import to_deg

        ax.plot(rec["cusum_windows"], to_deg(rec["cusum_L"]), color="#4c78a8", linewidth=1.6, label="unsupervised L")
        thresh = rec.get("cusum_loss_threshold")
        if thresh is not None:
            ax.axhline(to_deg(thresh), color="#f58518", linestyle="--", linewidth=1.2, label="abs threshold")
        if rec.get("w_true") is not None:
            ax.axvline(rec["w_true"], color="#111111", linestyle=":", linewidth=1.6, label="True drift")
        if rec.get("w_star") is not None:
            ax.axvline(rec["w_star"], color="#e45756", linestyle="-", linewidth=1.4, label="Identified")
        if rec.get("cusum_settled_window") is not None:
            ax.axvline(rec["cusum_settled_window"], color="#9d755d", linestyle="-.", linewidth=1.1, label="quiet-L settle")
        ax.set_title(title)
        ax.set_ylabel("L [deg]")
        ax.legend(frameon=False, fontsize=8, loc="best")
    axes[-1].set_xlabel("Window index")
    fig.tight_layout()
    dest = Path(output_dir) / "latency_example_cusum.png"
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    logger.info("Wrote %s", dest)

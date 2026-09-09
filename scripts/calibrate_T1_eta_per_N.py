#!/usr/bin/env python3
"""
Per-N η calibration for T1 antenna sweep.

Runs no-adapt OL smoke (ds=3) at each (N, η_jump) pair and measures supervised
RMSPE degradation on the frozen pretrained arm. Goal: find which η produces a
target degradation (default 8°) and document why N=12/18 barely move at η=0.9.

Usage:
  python3 scripts/calibrate_T1_eta_per_N.py
  python3 scripts/calibrate_T1_eta_per_N.py --target-deg 8 --eta-grid 0.5 0.9 1.2 1.5
  python3 scripts/calibrate_T1_eta_per_N.py --n 9 --quick   # single N, coarse grid
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = ROOT / "configs/Used_for_paper/paper_T1_antenna_sweep.yaml"
DEFAULT_OUT = ROOT / "experiments/results/smoke_runs/journal_paper/T1_eta_calibration"

# η jump size at w40 (= max_eta when starting from 0)
DEFAULT_ETA_GRID = [0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 1.8, 2.0]
N_VALUES = [6, 9, 12, 18]

PRE_WINDOWS = (30, 39)       # before η @ w40
POST_EARLY = (45, 55)        # just after η jump
POST_LATE = (80, 120)        # sustained mismatch
POST_PEAK = (40, 195)        # full post-η regime


def _rad_to_deg(x: float) -> float:
    return float(x) * 180.0 / math.pi


def _mean_loss_in_band(
    indices: List[int],
    losses: List[float],
    w_lo: int,
    w_hi: int,
) -> Optional[float]:
    vals = [l for w, l in zip(indices, losses) if w_lo <= w <= w_hi]
    if not vals:
        return None
    return float(np.mean(vals))


def measure_no_adapt_degradation(benchmark_path: Path) -> Dict[str, Any]:
    """Extract supervised RMSPE bands from benchmark_metrics (single-N run)."""
    data = json.loads(benchmark_path.read_text())
    if data.get("format") == "sweep":
        scenarios = data.get("scenarios") or {}
        if len(scenarios) != 1:
            # pick first non-null
            sc = next(v for v in scenarios.values() if v)
        else:
            sc = next(iter(scenarios.values()))
    else:
        sc = data

    na = (sc.get("benchmark_arms") or {}).get("no_adapt") or {}
    losses = na.get("reference_metric_losses") or []
    indices = na.get("window_indices") or list(range(len(losses)))
    if not losses:
        raise ValueError(f"No no_adapt losses in {benchmark_path}")

    pre = _mean_loss_in_band(indices, losses, *PRE_WINDOWS)
    early = _mean_loss_in_band(indices, losses, *POST_EARLY)
    late = _mean_loss_in_band(indices, losses, *POST_LATE)
    peak_band = _mean_loss_in_band(indices, losses, *POST_PEAK)
    peak_val = max(
        (_rad_to_deg(l) for w, l in zip(indices, losses) if w >= 40),
        default=float("nan"),
    )
    pre_deg = _rad_to_deg(pre) if pre is not None else float("nan")

    def delta(post_rad: Optional[float]) -> float:
        if post_rad is None or pre is None:
            return float("nan")
        return _rad_to_deg(post_rad - pre)

    return {
        "pre_deg": pre_deg,
        "post_early_deg": _rad_to_deg(early) if early is not None else float("nan"),
        "post_late_deg": _rad_to_deg(late) if late is not None else float("nan"),
        "post_peak_band_deg": _rad_to_deg(peak_band) if peak_band is not None else float("nan"),
        "peak_deg": peak_val,
        "delta_early_deg": delta(early),
        "delta_late_deg": delta(late),
        "delta_peak_deg": peak_val - pre_deg if not math.isnan(peak_val) and not math.isnan(pre_deg) else float("nan"),
        "training_start_window": sc.get("training_start_window"),
        "eta_jump_window": 40,
    }


def run_case(
    *,
    cfg: Path,
    n: float,
    eta: float,
    out_dir: Path,
    dataset_size: int,
) -> Path:
    """Run one (N, η) smoke; return path to benchmark_metrics.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "main_v2.py"),
        "run",
        "-c",
        str(cfg),
        "--goal",
        "online_learning",
        "--trajectory",
        "--sweep",
        "1d",
        "--axis",
        "n",
        "-v",
        str(n),
        "-o",
        str(out_dir),
        "-O",
        "simulation.save_plots=false",
        "-O",
        "online_learning.use_adaptive_learning_rate=false",
        "-O",
        f"online_learning.dataset_size={dataset_size}",
        "-O",
        f"online_learning.max_eta={eta}",
        "-O",
        f"online_learning.eta_increment={eta}",
    ]
    print(f"  RUN N={n:g} η={eta:g} -> {out_dir.name}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        raise RuntimeError(f"Run failed N={n} η={eta}: {tail}")

    bm = out_dir / "benchmark_metrics.json"
    if not bm.exists():
        raise FileNotFoundError(f"Missing {bm}")
    return bm


def interpolate_eta_for_target(
    rows: List[Dict[str, Any]],
    target_deg: float,
    metric: str = "delta_late_deg",
) -> Optional[float]:
    """Linear interpolate η to hit target degradation (uses late-window Δ)."""
    valid = sorted(
        [(r["eta"], r[metric]) for r in rows if not math.isnan(r.get(metric, float("nan")))],
        key=lambda x: x[0],
    )
    if len(valid) < 2:
        return None
    for i in range(len(valid) - 1):
        e0, d0 = valid[i]
        e1, d1 = valid[i + 1]
        if (d0 - target_deg) * (d1 - target_deg) <= 0 and d1 != d0:
            t = (target_deg - d0) / (d1 - d0)
            return e0 + t * (e1 - e0)
    # extrapolate only slightly above grid max (≤25% beyond last η)
    if valid[-1][1] < target_deg and len(valid) >= 2:
        e0, d0 = valid[-2]
        e1, d1 = valid[-1]
        if d1 != d0:
            t = (target_deg - d0) / (d1 - d0)
            eta_extrap = e0 + t * (e1 - e0)
            if eta_extrap <= e1 * 1.25:
                return eta_extrap
    return None


def summarize_findings(all_results: Dict[str, List[Dict[str, Any]]], target_deg: float) -> str:
    lines = [
        f"# T1 per-N η calibration (target late-window Δ ≈ {target_deg}°)",
        "",
        "| N | η@0.9 Δlate | η for 8° (interp) | notes |",
        "|---|-------------|-------------------|-------|",
    ]
    for n_key in sorted(all_results, key=lambda k: float(k)):
        rows = all_results[n_key]
        row09 = next((r for r in rows if abs(r["eta"] - 0.9) < 1e-6), None)
        eta8 = interpolate_eta_for_target(rows, target_deg, "delta_late_deg")
        d09 = row09["delta_late_deg"] if row09 else float("nan")
        note = ""
        if row09 and row09.get("delta_late_deg", 0) < 1.0:
            note = "EKF self-heal / low η sensitivity"
        elif eta8 and eta8 > 1.5:
            note = "needs severe η for target"
        max_late = max((r["delta_late_deg"] for r in rows), default=float("nan"))
        if max_late < target_deg:
            eta8_str = f"unreachable (max Δlate={max_late:.1f}° @ η≤2)"
        else:
            eta8_str = f"{eta8:.2f}" if eta8 is not None else "n/a"
        lines.append(f"| {n_key} | {d09:+.2f}° | {eta8_str} | {note} |")

    lines.extend(["", "## What drives 8+° degradation?", ""])
    hits = []
    for n_key, rows in all_results.items():
        for r in rows:
            if r.get("delta_late_deg", 0) >= target_deg or r.get("peak_deg", 0) >= target_deg + r.get("pre_deg", 0):
                hits.append(r)
    if hits:
        for h in sorted(hits, key=lambda x: (-x.get("delta_late_deg", 0), float(x["n"]))):
            lines.append(
                f"- N={h['n']:g} η={h['eta']:g}: Δlate={h['delta_late_deg']:+.2f}°, "
                f"peak={h['peak_deg']:.2f}° (pre={h['pre_deg']:.2f}°)"
            )
    else:
        lines.append(f"- No (N,η) in grid reached Δlate ≥ {target_deg}°; try η > 2 or longer post-η window.")

    lines.extend([
        "",
        "## Interpretation",
        "- **Δlate**: sustained supervised error after η jump (w80–120) vs pre-η baseline.",
        "- N=12/18 often flat at η=0.9 because EKF + large array absorbs steering mismatch.",
        "- N=9 sees largest sustained hit at moderate η (model trained at N=9, no geometry escape).",
        "- For matched-comparison T1: use per-N η from `eta_recommendations.json`, fixed LR, disable adaptive.",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate η per N for T1 antenna sweep")
    parser.add_argument("--config", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-deg", type=float, default=8.0)
    parser.add_argument("--dataset-size", type=int, default=3)
    parser.add_argument("--n", type=float, action="append", help="Restrict to specific N (repeatable)")
    parser.add_argument("--eta-grid", type=float, nargs="+", default=None)
    parser.add_argument("--quick", action="store_true", help="Smaller η grid for smoke")
    args = parser.parse_args()

    eta_grid = args.eta_grid or ( [0.5, 0.9, 1.2, 1.5, 2.0] if args.quick else DEFAULT_ETA_GRID)
    n_values = args.n if args.n else N_VALUES

    args.output.mkdir(parents=True, exist_ok=True)
    all_results: Dict[str, List[Dict[str, Any]]] = {}
    flat: List[Dict[str, Any]] = []

    for n in n_values:
        n_rows: List[Dict[str, Any]] = []
        for eta in eta_grid:
            sub = args.output / f"n_{n:g}_eta_{eta:g}"
            try:
                bm_path = run_case(
                    cfg=args.config,
                    n=n,
                    eta=eta,
                    out_dir=sub,
                    dataset_size=args.dataset_size,
                )
                metrics = measure_no_adapt_degradation(bm_path)
            except (RuntimeError, FileNotFoundError, ValueError) as exc:
                print(f"  SKIP N={n} η={eta}: {exc}", flush=True)
                continue
            try:
                out_rel = str(sub.resolve().relative_to(ROOT.resolve()))
            except ValueError:
                out_rel = str(sub)
            row = {"n": n, "eta": eta, "output_dir": out_rel, **metrics}
            n_rows.append(row)
            flat.append(row)
            print(
                f"    pre={metrics['pre_deg']:.2f}° "
                f"Δearly={metrics['delta_early_deg']:+.2f}° "
                f"Δlate={metrics['delta_late_deg']:+.2f}° "
                f"peak={metrics['peak_deg']:.2f}°",
                flush=True,
            )
        all_results[str(n)] = n_rows

    # Recommend η per N for target
    recommendations: Dict[str, Any] = {"target_delta_late_deg": args.target_deg, "per_n": {}}
    for n_key, rows in all_results.items():
        eta_star = interpolate_eta_for_target(rows, args.target_deg, "delta_late_deg")
        row09 = next((r for r in rows if abs(r["eta"] - 0.9) < 1e-6), None)
        max_late = max((r["delta_late_deg"] for r in rows if not math.isnan(r.get("delta_late_deg", float("nan")))), default=float("nan"))
        reachable = eta_star is not None and (max_late >= args.target_deg * 0.5)
        recommendations["per_n"][n_key] = {
            "eta_at_0.9_delta_late_deg": row09["delta_late_deg"] if row09 else None,
            "max_delta_late_in_grid_deg": max_late,
            "eta_for_target_late_deg": eta_star if reachable else None,
            "target_reachable_with_eta_le_2": bool(max_late >= args.target_deg),
            "eta_increment": eta_star if reachable else None,
            "max_eta": eta_star if reachable else None,
        }

    summary_path = args.output / "calibration_summary.json"
    summary_path.write_text(json.dumps({"results": flat, "recommendations": recommendations}, indent=2))

    md = summarize_findings(all_results, args.target_deg)
    (args.output / "CALIBRATION_REPORT.md").write_text(md)

    # YAML overrides snippet for author
    snippet_lines = [
        "# Paste under scenario_config or use -O overrides per N run",
        "# Matched late-window supervised degradation target: {:.1f}°".format(args.target_deg),
        "scenario_config:",
        "  type: n",
        "  values: [{}]".format(", ".join(str(int(float(k))) if float(k).is_integer() else k for k in sorted(all_results, key=float))),
        "  eta_by_n:  # use with calibrate runner or manual -O",
    ]
    for n_key in sorted(all_results, key=lambda k: float(k)):
        rec = recommendations["per_n"].get(n_key, {})
        eta_star = rec.get("eta_for_target_late_deg")
        if eta_star is not None:
            snippet_lines.append(f"    {n_key}: {{eta_increment: {eta_star:.3f}, max_eta: {eta_star:.3f}}}")
    (args.output / "eta_recommendations.yaml").write_text("\n".join(snippet_lines) + "\n")

    print(f"\nWrote {summary_path}", flush=True)
    print(f"Report: {args.output / 'CALIBRATION_REPORT.md'}", flush=True)
    print(md, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Report pre-EKF vs EKF stress from benchmark_metrics (T1 calibration helper)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PRE = (30, 39)
EARLY = (45, 55)
LATE = (80, 120)


def _band(losses, indices, w_lo, w_hi):
    vals = [l for w, l in zip(indices, losses) if w_lo <= w <= w_hi]
    return float(np.mean(vals)) if vals else float("nan")


def _deg(x: float) -> float:
    return x * 180.0 / math.pi


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "benchmark",
        type=Path,
        nargs="?",
        default=Path("experiments/results/smoke_runs/journal_paper/T1_final_recipe/benchmark_metrics.json"),
    )
    args = p.parse_args()
    data = json.loads(args.benchmark.read_text())
    scenarios = data.get("scenarios") or {None: data}

    print(f"{'N':>4}  {'pre_e':>8}  {'pre_peak':>8}  {'ekf_e':>8}  {'ekf_l':>8}  {'Δpre':>8}  {'Δekf_l':>8}")
    for key in sorted((k for k in scenarios if k is not None), key=lambda k: float(k)):
        na = scenarios[key]["benchmark_arms"]["no_adapt"]
        pre = na.get("pre_ekf_losses") or []
        ref = na.get("reference_metric_losses") or []
        idx = na.get("window_indices") or list(range(len(ref)))
        pre_b, pre_e = _band(pre, idx, *PRE), _band(pre, idx, *EARLY)
        ref_b, ref_l = _band(ref, idx, *PRE), _band(ref, idx, *LATE)
        peak = max((pre[i] for i, w in enumerate(idx) if w >= 40 and i < len(pre)), default=float("nan"))
        print(
            f"{float(key):4.0f}  {pre_e:8.3f}  {peak:8.3f}  {_deg(_band(ref, idx, *EARLY)):8.1f}  "
            f"{_deg(ref_l):8.1f}  {pre_e - pre_b:+8.3f}  {_deg(ref_l - ref_b):+8.1f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

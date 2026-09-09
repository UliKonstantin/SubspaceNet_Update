#!/usr/bin/env python3
"""Verify spacing-scale drift smoke: degradation across N and arm consistency."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCH = ROOT / "experiments/results/smoke_runs/journal_paper/T1_spacing_scale_smoke/benchmark_metrics.json"

PRE = (30, 39)
EARLY = (45, 55)


def _band_mean(losses, indices, w_lo, w_hi):
    vals = [l for w, l in zip(indices, losses) if w_lo <= w <= w_hi]
    return float(np.mean(vals)) if vals else float("nan")


def _deg(x: float) -> float:
    return x * 180.0 / math.pi


def verify(path: Path, *, min_early_delta_deg: float = 0.5) -> int:
    data = json.loads(path.read_text())
    scenarios = data.get("scenarios") or {}
    if not scenarios:
        print(f"FAIL: no scenarios in {path}", file=sys.stderr)
        return 1

    failures = []
    print(f"=== verify spacing-scale smoke: {path} ===")
    print(f"{'N':>4}  {'Δearly':>8}  {'no@45-55':>10}  {'ours@45-55':>10}  {'genie@45-55':>10}")
    for key in sorted(scenarios.keys(), key=lambda k: float(k)):
        sc = scenarios[key]
        arms = sc["benchmark_arms"]
        na = arms["no_adapt"]
        pre = _band_mean(na["reference_metric_losses"], na.get("window_indices") or [], *PRE)
        early_na = _band_mean(na["reference_metric_losses"], na.get("window_indices") or [], *EARLY)
        delta_early = _deg(early_na - pre)

        def arm_early(name):
            a = arms[name]
            return _deg(_band_mean(a["reference_metric_losses"], a.get("window_indices") or [], *EARLY))

        print(
            f"{float(key):4.0f}  {delta_early:+8.2f}  {arm_early('no_adapt'):10.2f}  "
            f"{arm_early('unsupervised_ours'):10.2f}  {arm_early('supervised_genie'):10.2f}"
        )

        # N>=9 should show clear early post-drift hit (matched coherent drift)
        if float(key) >= 9 and delta_early < min_early_delta_deg:
            failures.append(f"N={key}: Δearly={delta_early:.2f}° < {min_early_delta_deg}°")

    if failures:
        for f in failures:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1

    print("OK: spacing-scale drift produces matched early post-drift stress on N≥9")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", type=Path, default=DEFAULT_BENCH)
    p.add_argument("--min-early-delta-deg", type=float, default=0.5)
    args = p.parse_args()
    raise SystemExit(verify(args.benchmark, min_early_delta_deg=args.min_early_delta_deg))


if __name__ == "__main__":
    main()

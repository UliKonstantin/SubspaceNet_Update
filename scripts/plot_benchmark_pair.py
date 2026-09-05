#!/usr/bin/env python3
"""Plot SubspaceNet vs DeepCNN comparisons from benchmark_pair.json."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "DCD_MUSIC"))

from utils.plotting.benchmark_pair import (  # noqa: E402
    BENCHMARK_ARMS,
    plot_benchmark_pair_comparisons,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SN/CNN benchmark comparison plots from benchmark_pair.json",
    )
    parser.add_argument(
        "pair_path",
        type=Path,
        help="benchmark_pair.json or output dir containing subspacenet/ + deepcnn/",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write PNGs (default: same dir as pair file)",
    )
    parser.add_argument(
        "--scenario",
        type=float,
        default=None,
        help="Sweep point for side-by-side window plot (default: middle scenario)",
    )
    parser.add_argument(
        "--arm",
        choices=BENCHMARK_ARMS,
        default="unsupervised_ours",
        help="Benchmark arm to plot (default: online adapted)",
    )
    parser.add_argument(
        "--axis",
        choices=("snr", "n", "m", "t", "eta"),
        default=None,
        help="Sweep axis for average-RMSPE plot (default: infer from scenario values)",
    )
    parser.add_argument(
        "--tail-windows",
        type=int,
        default=None,
        help="Average RMSPE over last K windows only (default: all windows)",
    )
    args = parser.parse_args()

    pair_path = args.pair_path.resolve()
    output_dir = args.output_dir or (
        pair_path.parent if pair_path.is_file() else pair_path
    )

    paths = plot_benchmark_pair_comparisons(
        pair_path,
        output_dir,
        scenario=args.scenario,
        arm=args.arm,
        axis=args.axis,
        tail_windows=args.tail_windows,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

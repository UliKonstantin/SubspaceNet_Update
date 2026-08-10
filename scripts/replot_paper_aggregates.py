#!/usr/bin/env python3
"""Regenerate top-level aggregate plots from saved JSON in an OL sweep output dir."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "DCD_MUSIC"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Replot aggregate figures from sweep JSON artifacts")
    parser.add_argument("output_dir", type=Path, help="Sweep output directory (e.g. paper_eta_sweep_v3)")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        raise SystemExit(f"Not a directory: {output_dir}")

    from simulation.drift.drift_metrics import plot_drift_detection_metrics_in_output_dir
    from utils.lr_analysis import load_heatmap_data, postprocess_lr_sweep_analysis

    heatmap_path = output_dir / "lr_sweep_heatmap_data.json"
    if heatmap_path.exists():
        heatmap_data = load_heatmap_data(heatmap_path)
        created = postprocess_lr_sweep_analysis(output_dir, heatmap_data)
        print("LR analysis plots:", {k: str(v) for k, v in created.items()})
    else:
        print("Skip LR analysis: no lr_sweep_heatmap_data.json")

    drift_path = plot_drift_detection_metrics_in_output_dir(output_dir)
    if drift_path:
        print(f"Drift metrics plot: {drift_path}")
    else:
        print("Skip drift metrics: no drift_detection_dicts.json")

    scenario_stub = output_dir / "scenario_results_stub.json"
    if scenario_stub.exists():
        from utils.plotting.sweeps import (
            plot_eta_scenario_comparison,
            plot_performance_improvement_table_eta,
            plot_scenario_results,
        )

        with open(scenario_stub, "r", encoding="utf-8") as handle:
            scenario_results = json.load(handle)
        plot_scenario_results(scenario_results, output_dir, scenario_type="eta")
        plot_eta_scenario_comparison(scenario_results, output_dir)
        plot_performance_improvement_table_eta(scenario_results, output_dir)
        print("Scenario aggregate plots regenerated from scenario_results_stub.json")
    else:
        print(
            "Skip scenario aggregates: no scenario_results_stub.json "
            "(save during sweep postprocess to enable offline replot)"
        )


if __name__ == "__main__":
    main()

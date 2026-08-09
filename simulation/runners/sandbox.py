"""Backward-compat re-exports for GLRT drift helpers (prefer ``simulation.drift``)."""

from simulation.drift import (
    glrt_changepoint_detection,
    plot_drift_detection_metrics,
    plot_results,
)

__all__ = [
    "glrt_changepoint_detection",
    "plot_drift_detection_metrics",
    "plot_results",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot drift detection metrics from JSON file")
    parser.add_argument(
        "json_path",
        type=str,
        help="Path to the JSON file containing drift detection dicts",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for the plot (optional, displays if not provided)",
    )
    args = parser.parse_args()
    plot_drift_detection_metrics(args.json_path, args.output)

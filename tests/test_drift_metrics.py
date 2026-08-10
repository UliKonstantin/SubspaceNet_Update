"""Tests for drift metrics plotting."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "DCD_MUSIC"))

from simulation.drift.drift_metrics import (
    DRIFT_METRICS_PLOT_FILENAME,
    extract_adaptive_dg_rows_from_drift_dicts,
    extract_adaptive_glrt_metrics_from_scenario_results,
    plot_drift_detection_metrics,
    plot_drift_detection_metrics_from_dicts,
    plot_drift_detection_metrics_from_scenario_results,
    plot_drift_detection_metrics_in_output_dir,
)
from simulation.drift.glrt import glrt_changepoint_detection


def _sample_drift_dicts():
    return [
        {
            "eta": 0.0,
            "scenario_eta": 0.4,
            "window_idx": 10,
            "baseline_mean": 1.0,
            "main_log_glr": 2.0,
            "baseline_std": 0.1,
            "current_glrt_z_score": 3.0,
            "learning_rate_at_detection": 0.001,
        },
        {
            "eta": 0.5,
            "scenario_eta": 0.4,
            "window_idx": 12,
            "baseline_mean": 1.2,
            "main_log_glr": 2.5,
            "baseline_std": 0.15,
            "current_glrt_z_score": 3.5,
            "learning_rate_at_detection": 0.002,
        },
        {
            "eta": 0.0,
            "scenario_eta": 0.9,
            "window_idx": 15,
            "baseline_mean": 2.0,
            "main_log_glr": 4.0,
            "baseline_std": 0.2,
            "current_glrt_z_score": 4.0,
            "learning_rate_at_detection": 0.003,
        },
    ]


class TestDriftMetrics:
    def test_groups_by_scenario_eta(self, tmp_path):
        out = tmp_path / DRIFT_METRICS_PLOT_FILENAME
        fig = plot_drift_detection_metrics_from_dicts(_sample_drift_dicts(), out)
        assert fig is not None
        assert out.exists()

    def test_pipeline_helper_reads_json(self, tmp_path):
        json_path = tmp_path / "drift_detection_dicts.json"
        json_path.write_text(json.dumps(_sample_drift_dicts()), encoding="utf-8")
        created = plot_drift_detection_metrics_in_output_dir(tmp_path)
        assert created == tmp_path / DRIFT_METRICS_PLOT_FILENAME
        assert created.exists()

    def test_cli_loader(self, tmp_path):
        json_path = tmp_path / "drift_detection_dicts.json"
        json_path.write_text(json.dumps(_sample_drift_dicts()), encoding="utf-8")
        out = tmp_path / "manual.png"
        plot_drift_detection_metrics(json_path, out)
        assert out.exists()

    def test_sandbox_reexports(self):
        from simulation.runners.sandbox import glrt_changepoint_detection as sandbox_glrt

        assert sandbox_glrt is glrt_changepoint_detection

    def test_scenario_stub_plot(self, tmp_path):
        scenario = {
            "0.4": {
                "lr_sweep_results": {
                    "adaptive": {
                        "result": {
                            "glrt_results": {
                                "adaptation_loss": {
                                    "avg_changepoint_window": 33.0,
                                    "std_changepoint_window": 0.0,
                                    "avg_likelihood": 57.8,
                                    "std_likelihood": 0.0,
                                    "avg_z_score": 2.7,
                                    "std_z_score": 0.0,
                                    "avg_learning_rate": 0.0005,
                                    "std_learning_rate": 0.0,
                                }
                            }
                        }
                    }
                }
            }
        }
        drift_dicts = [
            {
                "eta": 0.4,
                "scenario_eta": 0.4,
                "window_idx": 33,
                "baseline_mean": 15.0,
                "main_log_glr": 23.0,
                "learning_rate_at_detection": 0.0005,
                "dG_at_detection": 8.0,
                "use_adaptive_learning_rate": True,
                "adaptive_lr_dG0": 69.26,
            }
        ]
        rows = extract_adaptive_glrt_metrics_from_scenario_results(scenario)
        assert len(rows) == 1
        assert rows[0]["eta"] == 0.4
        dg_rows = extract_adaptive_dg_rows_from_drift_dicts(drift_dicts, scenario)
        assert len(dg_rows) == 1
        assert dg_rows[0]["dG"] == 8.0
        out = tmp_path / DRIFT_METRICS_PLOT_FILENAME
        fig = plot_drift_detection_metrics_from_scenario_results(
            scenario, out, drift_dicts=drift_dicts
        )
        assert fig is not None
        assert out.exists()

    def test_glrt_detects_obvious_jump(self):
        losses = [0.01] * 10 + [0.5] * 10
        cp, _, _, _ = glrt_changepoint_detection(losses, min_segment_size=3)
        assert cp == 10

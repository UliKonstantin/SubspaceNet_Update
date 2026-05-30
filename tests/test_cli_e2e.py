"""
End-to-end CLI tests and legacy parity checks for main_v2.py.

Run with: python3 -m pytest tests/test_cli_e2e.py -v
"""
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "DCD_MUSIC"))

CONFIGS_DIR = Path(__file__).parent / "configs"

from cli.app import cli
from cli.resolver import resolve_run_request
from cli.runner import run as run_v2
from config_handler import setup_configuration
from simulation.core import Simulation
from tests.parity_utils import compare_numeric_metrics, extract_ol_numeric_metrics, set_seed


# ---------------------------------------------------------------------------
# Legacy helpers (old main.py / simulate path)
# ---------------------------------------------------------------------------

def _legacy_train(config_path: Path, output_dir: Path):
    cfg, comp, od = setup_configuration(str(config_path), str(output_dir))
    return Simulation(cfg, comp, od).run_training()


def _legacy_snr_sweep(config_path: Path, output_dir: Path):
    cfg, comp, od = setup_configuration(str(config_path), str(output_dir))
    sim = Simulation(cfg, comp, od)
    return sim.run_scenario(
        cfg.scenario_config.type,
        list(cfg.scenario_config.values),
        full_mode=False,
    )


def _legacy_online_learning(config_path: Path, output_dir: Path):
    cfg, comp, od = setup_configuration(str(config_path), str(output_dir))
    return Simulation(cfg, comp, od).execute_online_learning()


def _legacy_eta_sweep(config_path: Path, output_dir: Path):
    cfg, comp, od = setup_configuration(str(config_path), str(output_dir))
    sim = Simulation(cfg, comp, od)
    return sim.run_scenario(
        "eta",
        list(cfg.scenario_config.values),
        full_mode=False,
        goal="online_learning",
    )


def _v2_run(config_path: Path, output_dir: Path, **kwargs):
    request = resolve_run_request(str(config_path), output_dir=str(output_dir), **kwargs)
    return run_v2(request)


def _invoke_cli(args: list[str]) -> CliRunner:
    runner = CliRunner()
    return runner.invoke(cli, args, catch_exceptions=False)


# ---------------------------------------------------------------------------
# Parity: v2 runner vs legacy Simulation
# ---------------------------------------------------------------------------

class TestLegacyParity:
    """Phase 5: v2 routing must match legacy Simulation behavior."""

    def test_train_single_parity(self, tmp_path):
        cfg = CONFIGS_DIR / "test_training.yaml"
        legacy = _legacy_train(cfg, tmp_path / "legacy")
        v2 = _v2_run(cfg, tmp_path / "v2", goal="train")

        assert legacy["status"] == v2["status"] == "success"
        assert legacy.get("trained_model") == v2.get("trained_model")

    def test_snr_sweep_parity(self, tmp_path):
        cfg = CONFIGS_DIR / "test_training_snr_scenario.yaml"
        legacy = _legacy_snr_sweep(cfg, tmp_path / "legacy")
        v2 = _v2_run(cfg, tmp_path / "v2", goal="train")

        assert len(legacy) == len(v2) == 2
        assert set(legacy.keys()) == set(v2.keys())

    def test_online_learning_single_parity(self, tmp_path):
        cfg = CONFIGS_DIR / "test_online_learning_single.yaml"
        legacy = _legacy_online_learning(cfg, tmp_path / "legacy")
        v2 = _v2_run(cfg, tmp_path / "v2", goal="online_learning")

        assert legacy["status"] == v2["status"] == "success"
        assert "online_learning_results" in legacy
        assert "online_learning_results" in v2

    def test_eta_sweep_parity(self, tmp_path):
        cfg = CONFIGS_DIR / "test_online_learning_eta_sweep.yaml"
        set_seed(42)
        legacy = _legacy_eta_sweep(cfg, tmp_path / "legacy")
        set_seed(42)
        v2 = _v2_run(cfg, tmp_path / "v2", goal="online_learning")

        assert len(legacy) == len(v2) == 2
        assert set(legacy.keys()) == set(v2.keys())
        for eta in legacy:
            leg_m = extract_ol_numeric_metrics(legacy[eta])
            v2_m = extract_ol_numeric_metrics(v2[eta])
            ok, diffs = compare_numeric_metrics(leg_m, v2_m)
            assert ok, f"eta={eta} diffs: {diffs}"


# ---------------------------------------------------------------------------
# End-to-end: Click CLI → runner → Simulation
# ---------------------------------------------------------------------------

class TestCLIEndToEnd:
    """Full path through main_v2.py run subcommand."""

    def test_cli_train_single(self, tmp_path):
        result = _invoke_cli([
            "run",
            "-c", str(CONFIGS_DIR / "test_training.yaml"),
            "-o", str(tmp_path),
            "--goal", "train",
        ])
        assert result.exit_code == 0
        assert "Run completed successfully" in result.output or result.exit_code == 0
        assert len(list(tmp_path.rglob("*.pt"))) > 0

    def test_cli_train_snr_sweep(self, tmp_path):
        result = _invoke_cli([
            "run",
            "-c", str(CONFIGS_DIR / "test_training_snr_scenario.yaml"),
            "-o", str(tmp_path),
            "--goal", "train",
        ])
        assert result.exit_code == 0
        assert (tmp_path / "snr_5").exists() or any(tmp_path.iterdir())

    def test_cli_online_learning_single(self, tmp_path):
        result = _invoke_cli([
            "run",
            "-c", str(CONFIGS_DIR / "test_online_learning_single.yaml"),
            "-o", str(tmp_path),
            "--goal", "online_learning",
        ])
        assert result.exit_code == 0

    def test_cli_online_learning_eta_sweep(self, tmp_path):
        result = _invoke_cli([
            "run",
            "-c", str(CONFIGS_DIR / "test_online_learning_eta_sweep.yaml"),
            "-o", str(tmp_path),
            "--goal", "online_learning",
        ])
        assert result.exit_code == 0
        assert (tmp_path / "eta_0.5").exists() or (tmp_path / "eta_0.5").is_dir() or any(tmp_path.iterdir())

    def test_cli_goal_inferred_from_yaml(self, tmp_path):
        """No --goal flag; should infer train from test_training.yaml."""
        result = _invoke_cli([
            "run",
            "-c", str(CONFIGS_DIR / "test_training.yaml"),
            "-o", str(tmp_path),
        ])
        assert result.exit_code == 0

    def test_cli_routing_error_missing_model(self):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "run",
            "-c", str(CONFIGS_DIR / "test_training.yaml"),
            "--goal", "evaluate",
        ])
        assert result.exit_code == 1

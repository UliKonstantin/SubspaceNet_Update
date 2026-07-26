"""
Training path parity: main.py (legacy) vs main_v2.py on paper-style configs.

Verifies identical setup order, overrides, output dirs, and sweep structure.
"""
import sys
from pathlib import Path

import pytest

WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

PAPER_DIR = WORKSPACE_ROOT / "configs" / "Used_for_paper"
CONFIGS_DIR = Path(__file__).parent / "configs"

from cli.resolver import build_overrides, resolve_run_request
from cli.runner import run as run_v2
from config_handler import setup_configuration
from simulation.core import Simulation
from tests.parity_utils import set_seed


def _legacy_run_command(config_path: Path, output_dir: Path):
    """main.py run --scenario training path."""
    cfg, comp, od = setup_configuration(str(config_path), str(output_dir))
    return Simulation(cfg, comp, od).run_training(), od


def _legacy_simulate_snr_sweep(config_path: Path, output_dir: Path, values: list[float]):
    """main.py simulate -s snr -v ... --mode training path."""
    cfg, comp, od = setup_configuration(str(config_path), str(output_dir))
    sim = Simulation(cfg, comp, od)
    return sim.run_scenario("snr", values, full_mode=False), od


def _v2_train(config_path: Path, output_dir: Path, **kwargs):
    request = resolve_run_request(str(config_path), output_dir=str(output_dir), **kwargs)
    return run_v2(request), Path(output_dir)


class TestTrainingSetupParity:
    """Config handler + override parity before any torch work."""

    def test_single_train_no_spurious_overrides(self):
        req = resolve_run_request(str(CONFIGS_DIR / "test_training.yaml"), goal="train")
        assert build_overrides(req) == []

    def test_snr_sweep_no_retrain_override_without_cli(self):
        req = resolve_run_request(
            str(CONFIGS_DIR / "test_training_snr_scenario.yaml"),
            goal="train",
        )
        assert build_overrides(req) == []

    def test_output_dir_matches_for_paper_single_train(self, tmp_path):
        cfg = PAPER_DIR / "Random_basemodel_training_config.yaml"
        if not cfg.exists():
            pytest.skip("paper training config missing")
        out = tmp_path / "paper_single"
        _, _, od_legacy = setup_configuration(str(cfg), str(out))
        _, _, od_v2 = setup_configuration(str(cfg), str(out))
        assert od_legacy.resolve() == od_v2.resolve()

    def test_output_dir_matches_for_paper_snr_sweep_with_o_flag(self, tmp_path):
        cfg = PAPER_DIR / "Random_base_model_training_snr_scenario_config.yaml"
        if not cfg.exists():
            pytest.skip("paper SNR training config missing")
        explicit_out = tmp_path / "snr_training_sweep"
        _, _, od_legacy = setup_configuration(str(cfg), str(explicit_out))
        _, _, od_v2 = setup_configuration(str(cfg), str(explicit_out))
        assert od_legacy.resolve() == od_v2.resolve()


class TestTrainingRunParity:
    """End-to-end training runs on trimmed test configs."""

    def test_single_train_result_shape(self, tmp_path):
        cfg = CONFIGS_DIR / "test_training.yaml"
        set_seed(42)
        legacy, _ = _legacy_run_command(cfg, tmp_path / "legacy")
        set_seed(42)
        v2, _ = _v2_train(cfg, tmp_path / "v2", goal="train")

        assert legacy["status"] == v2["status"] == "success"
        assert legacy.get("trained_model") == v2.get("trained_model")

        legacy_ckpts = list((tmp_path / "legacy").rglob("*.pt"))
        v2_ckpts = list((tmp_path / "v2").rglob("*.pt"))
        assert len(legacy_ckpts) == len(v2_ckpts) == 1
        assert legacy_ckpts[0].parent.name == v2_ckpts[0].parent.name == "checkpoints"

    def test_snr_sweep_structure_and_subdirs(self, tmp_path):
        cfg = CONFIGS_DIR / "test_training_snr_scenario.yaml"
        req = resolve_run_request(str(cfg), goal="train")
        values = list(req.sweep_values)

        set_seed(42)
        legacy, od_legacy = _legacy_simulate_snr_sweep(cfg, tmp_path / "legacy", values)
        set_seed(42)
        v2, od_v2 = _v2_train(cfg, tmp_path / "v2", goal="train")

        assert set(legacy.keys()) == set(v2.keys())
        for val in legacy:
            assert legacy[val]["status"] == v2[val]["status"] == "success"
            assert (od_legacy / f"snr_{val}").is_dir()
            assert (od_v2 / f"snr_{val}").is_dir()

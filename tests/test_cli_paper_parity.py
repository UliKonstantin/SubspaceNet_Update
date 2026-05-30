"""
Numeric parity: legacy Simulation path vs CLI v2 on paper configs.

Run with: python3 -m pytest tests/test_cli_paper_parity.py -v
"""
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "DCD_MUSIC"))

from cli.resolver import resolve_run_request
from cli.runner import run as run_v2
from config_handler import setup_configuration
from simulation.core import Simulation

from tests.parity_utils import (
    PAPER_ETA_SWEEP_CONFIG,
    PAPER_ETA_TRIM_OVERRIDES,
    compare_numeric_metrics,
    extract_ol_numeric_metrics,
    set_seed,
)

pytestmark = pytest.mark.slow


def _run_legacy_eta_sweep(output_dir: Path, seed: int, overrides: list[str], values: list[float]):
    set_seed(seed)
    cfg, comp, od = setup_configuration(str(PAPER_ETA_SWEEP_CONFIG), str(output_dir), overrides)
    sim = Simulation(cfg, comp, od)
    return sim.run_scenario("eta", values, full_mode=False, goal="online_learning")


def _run_v2_eta_sweep(output_dir: Path, seed: int, overrides: list[str], values: list[float]):
    set_seed(seed)
    request = resolve_run_request(
        str(PAPER_ETA_SWEEP_CONFIG),
        output_dir=str(output_dir),
        goal="online_learning",
        overrides=overrides,
        values=values,
    )
    return run_v2(request)


class TestPaperEtaSweepNumericParity:
    """Bit-exact parity on SineAccel eta sweep paper YAML (trimmed)."""

    @pytest.fixture
    def trim_values(self):
        return [0.4, 0.6]

    def test_per_eta_main_losses_match_legacy(self, trim_values):
        base = Path(tempfile.mkdtemp(prefix="paper_parity_"))
        try:
            seed = 42
            legacy = _run_legacy_eta_sweep(base / "legacy", seed, PAPER_ETA_TRIM_OVERRIDES, trim_values)
            v2 = _run_v2_eta_sweep(base / "v2", seed, PAPER_ETA_TRIM_OVERRIDES, trim_values)

            assert set(legacy.keys()) == set(v2.keys()) == set(trim_values)

            for eta in trim_values:
                leg_m = extract_ol_numeric_metrics(legacy[eta])
                v2_m = extract_ol_numeric_metrics(v2[eta])
                ok, diffs = compare_numeric_metrics(leg_m, v2_m)
                assert ok, f"eta={eta} numeric diffs:\n" + "\n".join(diffs)

                # Must have per-window losses to compare (pretrained path always populated)
                assert "averaged_pretrained_trajectory_main_losses" in leg_m
                assert len(leg_m["averaged_pretrained_trajectory_main_losses"]) >= 1
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_legacy_without_goal_matches_v2(self, trim_values):
        """Old simulate path (no goal arg) vs explicit v2 goal."""
        base = Path(tempfile.mkdtemp(prefix="paper_parity_nogoal_"))
        try:
            seed = 123
            set_seed(seed)
            cfg, comp, od = setup_configuration(
                str(PAPER_ETA_SWEEP_CONFIG), str(base / "legacy"), PAPER_ETA_TRIM_OVERRIDES
            )
            sim = Simulation(cfg, comp, od)
            # Legacy: no goal → routes via parent config flags
            legacy = sim.run_scenario("eta", trim_values, full_mode=False)

            v2 = _run_v2_eta_sweep(base / "v2", seed, PAPER_ETA_TRIM_OVERRIDES, trim_values)

            for eta in trim_values:
                leg_m = extract_ol_numeric_metrics(legacy[eta])
                v2_m = extract_ol_numeric_metrics(v2[eta])
                ok, diffs = compare_numeric_metrics(leg_m, v2_m)
                assert ok, f"eta={eta} (no-goal legacy) diffs:\n" + "\n".join(diffs)
        finally:
            shutil.rmtree(base, ignore_errors=True)

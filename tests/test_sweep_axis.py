"""Tests for 1D sweep axis → system_model field mapping."""
import sys
from pathlib import Path

import pytest

WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from cli.types import SweepAxis
from config.sweep_axis import system_model_override_for_axis


class TestSweepAxisEnum:
    def test_axis_n(self):
        assert SweepAxis.from_string("n") == SweepAxis.N
        assert SweepAxis.N.value == "n"


class TestSystemModelOverride:
    @pytest.mark.parametrize(
        "axis, value, expected",
        [
            ("n", 9, "system_model.N=9"),
            ("m", 3, "system_model.M=3"),
            ("t", 200, "system_model.T=200"),
            ("snr", 10, "system_model.snr=10"),
        ],
    )
    def test_maps_cli_axis_to_schema_field(self, axis, value, expected):
        assert system_model_override_for_axis(axis, value) == expected

    def test_unknown_axis_raises(self):
        with pytest.raises(ValueError, match="Unsupported sweep axis"):
            system_model_override_for_axis("eta", 0.5)

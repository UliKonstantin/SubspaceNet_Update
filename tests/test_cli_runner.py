"""Tests for cli.runner and Click entry point."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from cli.app import cli
from cli.runner import run as run_simulation
from cli.types import Goal, RunRequest, SweepType


class TestRunner:
    def test_run_single_train(self, tmp_path):
        request = RunRequest(
            goal=Goal.TRAIN,
            config_path=Path("tests/configs/test_training.yaml"),
            output_dir=tmp_path,
        )
        mock_result = {"status": "success", "trained_model": True}
        mock_sim = MagicMock()
        mock_sim.run_training.return_value = mock_result

        with patch("cli.runner.setup_configuration") as mock_setup, patch(
            "cli.runner.Simulation", return_value=mock_sim
        ), patch("cli.runner.setup_logging_from_config"), patch(
            "cli.runner.postprocess"
        ) as mock_post:
            mock_setup.return_value = (MagicMock(), {}, tmp_path)
            result = run_simulation(request)

        assert result == mock_result
        mock_sim.run_training.assert_called_once()
        mock_post.assert_called_once_with(mock_result, request, tmp_path, mock_sim)

    def test_run_error_exits(self, tmp_path):
        request = RunRequest(
            goal=Goal.TRAIN,
            config_path=Path("tests/configs/test_training.yaml"),
            output_dir=tmp_path,
        )
        mock_sim = MagicMock()
        mock_sim.run_training.return_value = {"status": "error", "message": "boom"}

        with patch("cli.runner.setup_configuration") as mock_setup, patch(
            "cli.runner.Simulation", return_value=mock_sim
        ), patch("cli.runner.setup_logging_from_config"), pytest.raises(SystemExit):
            mock_setup.return_value = (MagicMock(), {}, tmp_path)
            run_simulation(request)


class TestClickCLI:
    def test_run_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--goal" in result.output
        assert "online_learning" in result.output

    def test_run_routing_error(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "-c",
                "tests/configs/test_training.yaml",
                "--goal",
                "evaluate",
            ],
        )
        assert result.exit_code == 1
        assert "model" in result.output.lower()

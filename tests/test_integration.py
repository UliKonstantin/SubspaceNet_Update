"""
Integration and unit tests for the SubspaceNet online learning pipeline.

Run with: python -m pytest tests/test_integration.py -v
From workspace root with PYTHONPATH set.
"""
import sys
import math
from pathlib import Path

import pytest
import torch
import numpy as np

# Ensure workspace root and DCD_MUSIC are on path
WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "DCD_MUSIC"))

CONFIGS_DIR = Path(__file__).parent / "configs"
PRETRAINED_MODEL_PATH = "experiments/results/base_model_random_data_snr_10_SubspaceNet_esprit_N9_M3_SNR10.0_Far_ESPRIT/checkpoints/final_SubspaceNet_20250916_084930.pt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_simulate(config_path: str, output_dir: Path, scenario: str = None, values=None, mode: str = "training"):
    """Run the simulate command programmatically (same as CLI entry point)."""
    from config_handler import setup_configuration
    from simulation.core import Simulation

    config_obj, components, _ = setup_configuration(str(config_path), str(output_dir))

    sim = Simulation(config_obj, components, output_dir)

    if scenario and values:
        return sim.run_scenario(scenario, list(values), full_mode=(mode == "full"))
    elif mode == "online_learning":
        return sim.execute_online_learning()
    else:
        return sim.run_training()


# ---------------------------------------------------------------------------
# Integration Tests: Training
# ---------------------------------------------------------------------------

class TestBaseModelTraining:
    """Test base model training pipeline."""

    def test_training_runs_to_completion(self, tmp_path):
        """Training pipeline completes and produces a model with valid weights."""
        config_path = CONFIGS_DIR / "test_training.yaml"
        result = run_simulate(config_path, tmp_path, mode="training")

        assert result["status"] == "success"
        assert result["trained_model"] is True

        # Model file should exist in tmp_path
        checkpoints = list(tmp_path.rglob("*.pt"))
        assert len(checkpoints) > 0, "No model checkpoint saved"

        # Verify the checkpoint contains actual model weights
        state_dict = torch.load(checkpoints[0], map_location="cpu")
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        assert len(state_dict) > 0, "State dict is empty"
        # Check at least one parameter has non-zero values (i.e. trained, not all zeros)
        has_nonzero = any(v.abs().sum() > 0 for v in state_dict.values() if isinstance(v, torch.Tensor))
        assert has_nonzero, "All model weights are zero — model likely not trained"

    def test_training_snr_scenario(self, tmp_path):
        """SNR scenario sweep with training mode produces results for each value."""
        config_path = CONFIGS_DIR / "test_training_snr_scenario.yaml"
        results = run_simulate(config_path, tmp_path, scenario="snr", values=[5, 10], mode="training")

        assert isinstance(results, dict)
        assert len(results) == 2, f"Expected 2 scenario results, got {len(results)}"

        for key, result in results.items():
            assert result["status"] == "success", f"Scenario {key} failed: {result.get('message', '')}"


# ---------------------------------------------------------------------------
# Integration Tests: Online Learning
# ---------------------------------------------------------------------------

class TestOnlineLearning:
    """Test online learning pipeline."""

    @pytest.fixture(autouse=True)
    def _check_pretrained_model(self):
        """Skip if pre-trained model doesn't exist."""
        model_path = WORKSPACE_ROOT / PRETRAINED_MODEL_PATH
        if not model_path.exists():
            pytest.skip(f"Pre-trained model not found: {model_path}")

    def test_single_trajectory(self, tmp_path):
        """Single trajectory online learning runs end-to-end."""
        config_path = CONFIGS_DIR / "test_online_learning_single.yaml"
        result = run_simulate(config_path, tmp_path, mode="online_learning")

        assert result["status"] == "success"
        ol_results = result.get("online_learning_results")
        assert ol_results is not None, "Missing online_learning_results"
        assert ol_results["dataset_size"] >= 1

        # Check trajectory results were populated (list of TrajectoryResults)
        pretrained_results = ol_results["pretrained_trajectory_results"]
        assert len(pretrained_results) > 0
        assert len(pretrained_results[0].window_results) > 0

    def test_eta_sweep(self, tmp_path):
        """Eta sweep produces results for each eta value."""
        config_path = CONFIGS_DIR / "test_online_learning_eta_sweep.yaml"
        results = run_simulate(config_path, tmp_path, scenario="eta", values=[0.5, 1.0], mode="online_learning")

        assert isinstance(results, dict)
        assert len(results) == 2, f"Expected 2 eta results, got {len(results)}"

        for key, result in results.items():
            assert result["status"] == "success", f"Eta {key} failed: {result.get('message', '')}"


# ---------------------------------------------------------------------------
# Unit Tests: Adaptive LR
# ---------------------------------------------------------------------------

class TestAdaptiveLR:
    """Unit tests for adaptive learning rate sigmoid computation."""

    def test_sigmoid_at_inflection_point(self):
        """At dG = dG0, LR should be geometric mean of lr_min and lr_max."""
        lr_min = 0.0005
        lr_max = 0.0356
        k_sig = 0.7336
        dG0 = 69.2599

        dG = dG0  # At inflection point
        log_lr_min = math.log10(lr_min)
        log_lr_max = math.log10(lr_max)
        log_lr = log_lr_min + (log_lr_max - log_lr_min) / (1.0 + math.exp(-k_sig * (dG - dG0)))
        lr = 10 ** log_lr

        # At inflection, sigmoid = 0.5, so log_lr = midpoint
        expected_log_lr = (log_lr_min + log_lr_max) / 2
        expected_lr = 10 ** expected_log_lr

        assert abs(lr - expected_lr) < 1e-8

    def test_sigmoid_monotonically_increasing(self):
        """LR increases as dG increases (within non-saturated range)."""
        lr_min = 0.0005
        lr_max = 0.0356
        k_sig = 0.7336
        dG0 = 69.2599

        def compute_lr(dG):
            log_lr_min = math.log10(lr_min)
            log_lr_max = math.log10(lr_max)
            log_lr = log_lr_min + (log_lr_max - log_lr_min) / (1.0 + math.exp(-k_sig * (dG - dG0)))
            return 10 ** log_lr

        # Test within the active region of the sigmoid (not saturated tails)
        dG_values = [0, 30, 50, 69.2599, 80, 90]
        lr_values = [compute_lr(dG) for dG in dG_values]

        for i in range(len(lr_values) - 1):
            assert lr_values[i] < lr_values[i + 1], f"LR not monotonic at dG={dG_values[i]}"

    def test_sigmoid_bounds(self):
        """LR stays within [lr_min, lr_max]."""
        lr_min = 0.0005
        lr_max = 0.0356
        k_sig = 0.7336
        dG0 = 69.2599

        def compute_lr(dG):
            log_lr_min = math.log10(lr_min)
            log_lr_max = math.log10(lr_max)
            exponent = -k_sig * (dG - dG0)
            # Clamp to avoid overflow (same as real code behavior)
            exponent = max(min(exponent, 500), -500)
            log_lr = log_lr_min + (log_lr_max - log_lr_min) / (1.0 + math.exp(exponent))
            return 10 ** log_lr

        # At extremes, LR should saturate near bounds
        assert compute_lr(-100) < lr_min * 1.1
        assert compute_lr(300) > lr_max * 0.99


# ---------------------------------------------------------------------------
# Unit Tests: GLRT Changepoint Detection
# ---------------------------------------------------------------------------

class TestGLRTChangepoint:
    """Unit tests for GLRT changepoint detection."""

    def test_flat_signal_low_glr(self):
        """Flat (no change) signal should produce much lower log-GLR than a step change."""
        from simulation.runners.sandbox import glrt_changepoint_detection

        np.random.seed(42)
        losses_flat = [1.0 + np.random.normal(0, 0.01) for _ in range(20)]
        losses_step = [1.0] * 10 + [5.0] * 10

        _, glr_flat, _, _ = glrt_changepoint_detection(losses_flat, min_segment_size=3)
        _, glr_step, _, _ = glrt_changepoint_detection(losses_step, min_segment_size=3)

        # Flat signal GLR should be much smaller than step change GLR
        assert glr_flat < glr_step * 0.1, f"Flat GLR ({glr_flat:.2f}) not much smaller than step GLR ({glr_step:.2f})"

    def test_step_change_detected(self):
        """Step change in mean should be detected near the true changepoint."""
        from simulation.runners.sandbox import glrt_changepoint_detection

        # Clear step change at index 15
        losses = [1.0] * 15 + [5.0] * 15

        changepoint, log_glr, _, _ = glrt_changepoint_detection(losses, min_segment_size=3)

        # Changepoint should be near index 15
        assert abs(changepoint - 15) <= 2, f"Changepoint at {changepoint}, expected near 15"
        # Log-GLR should be high for clear change
        assert log_glr > 10.0, f"Expected high GLR for step change, got {log_glr}"

    def test_min_segment_size_respected(self):
        """Changepoint should not be in the first or last min_segment_size windows."""
        from simulation.runners.sandbox import glrt_changepoint_detection

        losses = [1.0] * 5 + [10.0] * 15
        min_seg = 5
        changepoint, _, _, candidate_points = glrt_changepoint_detection(losses, min_segment_size=min_seg)

        assert changepoint >= min_seg
        assert changepoint <= len(losses) - min_seg


# ---------------------------------------------------------------------------
# Unit Tests: Config Schema
# ---------------------------------------------------------------------------

class TestConfigSchema:
    """Test config schema parses correctly with new fields."""

    def test_default_values(self):
        """OnlineLearningConfig has correct defaults for new fields."""
        from config.schema import OnlineLearningConfig

        config = OnlineLearningConfig()
        assert config.adaptive_lr_min == 0.0005
        assert config.adaptive_lr_max == 0.0356
        assert config.adaptive_lr_k_sigmoid == 0.7336
        assert config.adaptive_lr_dG0 == 69.2599
        assert config.glrt_history_exclusion == 5
        assert config.num_gd_steps == 3

    def test_override_values(self):
        """OnlineLearningConfig accepts overridden values."""
        from config.schema import OnlineLearningConfig

        config = OnlineLearningConfig(
            adaptive_lr_min=0.001,
            adaptive_lr_max=0.1,
            adaptive_lr_k_sigmoid=1.0,
            adaptive_lr_dG0=50.0,
            glrt_history_exclusion=10,
            num_gd_steps=5
        )
        assert config.adaptive_lr_min == 0.001
        assert config.adaptive_lr_max == 0.1
        assert config.adaptive_lr_k_sigmoid == 1.0
        assert config.adaptive_lr_dG0 == 50.0
        assert config.glrt_history_exclusion == 10
        assert config.num_gd_steps == 5

    def test_full_config_loads(self):
        """Full test config YAML loads without validation errors."""
        from config.loader import load_config

        config = load_config(str(CONFIGS_DIR / "test_online_learning_single.yaml"))
        assert config.online_learning.enabled is True
        assert config.online_learning.adaptive_lr_min == 0.0005
        assert config.online_learning.num_gd_steps == 1

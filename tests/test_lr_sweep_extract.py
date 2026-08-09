"""Unit tests for LR sweep post-training loss extraction."""
import sys
from pathlib import Path

import pytest

WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "DCD_MUSIC"))

from utils.utils import mean_reference_loss_after_training


class TestMeanReferenceLossAfterTraining:
    def test_filters_by_absolute_window_index_not_array_slice(self):
        window_indices = list(range(35, 45))
        losses = [1.0] * 5 + [2.0] * 5
        avg = mean_reference_loss_after_training(window_indices, losses, training_end_window=39)
        assert avg == pytest.approx(2.0)

    def test_old_slice_logic_would_fail_this_case(self):
        window_indices = list(range(35, 45))
        losses = [1.0] * 5 + [2.0] * 5
        # Broken: losses[39 + 1:] on length-10 array → empty
        assert len(losses[40:]) == 0
        avg = mean_reference_loss_after_training(window_indices, losses, training_end_window=39)
        assert avg is not None

    def test_fallback_last_n_when_no_post_training_windows(self):
        window_indices = list(range(35, 40))
        losses = [3.0, 4.0, 5.0, 6.0, 7.0]
        avg = mean_reference_loss_after_training(window_indices, losses, training_end_window=39, fallback_last_n=3)
        assert avg == pytest.approx(6.0)

    def test_extract_shape_matches_scenario_plot_last_n_fallback(self):
        window_indices = list(range(110, 120))
        losses = list(range(110, 120))
        avg = mean_reference_loss_after_training(
            window_indices, losses, training_end_window=119, fallback_last_n=10
        )
        assert avg == pytest.approx(114.5)

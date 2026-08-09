"""Unit tests for drift-detection gate helpers."""
import sys
from pathlib import Path

import numpy as np
import pytest

WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from utils.drift_gates import (
    GLRT_MIN_SEGMENT_SIZE,
    SCOPE_B_BASELINE_MIN_SAMPLES,
    first_g_window,
    first_z_window,
    baseline_g_values,
    can_compute_drift_z_score,
    compute_drift_z_score,
    has_enough_losses_for_changepoint_glr,
)


class TestDriftGates:
    def test_first_g_window(self):
        assert first_g_window(4, 5) == 14
        assert first_g_window(10, 5) == 20

    def test_first_z_window(self):
        # scope_a=4, m=5, guard=3, scope_b_min=7 -> 4+10+3+7-1=23
        assert first_z_window(4, 3) == 23
        # scope_a=10 -> 10+10+3+7-1=29
        assert first_z_window(10, 3) == 29

    def test_has_enough_losses(self):
        assert not has_enough_losses_for_changepoint_glr(10, min_segment_size=5)
        assert has_enough_losses_for_changepoint_glr(11, min_segment_size=5)

    def test_baseline_excludes_tail_guard_only(self):
        history = list(range(20))
        baseline = baseline_g_values(history, guard_samples=3)
        np.testing.assert_array_equal(baseline, list(range(17)))

    def test_can_compute_drift_z_score(self):
        assert not can_compute_drift_z_score(9, guard_samples=3)
        assert can_compute_drift_z_score(10, guard_samples=3)

    def test_z_score_with_guard_exclusion(self):
        stable = [1.0, 1.1, 1.0, 1.2, 1.0, 1.1, 1.0]
        tail = [50.0, 60.0, 70.0]
        history = stable + tail
        z, mean, std = compute_drift_z_score(70.0, history, guard_samples=3)
        assert z is not None
        assert z > 2.0

    def test_drift_detection_milestones(self):
        from utils.drift_gates import drift_detection_milestones

        m = drift_detection_milestones(10, 3)
        assert m["scope_a_loss_start"] == 10
        assert m["first_g"] == 20
        assert m["first_z"] == 29

        assert GLRT_MIN_SEGMENT_SIZE == 5
        assert SCOPE_B_BASELINE_MIN_SAMPLES == 7

    def test_schema_rejects_history_cap_too_small(self):
        from config.schema import OnlineLearningConfig

        with pytest.raises(ValueError, match="drift_history_max_size"):
            OnlineLearningConfig(drift_guard_samples=3, drift_history_max_size=5)

"""Tests for cross-window EKF handoff helpers."""
import pytest

from utils.ekf_handoff import (
    ekf_handoff_reuse_step0,
    ekf_handoff_step_index,
    ekf_next_predict_time_index,
)


class TestHandoffStepIndex:
    def test_overlapping_stride_3_window_5(self):
        # window 1 step 0 is global index 3 → prev window step 3
        assert ekf_handoff_step_index(3, 5) == 3

    def test_smoke_stride_2_window_3(self):
        # overlap step equals last step
        assert ekf_handoff_step_index(2, 3) == 2

    def test_non_overlapping_abutting(self):
        assert ekf_handoff_step_index(5, 5) == 4

    def test_stride_larger_than_window_gap(self):
        # ws=5, stride=6: prev last step
        assert ekf_handoff_step_index(6, 5) == 4

    def test_stride_1_heavy_overlap(self):
        assert ekf_handoff_step_index(1, 5) == 1

    def test_invalid_stride(self):
        with pytest.raises(ValueError):
            ekf_handoff_step_index(0, 5)


class TestHandoffReuse:
    def test_overlap(self):
        assert ekf_handoff_reuse_step0(3, 5) is True
        assert ekf_handoff_reuse_step0(1, 5) is True

    def test_no_overlap(self):
        assert ekf_handoff_reuse_step0(5, 5) is False
        assert ekf_handoff_reuse_step0(6, 5) is False


class TestNextPredictTime:
    def test_overlap_reuse(self):
        assert ekf_next_predict_time_index(3, True) == 4

    def test_sequential_handoff(self):
        assert ekf_next_predict_time_index(5, False) == 5

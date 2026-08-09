"""Tests for sine-accel DC offset (per-source attractor)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

_tp_path = WORKSPACE_ROOT / "simulation" / "runners" / "trajectory_physics.py"
_spec = importlib.util.spec_from_file_location("trajectory_physics", _tp_path)
trajectory_physics = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(trajectory_physics)

sine_accel_step = trajectory_physics.sine_accel_step
resolve_sine_accel_dc_offsets = trajectory_physics.resolve_sine_accel_dc_offsets

from config.schema import TrajectoryConfig, TrajectoryType


class TestSineAccelDcOffset:
    def test_resolve_random_range(self):
        cfg = TrajectoryConfig(
            trajectory_type=TrajectoryType.SINE_ACCEL_NONLINEAR,
            sine_accel_dc_offset_range=[-15, 15],
        )
        rng = np.random.default_rng(0)
        offsets = resolve_sine_accel_dc_offsets(cfg, num_sources=3, rng=rng)
        assert offsets.shape == (3,)
        assert np.all(offsets >= -15) and np.all(offsets <= 15)

    def test_gt_converges_to_dc_not_zero(self):
        dc = np.array([10.0, -8.0, 5.0])
        theta = np.zeros(3)
        omega0 = np.array([0.1, 0.1, 0.1])
        kappa = np.zeros(3)

        for t in range(800):
            theta = sine_accel_step(theta, t, omega0, kappa, 0.0, dc_offset=dc)

        np.testing.assert_allclose(theta, dc, atol=0.5)

    def test_disabled_when_range_unset(self):
        cfg = TrajectoryConfig(trajectory_type=TrajectoryType.SINE_ACCEL_NONLINEAR)
        assert resolve_sine_accel_dc_offsets(cfg, num_sources=3) is None

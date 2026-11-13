"""Measurement R is deg² → rad², matching process-noise conversion."""

import math

from config.loader import load_config
from simulation.kalman_filter.extended import ExtendedKalmanFilter1D
from simulation.kalman_filter.units import angle_std_deg_to_variance_rad


def test_angle_std_deg_to_variance_rad():
    expected = (0.03 * math.pi / 180.0) ** 2
    assert abs(angle_std_deg_to_variance_rad(0.03) - expected) < 1e-18


def test_t1_ekf_r_matches_deg_yaml():
    cfg = load_config("configs/Used_for_paper/paper_T1_antenna_sweep.yaml")
    _, kf_r, _ = ExtendedKalmanFilter1D.from_config(cfg)
    expected = angle_std_deg_to_variance_rad(cfg.kalman_filter.measurement_noise_std_dev)
    assert abs(float(kf_r) - expected) < 1e-12
    # Locked T1 R=0.30° → ~2.74e-5 rad² (0.30² deg would be 0.09; 0.30 as rad would be 0.09)
    assert 1e-6 < expected < 1e-4, f"0.30° R must be ~2.7e-5 rad², got {expected}"

"""CUSUM + absolute threshold on unsupervised adaptation_loss."""
from pathlib import Path

import pytest

from simulation.drift.loss_cusum import LossCusumDetector


# T6 ds50 averaged unsupervised L / EKF / SN (rad), windows 0–44.
T6_ADAPT = [
    0.0035, 0.0054, 0.0075, 0.0108, 0.0156, 0.0222, 0.0303, 0.0396, 0.0492, 0.0587,
    0.0674, 0.0743, 0.0790, 0.0814, 0.0810, 0.0777, 0.0716, 0.0636, 0.0545, 0.0454,
    0.0369, 0.0297, 0.0241, 0.0196, 0.0161, 0.0133, 0.0110, 0.0097, 0.0087, 0.0080,
    0.0074, 0.0074, 0.0067, 0.0060, 0.0053, 0.0047, 0.0041, 0.0039, 0.0040, 0.0043,
    0.1895, 0.1758, 0.1709, 0.1715, 0.1750,
]
T6_EKF = [
    0.0048, 0.0055, 0.0073, 0.0104, 0.0152, 0.0220, 0.0305, 0.0401, 0.0499, 0.0592,
    0.0674, 0.0737, 0.0779, 0.0798, 0.0790, 0.0755, 0.0694, 0.0616, 0.0529, 0.0442,
    0.0362, 0.0296, 0.0243, 0.0199, 0.0164, 0.0136, 0.0113, 0.0095, 0.0080, 0.0068,
    0.0059, 0.0051, 0.0045, 0.0041, 0.0037, 0.0034, 0.0033, 0.0031, 0.0029, 0.0028,
    0.1049, 0.1091, 0.1140, 0.1202, 0.1258,
]
T6_SN = [
    0.0051, 0.0052, 0.0054, 0.0055, 0.0058, 0.0061, 0.0064, 0.0069, 0.0071, 0.0074,
    0.0076, 0.0074, 0.0069, 0.0063, 0.0058, 0.0053, 0.0050, 0.0047, 0.0044, 0.0039,
    0.0035, 0.0032, 0.0030, 0.0030, 0.0031, 0.0034, 0.0040, 0.0048, 0.0055, 0.0061,
    0.0064, 0.0064, 0.0062, 0.0058, 0.0053, 0.0049, 0.0044, 0.0041, 0.0043, 0.0046,
    0.2319, 0.2309, 0.2339, 0.2442, 0.2531,
]


def _run(det, adapt, ekf, sn):
    out = []
    for w, (L, e, p) in enumerate(zip(adapt, ekf, sn)):
        out.append(det.step(w, L, e, p))
    return out


def test_ekf_hump_does_not_fire_and_step_does():
    det = LossCusumDetector()
    statuses = _run(det, T6_ADAPT, T6_EKF, T6_SN)
    assert det.settled_window == 19
    assert det.frozen
    assert statuses[19]["just_settled"]
    assert statuses[26]["just_froze"]
    fires = [s["window_idx"] for s in statuses if s["fired"]]
    assert fires[0] == 40
    assert all(not s["fired"] for s in statuses[:40])


def test_jump_during_ref_freezes_clean_null_and_fires():
    det = LossCusumDetector(ref_windows=8, settle_fallback_window=0)
    quiet = [0.004] * 4
    jump = [0.19]
    ekf = [0.003] * 5
    sn = [0.005] * 5
    statuses = _run(det, quiet + jump, ekf, sn)
    assert det.frozen
    assert abs(det.mu0 - 0.004) < 1e-12
    assert statuses[4]["fired"]
    assert statuses[4]["just_froze"]


def test_no_drift_stays_quiet_after_ekf_settle():
    det = LossCusumDetector()
    adapt = T6_ADAPT[:40] + [0.0042] * 80
    ekf = T6_EKF[:40] + [0.0030] * 80
    sn = T6_SN[:40] + [0.0045] * 80
    statuses = _run(det, adapt, ekf, sn)
    assert det.frozen
    assert all(not s["fired"] for s in statuses)


def test_abs_threshold_vetoes_cusum_on_tiny_sigma_wiggle():
    det = LossCusumDetector(loss_threshold=0.05, settle_fallback_window=0, ref_windows=8)
    quiet = [0.004] * 8
    bump = [0.01] * 20  # above CUSUM, below abs
    ekf = [0.003] * 28
    sn = [0.004] * 28
    statuses = _run(det, quiet + bump, ekf, sn)
    assert det.frozen
    assert any(s["cusum_ok"] for s in statuses)
    assert all(not s["fired"] for s in statuses)
    assert all(not s["abs_ok"] for s in statuses)


def test_late_step_still_fires():
    det = LossCusumDetector()
    adapt = T6_ADAPT[:40] + [0.0042] * 40 + [0.19] * 5
    ekf = T6_EKF[:40] + [0.0030] * 40 + [0.10] * 5
    sn = T6_SN[:40] + [0.0045] * 40 + [0.23] * 5
    statuses = _run(det, adapt, ekf, sn)
    fires = [s["window_idx"] for s in statuses if s["fired"]]
    assert fires[0] == 80


def test_early_ekf_blip_does_not_fire_on_setup_hump():
    det = LossCusumDetector(hump_confirm_windows=5)
    adapt = [0.004] * 6 + [0.02, 0.04, 0.06, 0.07, 0.06, 0.04, 0.02] + [0.008] * 10
    ekf = [0.003, 0.006, 0.003, 0.003, 0.003, 0.003] + [0.03] * 6 + [0.003] * 11
    sn = [0.004] * len(adapt)
    statuses = _run(det, adapt, ekf, sn)
    fires = [s["window_idx"] for s in statuses if s["fired"]]
    assert all(w >= 12 for w in fires) or not fires
    assert all(not s["fired"] for s in statuses[:12])


def test_sustained_ekf_hump_above_threshold_does_not_freeze():
    det = LossCusumDetector(settle_fallback_window=5)
    adapt = [0.08] * 40
    ekf = [0.03] * 40
    sn = [0.01] * 40
    statuses = _run(det, adapt, ekf, sn)
    assert det.saw_ekf_hump
    assert det.hump_confirmed
    assert not det.frozen
    assert all(not s["fired"] for s in statuses)


def test_schema_default_stays_glrt_z():
    from config.schema import OnlineLearningConfig

    cfg = OnlineLearningConfig()
    assert cfg.drift_detector == "glrt_z"
    assert cfg.cusum_loss_threshold == 0.05
    assert cfg.max_adapt_cycles == 1
    assert cfg.eta_step_windows is None
    assert cfg.dump_music_spectrum is False
    assert cfg.spectrum_pre_window == 0
    assert cfg.spectrum_min_sep_deg == 10.0


def test_schema_rejects_bad_cusum():
    from config.schema import OnlineLearningConfig

    with pytest.raises(ValueError, match="cusum_h"):
        OnlineLearningConfig(cusum_h=0.0)


def test_t1_cusum_yaml_loads():
    from config.loader import load_config

    cfg = load_config("configs/Used_for_paper/paper_T1_cusum.yaml")
    ol = cfg.online_learning
    assert ol.drift_detector == "loss_cusum"
    assert cfg.system_model.N == 9
    assert cfg.scenario_config.values == [6, 9, 12]
    assert ol.eta_update_interval_windows == 40
    assert abs(ol.cusum_loss_threshold - 0.05) < 1e-12
    assert ol.dump_music_spectrum is False
    assert ol.spectrum_pre_window == 0


def test_rearm_second_jump_fires():
    det = LossCusumDetector(settle_fallback_window=0, ref_windows=8)
    quiet = [0.004] * 8
    jump1 = [0.19] * 3
    after = [0.006] * 8
    jump2 = [0.18] * 3
    ekf = [0.003] * 22
    sn = [0.005] * 22
    L = quiet + jump1 + after + jump2
    statuses = _run(det, L, ekf, sn)
    fires = [s["window_idx"] for s in statuses if s["fired"]]
    assert fires[0] == 8
    det.rearm()
    rest_L = after + jump2
    rest_e = [0.003] * len(rest_L)
    rest_p = [0.005] * len(rest_L)
    statuses2 = []
    for i, (Lv, e, p) in enumerate(zip(rest_L, rest_e, rest_p)):
        statuses2.append(det.step(20 + i, Lv, e, p))
    fires2 = [s["window_idx"] for s in statuses2 if s["fired"]]
    assert det.frozen
    assert abs(det.mu0 - 0.006) < 1e-12
    assert fires2[0] == 28


def test_t1_2_cusum_yaml_loads():
    from config.loader import load_config

    cfg = load_config("configs/Used_for_paper/paper_T1_2_cusum.yaml")
    ol = cfg.online_learning
    assert ol.drift_detector == "loss_cusum"
    assert ol.drift_type == "sv_noise_var"
    assert cfg.scenario_config.sv_noise_increments == [1.80, 1.00, 12.00]
    assert ol.eta_update_interval_windows == 40
    assert ol.max_adapt_cycles == 1
    assert abs(ol.cusum_loss_threshold - 0.05) < 1e-12


def test_t2_cusum_yaml_loads():
    from config.loader import load_config

    cfg = load_config("configs/Used_for_paper/paper_T2_cusum.yaml")
    ol = cfg.online_learning
    assert ol.drift_detector == "loss_cusum"
    assert ol.drift_type == "position_eta"
    assert cfg.scenario_config.type == "snr"
    assert cfg.scenario_config.values == [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert cfg.scenario_config.eta_increments[-1] == 1.30
    assert len(cfg.scenario_config.model_paths) == 16
    assert ol.eta_update_interval_windows == 40
    assert ol.max_adapt_cycles == 1
    assert abs(ol.cusum_loss_threshold - 0.05) < 1e-12


def test_t3_cusum_yaml_loads():
    from config.loader import load_config

    cfg = load_config("configs/Used_for_paper/paper_T3_cusum.yaml")
    ol = cfg.online_learning
    assert ol.drift_detector == "loss_cusum"
    assert ol.drift_type == "position_eta"
    assert cfg.scenario_config.type == "m"
    assert cfg.scenario_config.values == [2, 3, 4, 5]
    assert cfg.scenario_config.eta_increments == [3.75, 1.50, 1.20, 2.09]
    assert len(cfg.scenario_config.model_paths) == 4
    assert ol.eta_update_interval_windows == 40
    assert ol.max_adapt_cycles == 1
    assert abs(ol.cusum_loss_threshold - 0.05) < 1e-12


T100_YAMLS = [
    "paper_T1_cusum_t100.yaml",
    "paper_T1_iqresnet_t100.yaml",
    "paper_T1_2_cusum_t100.yaml",
    "paper_T1_2_iqresnet_t100.yaml",
    "paper_T2_cusum_t100.yaml",
    "paper_T2_iqresnet_t100.yaml",
    "paper_T3_cusum_t100.yaml",
    "paper_T3_iqresnet_t100.yaml",
]
IQ_CKPT = (
    "experiments/results/basemodels_for_journal_paper/iqresnet_rmspe015/"
    "checkpoints/best_IQResNet_20260921_014736.pt"
)


def test_t100_yamls_are_cusum_and_do_not_alias_locked_dirs():
    from config.loader import load_config

    for name in T100_YAMLS:
        path = Path("configs/Used_for_paper") / name
        cfg = load_config(str(path))
        ol = cfg.online_learning
        assert ol.trajectory_length == 100, path.name
        assert ol.drift_detector == "loss_cusum", path.name
        assert ol.window_size == 5 and ol.stride == 1
        n_windows = (ol.trajectory_length - ol.window_size) // ol.stride + 1
        assert n_windows == 96
        assert cfg.training.simulation_name == path.stem
        raw = path.read_text()
        out_lines = [ln.strip() for ln in raw.splitlines() if ln.startswith("  output_dir:")]
        assert len(out_lines) == 1, path.name
        out = out_lines[0].split('"')[1]
        assert out == f"experiments/results/journal_paper/{path.stem}"
        if ol.eta_step_windows:
            assert max(ol.eta_step_windows) < n_windows
        if ol.drift_true_windows:
            present = [w for w in ol.drift_true_windows if w is not None]
            assert max(present) < n_windows
        if cfg.model.type == "IQResNet":
            assert abs(ol.adaptive_lr_max - 0.001) < 1e-12
            assert cfg.simulation.model_path == IQ_CKPT
        else:
            assert cfg.model.type == "SubspaceNet"
            assert abs(ol.adaptive_lr_max - 0.02) < 1e-12


def test_t2_t3_iq_t100_checkpoint_lists():
    from config.loader import load_config

    t2 = load_config("configs/Used_for_paper/paper_T2_iqresnet_t100.yaml")
    sn_t2 = load_config("configs/Used_for_paper/paper_T2_cusum_t100.yaml")
    assert t2.scenario_config.values == [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert t2.scenario_config.eta_increments == sn_t2.scenario_config.eta_increments
    assert t2.scenario_config.max_etas == sn_t2.scenario_config.max_etas
    assert len(t2.scenario_config.model_paths) == 16
    assert t2.scenario_config.model_paths[-1] == IQ_CKPT

    t3 = load_config("configs/Used_for_paper/paper_T3_iqresnet_t100.yaml")
    sn_t3 = load_config("configs/Used_for_paper/paper_T3_cusum_t100.yaml")
    assert t3.scenario_config.values == [2, 3, 4, 5]
    assert t3.scenario_config.eta_increments == sn_t3.scenario_config.eta_increments
    assert t3.scenario_config.max_etas == sn_t3.scenario_config.max_etas
    assert list(t3.scenario_config.model_paths) == [
        "experiments/results/basemodels_for_journal_paper/iqresnet_m/m_2.0/checkpoints/best_IQResNet_20260925_230811.pt",
        IQ_CKPT,
        "experiments/results/basemodels_for_journal_paper/iqresnet_m/m_4.0/checkpoints/best_IQResNet_20260925_231349.pt",
        "experiments/results/basemodels_for_journal_paper/iqresnet_m/m_5.0/checkpoints/best_IQResNet_20260925_233003.pt",
    ]
    assert "20260925_231930" not in Path("configs/Used_for_paper/paper_T3_iqresnet_t100.yaml").read_text()


def test_t6_latency_and_mixed_yamls_use_loss_cusum():
    from config.loader import load_config

    for name in ("paper_T6_latency.yaml", "paper_T6_mixed.yaml"):
        cfg = load_config(f"configs/Used_for_paper/{name}")
        ol = cfg.online_learning
        assert ol.drift_detector == "loss_cusum"
        assert ol.window_size == 5
        assert ol.stride == 1
        assert abs(cfg.kalman_filter.measurement_noise_std_dev - 0.30) < 1e-12
        assert abs(cfg.kalman_filter.process_noise_std_dev - 0.06) < 1e-12

"""T6 latency: wall-clock L_det / L_learn and τ* window histogram."""

from simulation.latency_metrics import compute_latency_record, summarize_latency_records


def test_compute_latency_matches_start_stop_clocks():
    rec = compute_latency_record(
        w_true=10,
        w_star=50,
        tau_star=15,
        w_done=56,
        t_true=100.0,
        t_star=104.2,
        t_done=107.5,
    )
    assert abs(rec["L_det_s"] - 4.2) < 1e-12
    assert abs(rec["L_learn_s"] - 7.5) < 1e-12
    assert rec["L_tau"] == 5
    assert rec["missed"] is False


def test_miss_has_no_det_and_flag():
    rec = compute_latency_record(
        w_true=40,
        w_star=None,
        tau_star=None,
        w_done=None,
        t_true=1.0,
        t_star=None,
        t_done=None,
    )
    assert rec["missed"] is True
    assert rec["L_det_s"] is None
    assert rec["L_learn_s"] is None
    assert rec["L_tau"] is None


def test_summarize_excludes_misses():
    fired = compute_latency_record(
        w_true=40, w_star=42, tau_star=40, w_done=48,
        t_true=0.0, t_star=0.5, t_done=2.0,
    )
    miss = compute_latency_record(
        w_true=40, w_star=None, tau_star=None, w_done=None,
        t_true=0.0, t_star=None, t_done=None,
    )
    s = summarize_latency_records([fired, miss])
    assert s["n_traj"] == 2
    assert s["n_miss"] == 1
    assert s["n_fired"] == 1
    assert abs(s["L_det_s_mean"] - 0.5) < 1e-12
    assert abs(s["L_learn_s_mean"] - 2.0) < 1e-12
    assert s["L_tau"] == [0]


def test_no_drift_false_alarm_and_tn():
    from simulation.latency_metrics import confusion_from_records

    fp = compute_latency_record(
        w_true=None, w_star=37, tau_star=30, w_done=43,
        t_true=None, t_star=1.0, t_done=2.0,
    )
    tn = compute_latency_record(
        w_true=None, w_star=None, tau_star=None, w_done=None,
        t_true=None, t_star=None, t_done=None,
    )
    tp = compute_latency_record(
        w_true=40, w_star=42, tau_star=40, w_done=48,
        t_true=0.0, t_star=0.5, t_done=2.0,
    )
    assert fp["false_alarm"] is True
    assert fp["missed"] is False
    assert tn["missed"] is False
    cm = confusion_from_records([fp, tn, tp])
    assert cm == {"tp": 1, "fp": 1, "fn": 0, "tn": 1, "n": 3}


def test_step_drift_schedule_null_and_window():
    from types import SimpleNamespace
    from simulation.latency_metrics import step_drift_is_due, scheduled_true_drift_window

    cfg = SimpleNamespace(
        drift_true_windows=[None, 40, 80],
        eta_update_interval_windows=40,
    )
    assert scheduled_true_drift_window(0, cfg) is None
    assert scheduled_true_drift_window(1, cfg) == 40
    assert step_drift_is_due(40, 0, cfg, "step") is False
    assert step_drift_is_due(40, 1, cfg, "step") is True
    assert step_drift_is_due(80, 1, cfg, "step") is False


def test_eta_step_windows_beats_modulo():
    from types import SimpleNamespace
    from simulation.latency_metrics import step_drift_is_due

    cfg = SimpleNamespace(
        eta_step_windows=[40, 80, 120],
        eta_update_interval_windows=40,
        drift_true_windows=None,
    )
    assert step_drift_is_due(40, 0, cfg, "step") is True
    assert step_drift_is_due(80, 0, cfg, "step") is True
    assert step_drift_is_due(120, 0, cfg, "step") is True
    assert step_drift_is_due(160, 0, cfg, "step") is False
    assert step_drift_is_due(39, 0, cfg, "step") is False


def test_pair_step_detections_assigns_hold():
    from simulation.latency_metrics import pair_step_detections

    paired = pair_step_detections([40, 80, 120], [41, 82])
    assert paired == [
        {"w_true": 40, "w_star": 41},
        {"w_true": 80, "w_star": 82},
        {"w_true": 120, "w_star": None},
    ]


def test_plot_latency_histograms_writes_pngs(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from utils.plotting.latency import plot_latency_histograms

    rec = compute_latency_record(
        w_true=10, w_star=50, tau_star=15, w_done=56,
        t_true=100.0, t_star=104.2, t_done=107.5,
    )
    plot_latency_histograms([rec], tmp_path)
    for name in (
        "latency_L_det_hist.png",
        "latency_L_learn_hist.png",
        "latency_L_tau_hist.png",
        "latency_metrics.json",
        "latency_identification_vs_true.png",
        "latency_confusion_matrix.png",
    ):
        assert (tmp_path / name).exists(), name


def test_collect_from_sweep_and_single():
    rec = compute_latency_record(
        w_true=40, w_star=42, tau_star=40, w_done=48,
        t_true=0.0, t_star=0.5, t_done=2.0,
    )
    from simulation.latency_metrics import collect_latency_records

    single = collect_latency_records({"latency_records": [rec]})
    assert single == [rec]
    nested = collect_latency_records({
        9.0: {"status": "success", "latency_records": [rec], "online_learning_results": {}},
    })
    assert nested == [rec]


def test_t6_yaml_loads():
    from config.loader import load_config

    cfg = load_config("configs/Used_for_paper/paper_T6_latency.yaml")
    ol = cfg.online_learning
    assert ol.drift_type == "position_eta"
    assert (ol.drift_profile or "step") == "step"
    assert ol.eta_update_interval_windows == 40
    assert abs(ol.max_eta - 1.30) < 1e-9
    assert cfg.system_model.N == 9


def test_t6_mixed_yaml_loads():
    from config.loader import load_config

    cfg = load_config("configs/Used_for_paper/paper_T6_mixed.yaml")
    ol = cfg.online_learning
    assert ol.drift_true_windows == [None, 40, 55, 80, 120]
    assert cfg.system_model.N == 9
    assert ol.drift_detector == "loss_cusum"

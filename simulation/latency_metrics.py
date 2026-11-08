"""T6 wall-clock detection / learning latency plus τ* window histogram."""

from __future__ import annotations

from typing import Any, Optional


def scheduled_true_drift_window(trajectory_idx: int, online_config: Any) -> Optional[int]:
    """Per-traj true drift window. None = no drift (negative class)."""
    schedule = getattr(online_config, "drift_true_windows", None)
    if schedule:
        w = schedule[int(trajectory_idx) % len(schedule)]
        return int(w) if w is not None else None
    return None


def step_drift_is_due(window_idx: int, trajectory_idx: int, online_config: Any, drift_profile: str) -> bool:
    """Whether to apply a one-shot step drift before this window."""
    if (drift_profile or "step") == "gradual":
        return False
    steps = getattr(online_config, "eta_step_windows", None)
    if steps:
        return int(window_idx) in {int(w) for w in steps}
    scheduled = scheduled_true_drift_window(trajectory_idx, online_config)
    if getattr(online_config, "drift_true_windows", None):
        return scheduled is not None and int(window_idx) == scheduled
    interval = getattr(online_config, "eta_update_interval_windows", None)
    return bool(interval and interval > 0 and window_idx > 0 and window_idx % interval == 0)


def pair_step_detections(
    true_windows: list[int],
    fire_windows: list[int],
) -> list[dict[str, Optional[int]]]:
    """Assign each fire to the hold [w_true[i], w_true[i+1]). Unused fires dropped."""
    remaining = [int(f) for f in fire_windows]
    out: list[dict[str, Optional[int]]] = []
    for i, wt in enumerate(true_windows):
        end = int(true_windows[i + 1]) if i + 1 < len(true_windows) else None
        match: Optional[int] = None
        for f in remaining:
            if f >= int(wt) and (end is None or f < end):
                match = f
                remaining.remove(f)
                break
        out.append({"w_true": int(wt), "w_star": match})
    return out


def confusion_from_records(records: list[dict]) -> dict[str, int]:
    """2×2: actual drift vs system identified. Timing-agnostic."""
    tp = fp = fn = tn = 0
    for r in records:
        has = r.get("w_true") is not None
        det = r.get("w_star") is not None
        if has and det:
            tp += 1
        elif has and not det:
            fn += 1
        elif not has and det:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": len(records)}


def collect_latency_records(result: Any) -> list[dict]:
    """Pull per-traj latency dicts from a single-run or 1d-sweep result."""
    if not isinstance(result, dict):
        return []
    recs = result.get("latency_records")
    if isinstance(recs, list) and recs:
        return list(recs)
    rec = (result.get("online_learning_results") or {}).get("latency_record")
    if rec:
        return [rec]
    out: list[dict] = []
    for v in result.values():
        if not isinstance(v, dict):
            continue
        if "lr_sweep_results" in v:
            for lr in v["lr_sweep_results"].values():
                if isinstance(lr, dict):
                    out.extend(collect_latency_records(lr.get("result") or {}))
        elif "latency_records" in v or "online_learning_results" in v:
            out.extend(collect_latency_records(v))
    return out


def compute_latency_record(
    *,
    w_true: Optional[int],
    w_star: Optional[int],
    tau_star: Optional[int],
    w_done: Optional[int],
    t_true: Optional[float],
    t_star: Optional[float],
    t_done: Optional[float],
) -> dict[str, Any]:
    """L_det / L_learn are seconds; L_tau is windows (τ* − w_true)."""
    has_drift = w_true is not None
    detected = w_star is not None
    missed = has_drift and not detected
    det = None
    learn = None
    tau = None
    if t_true is not None and t_star is not None:
        det = float(t_star) - float(t_true)
    if t_true is not None and t_done is not None:
        learn = float(t_done) - float(t_true)
    if w_true is not None and tau_star is not None:
        tau = int(tau_star) - int(w_true)
    return {
        "w_true": w_true,
        "w_star": w_star,
        "tau_star": tau_star,
        "w_done": w_done,
        "t_true": t_true,
        "t_star": t_star,
        "t_done": t_done,
        "L_det_s": det,
        "L_learn_s": learn,
        "L_tau": tau,
        "missed": missed,
        "false_alarm": (not has_drift) and detected,
    }


def summarize_latency_records(records: list[dict]) -> dict[str, Any]:
    tp_recs = [
        r for r in records
        if r.get("w_true") is not None and r.get("w_star") is not None
    ]
    det = [r["L_det_s"] for r in tp_recs if r.get("L_det_s") is not None]
    learn = [r["L_learn_s"] for r in tp_recs if r.get("L_learn_s") is not None]
    tau = [r["L_tau"] for r in tp_recs if r.get("L_tau") is not None]
    n = len(records)
    cm = confusion_from_records(records)

    def _mean_std(vals: list[float]) -> tuple[Optional[float], Optional[float]]:
        if not vals:
            return None, None
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        return m, var ** 0.5

    det_m, det_s = _mean_std(det)
    learn_m, learn_s = _mean_std(learn)
    tau_m, tau_s = _mean_std([float(v) for v in tau])
    return {
        "n_traj": n,
        "n_miss": cm["fn"],
        "n_fired": cm["tp"] + cm["fp"],
        "confusion": cm,
        "L_det_s_mean": det_m,
        "L_det_s_std": det_s,
        "L_learn_s_mean": learn_m,
        "L_learn_s_std": learn_s,
        "L_tau_mean": tau_m,
        "L_tau_std": tau_s,
        "L_det_s": det,
        "L_learn_s": learn,
        "L_tau": tau,
    }

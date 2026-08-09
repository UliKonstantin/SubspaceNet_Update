"""Cross-window EKF handoff helpers for sliding-window online learning."""


def ekf_handoff_step_index(stride: int, prev_window_len: int) -> int:
    """
    Step index in the previous window whose posterior matches this window's step 0.

    Window w>0 starts at trajectory index w*stride. In the previous window (start
    (w-1)*stride), that index is step ``stride`` when stride < window_size, or the
    last step when stride >= window_size (gap or abutting windows).
    """
    if prev_window_len <= 0:
        raise ValueError(f"prev_window_len must be positive, got {prev_window_len}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    return min(stride, prev_window_len - 1)


def ekf_handoff_reuse_step0(stride: int, window_size: int) -> bool:
    """True when step 0 revisits a trajectory index already filtered in the previous window."""
    if stride <= 0 or window_size <= 0:
        raise ValueError(f"stride and window_size must be positive, got {stride}, {window_size}")
    return stride < window_size


def ekf_next_predict_time_index(global_step: int, handoff_reuse: bool) -> int:
    """
    Motion-model time index for the next predict after handoff restore.

    - Overlap reuse: posterior is at global_step; next predict is for global_step+1.
    - Sequential handoff: predict runs immediately at step 0 → use global_step.
    """
    return global_step + 1 if handoff_reuse else global_step

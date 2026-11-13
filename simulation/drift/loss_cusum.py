"""CUSUM + absolute threshold on unsupervised adaptation_loss.

Confirm EKF setup (EKF worse than SN for hump_confirm_windows), then freeze
(μ0, σ0) on a consecutive block of unsupervised L below loss_threshold.
Fire only if both S_w > h σ0 and L_w > loss_threshold.

A sample with L > loss_threshold is never stored in the null. If that happens
mid-ref with at least one clean sample, freeze and CUSUM this window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LossCusumDetector:
    settle_consecutive: int = 3
    hump_confirm_windows: int = 5
    ref_windows: int = 8
    kappa: float = 0.5
    h: float = 5.0
    loss_threshold: float = 0.05
    settle_fallback_window: int = 24

    saw_ekf_hump: bool = False
    hump_confirmed: bool = False
    hump_streak: int = 0
    seen_high_L: bool = False
    collecting: bool = False
    settled: bool = False
    settled_window: Optional[int] = None
    frozen: bool = False
    mu0: Optional[float] = None
    sigma0: Optional[float] = None
    S: float = 0.0
    ref_losses: list[float] = field(default_factory=list)
    L_windows: list[int] = field(default_factory=list)
    L_values: list[float] = field(default_factory=list)
    S_values: list[float] = field(default_factory=list)

    def step(
        self,
        window_idx: int,
        adaptation_loss: float,
        ekf_loss: float,
        pre_ekf_loss: float,
    ) -> dict:
        L = float(adaptation_loss)
        was_settled = self.settled
        was_frozen = self.frozen
        self.L_windows.append(int(window_idx))
        self.L_values.append(L)

        if not self.hump_confirmed:
            if ekf_loss > pre_ekf_loss:
                self.saw_ekf_hump = True
                self.hump_streak += 1
                if self.hump_streak >= self.hump_confirm_windows:
                    self.hump_confirmed = True
            else:
                self.hump_streak = 0
            if not self.hump_confirmed and window_idx >= self.settle_fallback_window:
                self.hump_confirmed = True

        if L > self.loss_threshold:
            self.seen_high_L = True

        ready = self.hump_confirmed and (
            self.seen_high_L or window_idx >= self.settle_fallback_window
        )

        if ready and not self.frozen:
            jump = L > self.loss_threshold
            if jump and len(self.ref_losses) >= 1:
                self._freeze()
            elif jump:
                self.ref_losses = []
                self.collecting = False
                self.S_values.append(self.S)
                return self._status(
                    window_idx,
                    L,
                    fired=False,
                    abs_ok=True,
                    just_settled=False,
                )
            else:
                if not self.collecting:
                    self.collecting = True
                    self.settled = True
                    self.settled_window = int(window_idx)
                    self.ref_losses = []
                self.ref_losses.append(L)
                self.S_values.append(self.S)
                if len(self.ref_losses) >= self.ref_windows:
                    self._freeze()
                    return self._status(
                        window_idx,
                        L,
                        fired=False,
                        just_settled=self.settled and not was_settled,
                        just_froze=self.frozen and not was_frozen,
                    )
                return self._status(
                    window_idx,
                    L,
                    fired=False,
                    just_settled=self.settled and not was_settled,
                )

        if not self.frozen:
            self.S_values.append(self.S)
            return self._status(
                window_idx,
                L,
                fired=False,
                just_settled=self.settled and not was_settled,
            )

        sigma = max(float(self.sigma0), 1e-12)
        increment = L - float(self.mu0) - self.kappa * sigma
        self.S = max(0.0, self.S + increment)
        self.S_values.append(self.S)
        abs_ok = L > self.loss_threshold
        cusum_ok = self.S > self.h * sigma
        fired = abs_ok and cusum_ok
        return self._status(
            window_idx,
            L,
            fired=fired,
            abs_ok=abs_ok,
            cusum_ok=cusum_ok,
            just_settled=self.settled and not was_settled,
            just_froze=self.frozen and not was_frozen,
        )

    def rearm(self) -> None:
        """Start a new null after adaptation. Skip EKF-hump confirm.

        Keeps L/S history for plots. Next freeze needs 8 consecutive L ≤ abs.
        """
        self.hump_confirmed = True
        self.saw_ekf_hump = True
        self.hump_streak = self.hump_confirm_windows
        self.seen_high_L = False
        self.collecting = False
        self.settled = False
        self.settled_window = None
        self.frozen = False
        self.mu0 = None
        self.sigma0 = None
        self.S = 0.0
        self.ref_losses = []

    def series(self) -> dict:
        return {
            "cusum_windows": list(self.L_windows),
            "cusum_L": list(self.L_values),
            "cusum_S": list(self.S_values),
            "cusum_mu0": self.mu0,
            "cusum_sigma0": self.sigma0,
            "cusum_settled_window": self.settled_window,
            "cusum_frozen": self.frozen,
            "cusum_loss_threshold": self.loss_threshold,
            "cusum_h": self.h,
            "cusum_kappa": self.kappa,
        }

    def _freeze(self) -> None:
        n = len(self.ref_losses)
        mean = sum(self.ref_losses) / n
        var = sum((x - mean) ** 2 for x in self.ref_losses) / max(n - 1, 1)
        self.mu0 = float(mean)
        self.sigma0 = max(float(var ** 0.5), 1e-4)
        self.frozen = True
        self.S = 0.0

    def _status(
        self,
        window_idx: int,
        L: float,
        fired: bool,
        abs_ok=False,
        cusum_ok=False,
        just_settled=False,
        just_froze=False,
    ) -> dict:
        return {
            "window_idx": window_idx,
            "L": L,
            "S": self.S,
            "mu0": self.mu0,
            "sigma0": self.sigma0,
            "settled": self.settled,
            "settled_window": self.settled_window,
            "frozen": self.frozen,
            "fired": fired,
            "abs_ok": abs_ok,
            "cusum_ok": cusum_ok,
            "just_settled": just_settled,
            "just_froze": just_froze,
        }

"""CLI routing types for the v2 command surface."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional


class Goal(str, Enum):
    TRAIN = "train"
    EVALUATE = "evaluate"
    ONLINE_LEARNING = "online_learning"
    FULL = "full"


class SweepType(str, Enum):
    NONE = "none"
    ONE_D = "1d"
    KALMAN_2D = "2d_kalman"
    GRID_4D = "4d_grid"


class SweepAxis(str, Enum):
    SNR = "snr"
    M = "m"
    T = "t"
    ETA = "eta"
    TRAJECTORY_LENGTH = "trajectory_length"

    @classmethod
    def from_string(cls, value: str) -> SweepAxis:
        normalized = value.lower()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(f"Unknown sweep axis: {value}")


@dataclass
class Grid4DParams:
    eta_values: List[float] = field(default_factory=list)
    process_noise_values: List[float] = field(default_factory=list)
    kf_process_noise_values: List[float] = field(default_factory=list)
    kf_measurement_noise_values: List[float] = field(default_factory=list)


@dataclass
class RunRequest:
    goal: Goal
    config_path: Path
    output_dir: Optional[Path] = None
    overrides: List[str] = field(default_factory=list)
    model_path: Optional[Path] = None
    trajectory: bool = False
    sweep: SweepType = SweepType.NONE
    sweep_axis: Optional[SweepAxis] = None
    sweep_values: Optional[List[float]] = None
    lr_sweep: bool = False
    retrain_per_sweep: bool = True
    grid_params: Optional[Grid4DParams] = None

    @property
    def is_sweep(self) -> bool:
        return self.sweep != SweepType.NONE


class RoutingError(ValueError):
    """Invalid CLI routing combination per decision-tree rules."""

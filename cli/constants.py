"""CLI routing constants."""
from cli.types import Goal

DEFAULT_SWEEP_VALUES = {
    "eta": [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
    "snr": [-10, -5, 0, 5, 10, 15, 20],
    "m": [1, 2, 3, 4, 5],
}

GOAL_OVERRIDES: dict[Goal, list[str]] = {
    Goal.TRAIN: [],
    Goal.EVALUATE: [
        "simulation.train_model=false",
        "simulation.load_model=true",
        "simulation.evaluate_model=true",
    ],
    Goal.ONLINE_LEARNING: [
        "simulation.train_model=false",
        "simulation.load_model=true",
        "simulation.evaluate_model=false",
        "online_learning.enabled=true",
    ],
    Goal.FULL: [],
}

DEFAULT_KALMAN_MEASUREMENT_NOISE = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
DEFAULT_KALMAN_PROCESS_NOISE = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]

DEFAULT_GRID_ETA = [0.0, 0.01, 0.015, 0.02, 0.03]
DEFAULT_GRID_PROCESS_NOISE = [0.001, 0.01, 0.1]
DEFAULT_GRID_KF_NOISE = [0.001, 0.01, 0.1]

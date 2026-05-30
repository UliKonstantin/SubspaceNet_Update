"""Common CLI options for SubspaceNet commands."""
import click
from click import option

config_option = option(
    "--config",
    "-c",
    default="configs/default_config.yaml",
    help="Path to the configuration file",
)

output_option = option(
    "--output",
    "-o",
    default=None,
    help="Output directory for experiment results",
)

override_option = option(
    "--override",
    "-O",
    multiple=True,
    help="Override configuration parameter (format: key=value)",
)

goal_option = option(
    "--goal",
    type=click.Choice(["train", "evaluate", "online_learning", "full"], case_sensitive=False),
    default=None,
    help="Primary intent: train, evaluate, online_learning, or full pipeline. "
    "Inferred from YAML if omitted.",
)

model_option = option(
    "--model",
    "-m",
    default=None,
    help="Path to trained model checkpoint (required for evaluate / online_learning)",
)

trajectory_option = option(
    "--trajectory/--no-trajectory",
    default=False,
    help="Enable trajectory-based data generation",
)

sweep_option = option(
    "--sweep",
    type=click.Choice(["none", "1d", "2d_kalman", "4d_grid"], case_sensitive=False),
    default=None,
    help="Parameter sweep type. Inferred from YAML scenario_config if omitted.",
)

axis_option = option(
    "--axis",
    "-s",
    default=None,
    help="1D sweep axis: snr, m, t, eta, trajectory_length",
)

values_option = option(
    "--values",
    "-v",
    multiple=True,
    type=float,
    help="Sweep values (repeat -v for each value)",
)

lr_sweep_option = option(
    "--lr-sweep",
    is_flag=True,
    default=False,
    help="Nested learning-rate sweep (online_learning + axis=eta only)",
)

retrain_option = option(
    "--retrain-per-sweep/--no-retrain-per-sweep",
    default=None,
    help="Retrain a separate model per sweep value (train sweeps). "
    "Defaults to scenario_config.retrain_model in YAML.",
)

grid_eta_option = option(
    "--eta-values",
    multiple=True,
    type=float,
    help="4D grid: eta values",
)

grid_process_noise_option = option(
    "--process-noise-values",
    multiple=True,
    type=float,
    help="4D grid: trajectory process noise values",
)

grid_kf_process_noise_option = option(
    "--kf-process-noise-values",
    multiple=True,
    type=float,
    help="4D grid: Kalman filter process noise values",
)

grid_kf_measurement_noise_option = option(
    "--kf-measurement-noise-values",
    multiple=True,
    type=float,
    help="4D grid: Kalman filter measurement noise values",
)

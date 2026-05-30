"""Centralized --help text for CLI v2 (use \\b blocks to prevent Click reflow)."""

CLI_GROUP_HELP = """\
SubspaceNet — DOA estimation with SubspaceNet, EKF tracking, and online learning.

\b
Entry point (v2, recommended for new runs):
  python3 main_v2.py run [OPTIONS]

Legacy entry point (still supported):
  python3 main.py simulate|evaluate|online_learning|run ...

\b
Quick start:
  python3 main_v2.py run -c configs/default_config.yaml --goal train
  python3 main_v2.py run --help
  python3 main_v2.py run -c configs/default_config.yaml --goal train -O training.epochs=5

Full CLI guide: docs/CLI.md
"""

RUN_SHORT_HELP = "Run training, evaluation, online learning, or parameter sweeps."

RUN_HELP = """\
Run a SubspaceNet experiment. YAML holds the experiment recipe; --goal selects
the pipeline; --sweep/--axis/-v select parameter sweeps.

\b
GOALS (--goal, or inferred from YAML):
  train             Supervised base-model training
  evaluate          Evaluate a fixed checkpoint (no weight updates)
  online_learning   Online learning on a trajectory stream
  full              Full pipeline: train → evaluate → online learning

\b
SWEEPS (--sweep, or inferred from scenario_config):
  none              Single run at config defaults
  1d                One axis: --axis snr|m|t|eta|trajectory_length
  2d_kalman         Kalman meas × proc noise grid (evaluate only)
  4d_grid           4D research grid (online_learning only)

\b
YAML overrides (when CLI is omitted):
  goal        simulation.* + online_learning.enabled
  sweep/axis  scenario_config.type or evaluation.sweep_parameter
  values      -v > scenario_config.values > evaluation.sweep_values > defaults
  model       -m > simulation.model_path > scenario_config.model_paths[0]

\b
Examples:
  python3 main_v2.py run -c configs/default_config.yaml --goal train
  python3 main_v2.py run -c configs/training_config/Random_basemodel_training_config.yaml --goal train --sweep 1d --axis snr -v -10 -v 0 -v 10
  python3 main_v2.py run -c configs/evaluation_configs/snr_sweep_config.yaml --goal evaluate -m path/to/model.pt --sweep 1d --axis snr
  python3 main_v2.py run -c configs/Used_for_paper/SineAccel_base_model_Online_learning_eta_sweep_config.yaml --goal online_learning --trajectory --sweep 1d --axis eta --lr-sweep

\b
Legacy mapping:
  main.py run              →  main_v2.py run --goal train|evaluate|online_learning|full
  main.py simulate         →  main_v2.py run --goal train|online_learning [--sweep 1d ...]
  main.py evaluate         →  main_v2.py run --goal evaluate -m MODEL [--sweep ...]
  main.py online_learning  →  main_v2.py run --goal online_learning --trajectory [-m MODEL]

See docs/CLI.md for paper experiment recipes.
"""

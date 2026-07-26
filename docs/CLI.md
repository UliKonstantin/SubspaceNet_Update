# SubspaceNet CLI v2 (`main_v2.py`)

Parallel entry point to [`main.py`](../main.py). Old commands remain until cutover.

**Quick help:** `python3 main_v2.py --help` and `python3 main_v2.py run --help`  
**Project overview:** [README.md](../README.md)

## Decision tree

```
What is the primary goal?
├─ A) train          → --goal train
├─ B) evaluate       → --goal evaluate  (-m MODEL required)
├─ C) online_learning → --goal online_learning  (-m MODEL, --trajectory required)
└─ D) full           → --goal full  (no sweeps)
```

Sweep types (`--sweep`, inferred from `scenario_config.type` when omitted):

| `--sweep` | Valid goals | Notes |
|-----------|-------------|-------|
| `none` | all except full+sweep | Single run |
| `1d` | train, evaluate, online_learning | Requires `--axis` + values |
| `2d_kalman` | evaluate only | Kalman meas × proc noise grid |
| `4d_grid` | online_learning only | Research grid |

## YAML interaction

- **Goal**: `--goal` CLI → else infer from `simulation.*` / `online_learning.enabled` → else error
- **Sweep axis**: `--axis` → `scenario_config.type` → `evaluation.sweep_parameter`
- **Sweep values**: `-v` CLI → `scenario_config.values` → `evaluation.sweep_values` → axis defaults
- **Retrain per sweep**: `--retrain-per-sweep/--no-retrain-per-sweep` → else `scenario_config.retrain_model` → else `false` (matches legacy `run_scenario`)
- **Trajectory**: `--trajectory` adds override (legacy simulate behavior); YAML `trajectory.enabled` alone is enough for training without extra override

## Legacy command mapping

| Old (`main.py`) | New (`main_v2.py`) |
|-----------------|-------------------|
| `run -c CFG` | `run -c CFG --goal train` |
| `run --scenario evaluation` | `run -c CFG --goal evaluate -m MODEL` |
| `simulate -c CFG` | `run -c CFG --goal train` |
| `simulate --mode online_learning` | `run -c CFG --goal online_learning --trajectory` |
| `simulate -s snr -v ...` | `run -c CFG --goal train --sweep 1d --axis snr -v ...` |
| `evaluate -m MODEL -s snr` | `run -c CFG --goal evaluate -m MODEL --sweep 1d --axis snr` |
| `evaluate -s kalman_noise` | `run -c CFG --goal evaluate -m MODEL --sweep 2d_kalman` |
| `online_learning -c CFG` | `run -c CFG --goal online_learning --trajectory -m MODEL` |
| `online_learning -s 4d_grid` | `run -c CFG --goal online_learning --trajectory --sweep 4d_grid` |

## Recipes (from launch.json / paper configs)

### Train base model

```bash
python main_v2.py run \
  -c configs/Used_for_paper/Random_basemodel_training_config.yaml \
  --goal train
```

### Train SNR sweep

```bash
python main_v2.py run \
  -c configs/Used_for_paper/Random_base_model_training_snr_scenario_config.yaml \
  -o experiments/results/snr_training_sweep \
  --goal train --sweep 1d --axis snr \
  -v -10 -v -5 -v 0 -v 5 -v 10
```

Legacy equivalent:

```bash
python main.py simulate \
  -c configs/Used_for_paper/Random_base_model_training_snr_scenario_config.yaml \
  -o experiments/results/snr_training_sweep \
  -s snr -v -10 -v -5 -v 0 -v 5 -v 10 \
  --mode training
```

### Evaluate SNR sweep

```bash
python main_v2.py run \
  -c configs/evaluation_configs/snr_sweep_config.yaml \
  --goal evaluate -m path/to/checkpoint.pt \
  --sweep 1d --axis snr
```

### Evaluate calibration (eta) sweep

```bash
python main_v2.py run \
  -c configs/evaluation_configs/calibration_sweep_error_config.yaml \
  --goal evaluate -m path/to/checkpoint.pt \
  --sweep 1d --axis eta
```

### Evaluate Kalman noise 2D sweep

```bash
python main_v2.py run \
  -c configs/evaluation_configs/default_eval_config.yaml \
  --goal evaluate -m path/to/checkpoint.pt \
  --sweep 2d_kalman
```

### Online learning SNR sweep

```bash
python main_v2.py run \
  -c configs/Used_for_paper/SineAccel_base_model_Online_learning_snr_sweep_config.yaml \
  --goal online_learning --trajectory \
  --sweep 1d --axis snr
```

### Online learning eta sweep (+ LR heatmap)

```bash
python main_v2.py run \
  -c configs/Used_for_paper/SineAccel_base_model_Online_learning_eta_sweep_config.yaml \
  --goal online_learning --trajectory \
  --sweep 1d --axis eta --lr-sweep
```

### Online learning 4D grid

```bash
python main_v2.py run \
  -c configs/online_learning_config.yaml \
  --goal online_learning --trajectory \
  --sweep 4d_grid
```

## Other commands

```bash
python main_v2.py show -c configs/default_config.yaml
python main_v2.py save -c configs/default_config.yaml -o out.yaml
```

## Architecture

```
RunRequest → validate → build_overrides → setup_configuration → Simulation → dispatch → postprocess
```

See also: [`docs/cli_routing_spec.md`](cli_routing_spec.md), [`docs/cli_flow_diagram.md`](cli_flow_diagram.md).

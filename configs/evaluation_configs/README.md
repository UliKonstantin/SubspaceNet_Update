# Evaluation Configurations

This directory contains configuration files specifically designed for model evaluation.

## Configuration Files

- `default_eval_config.yaml`: Default configuration for evaluation only, with settings optimized for thorough model assessment.
- `snr_sweep_config.yaml`: Configuration for evaluating model performance across a range of SNR values.
- `calibration_error_config.yaml`: Configuration for evaluating model robustness to calibration errors.
- `source_count_config.yaml`: Configuration for evaluating model performance with different numbers of sources.

## Key Differences from Standard Configurations

Evaluation configurations have the following key differences:

1. **Dataset Settings**:
   - Optimized for evaluation with 100% test split `[1.0, 0.0, 0.0]`
   - Larger sample size for more thorough evaluation

2. **Training Settings**:
   - Training is disabled (`training.enabled: false`)
   - No model saving (`simulation.save_model: false`)

3. **Evaluation Settings**:
   - Added additional evaluation-specific parameters
   - Includes multiple subspace methods for comparison
   - Parameter sweep configurations for comparative analysis

4. **Trajectory Settings**:
   - Configured for longer trajectories
   - Optimized Kalman filter parameters for evaluation

## Usage

### Basic Evaluation

To evaluate a model with a specific configuration:

```bash
python main.py evaluate -c configs/evaluation_configs/default_eval_config.yaml -m <path_to_model>
```

or 

```bash
python main.py run -c configs/evaluation_configs/default_eval_config.yaml -s evaluation
```

### Parameter Sweeps

For parameter sweep evaluations, you can use the `--scenario` and `--values` options:

```bash
python main.py evaluate -c configs/evaluation_configs/snr_sweep_config.yaml -m <path_to_model> --scenario snr --values -10 --values -5 --values 0 --values 5 --values 10 --values 15 --values 20
```

Alternatively, you can specify the sweep parameters in the configuration file (under `evaluation.sweep_parameter` and `evaluation.sweep_values`) and run:

```bash
python main.py evaluate -c configs/evaluation_configs/snr_sweep_config.yaml -m <path_to_model> --scenario snr
```

This will automatically evaluate the model across all specified parameter values and generate comparative performance plots.

### Example Sweep Configurations

1. **SNR Sweep**:
   ```bash
   python main.py evaluate -c configs/evaluation_configs/snr_sweep_config.yaml -m <path_to_model> --scenario snr
   ```

2. **Calibration Error Sweep**:
   ```bash
   python main.py evaluate -c configs/evaluation_configs/calibration_error_config.yaml -m <path_to_model> --scenario eta
   ```

3. **Source Count Sweep**:
   ```bash
   python main.py evaluate -c configs/evaluation_configs/source_count_config.yaml -m <path_to_model> --scenario M
   ```

## Creating Custom Evaluation Configurations

To create a custom evaluation configuration:

1. Copy one of the existing configurations
2. Modify the parameters as needed
3. Ensure `training.enabled` is set to `false` and `simulation.evaluate_model` is set to `true`
4. If adding a parameter sweep, define `evaluation.sweep_parameter` and `evaluation.sweep_values` in the config (optional) 
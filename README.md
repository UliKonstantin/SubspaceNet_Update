# SubspaceNet Configuration Framework

A simplified configuration framework for working with the DCD-MUSIC direction-of-arrival (DOA) estimation library.

## Overview

This framework provides a streamlined way to configure and run experiments with the DCD-MUSIC library. It offers:

- A centralized configuration system based on YAML files
- Type-safe validation using Pydantic models
- Command-line interface for running experiments with parameter overrides
- Experiment management with result tracking
- Visualization and evaluation tools

## Installation

1. Clone this repository and the DCD-MUSIC submodule:
```bash
git clone --recursive [repository-url]
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running an Experiment

To run an experiment with the default configuration:

```bash
python main.py run
```

To use a specific configuration file:

```bash
python main.py run --config configs/my_experiment.yaml
```

To override configuration parameters:

```bash
python main.py run --override system_model.snr=15 training.learning_rate=0.001
```

To specify an output directory:

```bash
python main.py run --output experiments/results/my_experiment
```

### Displaying Configuration

To display the current configuration:

```bash
python main.py show
```

With overrides:

```bash
python main.py show --override system_model.snr=15
```

### Saving Configuration

To save a configuration to a new file:

```bash
python main.py save --output configs/new_config.yaml
```

## Configuration Structure

The configuration is organized into the following sections:

- `system_model`: Settings for the antenna array and signal model
- `dataset`: Data generation parameters
- `model`: Model type and specific parameters
- `training`: Training settings including optimizer and learning rate
- `simulation`: Simulation parameters
- `evaluation`: Evaluation metrics and visualization settings

Example:

```yaml
system_model:
  N: 127   # Number of antennas
  M: 2     # Number of sources
  T: 100   # Number of snapshots
  snr: 10  # Signal-to-noise ratio (dB)
  array_type: "ULA"
  
dataset:
  dataset_type: "src.data"
  dataset_class: "DOADataset"
  num_samples: 1000
  
# ... other sections
```

## Directory Structure

- `configs/`: Configuration YAML files
- `config/`: Configuration framework Python modules
- `experiments/`: Experiment execution and results
- `DCD_MUSIC/`: The DCD-MUSIC library (submodule)

## Extending the Framework

### Adding New Models

To add support for a new model:

1. Update the `config/schema.py` file to include the new model's parameters
2. Modify the `create_model` function in `config/factory.py`

### Adding New Metrics

To add new evaluation metrics:

1. Update the `_calculate_metrics` method in `experiments/runner.py`

## License

[Add license information here] 
# SubspaceNet Configuration Framework

## Overview

This repository is a wrapper around the [DCD-MUSIC](https://github.com/Sensing-Intelligent-System/DCD-MUSIC) project, providing a simplified configuration framework for running DOA (Direction of Arrival) and range estimation experiments.

The goal of this framework is to maintain all the powerful components of the original DCD-MUSIC project but make it more accessible and easier to configure for specific needs. Rather than modifying the source code directly, this wrapper provides a layer of abstraction that allows for simple configuration via YAML files.

## Features

- **Clean Configuration System**: Centralized configuration management using a single YAML file
- **Type Validation**: Automatic validation of configuration parameters using Pydantic models
- **Command-Line Interface**: Simple CLI for running experiments, generating configuration files, and more
- **Experiment Management**: Automatic tracking of experiment results and reproducibility
- **Interoperability**: Full compatibility with the original DCD-MUSIC codebase

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- All requirements from the original DCD-MUSIC project

### Installation

1. Clone this repository and the DCD-MUSIC submodule:

```bash
git clone --recurse-submodules https://github.com/your-username/SubspaceNet_Update.git
cd SubspaceNet_Update
```

2. Install the requirements:

```bash
pip install -r requirements.txt
```

## Working with Configurations

The framework uses a centralized configuration system based on YAML files. All experiment parameters are defined in these configuration files.

### Default Configuration

A default configuration file exists at `config/defaults/default_config.yaml`. This file contains sensible defaults for all parameters and is used as a template for new configurations.

To view the default configuration:

```bash
python main.py show-default
```

### Creating a New Configuration

To create a new configuration based on the defaults:

```bash
python main.py init -o my_config.yaml
```

This will generate a new configuration file at the specified path, which you can then edit to customize your experiment.

### Modifying the Default Configuration

You can modify the default configuration in two ways:

1. **Edit the file directly**: Open `config/defaults/default_config.yaml` in your text editor

2. **Use the CLI**:

```bash
# Set a specific parameter
python main.py set-default system_model.N 64
python main.py set-default training.epochs 100
python main.py set-default system_model.snr 15.0

# Update the entire default configuration from another file
python main.py update-defaults my_custom_config.yaml
```

### Resetting the Default Configuration

If you want to reset the default configuration to factory settings:

```bash
python main.py default-location --reset
```

## Running Experiments

### Basic Usage

To run an experiment with a configuration file:

```bash
python main.py run my_config.yaml
```

### Overriding Parameters

You can override specific parameters directly from the command line:

```bash
python main.py run my_config.yaml -o system_model.snr=20 -o training.epochs=100
```

This allows for quick parameter sweeps without creating multiple configuration files.

## Complete CLI Reference

The framework provides a comprehensive command-line interface:

```bash
# Show all available commands
python main.py --help

# View the default configuration
python main.py show-default

# See where the default configuration is stored
python main.py default-location

# Create a new configuration file from defaults
python main.py init -o my_config.yaml

# View a configuration file
python main.py show my_config.yaml

# View just one section of a configuration file
python main.py show my_config.yaml -s system_model

# Validate a configuration file
python main.py validate my_config.yaml

# Convert a configuration file to another format
python main.py convert my_config.yaml -f json -o my_config.json

# Set a value in the default configuration
python main.py set-default parameter.path value

# Update the default configuration from a file
python main.py update-defaults source_config.yaml

# Run an experiment
python main.py run my_config.yaml
```

## Configuration Structure

The configuration system is designed to be hierarchical and type-safe, organized into the following main sections:

### Main Configuration Sections

- **system_model**: Parameters for the system model (antennas, sources, SNR, etc.)
- **dataset**: Parameters for dataset generation and loading
- **model**: Parameters for model selection and configuration
- **training**: Parameters for model training
- **simulation_commands**: Control which parts of the experiment to run
- **evaluation**: Parameters for model evaluation

### Example: Configuring a Typical Experiment

Here's a typical workflow for setting up and running an experiment:

1. **Create a configuration file**:
   ```bash
   python main.py init -o experiments/my_experiment.yaml
   ```

2. **Edit the configuration file** to set up your experiment:
   ```yaml
   # In experiments/my_experiment.yaml
   system_model:
     N: 64                    # Use 64 antennas
     snr: 15.0                # Set SNR to 15 dB
   
   training:
     epochs: 100              # Train for 100 epochs
     batch_size: 64           # Use batch size of 64
   
   simulation_commands:
     train_model: true        # Enable model training
     save_model: true         # Save the trained model
   ```

3. **Run the experiment**:
   ```bash
   python main.py run experiments/my_experiment.yaml
   ```

4. **Try a parameter variation**:
   ```bash
   python main.py run experiments/my_experiment.yaml -o system_model.snr=20
   ```

## License

This project is licensed under the  License - see the LICENSE file for details.

## Acknowledgments

This project is a wrapper around the [DCD-MUSIC](https://github.com/Sensing-Intelligent-System/DCD-MUSIC) project, which provides the underlying algorithms and models for DOA and range estimation. 
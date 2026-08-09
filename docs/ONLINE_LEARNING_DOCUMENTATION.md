# Online Learning Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture and Flow](#architecture-and-flow)
3. [Configuration Options](#configuration-options)
4. [Code Flow and File Responsibilities](#code-flow-and-file-responsibilities)
5. [GLRT Drift Detection](#glrt-drift-detection)
6. [Adaptive Learning Rate](#adaptive-learning-rate)
7. [Dynamic Eta Updates](#dynamic-eta-updates)
8. [Examples](#examples)

---

## Overview

Online Learning is an adaptive training mechanism that allows pre-trained models to continuously adapt to changing conditions during inference. The system monitors model performance using a Generalized Likelihood Ratio Test (GLRT) for statistical drift detection, and automatically triggers retraining when performance degradation is detected.

### Key Features
- **Statistical Drift Detection**: Uses GLRT (Generalized Likelihood Ratio Test) with z-score normalization for relative, adaptive drift detection
- **Adaptive Learning Rate**: Optional adaptive learning rate that scales with the magnitude of detected changes
- **Dynamic Parameter Updates**: Supports dynamic updates to system parameters (e.g., `eta` for calibration errors)
- **Multi-Trajectory Support**: Runs multiple trajectories and averages results for robust performance evaluation
- **Dual Model Training**: Simultaneously trains online and supervised models for comparison
- **Sliding Window Processing**: Processes trajectory data in overlapping windows for sequential learning

---

## Architecture and Flow

### High-Level Flow

```
1. Configuration Loading (YAML → Config Object)
   ↓
2. Model Loading (Pre-trained model from checkpoint)
   ↓
3. Trajectory Generation (Time-series data with source movements)
   ↓
4. Window Processing Loop:
   ├─ Data Generation (Current eta/parameters)
   ├─ Model Inference (Pre-EKF predictions)
   ├─ EKF Filtering (State estimation)
   ├─ Loss Calculation (Performance metrics)
   ├─ GLRT Analysis (Drift detection)
   └─ Conditional Training (If drift detected)
   ↓
5. Results Aggregation (Average across trajectories)
   ↓
6. Plotting and Analysis (Visualization of results)
```

### Detailed Processing Flow

#### Per-Trajectory Flow:
1. **Initialization**: Copy pre-trained model for online training
2. **Window Iteration**: For each sliding window:
   - Generate trajectory data with current system parameters
   - Run model inference to get DOA predictions
   - Apply Extended Kalman Filter (EKF) for state estimation
   - Calculate loss metrics (pre-EKF, post-EKF)
   - Update GLRT history and compute z-score
   - **If GLRT z-score > threshold**: Mark drift detected
   - **If drift detected + delay elapsed**: Start online training
   - **During training**: Update model weights with gradient descent
   - **After training**: Continue evaluation with updated model
3. **Results Collection**: Aggregate window-level metrics

#### Multi-Trajectory Aggregation:
- Run multiple trajectories with different random seeds
- Average loss metrics, covariances, and performance indicators
- Generate comparative plots (pretrained vs. online vs. supervised)

---

## Configuration Options

All configuration options are defined in the YAML config file under the `online_learning:` section.

### Core Parameters

#### `enabled: bool` (default: `false`)
Enable or disable online learning functionality.

#### `window_size: int` (default: `10`)
Size of the sliding window in number of trajectory steps. Each window contains `window_size` consecutive time steps of the trajectory.

#### `stride: int` (default: `5`)
Step size between consecutive windows. Controls window overlap:
- `stride = window_size`: No overlap
- `stride < window_size`: Overlapping windows
- `stride = 1`: Maximum overlap (dense windowing)

**Example**: With `window_size=5` and `stride=3`, windows start at indices: [0, 3, 6, 9, ...]

#### `trajectory_length: int` (default: `1000`)
Total number of trajectory steps to generate for the online learning session.

**Number of windows** = `(trajectory_length - window_size) // stride + 1`

#### `dataset_size: int` (default: `1`)
Number of trajectories to run and average results over. Each trajectory uses different random seeds for robustness.

#### `learning_rate: float` (default: `1e-4`)
Base learning rate for online training. Used as-is if `use_adaptive_learning_rate=false`, or as base if adaptive LR is enabled.

### Loss and Threshold Parameters

#### `loss_threshold: float` (default: `0.5`)
⚠️ **Deprecated**: This threshold is no longer used for drift detection (replaced by GLRT z-score). Kept for backward compatibility.

#### `max_iterations: int` (default: `10`)
Maximum number of gradient descent iterations per window when training is triggered.

### Drift Detection Configuration

#### `time_to_learn: int | null` (default: `null`)
Delay in windows between GLRT drift detection and actual training start. If `null`, training starts immediately when drift is detected.

**Behavior**:
- GLRT detects drift at window `N` with z-score > threshold
- Training starts at window `N + time_to_learn`
- If `time_to_learn=10` and drift detected at window 15, training starts at window 25

#### `drift_z_threshold: float` (default: `2.5`)
Z-score threshold for drift detection. Drift is detected when:
```
current_GLRT_z_score > drift_z_threshold
```

**Statistical Meaning**: z-score of 2.5 corresponds to ~99.4% confidence that the value is significantly above baseline.

#### `drift_warmup_windows: int` (default: `7`)
Two roles:
1. **Scope A:** Skip the first N trajectory windows entirely (no changepoint GLRT, no g-history).
2. **Scope B:** Require at least N g-scalars in the baseline slice before z-score is computed.

First g at window `W + 2*GLRT_MIN_SEGMENT_SIZE`; first z at `2*W + 2*m + drift_guard_samples - 1` (internal `m=5`).

#### `drift_guard_samples: int` (default: `3`)
Exclude the last N log-GLR (g) values from the baseline when computing z-score (onset buffer).

#### `drift_history_max_size: int | null` (default: `null`)
Optional cap on rolling g-history length. `null` = unbounded.

### Adaptive Learning Rate Configuration

#### `use_adaptive_learning_rate: bool` (default: `false`)
Enable adaptive learning rate that scales with GLRT change magnitude.

**When enabled**:
- Learning rate = `base_lr * (1 + multiplier)`
- `multiplier = tanh(z_score / 3.0) * 2.0` (capped at 3x base LR)
- Larger detected changes → higher learning rate

**When disabled**:
- Uses fixed `learning_rate` for all training

### Dynamic Eta Update Parameters

#### `eta_update_interval_windows: int | null` (default: `null`)
Update `system_model.eta` (calibration error parameter) every N windows. If `null` or `0`, eta is not periodically updated.

**Update Logic**:
- At windows: `interval`, `2*interval`, `3*interval`, ...
- New eta = `current_eta + eta_increment`
- Clamped between `min_eta` and `max_eta`

#### `eta_increment: float` (default: `0.01`)
Amount to increment/decrement eta when an update occurs. Can be positive (increasing) or negative (decreasing).

#### `max_eta: float` (default: `0.5`)
Maximum allowed value for eta during dynamic updates.

#### `min_eta: float` (default: `0.0`)
Minimum allowed value for eta during dynamic updates.

#### `use_nominal: bool` (default: `true`)
If `true`, use nominal array configuration (no calibration errors) for sample generation. If `false`, apply calibration errors based on current `eta` value.

### Loss Configuration

#### `loss_config: dict`
Nested configuration for loss function selection:

```yaml
loss_config:
  metric: "rmspe" | "rmape"  # Loss metric for evaluation
  supervision: "supervised" | "unsupervised"  # Compare with ground truth or pre-EKF
  training_loss_type: "unsupervised_rmspe"  # Loss for online training
  supervised_loss_type: "supervised_rmspe"  # Loss for supervised model training
```

**Available Loss Types**:
- `"configured"`: Use `metric + supervision` combination
- `"kalman_innovation"`: Kalman gain times innovation
- `"y_s_inv_y"`: y^T * S^-1 * y (Mahalanobis distance)
- `"unsupervised_rmspe"`: RMSPE between EKF and pre-EKF predictions
- `"unsupervised_rmape"`: RMAPE between EKF and pre-EKF predictions
- `"supervised_rmspe"`: RMSPE between EKF and ground truth
- `"supervised_rmape"`: RMAPE between EKF and ground truth
- `"multimoment"`: Multi-Moment Innovation Consistency Loss

---

## Code Flow and File Responsibilities

### Entry Points

#### 1. Command Line Interface (`main.py`)
- **Entry Points**:
  - `python main.py run --scenario online_learning`
  - `python main.py simulate --mode online_learning`
  - `python main.py online_learning`

- **Responsibilities**:
  - Parse command-line arguments
  - Load configuration using `config_handler`
  - Create `Simulation` instance
  - Call `sim.execute_online_learning()`

**Key Functions**:
- `run_command()`: General run command handler
- `simulate_command()`: Scenario-based simulation
- `online_learning_command()`: Dedicated online learning command

#### 2. Configuration System

##### `config_handler.py`
- **Responsibilities**:
  - Orchestrates configuration loading
  - Sets up system components (model, data handler, etc.)
  - Returns config object and components

**Key Function**: `setup_configuration(config_path, output_dir, overrides)`

##### `config/loader.py`
- **Responsibilities**:
  - Loads YAML files
  - Validates against Pydantic schema
  - Applies command-line overrides

**Key Functions**:
- `load_config(config_file_path)`: Load and validate YAML
- `apply_overrides(config, overrides)`: Apply parameter overrides

##### `config/schema.py`
- **Responsibilities**:
  - Defines Pydantic models for all configuration sections
  - Validates configuration structure and types
  - Provides default values

**Key Class**: `OnlineLearningConfig(BaseModel)`
- Contains all online learning parameters with types and defaults

##### `config/factory.py`
- **Responsibilities**:
  - Creates model instances based on config
  - Handles model architecture selection
  - Initializes model with proper parameters

### Core Simulation

#### `simulation/core.py`

##### Class: `Simulation`
- **Responsibilities**:
  - Main simulation orchestrator
  - Coordinates training, evaluation, and online learning
  - Manages model loading and saving

**Key Method**: `execute_online_learning()`
```python
def execute_online_learning(self) -> Dict[str, Any]:
    # 1. Load pre-trained model from path
    # 2. Verify model availability
    # 3. Create OnlineLearning handler
    # 4. Run online learning pipeline
    # 5. Save results
```

**Flow**:
1. Checks `config.simulation.load_model` flag
2. Loads model weights from `config.simulation.model_path`
3. Creates `OnlineLearning` instance
4. Calls `online_learning_handler.run_online_learning()`
5. Saves results to output directory

### Online Learning Implementation

#### `simulation/runners/Online_learning.py`

##### Class: `OnlineLearning`
Main handler for all online learning functionality.

**Initialization (`__init__`)**:
- Stores config, system_model, trained_model
- Initializes state variables (drift_detected, learning_done, etc.)
- Reads GLRT detection parameters from config
- Sets up adaptive learning rate flag

**Main Method**: `run_online_learning()`
```python
def run_online_learning(self) -> Dict[str, Any]:
    # 1. Run online learning for each trajectory
    # 2. Aggregate results across trajectories
    # 3. Generate plots
    # 4. Return aggregated results
```

**Flow**:
1. Iterates `dataset_size` times (multiple trajectories)
2. For each trajectory:
   - Calls `_run_single_trajectory_online_learning()`
3. Aggregates results using `_aggregate_results()`
4. Generates plots using plotting utilities
5. Returns dictionary with all metrics

**Key Method**: `_run_single_trajectory_online_learning()`
```python
def _run_single_trajectory_online_learning(trajectory_idx: int) -> Dict[str, Any]:
    # 1. Create online and supervised model copies
    # 2. Create data loader with sliding windows
    # 3. Process each window:
    #    - Evaluate with pretrained model
    #    - Run GLRT analysis
    #    - Detect drift (z-score > threshold)
    #    - Trigger training after delay
    #    - Train online/supervised models
    # 4. Return trajectory results
```

**Per-Window Processing**:
1. **Data Generation**: Uses `OnlineLearningDataLoader` (from `data.py`)
2. **Model Evaluation**: Calls `_evaluate_window()` for pretrained model
3. **GLRT Calculation**: Computes GLRT on loss history, updates z-score
4. **Drift Detection**: Checks if z-score > threshold
5. **Training Trigger**: Starts training after `time_to_learn` delay
6. **Training**: Calls `_online_training_window()` for both models

**Key Method**: `_evaluate_window()`
```python
def _evaluate_window(...) -> WindowEvaluationResult:
    # 1. Process each step in window
    # 2. Run model inference (DOA predictions)
    # 3. Apply EKF filtering
    # 4. Calculate loss metrics
    # 5. Aggregate step-level results
    # 6. Return WindowEvaluationResult
```

**Key Method**: `_online_training_window()`
```python
def _online_training_window(..., glrt_z_score: float) -> WindowEvaluationResult:
    # 1. Calculate adaptive learning rate (if enabled)
    # 2. Set up optimizer with learning rate
    # 3. For each GD iteration:
    #    - Forward pass through model
    #    - EKF processing (with gradients)
    #    - Calculate window-level loss
    #    - Backpropagate and update weights
    # 4. Evaluate trained model
    # 5. Return WindowEvaluationResult
```

**GLRT Processing**:
- Maintains `glrt_history` list (rolling window)
- Computes baseline mean/std from history
- Calculates z-score: `(current_GLRT - mean) / std`
- Updates detection window when threshold exceeded

### Data Generation

#### `simulation/runners/data.py`

##### Function: `create_online_learning_dataset()`
- **Responsibilities**:
  - Creates on-demand dataset for online learning
  - Generates trajectory data with current system parameters
  - Implements sliding window logic

**Returns**: `OnlineLearningDataset` instance

##### Class: `OnlineLearningDataset`
- **Responsibilities**:
  - Generates trajectory data on-the-fly
  - Implements sliding windows over trajectory
  - Supports dynamic parameter updates (e.g., `update_eta()`)

**Key Method**: `update_eta(new_eta)`
- Updates `system_model.params.eta` for subsequent data generation
- Allows dynamic calibration error changes during online learning

**Key Method**: `__getitem__(window_idx)`
- Generates data for specific window index
- Returns: `(time_series, sources_num, labels)`

### Kalman Filtering

#### `simulation/kalman_filter/extended.py`

##### Class: `ExtendedKalmanFilter1D`
- **Responsibilities**:
  - 1D Extended Kalman Filter for DOA tracking
  - State estimation with nonlinear dynamics
  - Supports tensor inputs/outputs for gradient computation

**Key Method**: `predict_and_update(measurement, true_state)`
- Predicts next state based on trajectory model
- Updates state with measurement (model predictions)
- Returns: `(predicted_state, updated_state, covariance, kalman_gain, ...)`

**Note**: Modified to support gradient computation during training (tensor operations).

### Loss Functions

#### `simulation/losses/kalman_loss.py` and `DCD_MUSIC/src/metrics/`

**Available Loss Functions**:
- `RMSPELoss`: Root Mean Square Percentage Error
- `RMAPELoss`: Root Mean Absolute Percentage Error
- `MultiMomentInnovationConsistencyLoss`: Multi-moment consistency loss

### Plotting and Visualization

#### `utils/plotting.py`

**Key Functions**:
- `plot_online_learning_results()`: Per-trajectory plots
- `plot_averaged_online_learning_results()`: Multi-trajectory averaged plots
- `plot_glrt_results()`: GLRT changepoint detection plots

**Generated Plots**:
1. **Loss Comparison**: Pretrained vs. Online vs. Supervised
2. **Covariance Evolution**: EKF uncertainty over windows
3. **GLRT Analysis**: Changepoint detection visualizations
4. **Eta Evolution**: Dynamic parameter changes over time

### GLRT Implementation

#### `simulation/runners/sandbox.py`

##### Function: `glrt_changepoint_detection(losses, min_segment_size=5)`
- **Responsibilities**:
  - Implements Generalized Likelihood Ratio Test
  - Detects changepoints in time series of losses
  - Returns changepoint index and log-GLR value

**Algorithm**:
1. Compute log-likelihood under H0 (no change)
2. For each candidate changepoint:
   - Compute log-likelihood under H1 (change at this point)
   - Calculate log-GLR = log_L1 - log_L0
3. Return changepoint with maximum log-GLR

---

## GLRT Drift Detection

### Overview

GLRT (Generalized Likelihood Ratio Test) drift detection uses statistical hypothesis testing to detect significant changes in model performance. The system maintains a rolling baseline of GLRT values and detects drift when current performance deviates significantly from baseline.

### Algorithm

1. **Scope A (changepoint GLRT, each window k ≥ W)**:
   - Build post-warmup loss prefix `[L_W … L_k]`
   - Compute max log-GLR over valid segment splits (`GLRT_MIN_SEGMENT_SIZE=5` internal)
   - Append one g-scalar to `glrt_history`

2. **Baseline Estimation (Scope B)**:
   - `baseline = glrt_history[:-drift_guard_samples]` (all g except tail guard)
   - Require `len(baseline) >= drift_warmup_windows` before z-score
   - Compute mean (`μ`) and standard deviation (`σ`)

3. **Z-Score Calculation**:
   ```
   z_score = (current_g - μ) / σ
   ```

4. **Drift Detection**:
   ```
   if z_score > drift_z_threshold:
       drift_detected = True
       record_detection_window(current_window)
   ```

5. **Training Trigger** (with delay):
   ```
   if drift_detected and current_window >= (detection_window + time_to_learn):
       start_training()
   ```

### Advantages Over Fixed Threshold

- **Relative Detection**: Adapts to different loss scales automatically
- **Statistical Significance**: Z-score provides confidence level
- **Robust to Noise**: Warmup + guard samples reduce false positives
- **Clear Parameters**: Three user-facing drift knobs plus internal segment size

### Configuration

```yaml
online_learning:
  drift_warmup_windows: 7   # Scope A skip + min baseline g-count
  drift_guard_samples: 3    # Tail excluded from baseline
  drift_z_threshold: 2.5      # ~99.4% confidence
  time_to_learn: 10           # Delay after detection
```

---

## Adaptive Learning Rate

### Overview

When `use_adaptive_learning_rate=true`, the learning rate dynamically adjusts based on the magnitude of detected change (GLRT z-score). Larger changes result in higher learning rates, allowing faster adaptation to significant drift.

### Formula

```
if use_adaptive_learning_rate:
    z_normalized = clip(z_score / 3.0, -10, 10)
    multiplier_offset = tanh(z_normalized) * 2.0
    multiplier_offset = max(0, multiplier_offset)  # Only positive
    adaptive_lr = base_lr * (1 + multiplier_offset)
else:
    adaptive_lr = base_lr  # Fixed learning rate
```

### Characteristics

- **Bounded**: Maximum 3x base learning rate (when z-score → +∞)
- **Smooth**: Tanh function provides smooth transitions
- **Relative**: Scales with detected change magnitude
- **Safe**: Negative z-scores (below baseline) use base LR

### Example

```yaml
online_learning:
  learning_rate: 0.001  # Base LR
  use_adaptive_learning_rate: true
```

**Behavior**:
- z-score = 0.0 → LR = 0.001 (1x)
- z-score = 2.5 → LR ≈ 0.0015 (1.5x)
- z-score = 5.0 → LR ≈ 0.002 (2x)
- z-score = 10.0 → LR ≈ 0.003 (3x, capped)

---

## Dynamic Eta Updates

### Overview

The system supports dynamic updates to the `eta` parameter (calibration error variance) during online learning. This allows testing adaptation to gradual or sudden calibration changes.

### Update Mechanism

```yaml
online_learning:
  eta_update_interval_windows: 20  # Update every 20 windows
  eta_increment: 0.05              # Increase by 0.05 each time
  max_eta: 0.5                     # Clamp at 0.5
  min_eta: 0.0                     # Clamp at 0.0
```

**Update Logic**:
- At windows: 20, 40, 60, ...
- New eta = `min(max(current_eta + increment, min_eta), max_eta)`
- Updates `system_model.params.eta` (shared across data generation)
- Calls `dataset.update_eta()` to notify data generator

### Use Cases

1. **Gradual Calibration Drift**: Simulate slow degradation over time
2. **Sudden Calibration Change**: Large increment to test quick adaptation
3. **Eta Sweep Scenarios**: Set `eta_increment = target_value` and `max_eta = target_value` for discrete jumps

---

## Examples

### Example 1: Basic Online Learning

```yaml
online_learning:
  enabled: true
  window_size: 5
  stride: 5
  trajectory_length: 300
  dataset_size: 10
  learning_rate: 0.001
  max_iterations: 10
  
  # Drift detection
  drift_warmup_windows: 7
  drift_guard_samples: 3
  drift_z_threshold: 2.5
  time_to_learn: 10
  
  # Fixed learning rate
  use_adaptive_learning_rate: false
```

### Example 2: Adaptive Learning Rate Enabled

```yaml
online_learning:
  enabled: true
  window_size: 5
  stride: 3
  trajectory_length: 300
  learning_rate: 0.001
  
  # Enable adaptive LR
  use_adaptive_learning_rate: true
  
  # Drift detection
  drift_warmup_windows: 7
  drift_guard_samples: 3
  drift_z_threshold: 2.5
  time_to_learn: 5
```

### Example 3: Dynamic Eta Updates

```yaml
online_learning:
  enabled: true
  window_size: 5
  stride: 5
  
  # Dynamic eta updates
  eta_update_interval_windows: 10
  eta_increment: 0.05
  max_eta: 0.5
  min_eta: 0.0
  
  # Don't apply errors during generation
  use_nominal: true
```

### Example 4: Eta Sweep with Discrete Jump

For scenario sweeps where eta should jump from 0 to target value at specific window:

```yaml
# In scenario sweep config
scenario_config:
  type: "eta"
  values: [0.4, 0.9]

# In online_learning config
online_learning:
  eta_update_interval_windows: 10  # Jump at window 10
  # Scenario code sets: eta_increment = target_value, max_eta = target_value
  # This makes first update jump directly to target
```

### Running Examples

```bash
# Basic online learning
python main.py online_learning -c configs/online_learning_config.yaml

# With model override
python main.py online_learning -c configs/online_learning_config.yaml \
  -m experiments/results/checkpoints/model.pt

# Scenario sweep
python main.py simulate -c configs/eta_sweep_config.yaml \
  -s eta --mode online_learning
```

---

## Summary

The Online Learning system provides a comprehensive framework for adaptive model training with:

- **Statistical drift detection** via GLRT z-score analysis
- **Optional adaptive learning rates** scaling with change magnitude
- **Dynamic parameter updates** for realistic scenario simulation
- **Robust multi-trajectory evaluation** with result aggregation
- **Comprehensive visualization** and analysis tools

All configuration is handled through YAML files with full validation via Pydantic schemas, ensuring type safety and proper defaults.

"""
Configuration schema for SubspaceNet using Pydantic models.

This module defines the schema for validating configuration files using Pydantic.
"""

from typing import Optional, List, Literal, Union, Dict, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum


class SystemModelConfig(BaseModel):
    """System model configuration parameters."""
    N: int = 8
    M: int = 3
    T: int = 200
    snr: float = 0
    field_type: Literal["near", "far"] = "far"
    signal_nature: Literal["coherent", "non-coherent"] = "non-coherent"
    signal_type: Literal["narrowband", "broadband"] = "narrowband"
    wavelength: float = 0.06
    eta: float = 0.0
    bias: float = 0.0
    sv_noise_var: float = 0.0
    doa_range: int = 60
    doa_resolution: int = 1
    max_range_ratio_to_limit: float = 0.5
    range_resolution: int = 1


class DatasetConfig(BaseModel):
    """Dataset configuration parameters."""
    samples_size: int = 4096
    train_test_ratio: float = 1
    create_data: bool = False
    save_dataset: bool = False
    true_doa_train: Optional[List[float]] = None
    true_range_train: Optional[List[float]] = None
    true_doa_test: Optional[List[float]] = None
    true_range_test: Optional[List[float]] = None


class ModelParamsConfig(BaseModel):
    """Model parameters configuration."""
    diff_method: Union[Literal["music_2D", "music_1D", "esprit"], tuple[str, str]] = "esprit"
    train_loss_type: Union[Literal["music_spectrum", "rmspe"], tuple[str, str]] = "rmspe"
    tau: int = 8
    field_type: Literal["Near", "Far"] = "Far"
    regularization: Optional[Literal["aic", "mdl", "threshold", "null"]] = None
    variant: Literal["small", "big"] = "small"
    norm_layer: bool = False
    batch_norm: bool = False


class ModelConfig(BaseModel):
    """Model configuration."""
    type: Literal["SubspaceNet", "DCD-MUSIC"] = "SubspaceNet"
    params: ModelParamsConfig = ModelParamsConfig()


class TrainingConfig(BaseModel):
    """Training configuration parameters."""
    enabled: bool = True
    epochs: int = 10
    batch_size: int = 32
    optimizer: Literal["Adam", "SGD", "RMSprop"] = "Adam"
    scheduler: Literal["ReduceLROnPlateau", "StepLR", "CosineAnnealingLR"] = "ReduceLROnPlateau"
    learning_rate: float = 0.001
    weight_decay: float = 1e-9
    step_size: int = 50
    gamma: float = 0.5
    training_objective: Literal["angle", "range", "angle, range"] = "angle"
    use_wandb: bool = False
    simulation_name: Optional[str] = None
    save_checkpoint: bool = True


class SimulationConfig(BaseModel):
    """Simulation configuration parameters."""
    train_model: bool = True
    evaluate_model: bool = True
    load_model: bool = False
    save_model: bool = False
    plot_results: bool = True
    save_plots: bool = False
    model_path: Optional[str] = None  # Path to load pretrained model from
    subspace_methods: List[str] = Field(default_factory=list, description="List of classic subspace methods to evaluate (e.g., ['1D-MUSIC', 'Root-MUSIC','ESPRIT']).")


class EvaluationConfig(BaseModel):
    """Evaluation configuration parameters."""
    pass


class TrajectoryType(str, Enum):
    """Types of angle trajectories supported by the data handler."""
    RANDOM = "random"
    RANDOM_WALK = "random_walk"
    LINEAR = "linear"
    CIRCULAR = "circular"
    STATIC = "static"
    CUSTOM = "custom"
    FULL_RANDOM = "full_random"  # Completely random angles for each source in each trajectory step


class TrajectoryConfig(BaseModel):
    """Trajectory configuration parameters."""
    enabled: bool = False
    trajectory_type: TrajectoryType = TrajectoryType.RANDOM
    trajectory_length: int = 10
    save_trajectory: bool = False
    random_walk_std_dev: float = 1.0  # Standard deviation for random walk angles (degrees)


class KalmanFilterConfig(BaseModel):
    """Kalman Filter configuration parameters."""
    process_noise_std_dev: Optional[float] = None  # If None, uses trajectory.random_walk_std_dev
    measurement_noise_std_dev: float = 1e-3  # Near-zero for initial implementation (identity observation model)
    initial_covariance: float = 1.0  # Initial state covariance (uncertainty)


class OnlineLearningConfig(BaseModel):
    """Online learning configuration parameters."""
    enabled: bool = False
    window_size: int = Field(default=10, description="Size of the sliding window for trajectory data.")
    stride: int = Field(default=5, description="Stride between consecutive windows.")
    loss_threshold: float = Field(default=0.5, description="Threshold for detecting model drift and triggering online learning.")
    max_iterations: int = Field(default=10, description="Maximum number of training iterations per window.")
    learning_rate: float = Field(default=1e-4, description="Learning rate for online training (smaller than main training).")
    trajectory_length: Optional[int] = Field(default=None, description="Length of trajectory for online learning. If None, uses trajectory.trajectory_length.")


class ScenarioSystemModelOverride(BaseModel):
    """Defines overrides for SystemModelConfig for a specific scenario."""
    N: Optional[int] = None
    M: Optional[Union[int, Tuple[int, int]]] = None
    T: Optional[int] = None
    snr: Optional[float] = None
    field_type: Optional[Literal["near", "far"]] = None
    signal_nature: Optional[Literal["coherent", "non-coherent"]] = None
    signal_type: Optional[Literal["narrowband", "broadband"]] = None
    wavelength: Optional[float] = None
    eta: Optional[float] = None
    bias: Optional[float] = None
    sv_noise_var: Optional[float] = None
    doa_range: Optional[int] = None
    doa_resolution: Optional[int] = None
    max_range_ratio_to_limit: Optional[float] = None
    range_resolution: Optional[int] = None

    class Config:
        extra = 'forbid'  # Prevent unspecified fields


class ScenarioModelOverride(BaseModel):
    """Defines overrides for ModelConfig for a specific scenario."""
    type: Optional[str] = None  # e.g., "SubspaceNet", "DCD-MUSIC". If type changes, architecture changes.
    # params: Optional[ModelParamsConfig] = None # For overriding model-specific parameters

    class Config:
        extra = 'forbid'


class ScenarioDefinition(BaseModel):
    """Defines a single scenario to be run."""
    name: str = Field(..., description="Unique name for the scenario (used for output folders, results keys).")
    system_model_overrides: Optional[ScenarioSystemModelOverride] = None
    model_overrides: Optional[ScenarioModelOverride] = None
    retrain_model: bool = Field(default=False, description="If True, model is retrained for this scenario. If False, uses base/loaded model.")


class ScenarioConfig(BaseModel):
    """Configuration for running a scenario with multiple parameter values."""
    type: str = Field(..., description="Type of scenario to run (e.g., 'SNR', 'T', 'M', 'eta').")
    values: List[float] = Field(..., description="List of values to test for this scenario type.")
    retrain_model: bool = Field(default=False, description="Whether to retrain the model for each scenario value.")


class Config(BaseModel):
    """Complete configuration model."""
    system_model: SystemModelConfig = SystemModelConfig()
    dataset: DatasetConfig = DatasetConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    simulation: SimulationConfig = SimulationConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    trajectory: TrajectoryConfig = TrajectoryConfig()
    kalman_filter: KalmanFilterConfig = KalmanFilterConfig()
    online_learning: OnlineLearningConfig = OnlineLearningConfig()
    # New scenario-related fields
    scenarios: Optional[List[ScenarioDefinition]] = Field(default=None, description="List of scenarios to run.")
    train_base_model_only_once: bool = Field(default=True, description="If True and scenarios use a shared model, train it only once initially.")
    scenario_config: Optional[ScenarioConfig] = Field(default=None, description="Configuration for running a scenario with multiple parameter values.")


    #missing: critirion, balance factor, sequential_dataset_train, sequential_dataset_test, evaluation params, simulation_commands
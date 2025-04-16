"""
Configuration schema for SubspaceNet using Pydantic models.

This module defines the schema for validating configuration files using Pydantic.
"""

from typing import Optional, List, Literal, Union, Dict, Any
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
    train_loss_type: Union[Literal["music_spectrum", "rmspe"], tuple[str, str]] = "music_spectrum"
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


class EvaluationConfig(BaseModel):
    """Evaluation configuration parameters."""
    methods: List[str] = ["2D-MUSIC", "Beamformer"] #TBD: 


class TrajectoryType(str, Enum):
    """Types of angle trajectories supported by the data handler."""
    RANDOM = "random"
    RANDOM_WALK = "random_walk"
    LINEAR = "linear"
    CIRCULAR = "circular"
    STATIC = "static"
    CUSTOM = "custom"


class TrajectoryConfig(BaseModel):
    """Trajectory configuration parameters."""
    enabled: bool = False
    trajectory_type: TrajectoryType = TrajectoryType.RANDOM
    trajectory_length: int = 10
    save_trajectory: bool = False


class Config(BaseModel):
    """Complete configuration model."""
    system_model: SystemModelConfig = SystemModelConfig()
    dataset: DatasetConfig = DatasetConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    simulation: SimulationConfig = SimulationConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    trajectory: TrajectoryConfig = TrajectoryConfig() 


    #missing: critirion, balance factor, sequential_dataset_train, sequential_dataset_test, evaluation params, simulation_commands
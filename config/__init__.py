"""
SubspaceNet Configuration Framework

This package provides a simplified configuration system for SubspaceNet.
"""

from .schema import (
    Config, 
    SystemModelConfig, 
    DatasetConfig, 
    ModelConfig,
    ModelParamsConfig,
    TrainingConfig,
    SimulationConfig,
    EvaluationConfig,
    TrajectoryConfig,
    KalmanFilterConfig,
    OnlineLearningConfig
)

from .loader import (
    load_config,
    save_config,
    apply_overrides
)

__all__ = [
    'Config',
    'SystemModelConfig',
    'DatasetConfig',
    'ModelConfig',
    'ModelParamsConfig',
    'TrainingConfig',
    'SimulationConfig',
    'EvaluationConfig',
    'TrajectoryConfig',
    'KalmanFilterConfig',
    'OnlineLearningConfig',
    'load_config',
    'save_config',
    'apply_overrides'
] 
"""
State evolution models for Extended Kalman Filter.

This package contains the state evolution models that can be used with
the ExtendedKalmanFilter1D to model different types of non-linear dynamics.
"""

from .base import StateEvolutionModel
from .sine_accel import SineAccelStateModel
from .mult_noise import MultNoiseStateModel

__all__ = [
    'StateEvolutionModel',
    'SineAccelStateModel', 
    'MultNoiseStateModel'
] 
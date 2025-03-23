"""
Simulation framework for SubspaceNet experiments.

This module provides a clean interface for running simulations,
both single runs and parametric scenarios.
"""

from .core import Simulation
from .scenarios import ScenarioType

__all__ = ['Simulation', 'ScenarioType'] 
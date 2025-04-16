"""
Scenario definitions for parametric simulations.

This module defines the different types of scenarios that can be run,
such as SNR variations, snapshot count variations, etc.
"""

from enum import Enum, auto
from typing import Dict, Any, List

class ScenarioType(str, Enum):
    """Types of parametric scenarios supported by the simulation framework."""
    SNR = "SNR"
    SNAPSHOTS = "T"
    SOURCES = "M"
    STEERING_ERROR = "eta"
    
    @classmethod
    def get_param_name(cls, scenario_type: str) -> str:
        """Get the parameter name in the configuration for a scenario type."""
        mapping = {
            cls.SNR: "system_model.snr",
            cls.SNAPSHOTS: "system_model.T",
            cls.SOURCES: "system_model.M",
            cls.STEERING_ERROR: "system_model.eta"
        }
        return mapping.get(scenario_type, scenario_type) 
"""Reproducibility helpers for simulation runs."""

from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger("SubspaceNet.reproducibility")


def apply_simulation_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Applied simulation seed=%s", seed)

from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import logging
import numpy as np
import torch
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import time
import copy
import yaml
from itertools import permutations
import itertools

from config.schema import Config
from .runners.data import TrajectoryDataHandler
from .runners.training import Trainer, TrainingConfig, TrajectoryTrainer
from .runners.evaluation import Evaluator
from simulation.kalman_filter import KalmanFilter1D, BatchKalmanFilter1D, BatchExtendedKalmanFilter1D
from DCD_MUSIC.src.metrics.rmspe_loss import RMSPELoss
from DCD_MUSIC.src.signal_creation import Samples
from DCD_MUSIC.src.evaluation import get_model_based_method, evaluate_model_based
from simulation.kalman_filter.extended import ExtendedKalmanFilter1D

logger = logging.getLogger(__name__)

# Device setup for evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
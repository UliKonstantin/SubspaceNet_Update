"""Model adapters for journal / v2 pipeline."""

from models.deep_cnn_adapter import DeepCNNDoAAdapter, snapshots_to_covariance_channels

__all__ = ["DeepCNNDoAAdapter", "snapshots_to_covariance_channels"]

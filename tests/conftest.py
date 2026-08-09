"""Shared pytest configuration."""
import sys
from pathlib import Path

import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DCD_MUSIC_ROOT = WORKSPACE_ROOT / "DCD_MUSIC"

for path in (WORKSPACE_ROOT, DCD_MUSIC_ROOT):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (paper configs, full simulation runs)")

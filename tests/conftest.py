import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (paper configs, full simulation runs)")

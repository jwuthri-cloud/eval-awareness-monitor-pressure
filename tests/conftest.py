"""Shared pytest setup."""

import matplotlib
import pytest
import structlog

# Headless backend so figure tests run anywhere, including CI and remote GPU boxes.
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _reset_structlog():
    """Undo any structlog configuration a test made, so loggers never outlive its stderr."""
    yield
    structlog.reset_defaults()

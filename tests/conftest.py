"""Shared fixtures for the bench test suite; the tests never write into the repository tree."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "bench"


def pytest_configure(config):
    """Register the slow marker for tests that run a bench entry point end to end."""
    config.addinivalue_line("markers", "slow: runs a bench entry point end to end (seconds, not milliseconds)")


@pytest.fixture(scope="session")
def repo_dir():
    """Return the repository root."""
    return REPO


@pytest.fixture(scope="session")
def bench_dir():
    """Return the bench/ directory of the repository under test."""
    return BENCH

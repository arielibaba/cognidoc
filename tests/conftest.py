"""
Pytest configuration and fixtures for CogniDoc tests.
"""

import sys
from pathlib import Path

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (run with --run-slow)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --run-slow is provided."""
    if config.getoption("--run-slow", default=False):
        return

    import pytest
    skip_slow = pytest.mark.skip(reason="Slow test - use --run-slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add --run-slow command line option."""
    try:
        parser.addoption(
            "--run-slow",
            action="store_true",
            default=False,
            help="Run slow E2E tests with GraphRAG"
        )
    except ValueError:
        # Option already added
        pass

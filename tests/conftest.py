from __future__ import annotations

import asyncio
import pathlib
import tempfile

import pytest
import pytest_asyncio

from audex.lib.store import Store
from audex.lib.store.localfile import LocalFileStore


@pytest_asyncio.fixture
async def temp_dir():
    """Provide a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield pathlib.Path(tmpdir)


@pytest.fixture
def mock_store(temp_dir: pathlib.Path) -> Store:
    """Provide a mock store for testing."""
    return LocalFileStore(base_path=temp_dir / "store")


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test
    session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

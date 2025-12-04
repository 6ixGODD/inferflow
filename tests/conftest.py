from __future__ import annotations

import asyncio
import pathlib
import tempfile

import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def temp_dir():
    """Provide a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield pathlib.Path(tmpdir)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

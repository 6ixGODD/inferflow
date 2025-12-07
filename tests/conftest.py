from __future__ import annotations

import asyncio
import pathlib
import tempfile

import pytest
import pytest_asyncio
import torch


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


@pytest.fixture
def bbox_samples_small():
    """Generate small set of bounding boxes for quick tests."""
    num_boxes = 100
    return torch.rand(num_boxes, 4) * 100 + 50


@pytest.fixture
def bbox_samples_medium():
    """Generate medium set of bounding boxes (typical use case)."""
    num_boxes = 1000
    return torch.rand(num_boxes, 4) * 640  # Typical image size


@pytest.fixture
def bbox_samples_large():
    """Generate large set of bounding boxes (stress test)."""
    num_boxes = 10000
    return torch.rand(num_boxes, 4) * 1920  # HD image size


@pytest.fixture
def bbox_pair():
    """Generate two sets of boxes for IoU testing."""
    box1 = torch.tensor(
        [
            [0, 0, 10, 10],
            [5, 5, 15, 15],
            [20, 20, 30, 30],
            [100, 100, 150, 150],
        ],
        dtype=torch.float32,
    )

    box2 = torch.tensor(
        [
            [5, 5, 15, 15],
            [10, 10, 20, 20],
            [25, 25, 35, 35],
            [200, 200, 250, 250],
        ],
        dtype=torch.float32,
    )

    return box1, box2


@pytest.fixture
def bbox_batch():
    """Generate batched bounding boxes (3D tensor)."""
    batch_size = 8
    num_boxes = 100
    return torch.rand(batch_size, num_boxes, 4) * 640

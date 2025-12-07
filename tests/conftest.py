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
def yolo_prediction():
    """Generate realistic YOLOv5 prediction tensor."""
    batch_size = 1
    num_boxes = 8400  # Typical for YOLOv5 with 640x640 input
    num_classes = 80

    prediction = torch.randn(batch_size, num_boxes, 5 + num_classes)

    # Mock realistic bounding box coordinates and confidence scores
    prediction[:, :, 0:2] = torch.rand(batch_size, num_boxes, 2) * 640  # xc, yc
    prediction[:, :, 2:4] = torch.rand(batch_size, num_boxes, 2) * 100 + 10  # w, h
    prediction[:, :, 4] = torch.rand(batch_size, num_boxes) * 0.3 + 0.2  # obj conf (0.2-0.5)
    prediction[:, :, 5:] = torch.rand(batch_size, num_boxes, num_classes) * 0.8  # class conf

    return prediction


@pytest.fixture
def large_batch_prediction():
    """Generate large batch for throughput testing."""
    batch_size = 8
    num_boxes = 8400
    num_classes = 80

    prediction = torch.randn(batch_size, num_boxes, 5 + num_classes)
    prediction[:, :, 0:2] = torch.rand(batch_size, num_boxes, 2) * 640
    prediction[:, :, 2:4] = torch.rand(batch_size, num_boxes, 2) * 100 + 10
    prediction[:, :, 4] = torch.rand(batch_size, num_boxes) * 0.3 + 0.2
    prediction[:, :, 5:] = torch.rand(batch_size, num_boxes, num_classes) * 0.8

    return prediction


@pytest.fixture
def bbox_samples():
    """Generate bounding box samples for conversion tests."""
    num_boxes = 1000
    return torch.rand(num_boxes, 4) * 100 + 50

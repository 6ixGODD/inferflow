from __future__ import annotations

import asyncio
import contextlib
import pathlib
import tempfile

import pytest
import pytest_asyncio
import torch

from inferflow.asyncio.runtime import BatchableRuntime as AsyncBatchableRuntime
from inferflow.runtime import BatchableRuntime

# ============================================================================
# Event Loop
# ============================================================================


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Temporary Directory
# ============================================================================


@pytest_asyncio.fixture
async def temp_dir():
    """Provide a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield pathlib.Path(tmpdir)


# ============================================================================
# Bbox Test Data
# ============================================================================


@pytest.fixture
def bbox_samples_small():
    """Generate small set of bounding boxes for quick tests."""
    num_boxes = 100
    return torch.rand(num_boxes, 4) * 100 + 50


@pytest.fixture
def bbox_samples_medium():
    """Generate medium set of bounding boxes (typical use case)."""
    num_boxes = 1000
    return torch.rand(num_boxes, 4) * 640


@pytest.fixture
def bbox_samples_large():
    """Generate large set of bounding boxes (stress test)."""
    num_boxes = 10000
    return torch.rand(num_boxes, 4) * 1920


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


# ============================================================================
# Mock Runtime for Testing
# ============================================================================


# ============================================================================
# Mock Runtime for Testing
# ============================================================================


class MockRuntime(BatchableRuntime[torch.Tensor, torch.Tensor]):
    """Mock runtime for testing batch strategies and pipelines."""

    def __init__(self, processing_time_ms: float = 10.0, fail_rate: float = 0.0):
        self.processing_time_ms = processing_time_ms
        self.fail_rate = fail_rate
        self.call_count = 0
        self.total_items_processed = 0
        self._loaded = False

    def load(self) -> None:
        """Load resources."""
        self._loaded = True

    def unload(self) -> None:
        """Unload resources."""
        self._loaded = False

    def infer(self, item: torch.Tensor) -> torch.Tensor:
        """Single inference (sync)."""
        import random
        import time

        if not self._loaded:
            raise RuntimeError("Runtime not loaded")

        self.call_count += 1
        self.total_items_processed += 1

        time.sleep(self.processing_time_ms / 1000)

        if random.random() < self.fail_rate:
            raise RuntimeError("Mock inference failed")

        return item * 2

    def infer_batch(self, items: list[torch.Tensor]) -> list[torch.Tensor]:
        """Batch inference (sync)."""
        import random
        import time

        if not self._loaded:
            raise RuntimeError("Runtime not loaded")

        self.call_count += 1
        self.total_items_processed += len(items)

        batch_time = self.processing_time_ms * len(items) * 0.8 / 1000
        time.sleep(batch_time)

        if random.random() < self.fail_rate:
            raise RuntimeError("Mock batch inference failed")

        return [item * 2 for item in items]


class AsyncMockRuntime(AsyncBatchableRuntime[torch.Tensor, torch.Tensor]):
    """Async mock runtime for testing."""

    def __init__(self, processing_time_ms: float = 10.0, fail_rate: float = 0.0):
        self.processing_time_ms = processing_time_ms
        self.fail_rate = fail_rate
        self.call_count = 0
        self.total_items_processed = 0
        self._loaded = False

    async def load(self) -> None:
        """Load resources."""
        self._loaded = True

    async def unload(self) -> None:
        """Unload resources."""
        self._loaded = False

    async def infer(self, item: torch.Tensor) -> torch.Tensor:
        """Single inference (async)."""
        import asyncio
        import random

        if not self._loaded:
            raise RuntimeError("Runtime not loaded")

        self.call_count += 1
        self.total_items_processed += 1

        await asyncio.sleep(self.processing_time_ms / 1000)

        if random.random() < self.fail_rate:
            raise RuntimeError("Mock inference failed")

        return item * 2

    async def infer_batch(self, items: list[torch.Tensor]) -> list[torch.Tensor]:
        """Batch inference (async)."""
        import asyncio
        import random

        if not self._loaded:
            raise RuntimeError("Runtime not loaded")

        self.call_count += 1
        self.total_items_processed += len(items)

        batch_time = self.processing_time_ms * len(items) * 0.8 / 1000
        await asyncio.sleep(batch_time)

        if random.random() < self.fail_rate:
            raise RuntimeError("Mock batch inference failed")

        return [item * 2 for item in items]


@pytest.fixture
def mock_runtime():
    """Fast sync mock runtime for unit tests."""
    return MockRuntime(processing_time_ms=10.0)


@pytest.fixture
def slow_mock_runtime():
    """Slow sync mock runtime for stress tests."""
    return MockRuntime(processing_time_ms=100.0)


@pytest.fixture
def flaky_mock_runtime():
    """Flaky sync mock runtime that fails 10% of the time."""
    return MockRuntime(processing_time_ms=10.0, fail_rate=0.1)


@pytest_asyncio.fixture
async def async_mock_runtime():
    """Fast async mock runtime for unit tests."""
    return AsyncMockRuntime(processing_time_ms=10.0)


@pytest_asyncio.fixture
async def slow_async_mock_runtime():
    """Slow async mock runtime for stress tests."""
    return AsyncMockRuntime(processing_time_ms=100.0)


# ============================================================================
# Batch Strategy Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def batch_strategy(async_mock_runtime):
    """Create and start a batch strategy with mock runtime."""
    from inferflow.asyncio.batch.dynamic import DynamicBatchStrategy

    await async_mock_runtime.load()

    strategy = DynamicBatchStrategy(
        min_batch_size=1,
        max_batch_size=8,
        max_wait_ms=50.0,
        queue_size=100,
        block_on_full=True,
    )

    await strategy.start(async_mock_runtime)

    yield strategy

    await strategy.stop()
    await async_mock_runtime.unload()


@pytest_asyncio.fixture
async def large_batch_strategy(async_mock_runtime):
    """Batch strategy with larger capacity for stress tests."""
    from inferflow.asyncio.batch.dynamic import DynamicBatchStrategy

    await async_mock_runtime.load()

    strategy = DynamicBatchStrategy(
        min_batch_size=1,
        max_batch_size=32,
        max_wait_ms=100.0,
        queue_size=1000,
        block_on_full=True,
    )

    await strategy.start(async_mock_runtime)

    yield strategy

    await strategy.stop()
    await async_mock_runtime.unload()


class MockClassificationRuntime(BatchableRuntime[torch.Tensor, torch.Tensor]):
    """Mock runtime for classification pipeline testing."""

    def __init__(
        self,
        num_classes: int = 10,
        processing_time_ms: float = 10.0,
        fail_rate: float = 0.0,
    ):
        self.num_classes = num_classes
        self.processing_time_ms = processing_time_ms
        self.fail_rate = fail_rate
        self.call_count = 0
        self.total_items_processed = 0
        self._loaded = False

    @contextlib.contextmanager
    def context(self):
        """Context manager for runtime lifecycle (sync)."""
        self.load()
        try:
            yield self
        finally:
            self.unload()

    def load(self) -> None:
        """Load resources."""
        self._loaded = True

    def unload(self) -> None:
        """Unload resources."""
        self._loaded = False

    def infer(self, item: torch.Tensor) -> torch.Tensor:
        """Single inference (sync) - returns classification logits."""
        import random
        import time

        if not self._loaded:
            raise RuntimeError("Runtime not loaded")

        self.call_count += 1
        self.total_items_processed += 1

        time.sleep(self.processing_time_ms / 1000)

        if random.random() < self.fail_rate:
            raise RuntimeError("Mock inference failed")

        # Return random classification logits (batch_size, num_classes)
        batch_size = item.shape[0] if item.ndim == 4 else 1
        return torch.randn(batch_size, self.num_classes)

    def infer_batch(self, items: list[torch.Tensor]) -> list[torch.Tensor]:
        """Batch inference (sync)."""
        import random
        import time

        if not self._loaded:
            raise RuntimeError("Runtime not loaded")

        self.call_count += 1
        self.total_items_processed += len(items)

        batch_time = self.processing_time_ms * len(items) * 0.8 / 1000
        time.sleep(batch_time)

        if random.random() < self.fail_rate:
            raise RuntimeError("Mock batch inference failed")

        return [torch.randn(item.shape[0], self.num_classes) for item in items]


class AsyncMockClassificationRuntime(AsyncBatchableRuntime[torch.Tensor, torch.Tensor]):
    """Async mock runtime for classification."""

    def __init__(
        self,
        num_classes: int = 10,
        processing_time_ms: float = 10.0,
        fail_rate: float = 0.0,
    ):
        self.num_classes = num_classes
        self.processing_time_ms = processing_time_ms
        self.fail_rate = fail_rate
        self.call_count = 0
        self.total_items_processed = 0
        self._loaded = False

    @contextlib.asynccontextmanager
    async def context(self):
        """Context manager for runtime lifecycle (async)."""
        await self.load()
        try:
            yield self
        finally:
            await self.unload()

    async def load(self) -> None:
        """Load resources."""
        self._loaded = True

    async def unload(self) -> None:
        """Unload resources."""
        self._loaded = False

    async def infer(self, item: torch.Tensor) -> torch.Tensor:
        """Single inference (async) - returns classification logits."""
        import asyncio
        import random

        if not self._loaded:
            raise RuntimeError("Runtime not loaded")

        self.call_count += 1
        self.total_items_processed += 1

        await asyncio.sleep(self.processing_time_ms / 1000)

        if random.random() < self.fail_rate:
            raise RuntimeError("Mock inference failed")

        batch_size = item.shape[0] if item.ndim == 4 else 1
        return torch.randn(batch_size, self.num_classes)

    async def infer_batch(self, items: list[torch.Tensor]) -> list[torch.Tensor]:
        """Batch inference (async)."""
        import asyncio
        import random

        if not self._loaded:
            raise RuntimeError("Runtime not loaded")

        self.call_count += 1
        self.total_items_processed += len(items)

        batch_time = self.processing_time_ms * len(items) * 0.8 / 1000
        await asyncio.sleep(batch_time)

        if random.random() < self.fail_rate:
            raise RuntimeError("Mock batch inference failed")

        return [torch.randn(item.shape[0], self.num_classes) for item in items]


# ============================================================================
# Classification Fixtures
# ============================================================================


@pytest.fixture
def mock_classification_runtime():
    """Mock classification runtime (sync)."""
    return MockClassificationRuntime(num_classes=10, processing_time_ms=10.0)


@pytest_asyncio.fixture
async def async_mock_classification_runtime():
    """Mock classification runtime (async)."""
    return AsyncMockClassificationRuntime(num_classes=10, processing_time_ms=10.0)


__all__ = [
    "MockRuntime",
    "AsyncMockRuntime",
    "mock_runtime",
    "slow_mock_runtime",
    "flaky_mock_runtime",
    "async_mock_runtime",
    "slow_async_mock_runtime",
    "batch_strategy",
    "large_batch_strategy",
    "bbox_samples_small",
    "bbox_samples_medium",
    "bbox_samples_large",
    "bbox_pair",
    "bbox_batch",
    "MockRuntime",
    "AsyncMockRuntime",
    "MockClassificationRuntime",
    "AsyncMockClassificationRuntime",
    "mock_runtime",
    "slow_mock_runtime",
    "flaky_mock_runtime",
    "async_mock_runtime",
    "slow_async_mock_runtime",
    "mock_classification_runtime",
    "async_mock_classification_runtime",
    "batch_strategy",
    "large_batch_strategy",
]

from __future__ import annotations

import asyncio as aio
import typing as t

from inferflow._C import batch
from inferflow.batch import BatchMetrics
from inferflow.batch import BatchStrategy
from inferflow.types import P
from inferflow.types import R

if t.TYPE_CHECKING:
    from inferflow.runtime import Runtime


class CppBatchStrategy(BatchStrategy[P, R]):
    """C++-accelerated dynamic batching strategy.

    This strategy uses a C++ backend for:
    - Lock-free queue (minimal contention)
    - Dedicated worker thread (no GIL)
    - Efficient batch processing

    Args:
        min_batch_size: Minimum batch size.
        max_batch_size: Maximum batch size.
        max_wait_ms: Maximum wait time before processing.
        queue_size: Maximum queue capacity.
        block_on_full: Block when queue is full.

    Example:
        ```python
        strategy = CppBatchStrategy(
            max_batch_size=32, max_wait_ms=50
        )
        await strategy.start(runtime)

        result = await strategy.submit(preprocessed_input)

        await strategy.stop()
        ```
    """

    def __init__(
        self,
        min_batch_size: int = 1,
        max_batch_size: int = 32,
        max_wait_ms: float = 50.0,
        queue_size: int = 1000,
        block_on_full: bool = True,
    ):
        super().__init__()

        # Create C++ batcher config
        config = batch.BatcherConfig()
        config.min_batch_size = min_batch_size
        config.max_batch_size = max_batch_size
        config.max_wait_ms = max_wait_ms
        config.queue_capacity = queue_size
        config.block_on_full = block_on_full

        # Create C++ batcher
        self._batcher = batch.DynamicBatcher(config)

    async def submit(self, item: P) -> R:
        """Submit an item for batched processing."""
        if not self._running:
            raise RuntimeError("BatchStrategy not started.  Call start() first.")

        # Submit to C++ batcher (returns std::shared_future)
        cpp_future = self._batcher.submit(item)

        # Convert C++ future to Python asyncio future
        loop = aio.get_event_loop()
        return await loop.run_in_executor(None, cpp_future.get)  # type: ignore

    async def start(self, runtime: Runtime[P, R]) -> None:
        """Start the batch processing worker."""
        if self._running:
            return

        self.runtime = runtime
        self._batcher.start(runtime)
        self._running = True

    async def stop(self) -> None:
        """Stop the batch processing worker."""
        if not self._running:
            return

        self._batcher.stop()
        self._running = False

    def get_metrics(self) -> BatchMetrics:
        """Get current batch processing metrics."""
        cpp_metrics = self._batcher.get_metrics()

        # Convert C++ metrics to Python BatchMetrics
        py_metrics = BatchMetrics()
        py_metrics.total_requests = int(cpp_metrics.total_requests)
        py_metrics.total_batches = int(cpp_metrics.total_batches)
        py_metrics.rejected_requests = int(cpp_metrics.rejected_requests)
        py_metrics.current_queue_size = int(cpp_metrics.current_queue_size)
        py_metrics.current_batch_size = int(cpp_metrics.current_batch_size)

        return py_metrics

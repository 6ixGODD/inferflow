from __future__ import annotations

import typing as t

class BatcherConfig:
    """Configuration for C++ dynamic batcher.

    Attributes:
        min_batch_size: Minimum batch size (default: 1)
        max_batch_size: Maximum batch size (default: 32)
        max_wait_ms: Maximum wait time in milliseconds (default: 50. 0)
        queue_capacity: Maximum queue capacity (default: 1000)
        block_on_full: Block when queue is full instead of rejecting (default:
            True)
    """

    min_batch_size: int
    max_batch_size: int
    max_wait_ms: float
    queue_capacity: int
    block_on_full: bool

    def __init__(self) -> None:
        """Create a default batcher configuration."""

class BatchMetrics:
    """Real-time batch processing metrics from C++ backend.

    Attributes:
        total_requests: Total number of requests processed
        total_batches: Total number of batches executed
        rejected_requests: Number of rejected requests due to queue full
        current_queue_size: Current number of items in queue
        current_batch_size: Current batch size being used
    """

    total_requests: int
    total_batches: int
    rejected_requests: int
    current_queue_size: int
    current_batch_size: int

class DynamicBatcher:
    """High-performance C++ batcher with dynamic batching.

    This batcher uses:
    - Lock-free queue for minimal contention
    - Dedicated worker thread (no Python GIL)
    - Dynamic batch size adjustment based on queue depth
    - Efficient memory management

    Example:
        ```python
        config = BatcherConfig()
        config.max_batch_size = 32
        config.max_wait_ms = 50. 0

        batcher = DynamicBatcher(config)
        batcher.start(runtime)

        future = batcher.submit(input_tensor)
        result = future.get()  # Blocks until result is ready

        batcher.stop()
        ```
    """

    def __init__(self, config: BatcherConfig) -> None:
        """Create a dynamic batcher with given configuration.

        Args:
            config: Batcher configuration

        Raises:
            RuntimeError: If configuration is invalid
        """

    def start(self, runtime: t.Any) -> None:
        """Start the batcher with a Python runtime.

        The runtime must have an `infer_batch(items: list) -> list` method
        that processes a batch of items and returns results.

        Args:
            runtime: Python runtime object with `infer_batch` method

        Raises:
            RuntimeError: If batcher is already running

        Example:
            ```python
            class MyRuntime:
                def infer_batch(self, items):
                    return [process(item) for item in items]

            batcher.start(MyRuntime())
            ```
        """

    def stop(self) -> None:
        """Stop the batcher and cleanup resources.

        This will:
        - Stop accepting new requests
        - Wait for current batch to complete
        - Join worker thread
        - Log final metrics

        Example:
            ```python
            batcher.stop()
            # [C++ Batcher] Stopped.  Processed 1000 requests in 50 batches.
            ```
        """

    def submit(self, item: t.Any) -> SharedFuture:
        """Submit an item for batched processing.

        Args:
            item: Python object to process (e.g., torch.Tensor)

        Returns:
            A shared future that will contain the result.
            Call `. get()` to block until result is ready.

        Raises:
            RuntimeError: If batcher is not running
            RuntimeError: If queue is full and `block_on_full=False`

        Example:
            ```python
            future = batcher.submit(torch.randn(3, 224, 224))
            result = future.get()  # Blocks until processed
            ```
        """

    def is_running(self) -> bool:
        """Check if batcher is currently running.

        Returns:
            True if batcher is running, False otherwise
        """

    def get_metrics(self) -> BatchMetrics:
        """Get current batch processing metrics.

        Returns:
            Current metrics including throughput and queue status

        Example:
            ```python
            metrics = batcher.get_metrics()
            print(f"Processed {metrics.total_requests} requests")
            print(f"Current queue: {metrics.current_queue_size}")
            ```
        """

    def queue_size(self) -> int:
        """Get current queue size (approximate).

        Returns:
            Number of items currently in queue

        Note:
            This is an approximate value due to lock-free implementation.
        """

class SharedFuture:
    """Shared future for C++ asynchronous result.

    This wraps `std::shared_future<py::object>` from C++.
    Multiple copies can share the same result.
    """

    def get(self) -> t.Any:
        """Block until result is ready and return it.

        Returns:
            The result object from batch processing

        Raises:
            Exception: If batch processing failed

        Note:
            This blocks the calling thread.  For async usage,
            use `CppBatchStrategy` which wraps this in asyncio.
        """

    def wait(self) -> None:
        """Block until result is ready (but don't retrieve it).

        This is useful if you just want to wait without
        retrieving the result immediately.
        """

    def is_ready(self) -> bool:
        """Check if result is ready without blocking.

        Returns:
            True if result is ready, False otherwise
        """

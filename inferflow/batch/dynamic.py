from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
import typing as t

from inferflow.batch import BatchStrategy
from inferflow.types import P
from inferflow.types import R

if t.TYPE_CHECKING:
    from inferflow.runtime import Runtime

__doctitle__ = "Dynamic Batch Strategy"

logger = logging.getLogger("inferflow.batch.dynamic")


class QueueFullError(Exception):
    """Raised when queue is full and blocking is disabled."""


class DynamicBatchStrategy(BatchStrategy[P, R]):
    """Dynamic batching with adaptive batch size (sync version).

    This strategy:
        - Collects requests into batches
        - Adjusts batch size based on queue depth
        - Uses timeout to ensure low latency
        - Distributes results to individual requests

    Args:
        min_batch_size: Minimum batch size (default: 1).
        max_batch_size: Maximum batch size (default: 32).
        max_wait_ms: Maximum wait time before processing batch (default: 50ms).
        queue_size: Maximum queue size (default: 1000).
        block_on_full: Block when queue is full instead of raising error.

    Example:
        ```python
        strategy = DynamicBatchStrategy(
            max_batch_size=32,
            max_wait_ms=50,
        )
        ```
    """

    def __init__(
        self,
        min_batch_size: int = 1,
        max_batch_size: int = 32,
        max_wait_ms: float = 50,
        queue_size: int = 1000,
        block_on_full: bool = True,
    ):
        super().__init__()
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000
        self.queue_size = queue_size
        self.block_on_full = block_on_full

        self._queue: queue.Queue[tuple[P, queue.Queue[R]]] = queue.Queue(maxsize=queue_size)
        self._worker_thread: threading.Thread | None = None

        logger.info(
            f"DynamicBatchStrategy initialized:  "
            f"batch_size=[{min_batch_size}, {max_batch_size}], "
            f"max_wait={max_wait_ms}ms, queue_size={queue_size}"
        )

    def submit(self, item: P) -> R:
        """Submit an item for batched processing.

        Args:
            item: Preprocessed input.

        Returns:
            Inference result.

        Raises:
            RuntimeError: If strategy is not started.
            QueueFullError: If queue is full and blocking is disabled.
        """
        if not self._running:
            raise RuntimeError("BatchStrategy not started. Call start() first.")

        result_queue: queue.Queue[R] = queue.Queue(maxsize=1)

        if self._queue.full() and not self.block_on_full:
            self.metrics.rejected_requests += 1
            raise QueueFullError(
                f"Queue is full ({self.queue_size} items).Rejected {self.metrics.rejected_requests} requests."
            )

        self._queue.put((item, result_queue))
        self.metrics.current_queue_size = self._queue.qsize()

        return result_queue.get()

    def start(self, runtime: Runtime[P, R]) -> None:
        """Start the batch processing worker.

        Args:
            runtime: Runtime to use for inference.
        """
        if self._running:
            logger.warning("BatchStrategy already running")
            return

        self.runtime = runtime
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

        logger.info("DynamicBatchStrategy started")

    def stop(self) -> None:
        """Stop the batch processing worker (sync)."""
        if not self._running:
            return

        self._running = False

        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)

        logger.info(
            f"DynamicBatchStrategy stopped."
            f"Processed {self.metrics.total_requests} requests in "
            f"{self.metrics.total_batches} batches."
        )

    def _worker(self) -> None:
        """Background worker that processes batches."""
        logger.info("Batch worker started")

        while self._running:
            batch: list[tuple[P, queue.Queue[R]]] = []

            try:
                try:
                    first_item = self._queue.get(timeout=1.0)
                    batch.append(first_item)
                except queue.Empty:
                    continue

                batch_start = time.time()
                target_batch_size = min(self.max_batch_size, self._queue.qsize() + 1)

                while len(batch) < target_batch_size:
                    remaining_time = max(0.0, self.max_wait_ms - (time.time() - batch_start))
                    if remaining_time <= 0:
                        break

                    try:
                        item = self._queue.get(timeout=remaining_time)
                        batch.append(item)
                    except queue.Empty:
                        break

                if batch:
                    self._process_batch(batch)

            except Exception as e:
                logger.error(f"Error in batch worker:  {e}", exc_info=True)
                for _, result_queue in batch:
                    with contextlib.suppress(Exception):
                        result_queue.put(None)

        logger.info("Batch worker stopped")

    def _process_batch(self, batch: list[tuple[P, queue.Queue[R]]]) -> None:
        """Process a batch of items.

        Args:
            batch: List of (item, result_queue) tuples.
        """
        batch_size = len(batch)
        items = [item for item, _ in batch]
        result_queues = [rq for _, rq in batch]

        logger.debug(f"Processing batch of size {batch_size}")

        start_time = time.time()

        try:
            if hasattr(self.runtime, "infer_batch"):
                results = self.runtime.infer_batch(items)
            else:
                results = [self.runtime.infer(item) for item in items]

            processing_time = time.time() - start_time

            for result_queue, result in zip(result_queues, results, strict=False):
                result_queue.put(result)

            self.metrics.add_batch(batch_size, processing_time)
            self.metrics.current_batch_size = batch_size
            self.metrics.current_queue_size = self._queue.qsize()

            logger.debug(f"Batch processed: size={batch_size}, time={processing_time * 1000:.2f}ms")

        except Exception as e:
            logger.error(f"Batch processing failed: {e}", exc_info=True)
            for result_queue in result_queues:
                with contextlib.suppress(Exception):
                    result_queue.put(None)


__all__ = ["DynamicBatchStrategy", "QueueFullError"]

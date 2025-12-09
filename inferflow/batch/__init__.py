from __future__ import annotations

import abc
import dataclasses
import importlib
import typing as t

from inferflow.types import P
from inferflow.types import R

if t.TYPE_CHECKING:
    from inferflow.runtime import Runtime

__doctitle__ = "Batch Processing Strategies"


@dataclasses.dataclass
class BatchMetrics:
    """Real-time batch processing metrics."""

    total_requests: int = 0
    """Total number of requests processed."""

    total_batches: int = 0
    """Total number of batches executed."""

    total_latency_ms: float = 0.0
    """Cumulative latency in milliseconds."""

    current_queue_size: int = 0
    """Current number of items in queue."""

    current_batch_size: int = 0
    """Current batch size being used."""

    rejected_requests: int = 0
    """Number of rejected requests due to queue full."""

    processing_times: list[float] = dataclasses.field(default_factory=list)
    """Recent processing times (sliding window)."""

    batch_sizes: list[int] = dataclasses.field(default_factory=list)
    """Recent batch sizes (sliding window)."""

    @property
    def avg_batch_size(self) -> float:
        return sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    @property
    def avg_processing_time_ms(self) -> float:
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times) * 1000

    def add_batch(self, batch_size: int, processing_time: float, window_size: int = 50) -> None:
        """Record a batch execution.

        Args:
            batch_size: Size of the batch processed.
            processing_time: Time taken to process (seconds).
            window_size: Maximum number of recent measurements to keep.
        """
        self.total_batches += 1
        self.total_requests += batch_size
        self.total_latency_ms += processing_time * 1000

        self.processing_times.append(processing_time)
        self.batch_sizes.append(batch_size)

        # Keep only recent measurements
        if len(self.processing_times) > window_size:
            self.processing_times.pop(0)
        if len(self.batch_sizes) > window_size:
            self.batch_sizes.pop(0)


class BatchStrategy(abc.ABC, t.Generic[P, R]):
    """Abstract batch processing strategy.

    A batch strategy manages:
        - Request queuing
        - Batch formation
        - Batch execution
        - Result distribution

    Example:
        ```python
        strategy = DynamicBatchStrategy(
            max_batch_size=32, max_wait_ms=50
        )
        strategy.start(runtime)

        # Submit requests (automatically batched)
        result = strategy.submit(preprocessed_input)

        strategy.stop()
        ```
    """

    def __init__(self) -> None:
        self.runtime: Runtime[P, R] | None = None
        self.metrics = BatchMetrics()
        self._running = False

    @abc.abstractmethod
    def submit(self, item: P) -> R:
        """Submit an item for batched processing.

        Args:
            item: Preprocessed input.

        Returns:
            Inference result.

        Raises:
            RuntimeError: If strategy is not started.
        """

    @abc.abstractmethod
    def start(self, runtime: Runtime[P, R]) -> None:
        """Start the batch processing worker.

        Args:
            runtime: Runtime to use for inference.
        """

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop the batch processing worker and cleanup resources."""

    @property
    def is_running(self) -> bool:
        """Check if batch worker is running."""
        return self._running

    def get_metrics(self) -> BatchMetrics:
        """Get current batch processing metrics."""
        return self.metrics


__all__ = ["BatchStrategy", "BatchMetrics", "dynamic"]


def __getattr__(name: str) -> t.Any:
    if name in __all__:
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

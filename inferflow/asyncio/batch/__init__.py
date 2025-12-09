from __future__ import annotations

import abc
import importlib
import typing as t

from inferflow.batch import BatchMetrics
from inferflow.types import P
from inferflow.types import R

if t.TYPE_CHECKING:
    from inferflow.asyncio.runtime import Runtime


class BatchStrategy(abc.ABC, t.Generic[P, R]):
    """Abstract batch processing strategy (async version).

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
        await strategy.start(runtime)

        # Submit requests (automatically batched)
        result = await strategy.submit(preprocessed_input)

        await strategy.stop()
        ```
    """

    def __init__(self) -> None:
        self.runtime: Runtime[P, R] | None = None
        self.metrics = BatchMetrics()
        self._running = False

    @abc.abstractmethod
    async def submit(self, item: P) -> R:
        """Submit an item for batched processing.

        Args:
            item: Preprocessed input.

        Returns:
            Inference result.

        Raises:
            RuntimeError: If strategy is not started.
        """

    @abc.abstractmethod
    async def start(self, runtime: Runtime[P, R]) -> None:
        """Start the batch processing worker.

        Args:
            runtime: Runtime to use for inference.
        """

    @abc.abstractmethod
    async def stop(self) -> None:
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

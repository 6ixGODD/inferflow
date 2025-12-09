from __future__ import annotations

import abc
import contextlib
import importlib
import typing as t

from inferflow.runtime import RuntimeConfigMixin
from inferflow.types import P
from inferflow.types import R

if t.TYPE_CHECKING:
    import types

__doctitle__ = "Inference Runtimes (Async)"


class Runtime(abc.ABC, t.Generic[P, R]):
    """Abstract runtime for model inference (async version).

    A runtime encapsulates:
        - Model loading/unloading
        - Device management
        - Inference execution
        - Memory management

    Type Parameters:
        P: Preprocessed input type (e.g., torch.Tensor, np.ndarray)
        R: Raw output type from the model

    This is the synchronous version of the runtime. For async support,
    see `inferflow.asyncio.runtime.Runtime`.

    Example:
        ```python
        import inferflow.asyncio as iff

        runtime = iff.TorchScriptRuntime(
            model_path="model.pt",
            device="cuda: 0",
        )

        # Using async context manager
        async with runtime:
            result = runtime.infer(input_tensor)

        # Manual lifecycle
        await runtime.load()
        try:
            result = await runtime.infer(input_tensor)
        finally:
            await runtime.unload()
        ```
    """

    @abc.abstractmethod
    async def load(self) -> None:
        """Load model into memory and prepare for inference."""

    @abc.abstractmethod
    async def infer(self, input: P) -> R:
        """Run inference on preprocessed input.

        Args:
            input: Preprocessed input ready for model inference.
                Type depends on backend (e.g., torch.Tensor for PyTorch).

        Returns:
            Raw model output. Type depends on model architecture.

        Raises:
            RuntimeError: If model is not loaded.

        Example:
            ```python
            async with runtime:
                output = await runtime.infer(input_tensor)
            ```
        """

    @abc.abstractmethod
    async def unload(self) -> None:
        """Unload model and free resources.

        This method should:
            - Release model from memory
            - Clear device cache
            - Close any open handles

        Example:
            ```python
            await runtime.load()
            # ... do inference ...
            await runtime.unload()  # Free resources
            ```
        """

    @contextlib.asynccontextmanager
    async def context(self) -> t.AsyncIterator[t.Self]:
        """Async context manager for automatic lifecycle management.

        Automatically calls `load()` on entry and `unload()` on exit,
        even if an exception occurs.

        Yields:
            Self: The runtime instance.

        Example:
            ```python
            async with runtime.context():
                result = await runtime.infer(input)
            # Model is automatically unloaded here
            ```
        """
        await self.load()
        try:
            yield self
        finally:
            await self.unload()

    async def __aenter__(self) -> t.Self:
        """Async context manager entry.

        Loads the model.

        Returns:
            Self: The runtime instance.

        Example:
            ```python
            async with runtime:  # Calls __aenter__
                result = runtime.infer(input)
            ```
        """
        await self.load()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Async context manager exit.

        Unloads the model, even if an exception occurred.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        await self.unload()


class BatchableRuntime(Runtime[P, R], abc.ABC):
    """Runtime that supports batch inference natively (async version).

    Some runtimes (like TorchScript, ONNX) can process multiple inputs
    simultaneously for better throughput. This base class provides a
    common interface for batch inference.

    Type Parameters:
        P: Preprocessed input type (e.g., torch.Tensor, np.ndarray)
        R: Raw output type from the model

    The default `infer()` implementation delegates to `infer_batch()`,
    so subclasses only need to implement batch inference.

    Example:
        ```python
        async with runtime:
            # Single inference (delegates to batch)
            result = await runtime.infer(input)

            # Batch inference (more efficient)
            results = await runtime.infer_batch([
                input1,
                input2,
                input3,
            ])
        ```
    """

    @abc.abstractmethod
    async def infer_batch(self, inputs: list[P]) -> list[R]:
        """Run inference on a batch of inputs asynchronously.

        Process multiple inputs in a single forward pass for better
        throughput. Inputs should already have batch dimension.

        Args:
            inputs: List of preprocessed inputs. Each input should have
                shape (1, ...) for proper batching.

        Returns:
            List of raw outputs, one per input. Each output maintains
            the batch dimension (1, ...).

        Raises:
            RuntimeError: If model is not loaded.

        Example:
            ```python
            async with runtime:
                # Prepare batch
                batch = [
                    torch.randn(1, 3, 224, 224),
                    torch.randn(1, 3, 224, 224),
                    torch.randn(1, 3, 224, 224),
                ]

                # Batch inference
                results = await runtime.infer_batch(batch)

                # results[0], results[1], results[2] correspond to inputs
            ```
        """

    async def infer(self, input: P) -> R:
        """Single inference (delegates to batch inference).

        Wraps the input in a list, calls `infer_batch()`, and returns
        the first result. This provides a convenient single-input API
        while reusing the batch implementation.

        Args:
            input: Preprocessed input ready for model inference.

        Returns:
            Raw model output.

        Raises:
            RuntimeError: If model is not loaded.

        Example:
            ```python
            async with runtime:
                # These are equivalent:
                result = await runtime.infer(input)
                result = await runtime.infer_batch([input])[0]
            ```
        """
        results = await self.infer_batch([input])
        return results[0]


__all__ = [
    "Runtime",
    "BatchableRuntime",
    "RuntimeConfigMixin",
    "onnx",
    "tensorrt",
    "torch",
]


def __getattr__(name: str) -> t.Any:
    if name in __all__:
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from __future__ import annotations

import abc
import contextlib
import typing as t

from inferflow.types import P
from inferflow.types import R

if t.TYPE_CHECKING:
    import types


class Runtime(abc.ABC, t.Generic[P, R]):  # type: ignore[misc]
    """Abstract runtime for model inference.

    A runtime encapsulates:
    - Model loading/unloading
    - Device management
    - Inference execution
    - Memory management

    Type Parameters:
        P: Preprocessed input type (e.g., torch.Tensor, np.ndarray)
        R: Raw output type from the model

    Example:
        ```python
        runtime = TorchScriptRuntime(
            model_path="model.pt", device="cuda"
        )
        async with runtime:
            result = await runtime.infer(input_tensor)
        ```
    """

    @abc.abstractmethod
    async def load(self) -> None:
        """Load model into memory and prepare for inference.

        This method should:
        - Load model weights
        - Move model to target device
        - Perform warmup inference
        - Set model to evaluation mode
        """
        ...

    @abc.abstractmethod
    async def infer(self, input: P) -> R:
        """Run inference on preprocessed input.

        Args:
            input: Preprocessed input ready for model inference.

        Returns:
            Raw model output.

        Raises:
            RuntimeError: If model is not loaded.
        """

    @abc.abstractmethod
    async def unload(self) -> None:
        """Unload model and free resources.

        This method should:
        - Release model from memory
        - Clear device cache
        - Close any open handles
        """

    @contextlib.asynccontextmanager
    async def context(self) -> t.AsyncIterator[t.Self]:
        """Context manager for automatic lifecycle management.

        Example:
            ```python
            async with runtime.context():
                result = await runtime.infer(input)
            ```
        """
        await self.load()
        try:
            yield self
        finally:
            await self.unload()

    async def __aenter__(self) -> t.Self:
        """Async context manager entry."""
        await self.load()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Async context manager exit."""
        await self.unload()


class BatchableRuntime(Runtime[P, R], abc.ABC):
    """Runtime that supports batch inference natively.

    Some runtimes (like TorchScript, ONNX) can process multiple inputs
    simultaneously for better throughput.
    """

    @abc.abstractmethod
    async def infer_batch(self, inputs: list[P]) -> list[R]:
        """Run inference on a batch of inputs.

        Args:
            inputs: List of preprocessed inputs.

        Returns:
            List of raw outputs, one per input.

        Raises:
            RuntimeError: If model is not loaded.
        """

    async def infer(self, input: P) -> R:
        """Single inference (delegates to batch inference)."""
        results = await self.infer_batch([input])
        return results[0]

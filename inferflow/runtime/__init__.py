from __future__ import annotations

import abc
import contextlib
import importlib
import pathlib
import typing as t

from inferflow.types import Device
from inferflow.types import P
from inferflow.types import Precision
from inferflow.types import R

if t.TYPE_CHECKING:
    import os
    import types


__doctitle__ = "Inference Runtimes"


class RuntimeConfigMixin:
    """Shared configuration and validation logic for all runtimes.

    This mixin provides common configuration handling and validation that is
    shared across all runtime implementations (sync and async, all backends).

    It handles:
        - Model path validation
        - Device configuration
        - Precision settings
        - Warmup configuration
        - Input shape specification

    Attributes:
        model_path: Path to the model file.
        device: Device to run inference on.
        precision: Model precision (FP32, FP16, etc.).
        warmup_iterations: Number of warmup iterations.
        warmup_shape: Input shape for warmup.

    Args:
        model_path: Path to model file.
        device: Device specification (e.g., "cpu", "cuda: 0", "mps").
        precision: Model precision (default: FP32).
        warmup_iterations: Number of warmup iterations (default: 3).
        warmup_shape: Input shape for warmup (default: (1, 3, 224, 224)).

    Raises:
        FileNotFoundError: If model file does not exist.
        ValueError: If warmup_iterations is negative.

    Example:
        ```python
        class MyRuntime(RuntimeConfigMixin, Runtime):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # self.model_path, self.device, etc. are now available
        ```
    """

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        device: str | Device,
        precision: Precision = Precision.FP32,
        warmup_iterations: int = 3,
        warmup_shape: tuple[int, ...] = (1, 3, 224, 224),
    ):
        self.model_path = pathlib.Path(model_path)
        self.device = Device(device) if isinstance(device, str) else device
        self.precision = precision
        self.warmup_iterations = warmup_iterations
        self.warmup_shape = warmup_shape

        self._validate_config()

    def _validate_config(self) -> None:
        """Validate runtime configuration.

        Checks:
            - Model file exists
            - Warmup iterations is non-negative

        Raises:
            FileNotFoundError: If model file does not exist.
            ValueError: If warmup_iterations is negative.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found:  {self.model_path}")

        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations must be non-negative")


class Runtime(abc.ABC, t.Generic[P, R]):
    """Abstract runtime for model inference (sync version).

    A runtime encapsulates:
        - Model loading/unloading
        - Device management
        - Inference execution
        - Memory management

    Type Parameters:
        P:  Preprocessed input type (e.g., torch.Tensor, np.ndarray)
        R: Raw output type from the model

    This is the synchronous version of the runtime. For async support,
    see `inferflow.asyncio.runtime.Runtime`.

    Example:
        ```python
        import inferflow as iff

        runtime = iff.TorchScriptRuntime(
            model_path="model.pt",
            device="cuda: 0",
        )

        # Using context manager
        with runtime:
            result = runtime.infer(input_tensor)

        # Manual lifecycle
        runtime.load()
        try:
            result = runtime.infer(input_tensor)
        finally:
            runtime.unload()
        ```
    """

    @abc.abstractmethod
    def load(self) -> None:
        """Load model into memory and prepare for inference.

        This method should:
            - Load model weights from disk
            - Move model to target device
            - Perform warmup inference
            - Set model to evaluation mode

        Raises:
            FileNotFoundError: If model file does not exist.
            RuntimeError: If device is not available.
        """

    @abc.abstractmethod
    def infer(self, input: P) -> R:
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
            with runtime:
                output = runtime.infer(input_tensor)
            ```
        """

    @abc.abstractmethod
    def unload(self) -> None:
        """Unload model and free resources.

        This method should:
            - Release model from memory
            - Clear device cache
            - Close any open handles

        Example:
            ```python
            runtime.load()
            # ... do inference ...
            runtime.unload()  # Free resources
            ```
        """

    @contextlib.contextmanager
    def context(self) -> t.Iterator[t.Self]:
        """Context manager for automatic lifecycle management.

        Automatically calls `load()` on entry and `unload()` on exit,
        even if an exception occurs.

        Yields:
            Self: The runtime instance.

        Example:
            ```python
            with runtime.context():
                result = runtime.infer(input)
            # Model is automatically unloaded here
            ```
        """
        self.load()
        try:
            yield self
        finally:
            self.unload()

    def __enter__(self) -> t.Self:
        """Context manager entry.

        Loads the model.

        Returns:
            Self: The runtime instance.

        Example:
            ```python
            with runtime:  # Calls __enter__
                result = runtime.infer(input)
            ```
        """
        self.load()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Context manager exit.

        Unloads the model, even if an exception occurred.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        self.unload()


class BatchableRuntime(Runtime[P, R], abc.ABC):
    """Runtime that supports batch inference natively (sync version).

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
        with runtime:
            # Single inference (delegates to batch)
            result = runtime.infer(input)

            # Batch inference (more efficient)
            results = runtime.infer_batch([input1, input2, input3])
        ```
    """

    @abc.abstractmethod
    def infer_batch(self, inputs: list[P]) -> list[R]:
        """Run inference on a batch of inputs.

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
            with runtime:
                # Prepare batch
                batch = [
                    torch.randn(1, 3, 224, 224),
                    torch.randn(1, 3, 224, 224),
                    torch.randn(1, 3, 224, 224),
                ]

                # Batch inference
                results = runtime.infer_batch(batch)

                # results[0], results[1], results[2] correspond to inputs
            ```
        """

    def infer(self, input: P) -> R:
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
            with runtime:
                # These are equivalent:
                result = runtime.infer(input)
                result = runtime.infer_batch([input])[0]
            ```
        """
        results = self.infer_batch([input])
        return results[0]


__all__ = ["Runtime", "BatchableRuntime", "RuntimeConfigMixin", "onnx", "tensorrt", "torch"]


def __getattr__(name: str) -> t.Any:
    if name in __all__:
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

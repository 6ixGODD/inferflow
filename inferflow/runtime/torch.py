from __future__ import annotations

import logging
import os
import typing as t

try:
    import torch
except ImportError as e:
    raise ImportError("Torch is required. Install with: pip install 'inferflow[torch]'") from e

from inferflow.runtime import BatchableRuntime
from inferflow.runtime import RuntimeConfigMixin
from inferflow.types import Device
from inferflow.types import Precision
from inferflow.types import R

__doctitle__ = "TorchScript Runtime"

logger = logging.getLogger("inferflow.runtime.torch")


class TorchRuntimeMixin:
    """Shared TorchScript runtime logic for sync and async implementations.

    This mixin provides common TorchScript-specific logic that is shared
    between synchronous and asynchronous runtime implementations. It handles:

        - Device setup (CUDA, CPU, MPS)
        - Precision conversion (FP32, FP16)
        - Input preparation and validation
        - Batch dimension management
        - Output post-processing

    This mixin is pure logic with no I/O operations, making it safe to
    reuse across sync and async implementations.

    Attributes:
        device: Device configuration (provided by subclass).
        precision: Precision configuration (provided by subclass).

    Example:
        ```python
        # In sync runtime
        class TorchScriptRuntime(
            TorchRuntimeMixin, RuntimeConfigMixin, BatchableRuntime
        ):
            def load(self):
                self._torch_device = (
                    self._setup_torch_device()
                )  # Use mixin
                # ...


        # In async runtime
        class TorchScriptRuntime(
            TorchRuntimeMixin, RuntimeConfigMixin, BatchableRuntime
        ):
            async def load(self):
                self._torch_device = (
                    self._setup_torch_device()
                )  # Same mixin!
                # ...
        ```
    """

    device: t.Any
    precision: Precision

    def _setup_torch_device(self) -> torch.device:
        """Setup torch device based on configuration.

        Validates device availability and returns a torch.device object.
        This is pure logic with no I/O.

        Returns:
            torch.device: Configured device.

        Raises:
            RuntimeError: If CUDA/MPS is requested but not available.

        Example:
            ```python
            # In load() method
            self._torch_device = self._setup_torch_device()
            self.model.to(self._torch_device)
            ```
        """
        if self.device.type.value == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            return torch.device(f"cuda:{self.device.index}")

        if self.device.type.value == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS is not available")
            return torch.device("mps")

        return torch.device("cpu")

    def _apply_precision_to_model(
        self,
        model: torch.jit.ScriptModule,
    ) -> torch.jit.ScriptModule:
        """Apply precision conversion to model.

        Converts model to specified precision (e.g., FP16). This is an
        in-place operation on the model.

        Args:
            model: TorchScript model to convert.

        Returns:
            torch.jit.ScriptModule: Model with precision applied.

        Example:
            ```python
            # In load() method
            self.model = torch.jit.load("model.pt")
            self.model = self._apply_precision_to_model(self.model)
            ```
        """
        if self.precision == Precision.FP16:
            return model.half()
        return model

    def _prepare_input(
        self,
        input: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Prepare input tensor for inference.

        Handles:
            - Moving tensor to target device
            - Converting to target precision

        Args:
            input: Input tensor.
            device: Target device.

        Returns:
            torch.Tensor: Prepared input tensor.

        Example:
            ```python
            # In infer() method
            input = self._prepare_input(input, self._torch_device)
            output = self.model(input)
            ```
        """
        if input.device != device:
            input = input.to(device)

        if self.precision == Precision.FP16 and input.dtype != torch.float16:
            input = input.half()

        return input

    def _add_batch_dim_if_needed(
        self,
        input: torch.Tensor,
        auto_add: bool,
    ) -> tuple[torch.Tensor, bool]:
        """Add batch dimension to input if needed.

        Some models expect 4D input (batch, channel, height, width) but
        users may provide 3D input (channel, height, width). This method
        automatically adds the batch dimension if configured.

        Args:
            input: Input tensor.
            auto_add: Whether to automatically add batch dimension.

        Returns:
            Tuple of (prepared_input, was_batch_added).

        Example:
            ```python
            # Input:  (3, 224, 224)
            input, added = self._add_batch_dim_if_needed(input, True)
            # Output: (1, 3, 224, 224), True
            ```
        """
        added_batch = False
        if auto_add and input.ndim == 3:
            input = input.unsqueeze(0)
            added_batch = True
        return input, added_batch

    def _remove_batch_dim_if_added(
        self,
        output: t.Any,
        added_batch: bool,
    ) -> t.Any:
        """Remove batch dimension if it was automatically added.

        If we added a batch dimension in `_add_batch_dim_if_needed()`,
        remove it from the output to maintain consistency.

        Args:
            output: Model output (tensor or tuple of tensors).
            added_batch: Whether batch dimension was added.

        Returns:
            Output with batch dimension removed if applicable.

        Example:
            ```python
            # If added_batch=True and output shape is (1, 1000)
            output = self._remove_batch_dim_if_added(
                output, added_batch
            )
            # Output shape becomes (1000,)
            ```
        """
        if not added_batch:
            return output

        if isinstance(output, torch.Tensor):
            return output.squeeze(0)

        if isinstance(output, (tuple, list)):
            return type(output)(o.squeeze(0) if isinstance(o, torch.Tensor) else o for o in output)

        return output

    def _split_batch_output(
        self,
        batch_output: t.Any,
        batch_size: int,
    ) -> list[t.Any]:
        """Split batched output into list of individual outputs.

        After batch inference, split the output tensor(s) back into a
        list, one entry per input. Maintains batch dimension for each output.

        Args:
            batch_output: Batched model output.
            batch_size: Number of inputs in the batch.

        Returns:
            List of outputs, one per input.

        Raises:
            TypeError: If output type is not supported.

        Example:
            ```python
            # batch_output shape: (3, 1000)
            results = self._split_batch_output(batch_output, 3)
            # results[0] shape: (1, 1000)
            # results[1] shape: (1, 1000)
            # results[2] shape: (1, 1000)
            ```
        """
        if isinstance(batch_output, torch.Tensor):
            return [batch_output[i : i + 1] for i in range(batch_size)]

        if isinstance(batch_output, (tuple, list)):
            results = []
            for i in range(batch_size):
                result_tuple = tuple(
                    output[i : i + 1] if isinstance(output, torch.Tensor) else output for output in batch_output
                )
                results.append(result_tuple)
            return results

        raise TypeError(f"Unexpected output type: {type(batch_output)}")


class TorchScriptRuntime(
    RuntimeConfigMixin,
    TorchRuntimeMixin,
    BatchableRuntime[torch.Tensor, R],
):
    """TorchScript model runtime (sync version).

    Supports:
        - TorchScript (.pt, .pth) models
        - CUDA, CPU, MPS devices
        - FP32, FP16 precision
        - Batch inference
        - Automatic warmup
        - Optional automatic batch dimension handling

    Attributes:
        model: Loaded TorchScript model (None before load()).
        auto_add_batch_dim: Whether to auto-add batch dimension for 3D inputs.

    Args:
        model_path: Path to TorchScript model file.
        device: Device to run inference on (default: "cpu").
        precision: Model precision (default: FP32).
        warmup_iterations: Number of warmup iterations (default: 3).
        warmup_shape: Input shape for warmup (default: (1, 3, 224, 224)).
        auto_add_batch_dim: Whether to automatically add batch dimension
            if input is 3D (default: False).

    Raises:
        FileNotFoundError: If model file does not exist.
        RuntimeError: If CUDA/MPS is requested but not available.
        ImportError: If torch is not installed.

    Example:
        ```python
        import inferflow as iff
        import torch

        # Initialize runtime
        runtime = iff.TorchScriptRuntime(
            model_path="model.pt",
            device="cuda: 0",
            precision=iff.Precision.FP16,
            auto_add_batch_dim=True,
        )

        # Single inference
        with runtime:
            input_tensor = torch.randn(3, 224, 224)  # 3D input
            output = runtime.infer(
                input_tensor
            )  # Batch dim auto-added

        # Batch inference
        with runtime:
            batch = [
                torch.randn(1, 3, 224, 224),
                torch.randn(1, 3, 224, 224),
            ]
            outputs = runtime.infer_batch(batch)
        ```
    """

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        device: str | Device,
        precision: Precision = Precision.FP32,
        warmup_iterations: int = 3,
        warmup_shape: tuple[int, ...] = (1, 3, 224, 224),
        auto_add_batch_dim: bool = False,
    ):
        super().__init__(
            model_path=model_path,
            device=device,
            precision=precision,
            warmup_iterations=warmup_iterations,
            warmup_shape=warmup_shape,
        )

        self.auto_add_batch_dim = auto_add_batch_dim

        self.model: torch.jit.ScriptModule | None = None
        self._torch_device: torch.device | None = None

        logger.info(
            f"TorchScriptRuntime initialized:  "
            f"model={self.model_path}, device={self.device}, "
            f"precision={self.precision.value}"
        )

    def load(self) -> None:
        """Load TorchScript model and prepare for inference.

        Performs:
        - Load model from disk
        - Setup device
        - Move model to device
        - Set evaluation mode
        - Apply precision
        - Warmup inference

        Raises:
            FileNotFoundError: If model file does not exist.
            RuntimeError: If device is not available.
        """
        logger.info(f"Loading model from {self.model_path}")

        # Load model
        self.model = torch.jit.load(str(self.model_path))

        # Setup device (reuse mixin)
        self._torch_device = self._setup_torch_device()

        # Configure model
        self.model.to(self._torch_device)
        self.model.eval()
        self.model = self._apply_precision_to_model(self.model)

        logger.info(f"Model loaded on {self._torch_device}")

        # Warmup
        self._warmup()

    def _warmup(self) -> None:
        """Warmup model with dummy inputs.

        Runs several inference iterations to:
        - Initialize CUDA contexts
        - Optimize kernel selection
        - Stabilize inference latency

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self.model is None or self._torch_device is None:
            raise RuntimeError("Model not loaded.  Call load() first.")

        dummy = torch.zeros(*self.warmup_shape).to(self._torch_device)

        if self.precision == Precision.FP16:
            dummy = dummy.half()

        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                self.model(dummy)

        logger.info("Warmup completed")

    def infer(self, input: torch.Tensor) -> R:
        """Run inference on a single input.

        Automatically handles:
        - Moving input to correct device
        - Converting to correct precision
        - Adding batch dimension (if configured)
        - Removing batch dimension (if added)

        Args:
            input: Input tensor. Can be 3D (C, H, W) if auto_add_batch_dim=True,
                or 4D (1, C, H, W) otherwise.

        Returns:
            Model output.  Type depends on model architecture (tensor or tuple).

        Raises:
            RuntimeError: If model is not loaded.

        Example:
            ```python
            with runtime:
                # 3D input (auto_add_batch_dim=True)
                input = torch.randn(3, 224, 224)
                output = runtime.infer(input)

                # 4D input
                input = torch.randn(1, 3, 224, 224)
                output = runtime.infer(input)
            ```
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Prepare input (reuse mixin)
        input = self._prepare_input(input, self._torch_device)
        input, added_batch = self._add_batch_dim_if_needed(input, self.auto_add_batch_dim)

        # Inference
        with torch.no_grad():
            output = self.model(input)

        # Post-process (reuse mixin)
        return self._remove_batch_dim_if_added(output, added_batch)

    def infer_batch(self, inputs: list[torch.Tensor]) -> list[R]:
        """Run inference on a batch of inputs.

        Concatenates inputs into a single batch tensor for efficient
        processing, then splits the output back into individual results.

        Args:
            inputs: List of input tensors. Each should have shape (1, C, H, W).

        Returns:
            List of outputs, one per input. Each maintains batch dimension.

        Raises:
            RuntimeError: If model is not loaded.

        Example:
            ```python
            with runtime:
                batch = [
                    torch.randn(1, 3, 224, 224),
                    torch.randn(1, 3, 224, 224),
                    torch.randn(1, 3, 224, 224),
                ]

                # Efficient batch processing
                outputs = runtime.infer_batch(batch)

                # outputs[0], outputs[1], outputs[2]
            ```
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Concatenate
        batch = torch.cat(inputs, dim=0).to(self._torch_device)

        if self.precision == Precision.FP16:
            batch = batch.half()

        # Inference
        with torch.no_grad():
            batch_output = self.model(batch)

        # Split (reuse mixin)
        return self._split_batch_output(batch_output, len(inputs))

    def unload(self) -> None:
        """Unload model and free resources.

        Performs:
        - Release model from memory
        - Clear CUDA cache (if using CUDA)

        Safe to call multiple times.
        """
        logger.info("Unloading model")
        self.model = None

        if self.device.type.value == "cuda":
            torch.cuda.empty_cache()

        logger.info("Model unloaded")


__all__ = [
    "TorchRuntimeMixin",
    "TorchScriptRuntime",
]

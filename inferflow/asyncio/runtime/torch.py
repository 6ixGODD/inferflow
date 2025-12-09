from __future__ import annotations

import asyncio
import logging
import typing as t

try:
    import torch
except ImportError as e:
    raise ImportError("Torch is required. Install with: pip install 'inferflow[torch]'") from e

from inferflow.asyncio.runtime import BatchableRuntime
from inferflow.asyncio.runtime import RuntimeConfigMixin
from inferflow.runtime.torch import TorchRuntimeMixin
from inferflow.types import Precision

logger = logging.getLogger("inferflow.asyncio.runtime.torch")


class TorchScriptRuntime(
    RuntimeConfigMixin,
    TorchRuntimeMixin,
    BatchableRuntime[torch.Tensor, t.Any],
):
    """TorchScript model runtime (async version).

    **Asynchronous version** of `inferflow.runtime.torch.TorchScriptRuntime`.

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
        import inferflow.asyncio as iff
        import torch

        # Initialize runtime
        runtime = iff.TorchScriptRuntime(
            model_path="model.pt",
            device="cuda: 0",
            precision=iff.Precision.FP16,
            auto_add_batch_dim=True,
        )

        # Single inference
        async with runtime:
            input_tensor = torch.randn(3, 224, 224)  # 3D input
            output = await runtime.infer(
                input_tensor
            )  # Batch dim auto-added

        # Batch inference
        async with runtime:
            batch = [
                torch.randn(1, 3, 224, 224),
                torch.randn(1, 3, 224, 224),
            ]
            outputs = await runtime.infer_batch(batch)
        ```
    """

    def __init__(self, *args, auto_add_batch_dim: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.auto_add_batch_dim = auto_add_batch_dim

        self.model: torch.jit.ScriptModule | None = None
        self._torch_device: torch.device | None = None

        logger.info(
            f"TorchScriptRuntime (async) initialized: "
            f"model={self.model_path}, device={self.device}, "
            f"precision={self.precision.value}"
        )

    async def load(self) -> None:
        """Load TorchScript model and prepare for inference (async).

        Performs:
            - Load model from disk (in thread pool)
            - Setup device
            - Move model to device
            - Set evaluation mode
            - Apply precision
            - Warmup inference (in thread pool)

        Raises:
            FileNotFoundError: If model file does not exist.
            RuntimeError: If device is not available.
        """
        logger.info(f"Loading model from {self.model_path}")

        # Load model in thread pool
        loop = asyncio.get_event_loop()
        self.model = await loop.run_in_executor(None, torch.jit.load, str(self.model_path))  # type: ignore

        # Setup device (reuse mixin)
        self._torch_device = self._setup_torch_device()

        # Configure model (reuse mixin)
        self.model.to(self._torch_device)
        self.model.eval()
        self.model = self._apply_precision_to_model(self.model)

        logger.info(f"Model loaded on {self._torch_device}")

        # Warmup
        await self._warmup()

    async def _warmup(self) -> None:
        """Warmup model with dummy inputs (async).

        Runs several inference iterations to:
            - Initialize CUDA contexts
            - Optimize kernel selection
            - Stabilize inference latency

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self.model is None or self._torch_device is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        dummy = torch.zeros(*self.warmup_shape).to(self._torch_device)

        if self.precision == Precision.FP16:
            dummy = dummy.half()

        loop = asyncio.get_event_loop()

        for _ in range(self.warmup_iterations):
            await loop.run_in_executor(None, self._infer_sync, dummy)

        logger.info("Warmup completed")

    def _infer_sync(self, input: torch.Tensor) -> t.Any:
        """Sync inference (runs in thread pool)."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        with torch.no_grad():
            return self.model(input)

    async def infer(self, input: torch.Tensor) -> t.Any:
        """Run inference on a single input (async).

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
            async with runtime:
                # 3D input (auto_add_batch_dim=True)
                input = torch.randn(3, 224, 224)
                output = await runtime.infer(input)

                # 4D input
                input = torch.randn(1, 3, 224, 224)
                output = await runtime.infer(input)
            ```
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Prepare input (reuse mixin)
        input = self._prepare_input(input, self._torch_device)
        input, added_batch = self._add_batch_dim_if_needed(input, self.auto_add_batch_dim)

        # Inference in thread pool
        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(None, self._infer_sync, input)

        # Post-process (reuse mixin)
        return self._remove_batch_dim_if_added(output, added_batch)

    async def infer_batch(self, inputs: list[torch.Tensor]) -> list[t.Any]:
        """Run inference on a batch of inputs (async).

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
            async with runtime:
                batch = [
                    torch.randn(1, 3, 224, 224),
                    torch.randn(1, 3, 224, 224),
                    torch.randn(1, 3, 224, 224),
                ]

                # Efficient batch processing
                outputs = await runtime.infer_batch(batch)

                # outputs[0], outputs[1], outputs[2]
            ```
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Concatenate
        batch = torch.cat(inputs, dim=0).to(self._torch_device)

        if self.precision == Precision.FP16:
            batch = batch.half()

        # Inference in thread pool
        loop = asyncio.get_event_loop()
        batch_output = await loop.run_in_executor(None, self._infer_sync, batch)

        # Split (reuse mixin)
        return self._split_batch_output(batch_output, len(inputs))

    async def unload(self) -> None:
        """Unload model and free resources (async).

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
    "TorchScriptRuntime",
]

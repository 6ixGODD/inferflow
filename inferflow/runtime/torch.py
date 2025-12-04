from __future__ import annotations

import asyncio as aio
import logging
import typing as t

import torch

from inferflow.runtime import BatchableRuntime
from inferflow.types import Device
from inferflow.types import Precision

logger = logging.getLogger("inferflow.runtime.torch")


class TorchScriptRuntime(BatchableRuntime[torch.Tensor, t.Any]):
    """TorchScript model runtime.

    Supports:
    - TorchScript (. pt, .pth) models
    - CUDA, CPU, MPS devices
    - FP32, FP16 precision
    - Batch inference
    - Automatic warmup

    Args:
        model_path: Path to TorchScript model file.
        device: Device to run inference on (default: "cpu").
        precision: Model precision (default: FP32).
        warmup_iterations: Number of warmup iterations (default: 3).
        warmup_shape: Input shape for warmup (default: (1, 3, 224, 224)).
        auto_add_batch_dim: Whether to automatically add batch dimension if
            input is 3D (default: False).

    Example:
        ```python
        runtime = TorchScriptRuntime(
            model_path="model.pt",
            device="cuda:0",
            precision=Precision.FP16,
        )

        async with runtime:
            result = await runtime.infer(input_tensor)
        ```
    """

    def __init__(
        self,
        model_path: str,
        device: str | Device = "cpu",
        precision: Precision = Precision.FP32,
        warmup_iterations: int = 3,
        warmup_shape: tuple[int, ...] = (1, 3, 224, 224),
        auto_add_batch_dim: bool = False,
    ):
        self.model_path = model_path
        self.device = Device(device) if isinstance(device, str) else device
        self.precision = precision
        self.warmup_iterations = warmup_iterations
        self.warmup_shape = warmup_shape
        self.auto_add_batch_dim = auto_add_batch_dim

        self.model: torch.jit.ScriptModule | None = None
        self._torch_device: torch.device | None = None

        logger.info(
            f"TorchScriptRuntime initialized: "
            f"model={model_path}, device={self.device}, precision={precision.value}, "
            f"auto_add_batch_dim={auto_add_batch_dim}"
        )

    async def load(self) -> None:
        """Load TorchScript model and prepare for inference."""
        logger.info(f"Loading TorchScript model from {self.model_path}")

        # Load model in thread pool to avoid blocking
        loop = aio.get_event_loop()
        self.model = await loop.run_in_executor(None, torch.jit.load, self.model_path)  # type: ignore

        # Setup device
        if self.device.type.value == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            self._torch_device = torch.device(f"cuda:{self.device.index}")
        elif self.device.type.value == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS is not available")
            self._torch_device = torch.device("mps")
        else:
            self._torch_device = torch.device("cpu")

        # Move model to device
        self.model.to(self._torch_device)
        self.model.eval()

        # Apply precision
        if self.precision == Precision.FP16:
            self.model.half()

        logger.info(f"Model loaded on device: {self._torch_device}")

        # Warmup
        await self._warmup()

    async def _warmup(self) -> None:
        """Warmup the model with dummy inputs."""
        logger.info(f"Warming up model with {self.warmup_iterations} iterations")

        # For warmup, use the shape as-is
        dummy_input = torch.zeros(*self.warmup_shape).to(self._torch_device)
        if self.precision == Precision.FP16:
            dummy_input = dummy_input.half()

        loop = aio.get_event_loop()

        for _i in range(self.warmup_iterations):
            await loop.run_in_executor(None, self._infer_sync, dummy_input)

        logger.info("Model warmup completed")

    def _infer_sync(self, input: torch.Tensor) -> t.Any:
        """Synchronous inference (runs in thread pool)."""
        if self.model is None:
            raise RuntimeError("Model not loaded.  Call load() first.")

        with torch.no_grad():
            return self.model(input)

    async def infer(self, input: torch.Tensor) -> t.Any:
        """Run inference on a single input.

        Args:
            input: Input tensor (will be moved to device automatically).

        Returns:
            Output tensor or tuple of tensors.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Move input to device if needed
        if input.device != self._torch_device:
            input = input.to(self._torch_device)

        # Apply precision if needed
        if self.precision == Precision.FP16 and input.dtype != torch.float16:
            input = input.half()

        # Auto add batch dimension if needed
        added_batch = False
        if self.auto_add_batch_dim and input.ndim == 3:
            input = input.unsqueeze(0)
            added_batch = True

        # Run inference in thread pool
        loop = aio.get_event_loop()
        result = await loop.run_in_executor(None, self._infer_sync, input)

        # Remove batch dimension if we added it
        if added_batch:
            if isinstance(result, torch.Tensor):
                result = result.squeeze(0)
            elif isinstance(result, (tuple, list)):
                result = type(result)(r.squeeze(0) if isinstance(r, torch.Tensor) else r for r in result)

        return result

    async def infer_batch(self, inputs: list[torch.Tensor]) -> list[t.Any]:
        """Run inference on a batch of inputs.

        Args:
            inputs: List of input tensors.

        Returns:
            List of output tensors.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.  Call load() first.")

        # Concatenate inputs into batch (inputs already have batch dim)
        # inputs: [(1, 3, 640, 640), (1, 3, 640, 640), ...]
        # -> batch: (N, 3, 640, 640)
        batch = torch.cat(inputs, dim=0).to(self._torch_device)

        if self.precision == Precision.FP16 and batch.dtype != torch.float16:
            batch = batch.half()

        # Run batch inference
        loop = aio.get_event_loop()
        batch_output = await loop.run_in_executor(None, self._infer_sync, batch)

        # Split batch output back to list
        batch_size = len(inputs)

        if isinstance(batch_output, torch.Tensor):
            # Single tensor output: split along batch dimension
            return [batch_output[i : i + 1] for i in range(batch_size)]
        if isinstance(batch_output, (tuple, list)):
            # Multiple outputs (e.g., detection + mask protos)
            # Each output tensor has batch dimension
            results = []
            for i in range(batch_size):
                result_tuple = tuple(
                    output[i : i + 1] if isinstance(output, torch.Tensor) else output for output in batch_output
                )
                results.append(result_tuple)
            return results
        raise TypeError(f"Unexpected output type: {type(batch_output)}")

    async def unload(self) -> None:
        """Unload model and free resources."""
        logger.info("Unloading model and freeing resources")

        self.model = None

        # Clear CUDA cache if using CUDA
        if self.device.type.value == "cuda":
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")

        logger.info("Model unloaded successfully")

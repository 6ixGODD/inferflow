from __future__ import annotations

import asyncio as aio
import logging

import torch

from inferflow.runtime import BatchableRuntime
from inferflow.types import Device
from inferflow.types import Precision

logger = logging.getLogger("inferflow.runtime.torch")


class TorchScriptRuntime(BatchableRuntime[torch.Tensor, torch.Tensor]):
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
    ):
        self.model_path = model_path
        self.device = Device(device) if isinstance(device, str) else device
        self.precision = precision
        self.warmup_iterations = warmup_iterations
        self.warmup_shape = warmup_shape

        self.model: torch.jit.ScriptModule | None = None
        self._torch_device: torch.device | None = None

        logger.info(
            f"TorchScriptRuntime initialized: model={model_path}, device={self.device}, precision={precision.value}"
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

        dummy_input = torch.zeros(*self.warmup_shape).to(self._torch_device)
        if self.precision == Precision.FP16:
            dummy_input = dummy_input.half()

        loop = aio.get_event_loop()

        for _i in range(self.warmup_iterations):
            await loop.run_in_executor(None, self._infer_sync, dummy_input)

        logger.info("Model warmup completed")

    def _infer_sync(self, input: torch.Tensor) -> torch.Tensor:
        """Synchronous inference (runs in thread pool)."""
        if self.model is None:
            raise RuntimeError("Model not loaded.  Call load() first.")

        with torch.no_grad():
            return self.model(input)

    async def infer(self, input: torch.Tensor) -> torch.Tensor:
        """Run inference on a single input.

        Args:
            input: Input tensor (will be moved to device automatically).

        Returns:
            Output tensor.

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

        # Run inference in thread pool
        loop = aio.get_event_loop()
        return await loop.run_in_executor(None, self._infer_sync, input)

    async def infer_batch(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """Run inference on a batch of inputs.

        Args:
            inputs: List of input tensors.

        Returns:
            List of output tensors.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Stack inputs into batch
        batch = torch.stack(inputs).to(self._torch_device)

        if self.precision == Precision.FP16 and batch.dtype != torch.float16:
            batch = batch.half()

        # Run batch inference
        loop = aio.get_event_loop()
        batch_output = await loop.run_in_executor(None, self._infer_sync, batch)

        # Split batch output back to list
        return [batch_output[i] for i in range(len(inputs))]

    async def unload(self) -> None:
        """Unload model and free resources."""
        logger.info("Unloading model and freeing resources")

        self.model = None

        # Clear CUDA cache if using CUDA
        if self.device.type.value == "cuda":
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")

        logger.info("Model unloaded successfully")

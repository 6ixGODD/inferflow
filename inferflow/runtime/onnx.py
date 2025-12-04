from __future__ import annotations

import asyncio as aio
import logging
import os
import pathlib
import typing as t

try:
    import numpy as np
    import onnxruntime as ort
except ImportError as e:
    raise ImportError("ONNX Runtime is required. Install with: pip install 'inferflow[onnx]'") from e

from inferflow.runtime import BatchableRuntime
from inferflow.types import Device
from inferflow.types import Precision

logger = logging.getLogger("inferflow.runtime.onnx")


class ONNXRuntime(BatchableRuntime[np.ndarray, t.Any]):
    """ONNX Runtime for model inference.

    Supports:
    - ONNX (.onnx) models
    - CPU, CUDA execution providers
    - FP32, FP16 precision
    - Batch inference
    - Automatic warmup

    Args:
        model_path: Path to ONNX model file.
        device: Device to run inference on (default: "cpu").
        precision: Model precision (default: FP32).
        warmup_iterations: Number of warmup iterations (default: 3).
        warmup_shape: Input shape for warmup (default: (1, 3, 224, 224)).
        providers: ONNX execution providers (default: auto-detect).

    Example:
        ```python
        runtime = ONNXRuntime(
            model_path="model.onnx",
            device="cuda:0",
        )

        async with runtime:
            result = await runtime.infer(input_array)
        ```
    """

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        device: str | Device = "cpu",
        precision: Precision = Precision.FP32,
        warmup_iterations: int = 3,
        warmup_shape: tuple[int, ...] = (1, 3, 224, 224),
        providers: list[str] | None = None,
    ):
        self.model_path = pathlib.Path(model_path)
        self.device = Device(device) if isinstance(device, str) else device
        self.precision = precision
        self.warmup_iterations = warmup_iterations
        self.warmup_shape = warmup_shape

        # Auto-detect providers
        if providers is None:
            if self.device.type.value == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
        self.providers = providers

        self.session: ort.InferenceSession | None = None
        self.input_name: str | None = None
        self.output_names: list[str] | None = None

        logger.info(
            f"ONNXRuntime initialized: "
            f"model={model_path}, device={self.device}, "
            f"providers={providers}, precision={precision.value}"
        )

    async def load(self) -> None:
        """Load ONNX model and prepare for inference."""
        logger.info(f"Loading ONNX model from {self.model_path}")

        # Session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load in thread pool
        loop = aio.get_event_loop()
        self.session = await loop.run_in_executor(
            None,
            lambda _: ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=self.providers,
            ),
            None,
        )

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        logger.info(
            f"Model loaded: input={self.input_name}, "
            f"outputs={self.output_names}, "
            f"providers={self.session.get_providers()}"
        )

        # Warmup
        await self._warmup()

    async def _warmup(self) -> None:
        """Warmup the model with dummy inputs."""
        logger.info(f"Warming up model with {self.warmup_iterations} iterations")

        dummy_input = np.zeros(self.warmup_shape, dtype=np.float32)
        if self.precision == Precision.FP16:
            dummy_input = dummy_input.astype(np.float16)

        loop = aio.get_event_loop()

        for _ in range(self.warmup_iterations):
            await loop.run_in_executor(None, self._infer_sync, dummy_input)

        logger.info("Model warmup completed")

    def _infer_sync(self, input: np.ndarray) -> t.Any:
        """Synchronous inference."""
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        outputs = self.session.run(
            self.output_names,
            {self.input_name: input},
        )

        # Return single output or tuple
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    async def infer(self, input: np.ndarray) -> t.Any:
        """Run inference on a single input.

        Args:
            input: Input numpy array (will be converted to correct dtype).

        Returns:
            Output array or tuple of arrays.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self.session is None:
            raise RuntimeError("Model not loaded.  Call load() first.")

        # Apply precision
        if self.precision == Precision.FP16 and input.dtype != np.float16:
            input = input.astype(np.float16)
        elif input.dtype != np.float32:
            input = input.astype(np.float32)

        # Run inference in thread pool
        loop = aio.get_event_loop()
        return await loop.run_in_executor(None, self._infer_sync, input)

    async def infer_batch(self, inputs: list[np.ndarray]) -> list[t.Any]:
        """Run inference on a batch of inputs.

        Args:
            inputs: List of input arrays.

        Returns:
            List of output arrays.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Concatenate inputs
        batch = np.concatenate(inputs, axis=0)

        # Apply precision
        if self.precision == Precision.FP16 and batch.dtype != np.float16:
            batch = batch.astype(np.float16)
        elif batch.dtype != np.float32:
            batch = batch.astype(np.float32)

        # Run batch inference
        loop = aio.get_event_loop()
        batch_output = await loop.run_in_executor(None, self._infer_sync, batch)

        batch_size = len(inputs)

        # Split outputs
        if isinstance(batch_output, np.ndarray):
            return [batch_output[i : i + 1] for i in range(batch_size)]
        if isinstance(batch_output, tuple):
            results = []
            for i in range(batch_size):
                result_tuple = tuple(
                    output[i : i + 1] if isinstance(output, np.ndarray) else output for output in batch_output
                )
                results.append(result_tuple)
            return results
        raise TypeError(f"Unexpected output type: {type(batch_output)}")

    async def unload(self) -> None:
        """Unload model and free resources."""
        logger.info("Unloading model and freeing resources")
        self.session = None
        logger.info("Model unloaded successfully")

# inferflow/asyncio/runtime/onnx.py

from __future__ import annotations

import asyncio
import logging
import typing as t

try:
    import numpy as np
    import onnxruntime as ort
except ImportError as e:
    raise ImportError("ONNX Runtime is required. Install with: pip install 'inferflow[onnx]'") from e

from inferflow.asyncio.runtime import BatchableRuntime
from inferflow.asyncio.runtime import RuntimeConfigMixin
from inferflow.runtime.onnx import ONNXRuntimeMixin

logger = logging.getLogger("inferflow.asyncio.runtime.onnx")


class ONNXRuntime(
    RuntimeConfigMixin,
    ONNXRuntimeMixin,
    BatchableRuntime[np.ndarray, t.Any],
):
    """ONNX Runtime for model inference (async version).

    **Asynchronous version** of `inferflow.runtime.onnx.ONNXRuntime`.

    All I/O operations (model loading, inference) are executed in a thread pool
    to avoid blocking the event loop. The API is identical to the sync version,
    but all methods are async.

    Supports:
        - ONNX (.onnx) models
        - CPU, CUDA execution providers
        - FP32, FP16 precision
        - Batch inference
        - Automatic warmup

    Attributes:
        session:  Loaded ONNX inference session (None before load()).
        input_name: Name of the model's input tensor.
        output_names: Names of the model's output tensors.
        providers: List of execution providers to use.

    Args:
        model_path: Path to ONNX model file.
        device: Device to run inference on (default: "cpu").
        precision: Model precision (default: FP32).
        warmup_iterations: Number of warmup iterations (default:  3).
        warmup_shape: Input shape for warmup (default: (1, 3, 224, 224)).
        providers:  ONNX execution providers (default:  auto-detect).

    Raises:
        FileNotFoundError: If model file does not exist.
        ImportError: If onnxruntime is not installed.

    Example:
        ```python
        import inferflow.asyncio as iff
        import numpy as np

        # Initialize runtime
        runtime = iff.ONNXRuntime(
            model_path="model.onnx",
            device="cuda:0",
            precision=iff.Precision.FP16,
        )

        # Single inference
        async with runtime:
            input_array = np.random.randn(1, 3, 224, 224).astype(
                np.float32
            )
            output = await runtime.infer(input_array)

        # Batch inference
        async with runtime:
            batch = [
                np.random.randn(1, 3, 224, 224).astype(np.float32),
                np.random.randn(1, 3, 224, 224).astype(np.float32),
            ]
            outputs = await runtime.infer_batch(batch)
        ```
    """

    def __init__(self, *args, providers: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)

        # Get providers (reuse mixin)
        self.providers = self._get_onnx_providers(providers)

        self.session: ort.InferenceSession | None = None
        self.input_name: str | None = None
        self.output_names: list[str] | None = None

        logger.info(
            f"ONNXRuntime (async) initialized: "
            f"model={self.model_path}, device={self.device}, "
            f"providers={self.providers}, precision={self.precision.value}"
        )

    async def load(self) -> None:
        """Load ONNX model and prepare for inference (async).

        Performs:
            - Configure session options
            - Load model from disk (in thread pool)
            - Extract input/output names
            - Warmup inference (in thread pool)

        Raises:
            FileNotFoundError: If model file does not exist.
            RuntimeError: If ONNX Runtime fails to load model.
        """
        logger.info(f"Loading ONNX model from {self.model_path}")

        # Session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load in thread pool
        loop = asyncio.get_event_loop()
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
            f"Model loaded:  input={self.input_name}, "
            f"outputs={self.output_names}, "
            f"providers={self.session.get_providers()}"
        )

        # Warmup
        await self._warmup()

    async def _warmup(self) -> None:
        """Warmup model with dummy inputs (async).

        Runs several inference iterations to:
            - Initialize execution providers
            - Optimize kernel selection
            - Stabilize inference latency

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self.session is None or self.input_name is None or self.output_names is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        logger.info(f"Warming up model with {self.warmup_iterations} iterations")

        dummy_input = np.zeros(self.warmup_shape, dtype=np.float32)

        # Apply precision (reuse mixin)
        dummy_input = self._prepare_onnx_input(dummy_input)

        loop = asyncio.get_event_loop()

        for _ in range(self.warmup_iterations):
            await loop.run_in_executor(None, self._infer_sync, dummy_input)

        logger.info("Warmup completed")

    def _infer_sync(self, input: np.ndarray) -> t.Any:
        """Sync inference (runs in thread pool)."""
        if self.session is None or self.input_name is None or self.output_names is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        outputs = self.session.run(self.output_names, {self.input_name: input})

        # Parse output (reuse mixin)
        return self._parse_onnx_output(list(outputs))

    async def infer(self, input: np.ndarray) -> t.Any:
        """Run inference on a single input (async).

        Automatically handles:
            - Converting to correct dtype

        Args:
            input: Input numpy array.

        Returns:
            Output array or tuple of arrays (if multi-output model).

        Raises:
            RuntimeError: If model is not loaded.

        Example:
            ```python
            async with runtime:
                input = np.random.randn(1, 3, 224, 224).astype(
                    np.float32
                )
                output = await runtime.infer(input)
            ```
        """
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Prepare input (reuse mixin)
        input = self._prepare_onnx_input(input)

        # Run inference in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._infer_sync, input)

    async def infer_batch(self, inputs: list[np.ndarray]) -> list[t.Any]:
        """Run inference on a batch of inputs (async).

        Concatenates inputs into a single batch array for efficient
        processing, then splits the output back into individual results.

        Args:
            inputs: List of input arrays. Each should have shape (1, C, H, W).

        Returns:
            List of outputs, one per input. Each maintains batch dimension.

        Raises:
            RuntimeError: If model is not loaded.

        Example:
            ```python
            async with runtime:
                batch = [
                    np.random.randn(1, 3, 224, 224).astype(np.float32),
                    np.random.randn(1, 3, 224, 224).astype(np.float32),
                ]
                outputs = await runtime.infer_batch(batch)
            ```
        """
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Concatenate inputs
        batch = np.concatenate(inputs, axis=0)

        # Prepare input (reuse mixin)
        batch = self._prepare_onnx_input(batch)

        # Run batch inference
        batch_output = await self.infer(batch)

        # Split outputs (reuse mixin)
        return self._split_onnx_batch_output(batch_output, len(inputs))

    async def unload(self) -> None:
        """Unload model and free resources (async).

        Performs:
            - Release session from memory

        Safe to call multiple times.
        """
        logger.info("Unloading model")
        self.session = None
        self.input_name = None
        self.output_names = None
        logger.info("Model unloaded")


__all__ = [
    "ONNXRuntime",
]

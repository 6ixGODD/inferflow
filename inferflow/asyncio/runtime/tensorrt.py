from __future__ import annotations

import asyncio
import logging
import typing as t

try:
    import numpy as np
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import tensorrt as trt
except ImportError as e:
    raise ImportError("TensorRT is required. Install with: pip install 'inferflow[tensorrt]'") from e

from inferflow.asyncio.runtime import BatchableRuntime
from inferflow.asyncio.runtime import RuntimeConfigMixin
from inferflow.runtime.tensorrt import TensorRTRuntimeMixin

__doctitle__ = "TensorRT Runtime (Async)"

logger = logging.getLogger("inferflow.asyncio.runtime.tensorrt")


class TensorRTRuntime(
    RuntimeConfigMixin,
    TensorRTRuntimeMixin,
    BatchableRuntime[np.ndarray, t.Any],
):
    """TensorRT Runtime for optimized inference (async version).

    **Asynchronous version** of `inferflow.runtime.tensorrt.TensorRTRuntime`.

    All I/O operations (engine loading, inference) are executed in a thread pool
    to avoid blocking the event loop. The API is identical to the sync version,
    but all methods are async.

    Supports:
        - TensorRT (.engine, .trt) models
        - CUDA devices only
        - FP32, FP16, INT8 precision
        - Batch inference
        - Automatic warmup

    Attributes:
        runtime: TensorRT runtime instance (None before load()).
        engine: TensorRT engine (None before load()).
        context: TensorRT execution context (None before load()).
        inputs: List of input device memory allocations.
        outputs: List of output device memory allocations.
        bindings: List of binding pointers for execution.
        stream:  CUDA stream for async operations.

    Args:
        model_path: Path to TensorRT engine file.
        device: CUDA device (default: "cuda:0").
        warmup_iterations: Number of warmup iterations (default: 3).
        warmup_shape: Input shape for warmup (default: (1, 3, 640, 640)).

    Raises:
        FileNotFoundError: If model file does not exist.
        ValueError: If device is not CUDA.
        ImportError: If tensorrt or pycuda is not installed.

    Example:
        ```python
        import inferflow.asyncio as iff
        import numpy as np

        # Initialize runtime
        runtime = iff.TensorRTRuntime(
            model_path="model.engine",
            device="cuda:0",
        )

        # Single inference
        async with runtime:
            input_array = np.random.randn(1, 3, 640, 640).astype(
                np.float32
            )
            output = await runtime.infer(input_array)

        # Batch inference
        async with runtime:
            batch = [
                np.random.randn(1, 3, 640, 640).astype(np.float32),
                np.random.randn(1, 3, 640, 640).astype(np.float32),
            ]
            outputs = await runtime.infer_batch(batch)
        ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Validate device (reuse mixin)
        self._validate_tensorrt_device()

        self.logger_trt = trt.Logger(trt.Logger.WARNING)
        self.runtime: trt.Runtime | None = None
        self.engine: trt.ICudaEngine | None = None
        self.context: trt.IExecutionContext | None = None

        # CUDA memory
        self.inputs: list[cuda.DeviceAllocation] = []
        self.outputs: list[cuda.DeviceAllocation] = []
        self.bindings: list[int] = []
        self.stream: cuda.Stream | None = None

        logger.info(f"TensorRTRuntime (async) initialized: model={self.model_path}, device={self.device}")

    async def load(self) -> None:
        """Load TensorRT engine and prepare for inference (async).

        Performs:
            - Load engine from disk (in thread pool)
            - Create execution context
            - Allocate device memory for inputs/outputs
            - Create CUDA stream
            - Warmup inference (in thread pool)

        Raises:
            FileNotFoundError: If engine file does not exist.
            RuntimeError: If TensorRT fails to deserialize engine.
        """
        logger.info(f"Loading TensorRT engine from {self.model_path}")

        # Load engine in thread pool
        loop = asyncio.get_event_loop()

        def _load_engine():
            self.runtime = trt.Runtime(self.logger_trt)
            with self.model_path.open("rb") as f:
                engine_data = f.read()
            return self.runtime.deserialize_cuda_engine(engine_data)

        self.engine = await loop.run_in_executor(None, _load_engine)
        self.context = self.engine.create_execution_context()

        # Create CUDA stream
        self.stream = cuda.Stream()

        # Allocate memory for all bindings
        for i in range(self.engine.num_bindings):
            binding_shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))

            # Calculate size (reuse mixin)
            size = self._calculate_binding_size(binding_shape, dtype)

            # Allocate device memory
            mem = cuda.mem_alloc(size)
            self.bindings.append(int(mem))

            if self.engine.binding_is_input(i):
                self.inputs.append(mem)
            else:
                self.outputs.append(mem)

        logger.info(f"Engine loaded: inputs={len(self.inputs)}, outputs={len(self.outputs)}")

        # Warmup
        await self._warmup()

    async def _warmup(self) -> None:
        """Warmup engine with dummy inputs (async).

        Runs several inference iterations to:
            - Initialize CUDA contexts
            - Optimize kernel selection
            - Stabilize inference latency

        Raises:
            RuntimeError: If engine is not loaded.
        """
        if self.context is None or self.stream is None:
            raise RuntimeError("Engine not loaded. Call load() first.")

        logger.info(f"Warming up engine with {self.warmup_iterations} iterations")

        dummy_input = np.zeros(self.warmup_shape, dtype=np.float32)

        loop = asyncio.get_event_loop()

        for _ in range(self.warmup_iterations):
            await loop.run_in_executor(None, self._infer_sync, dummy_input)

        logger.info("Warmup completed")

    def _infer_sync(self, input: np.ndarray) -> t.Any:
        """Sync inference (runs in thread pool)."""
        if self.context is None or self.stream is None:
            raise RuntimeError("Engine not loaded. Call load() first.")

        # Copy input to device
        cuda.memcpy_htod_async(self.inputs[0], input, self.stream)

        # Execute
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # Copy output to host
        output_shape = self.engine.get_binding_shape(1)
        output_dtype = trt.nptype(self.engine.get_binding_dtype(1))
        output = np.empty(trt.volume(output_shape), dtype=output_dtype)

        cuda.memcpy_dtoh_async(output, self.outputs[0], self.stream)
        self.stream.synchronize()

        return output.reshape(output_shape)

    async def infer(self, input: np.ndarray) -> t.Any:
        """Run inference on a single input (async).

        Uses CUDA async operations for efficient processing.

        Args:
            input: Input numpy array.

        Returns:
            Output array with shape determined by model.

        Raises:
            RuntimeError: If engine is not loaded.

        Example:
            ```python
            async with runtime:
                input = np.random.randn(1, 3, 640, 640).astype(
                    np.float32
                )
                output = await runtime.infer(input)
            ```
        """
        if self.context is None:
            raise RuntimeError("Engine not loaded. Call load() first.")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._infer_sync, input)

    async def infer_batch(self, inputs: list[np.ndarray]) -> list[t.Any]:
        """Run inference on a batch of inputs (async).

        Concatenates inputs and runs batch inference, then splits
        the output back into individual results.

        Args:
            inputs:  List of input arrays. Each should have shape (1, C, H, W).

        Returns:
            List of outputs, one per input. Each maintains batch dimension.

        Raises:
            RuntimeError: If engine is not loaded.

        Example:
            ```python
            async with runtime:
                batch = [
                    np.random.randn(1, 3, 640, 640).astype(np.float32),
                    np.random.randn(1, 3, 640, 640).astype(np.float32),
                ]
                outputs = await runtime.infer_batch(batch)
            ```
        """
        if self.context is None:
            raise RuntimeError("Engine not loaded. Call load() first.")

        # Concatenate and infer
        batch = np.concatenate(inputs, axis=0)
        batch_output = await self.infer(batch)

        # Split outputs
        batch_size = len(inputs)
        if isinstance(batch_output, np.ndarray):
            return [batch_output[i : i + 1] for i in range(batch_size)]

        raise TypeError(f"Unexpected output type: {type(batch_output)}")

    async def unload(self) -> None:
        """Unload engine and free resources (async).

        Performs:
            - Free CUDA device memory
            - Release engine and context

        Safe to call multiple times.
        """
        logger.info("Unloading engine")

        # Free CUDA memory
        for mem in self.inputs + self.outputs:
            mem.free()

        self.inputs.clear()
        self.outputs.clear()
        self.bindings.clear()

        self.context = None
        self.engine = None
        self.runtime = None
        self.stream = None

        logger.info("Engine unloaded")


__all__ = [
    "TensorRTRuntime",
]

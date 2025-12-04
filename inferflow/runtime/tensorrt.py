from __future__ import annotations

import asyncio as aio
import logging
import os
import pathlib
import typing as t

try:
    import numpy as np
    import pycuda.autoinit  # noqa
    import pycuda.driver as cuda
    import tensorrt as trt

except ImportError as e:
    raise ImportError("TensorRT is required. Install with: pip install 'inferflow[tensorrt]'") from e

from inferflow.runtime import BatchableRuntime
from inferflow.types import Device

logger = logging.getLogger("inferflow.runtime.tensorrt")


class TensorRTRuntime(BatchableRuntime[np.ndarray, t.Any]):
    """TensorRT Runtime for optimized inference.

    Supports:
    - TensorRT (.engine, .trt) models
    - CUDA devices only
    - FP32, FP16, INT8 precision
    - Batch inference
    - Automatic warmup

    Args:
        model_path: Path to TensorRT engine file.
        device: CUDA device index (default: 0).
        warmup_iterations: Number of warmup iterations (default: 3).
        warmup_shape: Input shape for warmup (default: (1, 3, 224, 224)).

    Example:
        ```python
        runtime = TensorRTRuntime(
            model_path="model.engine",
            device="cuda:0",
        )

        async with runtime:
            result = await runtime.infer(input_array)
        ```
    """

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        device: str | Device = "cuda:0",
        warmup_iterations: int = 3,
        warmup_shape: tuple[int, ...] = (1, 3, 640, 640),
    ):
        self.model_path = pathlib.Path(model_path)
        self.device = Device(device) if isinstance(device, str) else device
        self.warmup_iterations = warmup_iterations
        self.warmup_shape = warmup_shape

        if self.device.type.value != "cuda":
            raise ValueError("TensorRT only supports CUDA devices")

        self.logger_trt = trt.Logger(trt.Logger.WARNING)
        self.runtime: trt.Runtime | None = None
        self.engine: trt.ICudaEngine | None = None
        self.context: trt.IExecutionContext | None = None

        # CUDA memory
        self.inputs: list[cuda.DeviceAllocation] = []
        self.outputs: list[cuda.DeviceAllocation] = []
        self.bindings: list[int] = []
        self.stream: cuda.Stream | None = None

        logger.info(f"TensorRTRuntime initialized: model={model_path}, device={self.device}")

    async def load(self) -> None:
        """Load TensorRT engine and prepare for inference."""
        logger.info(f"Loading TensorRT engine from {self.model_path}")

        # Load engine
        loop = aio.get_event_loop()

        def _load_engine():
            self.runtime = trt.Runtime(self.logger_trt)
            with self.model_path.open("rb") as f:
                engine_data = f.read()
            return self.runtime.deserialize_cuda_engine(engine_data)

        self.engine = await loop.run_in_executor(None, _load_engine)
        self.context = self.engine.create_execution_context()

        # Create CUDA stream
        self.stream = cuda.Stream()

        # Allocate memory
        for i in range(self.engine.num_bindings):
            binding_shape = self.engine.get_binding_shape(i)
            size = trt.volume(binding_shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))

            # Allocate device memory
            mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)

            self.bindings.append(int(mem))

            if self.engine.binding_is_input(i):
                self.inputs.append(mem)
            else:
                self.outputs.append(mem)

        logger.info(f"Engine loaded: inputs={len(self.inputs)}, outputs={len(self.outputs)}")

        # Warmup
        await self._warmup()

    async def _warmup(self) -> None:
        """Warmup the engine with dummy inputs."""
        logger.info(f"Warming up engine with {self.warmup_iterations} iterations")

        dummy_input = np.zeros(self.warmup_shape, dtype=np.float32)

        loop = aio.get_event_loop()

        for _ in range(self.warmup_iterations):
            await loop.run_in_executor(None, self._infer_sync, dummy_input)

        logger.info("Engine warmup completed")

    def _infer_sync(self, input: np.ndarray) -> t.Any:
        """Synchronous inference."""
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
        """Run inference on a single input."""
        if self.context is None:
            raise RuntimeError("Engine not loaded. Call load() first.")

        loop = aio.get_event_loop()
        return await loop.run_in_executor(None, self._infer_sync, input)

    async def infer_batch(self, inputs: list[np.ndarray]) -> list[t.Any]:
        """Run inference on a batch of inputs."""
        # Concatenate and infer
        batch = np.concatenate(inputs, axis=0)
        batch_output = await self.infer(batch)

        # Split outputs
        batch_size = len(inputs)
        if isinstance(batch_output, np.ndarray):
            return [batch_output[i : i + 1] for i in range(batch_size)]
        raise TypeError(f"Unexpected output type: {type(batch_output)}")

    async def unload(self) -> None:
        """Unload engine and free resources."""
        logger.info("Unloading engine and freeing resources")

        # Free CUDA memory
        for mem in self.inputs + self.outputs:
            mem.free()

        self.inputs.clear()
        self.outputs.clear()
        self.bindings.clear()

        self.context = None
        self.engine = None
        self.runtime = None

        logger.info("Engine unloaded successfully")

from __future__ import annotations

import logging
import typing as t

try:
    import numpy as np
    import onnxruntime as ort
except ImportError as e:
    raise ImportError("ONNX Runtime is required. Install with: pip install 'inferflow[onnx]'") from e

from inferflow.runtime import BatchableRuntime
from inferflow.runtime import RuntimeConfigMixin
from inferflow.types import Precision

logger = logging.getLogger("inferflow.runtime.onnx")


class ONNXRuntimeMixin:
    """Shared ONNX runtime logic for sync and async implementations.

    This mixin provides common ONNX-specific logic that is shared between
    synchronous and asynchronous runtime implementations. It handles:

        - Execution provider selection (CPU, CUDA)
        - Input precision conversion
        - Output parsing
        - Batch output splitting

    This mixin is pure logic with no I/O operations, making it safe to
    reuse across sync and async implementations.

    Attributes:
        device: Device configuration (provided by subclass).
        precision:  Precision configuration (provided by subclass).

    Example:
        ```python
        # In sync runtime
        class ONNXRuntime(
            ONNXRuntimeMixin, RuntimeConfigMixin, BatchableRuntime
        ):
            def load(self):
                providers = self._get_onnx_providers()  # Use mixin
                # ...


        # In async runtime
        class ONNXRuntime(
            ONNXRuntimeMixin, RuntimeConfigMixin, BatchableRuntime
        ):
            async def load(self):
                providers = (
                    self._get_onnx_providers()
                )  # Same mixin!
                # ...
        ```
    """

    device: t.Any
    precision: Precision

    def _get_onnx_providers(self, custom_providers: list[str] | None = None) -> list[str]:
        """Get ONNX execution providers based on device configuration.

        Auto-detects appropriate providers based on device type. Supports
        custom provider lists for advanced use cases.

        Args:
            custom_providers: Optional list of custom providers. If None,
                auto-detect based on device.

        Returns:
            List of execution provider names in priority order.

        Example:
            ```python
            # Auto-detect
            providers = self._get_onnx_providers()
            # Returns ["CUDAExecutionProvider", "CPUExecutionProvider"] for CUDA

            # Custom
            providers = self._get_onnx_providers([
                "TensorrtExecutionProvider"
            ])
            ```
        """
        if custom_providers is not None:
            return custom_providers

        if self.device.type.value == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        return ["CPUExecutionProvider"]

    def _prepare_onnx_input(self, input: np.ndarray) -> np.ndarray:
        """Prepare ONNX input array with correct dtype.

        Converts input to the precision specified in configuration.

        Args:
            input: Input numpy array.

        Returns:
            Input array with correct dtype.

        Example:
            ```python
            # In infer() method
            input = self._prepare_onnx_input(input)
            outputs = self.session.run(...)
            ```
        """
        if self.precision == Precision.FP16 and input.dtype != np.float16:
            return input.astype(np.float16)

        if self.precision == Precision.FP32 and input.dtype != np.float32:
            return input.astype(np.float32)

        return input

    def _parse_onnx_output(self, outputs: list) -> t.Any:
        """Parse ONNX session output.

        ONNX sessions return a list of outputs. This method converts
        single-output models to a single array, and multi-output models
        to a tuple.

        Args:
            outputs: List of output arrays from ONNX session.

        Returns:
            Single array if one output, tuple of arrays if multiple.

        Example:
            ```python
            outputs = self.session.run(...)
            return self._parse_onnx_output(outputs)
            # Returns np.ndarray or tuple[np.ndarray, ...]
            ```
        """
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def _split_onnx_batch_output(
        self,
        batch_output: t.Any,
        batch_size: int,
    ) -> list[t.Any]:
        """Split ONNX batch output into list of individual outputs.

        After batch inference, split the output array(s) back into a list,
        one entry per input. Maintains batch dimension for each output.

        Args:
            batch_output:  Batched model output (array or tuple).
            batch_size: Number of inputs in the batch.

        Returns:
            List of outputs, one per input.

        Raises:
            TypeError: If output type is not supported.

        Example:
            ```python
            # batch_output shape: (3, 1000)
            results = self._split_onnx_batch_output(batch_output, 3)
            # results[0] shape: (1, 1000)
            # results[1] shape: (1, 1000)
            # results[2] shape: (1, 1000)
            ```
        """
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


class ONNXRuntime(
    RuntimeConfigMixin,
    ONNXRuntimeMixin,
    BatchableRuntime[np.ndarray, t.Any],
):
    """ONNX Runtime for model inference (sync version).

    Supports:
        - ONNX (.onnx) models
        - CPU, CUDA execution providers
        - FP32, FP16 precision
        - Batch inference
        - Automatic warmup

    This is the synchronous version. For async support, see
    `inferflow.asyncio.runtime.onnx.ONNXRuntime`.

    Attributes:
        session:  Loaded ONNX inference session (None before load()).
        input_name: Name of the model's input tensor.
        output_names: Names of the model's output tensors.
        providers: List of execution providers to use.

    Args:
        model_path: Path to ONNX model file.
        device: Device to run inference on (default:  "cpu").
        precision: Model precision (default: FP32).
        warmup_iterations: Number of warmup iterations (default: 3).
        warmup_shape: Input shape for warmup (default: (1, 3, 224, 224)).
        providers:  ONNX execution providers (default:  auto-detect).

    Raises:
        FileNotFoundError: If model file does not exist.
        ImportError: If onnxruntime is not installed.

    Example:
        ```python
        import inferflow as iff
        import numpy as np

        # Initialize runtime
        runtime = iff.ONNXRuntime(
            model_path="model.onnx",
            device="cuda:0",
            precision=iff.Precision.FP16,
        )

        # Single inference
        with runtime:
            input_array = np.random.randn(1, 3, 224, 224).astype(
                np.float32
            )
            output = runtime.infer(input_array)

        # Batch inference
        with runtime:
            batch = [
                np.random.randn(1, 3, 224, 224).astype(np.float32),
                np.random.randn(1, 3, 224, 224).astype(np.float32),
            ]
            outputs = runtime.infer_batch(batch)
        ```
    """

    def __init__(
        self,
        *args,
        providers: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Get providers (reuse mixin)
        self.providers = self._get_onnx_providers(providers)

        self.session: ort.InferenceSession | None = None
        self.input_name: str | None = None
        self.output_names: list[str] | None = None

        logger.info(
            f"ONNXRuntime initialized: "
            f"model={self.model_path}, device={self.device}, "
            f"providers={self.providers}, precision={self.precision.value}"
        )

    def load(self) -> None:
        """Load ONNX model and prepare for inference.

        Performs:
            - Configure session options
            - Load model from disk
            - Extract input/output names
            - Warmup inference

        Raises:
            FileNotFoundError: If model file does not exist.
            RuntimeError: If ONNX Runtime fails to load model.
        """
        logger.info(f"Loading ONNX model from {self.model_path}")

        # Session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load model
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=self.providers,
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
        self._warmup()

    def _warmup(self) -> None:
        """Warmup model with dummy inputs.

        Runs several inference iterations to:
            - Initialize execution providers
            - Optimize kernel selection
            - Stabilize inference latency

        Raises:
            RuntimeError:  If model is not loaded.
        """
        if self.session is None or self.input_name is None or self.output_names is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        logger.info(f"Warming up model with {self.warmup_iterations} iterations")

        dummy_input = np.zeros(self.warmup_shape, dtype=np.float32)

        # Apply precision (reuse mixin)
        dummy_input = self._prepare_onnx_input(dummy_input)

        for _ in range(self.warmup_iterations):
            self.session.run(self.output_names, {self.input_name: dummy_input})

        logger.info("Warmup completed")

    def infer(self, input: np.ndarray) -> t.Any:
        """Run inference on a single input.

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
            with runtime:
                input = np.random.randn(1, 3, 224, 224).astype(
                    np.float32
                )
                output = runtime.infer(input)
            ```
        """
        if self.session is None or self.input_name is None or self.output_names is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Prepare input (reuse mixin)
        input = self._prepare_onnx_input(input)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input})

        # Parse output (reuse mixin)
        return self._parse_onnx_output(list(outputs))

    def infer_batch(self, inputs: list[np.ndarray]) -> list[t.Any]:
        """Run inference on a batch of inputs.

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
            with runtime:
                batch = [
                    np.random.randn(1, 3, 224, 224).astype(np.float32),
                    np.random.randn(1, 3, 224, 224).astype(np.float32),
                ]
                outputs = runtime.infer_batch(batch)
            ```
        """
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Concatenate inputs
        batch = np.concatenate(inputs, axis=0)

        # Prepare input (reuse mixin)
        batch = self._prepare_onnx_input(batch)

        # Run batch inference
        batch_output = self.infer(batch)

        # Split outputs (reuse mixin)
        return self._split_onnx_batch_output(batch_output, len(inputs))

    def unload(self) -> None:
        """Unload model and free resources.

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
    "ONNXRuntimeMixin",
    "ONNXRuntime",
]

from __future__ import annotations

import abc
import contextlib
import importlib
import typing as t

from inferflow.types import O
from inferflow.types import P
from inferflow.types import R

if t.TYPE_CHECKING:
    from inferflow.asyncio.batch import BatchStrategy
    from inferflow.asyncio.runtime import Runtime
    from inferflow.types import ImageInput

__doctitle__ = "Pipeline (Async)"


class Pipeline(abc.ABC, t.Generic[P, R, O]):
    """Abstract inference pipeline (async version).

    A pipeline combines:
    - Preprocessing: Convert raw input to model-ready format
    - Inference: Run model inference via runtime
    - Postprocessing: Convert raw output to structured result

    Example:
        ```python
        pipeline = ClassificationPipeline(runtime=runtime)
        async with pipeline.serve():
            result = await pipeline(image)
            print(
                f"Class: {result.class_name}, Confidence: {result.confidence}"
            )
        ```
    """

    def __init__(
        self,
        runtime: Runtime[P, R],
        batch_strategy: BatchStrategy[P, R] | None = None,
    ):
        """Initialize pipeline.

        Args:
            runtime: Inference runtime.
            batch_strategy: Optional batching strategy for improved throughput.
        """
        self.runtime = runtime
        self.batch_strategy = batch_strategy

    @abc.abstractmethod
    async def preprocess(self, input: ImageInput) -> P:
        """Preprocess raw input into model-ready format.

        Args:
            input: Raw input (image bytes, numpy array, PIL Image, etc.)

        Returns:
            Preprocessed input ready for inference.
        """

    @abc.abstractmethod
    async def postprocess(self, raw: R) -> O:
        """Postprocess raw model output into structured result.

        Args:
            raw: Raw output from model inference.

        Returns:
            Structured output (classification result, detections, etc.)
        """

    async def infer(self, preprocessed: P) -> R:
        """Run inference on preprocessed input.

        This method automatically uses batching if a batch strategy is configured.

        Args:
            preprocessed: Preprocessed input.

        Returns:
            Raw inference result.
        """
        if self.batch_strategy:
            return await self.batch_strategy.submit(preprocessed)
        return await self.runtime.infer(preprocessed)

    async def __call__(self, input: ImageInput) -> O:
        """End-to-end inference.

        Args:
            input: Raw input.

        Returns:
            Structured output.

        Example:
            ```python
            result = await pipeline(image_bytes)
            ```
        """
        preprocessed = await self.preprocess(input)
        raw = await self.infer(preprocessed)
        return await self.postprocess(raw)

    @contextlib.asynccontextmanager
    async def serve(self) -> t.AsyncIterator[t.Self]:
        """Start serving pipeline with automatic lifecycle management.

        This method:
            - Loads the runtime
            - Starts batch processing (if enabled)
            - Yields the pipeline for inference
            - Cleans up resources on exit

        Example:
            ```python
            async with pipeline.serve():
                result = await pipeline(image)
            ```
        """
        async with self.runtime.context():
            if self.batch_strategy:
                await self.batch_strategy.start(self.runtime)

            try:
                yield self
            finally:
                if self.batch_strategy:
                    await self.batch_strategy.stop()


__all__ = ["Pipeline", "classification", "detection", "segmentation"]


def __getattr__(name: str) -> t.Any:
    if name in __all__:
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from __future__ import annotations

import typing as t

import numpy as np

from inferflow.asyncio.pipeline import Pipeline
from inferflow.pipeline.detection.onnx import YOLODetectionMixin
from inferflow.types import DetectionOutput

if t.TYPE_CHECKING:
    from inferflow.asyncio.batch import BatchStrategy
    from inferflow.asyncio.runtime.onnx import ONNXRuntime
    from inferflow.types import ImageInput


class YOLOv5DetectionPipeline(
    YOLODetectionMixin,
    Pipeline[np.ndarray, tuple[np.ndarray, ...] | np.ndarray, list[DetectionOutput]],
):
    """YOLOv5 object detection pipeline (async ONNX version)."""

    def __init__(
        self,
        runtime: ONNXRuntime,
        image_size: tuple[int, int] = (640, 640),
        stride: int = 32,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_names: dict[int, str] | None = None,
        batch_strategy: BatchStrategy[np.ndarray, tuple[np.ndarray, ...] | np.ndarray] | None = None,
    ):
        super().__init__(runtime=runtime, batch_strategy=batch_strategy)

        self.image_size = image_size
        self.stride = stride
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or {}

        self._runtime = runtime
        self._original_size = None
        self._padding = None

    async def preprocess(self, input: ImageInput) -> np.ndarray:
        """Preprocess image input (async)."""
        image = self._convert_to_numpy(input)
        return self._preprocess_numpy(image)

    async def postprocess(self, raw: tuple[np.ndarray, ...] | np.ndarray) -> list[DetectionOutput]:
        """Postprocess YOLOv5 output (async)."""
        predictions = raw if isinstance(raw, np.ndarray) else raw[0]
        return self._postprocess_detections(predictions)


__all__ = ["YOLOv5DetectionPipeline"]

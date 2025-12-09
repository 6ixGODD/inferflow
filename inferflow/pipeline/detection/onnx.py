from __future__ import annotations

import typing as t

import cv2
import numpy as np
import PIL.Image as Image

from inferflow._utils.yolo.np import nms
from inferflow._utils.yolo.np import padding_resize
from inferflow._utils.yolo.np import scale_bbox
from inferflow._utils.yolo.np import xyxy2xywh
from inferflow.pipeline import Pipeline
from inferflow.runtime.onnx import ONNXRuntime
from inferflow.types import Box
from inferflow.types import DetectionOutput

if t.TYPE_CHECKING:
    from inferflow.batch import BatchStrategy
    from inferflow.runtime import RuntimeConfigMixin
    from inferflow.types import ImageInput


class YOLODetectionMixin:
    """Shared YOLOv5 detection logic (ONNX version)."""

    image_size: tuple[int, int]
    stride: int
    conf_threshold: float
    iou_threshold: float
    class_names: dict[int, str]

    _runtime: RuntimeConfigMixin

    _original_size: tuple[int, int] | None
    _padding: tuple[int, int] | None

    def _convert_to_numpy(self, input: ImageInput) -> np.ndarray:
        """Convert input to numpy array (BGR)."""
        if isinstance(input, bytes):
            nparr = np.frombuffer(input, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode image")
            return image

        if isinstance(input, Image.Image):
            image = np.array(input.convert("RGB"))
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if isinstance(input, np.ndarray):
            return input.copy()

        raise ValueError(f"Unsupported input type:  {type(input)}")

    def _preprocess_numpy(self, image: np.ndarray) -> np.ndarray:
        """Preprocess numpy image to numpy array."""
        h, w = image.shape[:2]
        self._original_size = (w, h)

        image, padding = padding_resize(image, self.image_size, self.stride, full=True)
        self._padding = padding

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.ascontiguousarray(image.transpose(2, 0, 1))

        # Normalize to [0, 1]
        array = image.astype(np.float32) / 255.0
        array = np.expand_dims(array, axis=0)

        from inferflow.types import Precision

        if self._runtime.precision == Precision.FP16:
            array = array.astype(np.float16)

        return array

    def _postprocess_detections(self, predictions: np.ndarray) -> list[DetectionOutput]:
        """Postprocess YOLO detections."""
        if predictions.dtype == np.float16:
            predictions = predictions.astype(np.float32)

        filtered = nms(
            predictions,
            conf_thres=self.conf_threshold,
            iou_thres=self.iou_threshold,
            nm=0,
            max_det=1000,
        )

        bbox = filtered[0]
        if len(bbox) == 0:
            return []

        boxes_xyxy = bbox[:, :4]
        confidences = bbox[:, 4]
        class_ids = bbox[:, 5].astype(int)

        boxes_xywh = xyxy2xywh(boxes_xyxy)

        detections = []
        for box_xywh, conf, class_id in zip(boxes_xywh, confidences, class_ids, strict=False):
            scaled_box = scale_bbox(
                tuple(box_xywh.tolist()),  # type: ignore
                self.image_size,
                self._original_size,
                self._padding,
            )

            cx, cy, w, h = scaled_box

            detections.append(
                DetectionOutput(
                    box=Box(xc=cx, yc=cy, w=w, h=h),
                    class_id=int(class_id),
                    confidence=float(conf),
                    class_name=self.class_names.get(int(class_id)),
                )
            )

        return detections


class YOLOv5DetectionPipeline(
    YOLODetectionMixin,
    Pipeline[np.ndarray, tuple[np.ndarray, ...] | np.ndarray, list[DetectionOutput]],
):
    """YOLOv5 object detection pipeline (ONNX version)."""

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

    def preprocess(self, input: ImageInput) -> np.ndarray:
        """Preprocess image input."""
        image = self._convert_to_numpy(input)
        return self._preprocess_numpy(image)

    def postprocess(self, raw: tuple[np.ndarray, ...] | np.ndarray) -> list[DetectionOutput]:
        """Postprocess YOLOv5 output."""
        predictions = raw if isinstance(raw, np.ndarray) else raw[0]
        return self._postprocess_detections(predictions)


__all__ = ["YOLODetectionMixin", "YOLOv5DetectionPipeline"]

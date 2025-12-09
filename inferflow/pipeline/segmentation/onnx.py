from __future__ import annotations

import typing as t

import numpy as np

from inferflow._utils.yolo.np import nms
from inferflow._utils.yolo.np import process_mask
from inferflow._utils.yolo.np import scale_bbox
from inferflow._utils.yolo.np import scale_mask
from inferflow._utils.yolo.np import xyxy2xywh
from inferflow.pipeline import Pipeline
from inferflow.pipeline.detection.onnx import YOLODetectionMixin
from inferflow.types import Box
from inferflow.types import SegmentationOutput

if t.TYPE_CHECKING:
    from inferflow.batch import BatchStrategy
    from inferflow.runtime import Runtime
    from inferflow.types import ImageInput


class YOLOSegmentationMixin(YOLODetectionMixin):
    """Shared YOLOv5 segmentation logic (ONNX version)."""

    def _postprocess_segmentation(
        self,
        detections: np.ndarray,
        protos: np.ndarray,
    ) -> list[SegmentationOutput]:
        """Postprocess segmentation output."""
        if detections.dtype == np.float16:
            detections = detections.astype(np.float32)
        if protos.dtype == np.float16:
            protos = protos.astype(np.float32)

        filtered = nms(
            detections,
            conf_thres=self.conf_threshold,
            iou_thres=self.iou_threshold,
            nm=32,
            max_det=1000,
        )

        bbox = filtered[0]
        proto = protos[0] if protos.ndim == 4 else protos

        if len(bbox) == 0:
            return []

        masks = process_mask(proto, bbox[:, 6:], bbox[:, :4], self.image_size, upsample=True)

        if masks.ndim == 2:
            masks = np.expand_dims(masks, axis=0)

        bbox = bbox[:, :6]

        results = []
        for mask_np, (*xyxy, conf, cls) in zip(masks, bbox, strict=False):
            xywh = xyxy2xywh(np.array(xyxy).reshape(1, 4)).ravel()

            scaled_box = scale_bbox(
                tuple(xywh.tolist()),  # type: ignore
                self.image_size,
                self._original_size,
                self._padding,
            )

            scaled_mask = scale_mask(mask_np, self._original_size, self._padding)
            scaled_mask = scaled_mask.astype(bool)

            cx, cy, w, h = scaled_box

            results.append(
                SegmentationOutput(
                    mask=scaled_mask,
                    box=Box(xc=cx, yc=cy, w=w, h=h),
                    class_id=int(cls),
                    confidence=float(conf),
                    class_name=self.class_names.get(int(cls)),
                )
            )

        return results


class YOLOv5SegmentationPipeline(
    YOLOSegmentationMixin,
    Pipeline[np.ndarray, tuple[np.ndarray, np.ndarray], list[SegmentationOutput]],
):
    """YOLOv5 instance segmentation pipeline (ONNX version)."""

    def __init__(
        self,
        runtime: Runtime[np.ndarray, tuple[np.ndarray, np.ndarray]],
        image_size: tuple[int, int] = (640, 640),
        stride: int = 32,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_names: dict[int, str] | None = None,
        batch_strategy: BatchStrategy[np.ndarray, tuple[np.ndarray, np.ndarray]] | None = None,
    ):
        super().__init__(runtime=runtime, batch_strategy=batch_strategy)

        self.image_size = image_size
        self.stride = stride
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or {}

        # 👇 保存 runtime 引用
        self._runtime = runtime

        self._original_size = None
        self._padding = None

    def preprocess(self, input: ImageInput) -> np.ndarray:
        """Preprocess image for YOLOv5-Seg."""
        image = self._convert_to_numpy(input)
        return self._preprocess_numpy(image)

    def postprocess(self, raw: tuple[np.ndarray, np.ndarray]) -> list[SegmentationOutput]:
        """Postprocess YOLOv5-Seg output."""
        detections, protos = raw
        return self._postprocess_segmentation(detections, protos)


__all__ = ["YOLOSegmentationMixin", "YOLOv5SegmentationPipeline"]

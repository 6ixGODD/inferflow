from __future__ import annotations

import typing as t

import torch

from inferflow._utils.yolo.torch import nms
from inferflow._utils.yolo.torch import process_mask
from inferflow._utils.yolo.torch import scale_bbox
from inferflow._utils.yolo.torch import scale_mask
from inferflow._utils.yolo.torch import xyxy2xywh
from inferflow.pipeline import Pipeline
from inferflow.pipeline.detection.torch import YOLODetectionMixin
from inferflow.types import Box
from inferflow.types import SegmentationOutput

if t.TYPE_CHECKING:
    from inferflow.batch import BatchStrategy
    from inferflow.runtime import Runtime
    from inferflow.types import ImageInput


class YOLOSegmentationMixin(YOLODetectionMixin):
    """Shared YOLOv5 segmentation logic (extends detection).

    Attributes:
        image_size: Target image size.
        stride: Model stride.
        conf_threshold: Confidence threshold for detections.
        iou_threshold: IoU threshold for NMS.
        class_names: Mapping from class ID to class name.
    """

    def _postprocess_segmentation(self, detections: torch.Tensor, protos: torch.Tensor) -> list[SegmentationOutput]:
        """Postprocess YOLO segmentation."""
        filtered = nms(
            detections,
            conf_thres=self.conf_threshold,
            iou_thres=self.iou_threshold,
            nm=32,
            max_det=1000,
        )

        bbox = filtered[0]
        proto = protos[0]

        if len(bbox) == 0:
            return []

        masks = process_mask(proto, bbox[:, 6:], bbox[:, :4], self.image_size, upsample=True)

        if masks.ndimension() == 2:
            masks = masks.unsqueeze(0)

        masks_np = masks.cpu().numpy()
        bbox = bbox[:, :6]

        results = []
        for mask_np, (*xyxy, conf, cls) in zip(masks_np, bbox, strict=False):
            xywh = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1).tolist()

            scaled_box = scale_bbox(
                tuple(xywh),  # type: ignore[arg-type]
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
                    class_id=int(cls.item()),
                    confidence=float(conf.item()),
                    class_name=self.class_names.get(int(cls.item())),
                )
            )

        return results


class YOLOv5SegmentationPipeline(
    YOLOSegmentationMixin,
    Pipeline[torch.Tensor, tuple[torch.Tensor, torch.Tensor], list[SegmentationOutput]],
):
    """YOLOv5 instance segmentation pipeline (sync version).

    Performs:
        - Image decoding and conversion
        - Image resizing and normalization
        - YOLOv5 inference
        - Instance segmentation mask extraction with NMS

    Args:
        runtime: Inference runtime.
        image_size: Target image size (default: (640, 640)).
        stride: Model stride (default: 32).
        conf_threshold: Confidence threshold for detections (default: 0.25).
        iou_threshold: IoU threshold for NMS (default: 0.45).
        class_names: Optional mapping from class ID to class name.
        batch_strategy: Optional batching strategy.

    Example:
        ```python
        runtime = TorchScriptRuntime(
            model_path="yolov5s.pt", device="cuda"
        )
        pipeline = YOLOv5SegmentationPipeline(
            runtime=runtime,
            class_names={0: "person", 1: "bicycle", 2: "car"},
        )
        with pipeline.serve():
            results = pipeline(image_bytes)
            for result in results:
                print(
                    f"{result.class_name}: {result.confidence:.2%} at {result.box.to_xywh()}"
                )
        ```
    """

    def __init__(
        self,
        runtime: Runtime[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        image_size: tuple[int, int] = (640, 640),
        stride: int = 32,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_names: dict[int, str] | None = None,
        batch_strategy: BatchStrategy[torch.Tensor, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ):
        super().__init__(runtime=runtime, batch_strategy=batch_strategy)

        self.image_size = image_size
        self.stride = stride
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or {}

        self._original_size = None
        self._padding = None

    def preprocess(self, input: ImageInput) -> torch.Tensor:
        """Preprocess image input for YOLOv5-seg.

        Args:
            input: Raw image input (bytes, numpy array, PIL Image, or tensor).

        Returns:
            Preprocessed tensor ready for model inference.
        """
        image = self._convert_to_numpy(input)
        return self._preprocess_numpy(image)

    def postprocess(self, raw: tuple[torch.Tensor, torch.Tensor]) -> list[SegmentationOutput]:
        """Postprocess YOLOv5-Seg output to segmentation results.

        Args:
            raw: Raw model output (detections and protos).

        Returns:
            SegmentationOutput list with masks and bounding boxes.
        """
        detections, protos = raw
        return self._postprocess_segmentation(detections, protos)


__all__ = ["YOLOv5SegmentationPipeline", "YOLOSegmentationMixin"]

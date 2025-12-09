from __future__ import annotations

import typing as t

import torch

from inferflow.asyncio.pipeline import Pipeline
from inferflow.pipeline.segmentation import YOLOSegmentationMixin
from inferflow.types import SegmentationOutput

if t.TYPE_CHECKING:
    from inferflow.asyncio.runtime import Runtime
    from inferflow.batch import BatchStrategy
    from inferflow.types import ImageInput


class YOLOv5SegmentationPipeline(
    YOLOSegmentationMixin,
    Pipeline[torch.Tensor, tuple[torch.Tensor, torch.Tensor], list[SegmentationOutput]],
):
    """YOLOv5 instance segmentation pipeline (async version).

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
        async with pipeline.serve():
            results = await pipeline(image_bytes)
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

    async def preprocess(self, input: ImageInput) -> torch.Tensor:
        """Preprocess image input for YOLOv5-seg.

        Args:
            input: Raw image input (bytes, numpy array, PIL Image, or tensor).

        Returns:
            Preprocessed tensor ready for model inference.
        """
        image = self._convert_to_numpy(input)
        return self._preprocess_numpy(image)

    async def postprocess(self, raw: tuple[torch.Tensor, torch.Tensor]) -> list[SegmentationOutput]:
        """Postprocess YOLOv5-Seg output to segmentation results.

        Args:
            raw: Raw model output (detections and protos).

        Returns:
            SegmentationOutput list with masks and bounding boxes.
        """
        detections, protos = raw
        return self._postprocess_segmentation(detections, protos)


__all__ = ["YOLOv5SegmentationPipeline"]

"""Object detection pipeline implementation."""

from __future__ import annotations

import typing as t

import cv2
import numpy as np
from PIL import Image
import torch

from inferflow.pipeline import Pipeline
from inferflow.types import Box
from inferflow.types import DetectionOutput
from inferflow.types import ImageInput
from inferflow.utils.yolo import nms
from inferflow.utils.yolo import padding_resize
from inferflow.utils.yolo import scale_bbox
from inferflow.utils.yolo import xyxy2xywh

if t.TYPE_CHECKING:
    from inferflow.batch import BatchStrategy
    from inferflow.runtime import Runtime


class YOLOv5DetectionPipeline(Pipeline[torch.Tensor, tuple[torch.Tensor, ...], list[DetectionOutput]]):
    """YOLOv5 object detection pipeline.

    Performs:
    - Image preprocessing (resize with padding, normalization)
    - YOLOv5 inference
    - NMS (Non-Maximum Suppression)
    - Bounding box rescaling to original image size

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
        pipeline = YOLOv5DetectionPipeline(
            runtime=runtime,
            conf_threshold=0.5,
            class_names={0: "person", 1: "car", 2: "dog"},
        )

        async with pipeline.serve():
            detections = await pipeline(image_bytes)
            for det in detections:
                print(
                    f"{det.class_name}: {det.confidence:.2%} at {det.box}"
                )
        ```
    """

    def __init__(
        self,
        runtime: Runtime[torch.Tensor, tuple[torch.Tensor, ...]],
        image_size: tuple[int, int] = (640, 640),
        stride: int = 32,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_names: dict[int, str] | None = None,
        batch_strategy: BatchStrategy[torch.Tensor, tuple[torch.Tensor, ...]] | None = None,
    ):
        super().__init__(runtime=runtime, batch_strategy=batch_strategy)

        self.image_size = image_size
        self.stride = stride
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or {}

        self._original_size: tuple[int, int] | None = None
        self._padding: tuple[int, int] | None = None

    async def preprocess(self, input: ImageInput) -> torch.Tensor:
        """Preprocess image for YOLOv5.

        Args:
            input: Image as bytes, numpy array, PIL Image, or torch.Tensor.

        Returns:
            Preprocessed tensor (1, 3, H, W) ready for YOLOv5 inference.
        """
        # Convert to numpy
        if isinstance(input, bytes):
            nparr = np.frombuffer(input, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode image")
        elif isinstance(input, Image.Image):
            image = np.array(input.convert("RGB"))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif isinstance(input, np.ndarray):
            image = input.copy()
        elif isinstance(input, torch.Tensor):
            if input.ndim == 4:
                input = input.squeeze(0)
            if input.ndim == 3 and input.shape[0] in [1, 3]:
                image = input.permute(1, 2, 0).cpu().numpy()
                image = (image * 255).astype(np.uint8)
            else:
                raise ValueError(f"Unsupported tensor shape: {input.shape}")
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")

        # Store original size
        h, w = image.shape[:2]
        self._original_size = (w, h)

        # Padding resize
        image, padding = padding_resize(image, self.image_size, self.stride, full=True)
        self._padding = padding

        # Convert to RGB and normalize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.ascontiguousarray(image.transpose(2, 0, 1))  # HWC -> CHW

        # To tensor and normalize
        tensor = torch.from_numpy(image).float() / 255.0

        # Add batch dimension
        return tensor.unsqueeze(0)

    async def postprocess(self, raw: tuple[torch.Tensor, ...]) -> list[DetectionOutput]:
        """Postprocess YOLOv5 output to detection results.

        Args:
            raw: Tuple of model outputs. First element is predictions (batch, N, 6)
                 where each detection is [xyxy, conf, class_id].

        Returns:
            List of DetectionOutput objects.
        """
        # YOLOv5 detection returns tuple, take first element (predictions)
        predictions = raw[0]

        # Apply NMS
        filtered = nms(
            predictions,
            conf_thres=self.conf_threshold,
            iou_thres=self.iou_threshold,
            nm=0,  # No mask coefficients for detection
            max_det=1000,
        )

        # Take first batch item
        bbox = filtered[0]

        if len(bbox) == 0:
            return []

        # Extract components [xyxy, conf, cls]
        boxes_xyxy = bbox[:, :4]
        confidences = bbox[:, 4]
        class_ids = bbox[:, 5].long()

        # Convert xyxy to xywh (center format)
        boxes_xywh = xyxy2xywh(boxes_xyxy)

        # Scale boxes to original image
        detections = []
        for box_xywh, conf, class_id in zip(boxes_xywh, confidences, class_ids, strict=False):
            scaled_box = scale_bbox(
                tuple(box_xywh.tolist()),  # type: ignore
                self.image_size,
                self._original_size,  # type: ignore
                self._padding,  # type: ignore
            )

            cx, cy, w, h = scaled_box

            detections.append(
                DetectionOutput(
                    box=Box(xc=cx, yc=cy, w=w, h=h),
                    class_id=int(class_id.item()),
                    confidence=float(conf.item()),
                    class_name=self.class_names.get(int(class_id.item())),
                )
            )

        return detections

from __future__ import annotations

import typing as t

import cv2
import numpy as np
import PIL.Image as Image
import torch

from inferflow.pipeline import Pipeline
from inferflow.types import Box
from inferflow.types import ImageInput
from inferflow.types import SegmentationOutput
from inferflow.utils.yolo import nms
from inferflow.utils.yolo import padding_resize
from inferflow.utils.yolo import process_mask
from inferflow.utils.yolo import scale_bbox
from inferflow.utils.yolo import scale_mask
from inferflow.utils.yolo import xyxy2xywh

if t.TYPE_CHECKING:
    from inferflow.batch import BatchStrategy
    from inferflow.runtime import Runtime


class YOLOv5SegmentationPipeline(Pipeline[torch.Tensor, tuple[torch.Tensor, torch.Tensor], list[SegmentationOutput]]):
    """YOLOv5 instance segmentation pipeline."""

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

        self._original_size: tuple[int, int] | None = None
        self._padding: tuple[int, int] | None = None

    async def preprocess(self, input: ImageInput) -> torch.Tensor:
        """Preprocess image for YOLOv5-Seg."""
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
        image = np.ascontiguousarray(image.transpose(2, 0, 1))

        # To tensor and normalize
        tensor = torch.from_numpy(image).float() / 255.0

        # Add batch dimension
        return tensor.unsqueeze(0)

    async def postprocess(self, raw: tuple[torch.Tensor, torch.Tensor]) -> list[SegmentationOutput]:
        """Postprocess YOLOv5-Seg output."""
        detections, protos = raw

        # Apply NMS
        filtered = nms(
            detections,
            conf_thres=self.conf_threshold,
            iou_thres=self.iou_threshold,
            nm=32,
            max_det=1000,
        )

        results = []

        bbox = filtered[0]
        proto = protos[0]

        if len(bbox) == 0:
            return []

        # Process masks
        masks = process_mask(proto, bbox[:, 6:], bbox[:, :4], self.image_size, upsample=True)

        if masks.ndimension() == 2:
            masks = masks.unsqueeze(0)

        masks_np = masks.cpu().numpy()
        bbox = bbox[:, :6]

        for mask_np, (*xyxy, conf, cls) in zip(masks_np, bbox, strict=False):
            xywh = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1).tolist()

            scaled_box = scale_bbox(
                tuple(xywh),  # type: ignore
                self.image_size,
                self._original_size,  # type: ignore
                self._padding,  # type: ignore
            )

            scaled_mask = scale_mask(
                mask_np,
                self._original_size,  # type: ignore
                self._padding,  # type: ignore
            )

            # Ensure boolean mask
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

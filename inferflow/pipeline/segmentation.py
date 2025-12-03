"""Instance segmentation pipeline implementation."""

from __future__ import annotations

import typing as t

import cv2
import numpy as np
from PIL import Image
import torch

from inferflow.pipeline import Pipeline
from inferflow.types import Box
from inferflow.types import ImageInput
from inferflow.types import SegmentationOutput

if t.TYPE_CHECKING:
    from inferflow.batch import BatchStrategy
    from inferflow.runtime import Runtime


class YOLOv5SegmentationPipeline(Pipeline[torch.Tensor, tuple[torch.Tensor, torch.Tensor], list[SegmentationOutput]]):
    """YOLOv5 instance segmentation pipeline.

    Performs:
    - Image preprocessing with padding
    - YOLOv5-Seg inference (detection + mask prototypes)
    - Mask reconstruction
    - Rescaling to original image size

    Args:
        runtime: Inference runtime (must return (detections, protos) tuple).
        image_size: Target image size (default: (640, 640)).
        stride: Model stride (default: 32).
        conf_threshold: Confidence threshold (default: 0.25).
        iou_threshold: IoU threshold for NMS (default: 0.45).
        class_names: Optional mapping from class ID to class name.
        batch_strategy: Optional batching strategy.

    Example:
        ```python
        runtime = TorchScriptRuntime(
            model_path="yolov5s-seg.pt", device="cuda"
        )
        pipeline = YOLOv5SegmentationPipeline(
            runtime=runtime,
            conf_threshold=0.5,
            class_names={0: "person", 1: "car"},
        )

        async with pipeline.serve():
            results = await pipeline(image_bytes)
            for seg in results:
                print(f"{seg.class_name}: {seg.confidence:.2%}")
                cv2.imshow("mask", seg.mask.astype(np.uint8) * 255)
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

        self._original_size: tuple[int, int] | None = None
        self._padding: tuple[int, int] | None = None

    def _padding_resize(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """Resize image with padding (same as detection pipeline)."""
        h, w = image.shape[:2]
        self._original_size = (w, h)

        scale = min(self.image_size[0] / h, self.image_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        target_h = ((self.image_size[0] + self.stride - 1) // self.stride) * self.stride
        target_w = ((self.image_size[1] + self.stride - 1) // self.stride) * self.stride

        pad_h = target_h - new_h
        pad_w = target_w - new_w

        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        self._padding = (left, top)
        return padded, (left, top)

    async def preprocess(self, input: ImageInput) -> torch.Tensor:
        """Preprocess image (same as detection pipeline)."""
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

        # Padding resize
        image, _ = self._padding_resize(image)

        # Convert and normalize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        return torch.from_numpy(image).float() / 255.0

    def _process_masks(
        self,
        protos: torch.Tensor,
        mask_coeffs: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct instance masks from prototypes.

        Args:
            protos: Mask prototypes (32, H, W).
            mask_coeffs: Mask coefficients per detection (N, 32).

        Returns:
            Instance masks (N, H, W).
        """
        # Matrix multiplication to get full masks
        masks = torch.matmul(mask_coeffs, protos.view(protos.shape[0], -1))
        masks = masks.view(-1, protos.shape[1], protos.shape[2])
        return torch.sigmoid(masks)

        # Crop masks to bounding boxes
        # ... (simplified, in practice use efficient cropping)

    def _rescale_mask(self, mask: np.ndarray) -> np.ndarray:
        """Rescale mask from resized image to original size."""
        if self._original_size is None or self._padding is None:
            raise RuntimeError("Original size not set")

        orig_w, orig_h = self._original_size
        pad_left, pad_top = self._padding

        # Remove padding
        h, w = mask.shape
        mask_cropped = mask[
            pad_top : h - pad_top,
            pad_left : w - pad_left,
        ]

        # Resize to original
        mask_resized = cv2.resize(
            mask_cropped.astype(np.uint8),
            (orig_w, orig_h),
            interpolation=cv2.INTER_LINEAR,
        )

        return (mask_resized > 0.5).astype(bool)

    async def postprocess(self, raw: tuple[torch.Tensor, torch.Tensor]) -> list[SegmentationOutput]:
        """Postprocess YOLOv5-Seg output.

        Args:
            raw: Tuple of (detections, mask_prototypes).
                 detections: (N, 6 + 32) [x, y, w, h, conf, class_id, mask_coeffs...]
                 protos: (32, H/4, W/4) mask prototypes.

        Returns:
            List of SegmentationOutput objects.
        """
        detections, protos = raw

        # Handle batch dimension
        if detections.ndim == 3:
            detections = detections[0]
        if protos.ndim == 4:
            protos = protos[0]

        # Filter by confidence
        mask = detections[:, 4] >= self.conf_threshold
        detections = detections[mask]

        if len(detections) == 0:
            return []

        # Extract components
        boxes = detections[:, :4]
        confidences = detections[:, 4]
        class_ids = detections[:, 5].long()
        mask_coeffs = detections[:, 6:]  # 32 coefficients

        # Reconstruct masks
        masks = self._process_masks(protos, mask_coeffs)

        # Build outputs
        results = []
        for _i, (box_coords, conf, class_id, mask) in enumerate(
            zip(boxes, confidences, class_ids, masks, strict=False)
        ):
            # Convert box to Box object
            x, y, w, h = box_coords.tolist()
            box = Box(xc=x, yc=y, w=w, h=h)

            # Rescale mask
            mask_np = mask.cpu().numpy()
            mask_rescaled = self._rescale_mask(mask_np)

            results.append(
                SegmentationOutput(
                    mask=mask_rescaled,
                    box=box,
                    class_id=int(class_id.item()),
                    confidence=float(conf.item()),
                    class_name=self.class_names.get(int(class_id.item())),
                )
            )

        return results

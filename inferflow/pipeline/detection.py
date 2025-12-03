from __future__ import annotations

import typing as t

import cv2
import numpy as np
import PIL.Image as Image
import torch

from inferflow.pipeline import Pipeline
from inferflow.types import Box
from inferflow.types import DetectionOutput
from inferflow.types import ImageInput

if t.TYPE_CHECKING:
    from inferflow.batch import BatchStrategy
    from inferflow.runtime import Runtime


class YOLOv5DetectionPipeline(Pipeline[torch.Tensor, torch.Tensor, list[DetectionOutput]]):
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
        runtime: Runtime[torch.Tensor, torch.Tensor],
        image_size: tuple[int, int] = (640, 640),
        stride: int = 32,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_names: dict[int, str] | None = None,
        batch_strategy: BatchStrategy[torch.Tensor, torch.Tensor] | None = None,
    ):
        super().__init__(runtime=runtime, batch_strategy=batch_strategy)

        self.image_size = image_size
        self.stride = stride
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or {}

        # Store original image size for rescaling
        self._original_size: tuple[int, int] | None = None
        self._padding: tuple[int, int] | None = None

    def _padding_resize(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """Resize image with padding to maintain aspect ratio.

        Args:
            image: Input image (H, W, C).

        Returns:
            Tuple of (resized_image, padding).
        """
        h, w = image.shape[:2]
        self._original_size = (w, h)

        # Calculate scaling factor
        scale = min(self.image_size[0] / h, self.image_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Calculate padding to reach target size
        # Make sure dimensions are divisible by stride
        target_h = ((self.image_size[0] + self.stride - 1) // self.stride) * self.stride
        target_w = ((self.image_size[1] + self.stride - 1) // self.stride) * self.stride

        pad_h = target_h - new_h
        pad_w = target_w - new_w

        # Pad (top, bottom, left, right)
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        self._padding = (left, top)

        return padded, (left, top)

    async def preprocess(self, input: ImageInput) -> torch.Tensor:
        """Preprocess image for YOLOv5.

        Args:
            input: Image as bytes, numpy array, PIL Image, or torch.Tensor.

        Returns:
            Preprocessed tensor (C, H, W) ready for YOLOv5 inference.
        """
        # Convert to numpy array (OpenCV format)
        if isinstance(input, bytes):
            try:
                nparr = np.frombuffer(input, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Failed to decode image")
            except Exception as e:
                raise ValueError(f"Failed to decode image from bytes: {e}") from e

        elif isinstance(input, Image.Image):
            image = np.array(input.convert("RGB"))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        elif isinstance(input, np.ndarray):
            image = input.copy()

        elif isinstance(input, torch.Tensor):
            # Convert tensor to numpy
            if input.ndim == 4:  # (B, C, H, W)
                input = input.squeeze(0)
            if input.ndim == 3 and input.shape[0] in [1, 3]:  # (C, H, W)
                image = input.permute(1, 2, 0).cpu().numpy()
                image = (image * 255).astype(np.uint8)
            else:
                raise ValueError(f"Unsupported tensor shape: {input.shape}")

        else:
            raise ValueError(f"Unsupported input type: {type(input)}")

        # Padding resize
        image, padding = self._padding_resize(image)

        # Convert to RGB and normalize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.ascontiguousarray(image.transpose(2, 0, 1))  # HWC -> CHW

        # Convert to tensor and normalize
        return torch.from_numpy(image).float() / 255.0

    def _nms(self, predictions: torch.Tensor) -> torch.Tensor:
        """Apply Non-Maximum Suppression.

        Args:
            predictions: Raw predictions from YOLOv5 (detections, 6+)
                         Format: [x, y, w, h, conf, class_id, ...]

        Returns:
            Filtered predictions after NMS.
        """
        # Filter by confidence
        mask = predictions[:, 4] >= self.conf_threshold
        predictions = predictions[mask]

        if len(predictions) == 0:
            return predictions

        # Get boxes and scores
        boxes = predictions[:, :4]  # xyxy format expected
        scores = predictions[:, 4]
        class_ids = predictions[:, 5]

        # Apply NMS per class
        keep_indices = []
        for class_id in class_ids.unique():
            class_mask = class_ids == class_id
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]

            # torchvision NMS
            import torchvision

            keep = torchvision.ops.nms(class_boxes, class_scores, self.iou_threshold)
            keep_indices.extend(torch.where(class_mask)[0][keep].tolist())

        return predictions[keep_indices]

    def _rescale_boxes(self, boxes: torch.Tensor) -> list[Box]:
        """Rescale boxes from resized image to original image coordinates.

        Args:
            boxes: Boxes in resized image space (N, 4) [x, y, w, h].

        Returns:
            List of Box objects in original image coordinates.
        """
        if self._original_size is None or self._padding is None:
            raise RuntimeError("Original size not set. Call preprocess first.")

        orig_w, orig_h = self._original_size
        pad_left, pad_top = self._padding

        # Calculate scale
        scale_w = orig_w / (self.image_size[1] - 2 * pad_left)
        scale_h = orig_h / (self.image_size[0] - 2 * pad_top)

        rescaled_boxes = []
        for box in boxes:
            x, y, w, h = box.tolist()

            # Remove padding
            x = (x - pad_left) * scale_w
            y = (y - pad_top) * scale_h
            w = w * scale_w
            h = h * scale_h

            rescaled_boxes.append(Box(xc=x, yc=y, w=w, h=h))

        return rescaled_boxes

    async def postprocess(self, raw: torch.Tensor) -> list[DetectionOutput]:
        """Postprocess YOLOv5 output to detection results.

        Args:
            raw: Raw model output (N, 6+) where each detection is
                 [x, y, w, h, conf, class_id, ...].

        Returns:
            List of DetectionOutput objects.
        """
        # Handle batch dimension
        if raw.ndim == 3:  # (batch, N, 6+)
            raw = raw[0]  # Take first item

        # Apply NMS
        filtered = self._nms(raw)

        if len(filtered) == 0:
            return []

        # Extract components
        boxes_xywh = filtered[:, :4]  # Assuming center format
        confidences = filtered[:, 4]
        class_ids = filtered[:, 5].long()

        # Rescale boxes to original image size
        rescaled_boxes = self._rescale_boxes(boxes_xywh)

        # Build detection outputs
        detections = []
        for box, conf, class_id in zip(rescaled_boxes, confidences, class_ids, strict=False):
            detections.append(
                DetectionOutput(
                    box=box,
                    class_id=int(class_id.item()),
                    confidence=float(conf.item()),
                    class_name=self.class_names.get(int(class_id.item())),
                )
            )

        return detections

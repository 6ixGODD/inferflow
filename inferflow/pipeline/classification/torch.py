from __future__ import annotations

import io
import typing as t

import cv2
import numpy as np
import PIL.Image as Image
import torch
import torchvision.transforms as T

from inferflow.pipeline import Pipeline
from inferflow.types import ClassificationOutput

if t.TYPE_CHECKING:
    from inferflow.batch import BatchStrategy
    from inferflow.runtime import Runtime
    from inferflow.types import ImageInput


class ClassificationMixin:
    """Shared classification preprocessing/postprocessing logic.

    Attributes:
        image_size: Target image size.
        mean: Normalization mean.
        std: Normalization std.
        class_names: Mapping from class ID to class name.
        transform: Image transformation pipeline.
    """

    image_size: tuple[int, int]
    mean: torch.Tensor
    std: torch.Tensor
    class_names: dict[int, str]
    transform: T.Compose

    def _convert_to_pil(self, input: ImageInput) -> Image.Image:
        """Convert various input types to PIL Image.

        Args:
            input: Input image (bytes, numpy array, or PIL Image).

        Returns:
            PIL Image in RGB mode.
        """
        if isinstance(input, bytes):
            try:
                return Image.open(io.BytesIO(input)).convert("RGB")
            except Exception as e:
                raise ValueError(f"Failed to decode image from bytes: {e}") from e

        elif isinstance(input, np.ndarray):
            if input.ndim == 3 and input.shape[2] == 3:
                return Image.fromarray(cv2.cvtColor(input, cv2.COLOR_BGR2RGB))
            if input.ndim == 2:
                return Image.fromarray(input).convert("RGB")
            raise ValueError(f"Unsupported numpy array shape: {input.shape}")

        elif isinstance(input, Image.Image):
            return input.convert("RGB")

        else:
            raise ValueError(f"Unsupported input type: {type(input)}")

    def _preprocess_tensor(self, input: torch.Tensor) -> torch.Tensor:
        """Preprocess tensor input.

        Args:
            input: Input tensor to preprocess.

        Returns:
            Preprocessed tensor ready for model inference.
        """
        if input.ndim == 4:
            tensor = input
        elif input.ndim == 3:
            tensor = input.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported tensor shape: {input.shape}")

        return (tensor - self.mean) / self.std

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess PIL Image.

        Args:
            image: PIL Image to preprocess.

        Returns:
            Preprocessed tensor ready for model inference.
        """
        tensor = self.transform(image)
        tensor = (tensor - self.mean) / self.std
        return tensor.unsqueeze(0)

    def _postprocess_raw(self, raw: torch.Tensor) -> ClassificationOutput:
        """Postprocess raw output.

        Args:
            raw: Raw output tensor from model inference.

        Returns:
            ClassificationOutput with class ID, confidence, and optional class name.
        """
        if raw.ndim == 2:
            raw = raw[0]
        elif raw.ndim != 1:
            raise ValueError(f"Expected 1D or 2D tensor, got shape: {raw.shape}")

        probs = torch.softmax(raw, dim=0)
        class_id = int(probs.argmax().item())
        confidence = float(probs[class_id].item())
        class_name = self.class_names.get(class_id)

        return ClassificationOutput(class_id=class_id, confidence=confidence, class_name=class_name)


class ClassificationPipeline(
    ClassificationMixin,
    Pipeline[torch.Tensor, torch.Tensor, ClassificationOutput],
):
    """Image classification pipeline (sync version).

    Performs:
        - Image decoding and conversion
        - Resizing and normalization
        - Model inference
        - Class prediction with confidence

    Args:
        runtime: Inference runtime.
        image_size: Target image size (default: (224, 224)).
        mean: Normalization mean (default: ImageNet mean).
        std: Normalization std (default: ImageNet std).
        class_names: Optional mapping from class ID to class name.
        batch_strategy: Optional batching strategy.

    Example:
        ```python
        runtime = TorchScriptRuntime(
            model_path="resnet50.pt", device="cuda"
        )
        pipeline = ClassificationPipeline(
            runtime=runtime,
            class_names={0: "cat", 1: "dog", 2: "bird"},
        )

        with pipeline.serve():
            result = pipeline(image_bytes)
            print(f"{result.class_name}: {result.confidence:.2%}")
        ```
    """

    def __init__(
        self,
        runtime: Runtime[torch.Tensor, torch.Tensor],
        image_size: tuple[int, int] = (224, 224),
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        class_names: dict[int, str] | None = None,
        batch_strategy: BatchStrategy[torch.Tensor, torch.Tensor] | None = None,
    ):
        super().__init__(runtime=runtime, batch_strategy=batch_strategy)

        self.image_size = image_size
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.class_names = class_names or {}

        self.transform = T.Compose([T.Resize(image_size), T.ToTensor()])

    def preprocess(self, input: ImageInput) -> torch.Tensor:
        """Preprocess image input.

        Args:
            input: Raw image input (bytes, numpy array, PIL Image, or tensor).

        Returns:
            Preprocessed tensor ready for model inference.
        """
        if isinstance(input, torch.Tensor):
            return self._preprocess_tensor(input)

        image = self._convert_to_pil(input)
        return self._preprocess_image(image)

    def postprocess(self, raw: torch.Tensor) -> ClassificationOutput:
        """Postprocess model output to classification result.

        Args:
            raw: Raw output tensor from model inference.

        Returns:
            ClassificationOutput with class ID, confidence, and optional class name.
        """
        return self._postprocess_raw(raw)


__all__ = ["ClassificationPipeline", "ClassificationMixin"]

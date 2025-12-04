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


class ClassificationPipeline(Pipeline[torch.Tensor, torch.Tensor, ClassificationOutput]):
    """Image classification pipeline.

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

        async with pipeline.serve():
            result = await pipeline(image_bytes)
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

        # Build transform pipeline
        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
        ])

    async def preprocess(self, input: ImageInput) -> torch.Tensor:
        """Preprocess image input.

        Args:
            input: Image as bytes, numpy array, PIL Image, or torch.Tensor.

        Returns:
            Preprocessed tensor (1, 3, H, W) normalized and ready for inference.

        Raises:
            ValueError: If input type is unsupported or image is invalid.
        """
        # Convert to PIL Image
        if isinstance(input, bytes):
            try:
                image = Image.open(io.BytesIO(input)).convert("RGB")
            except Exception as e:
                raise ValueError(f"Failed to decode image from bytes: {e}") from e

        elif isinstance(input, np.ndarray):
            # OpenCV format (BGR) -> PIL (RGB)
            if input.ndim == 3 and input.shape[2] == 3:
                image = Image.fromarray(cv2.cvtColor(input, cv2.COLOR_BGR2RGB))
            elif input.ndim == 2:  # Grayscale
                image = Image.fromarray(input).convert("RGB")
            else:
                raise ValueError(f"Unsupported numpy array shape: {input.shape}")

        elif isinstance(input, Image.Image):
            image = input.convert("RGB")

        elif isinstance(input, torch.Tensor):
            # Already a tensor
            if input.ndim == 4:  # (1, C, H, W) - already has batch
                tensor = input
            elif input.ndim == 3:  # (C, H, W) - add batch
                tensor = input.unsqueeze(0)
            else:
                raise ValueError(f"Unsupported tensor shape: {input.shape}")

            # Normalize
            return (tensor - self.mean) / self.std

        else:
            raise ValueError(f"Unsupported input type: {type(input)}")

        # Apply transforms
        tensor = self.transform(image)  # (C, H, W)

        # Normalize
        tensor = (tensor - self.mean) / self.std

        # Add batch dimension
        return tensor.unsqueeze(0)  # (1, C, H, W)

    async def postprocess(self, raw: torch.Tensor) -> ClassificationOutput:
        """Postprocess model output to classification result.

        Args:
            raw: Raw model output tensor (logits or probabilities).

        Returns:
            Classification result with class ID, confidence, and name.
        """
        # Handle batch dimension
        if raw.ndim == 2:  # (batch, classes)
            raw = raw[0]  # Take first item
        elif raw.ndim != 1:
            raise ValueError(f"Expected 1D or 2D tensor, got shape: {raw.shape}")

        # Apply softmax to get probabilities
        probs = torch.softmax(raw, dim=0)

        # Get predicted class
        class_id = int(probs.argmax().item())
        confidence = float(probs[class_id].item())

        # Get class name if available
        class_name = self.class_names.get(class_id)

        return ClassificationOutput(class_id=class_id, confidence=confidence, class_name=class_name)

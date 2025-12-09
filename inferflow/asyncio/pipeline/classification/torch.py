from __future__ import annotations

import typing as t

import torch

from inferflow.asyncio.pipeline import Pipeline
from inferflow.pipeline.classification.torch import ClassificationMixin
from inferflow.types import ClassificationOutput

if t.TYPE_CHECKING:
    from inferflow.asyncio.batch import BatchStrategy
    from inferflow.asyncio.runtime import Runtime
    from inferflow.types import ImageInput


class ClassificationPipeline(
    ClassificationMixin,
    Pipeline[torch.Tensor, torch.Tensor, ClassificationOutput],
):
    """Image classification pipeline (async version).

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

        import torchvision.transforms as T

        self.transform = T.Compose([T.Resize(image_size), T.ToTensor()])

    async def preprocess(self, input: ImageInput) -> torch.Tensor:
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

    async def postprocess(self, raw: torch.Tensor) -> ClassificationOutput:
        """Postprocess model output to classification result.

        Args:
            raw: Raw output tensor from model inference.

        Returns:
            ClassificationOutput with class ID, confidence, and optional class name.
        """
        return self._postprocess_raw(raw)


__all__ = ["ClassificationPipeline"]

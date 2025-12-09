from __future__ import annotations

import io

import numpy as np
import PIL.Image as Image
import pytest
import torch

from inferflow.pipeline.classification.torch import ClassificationPipeline


@pytest.mark.unit
class TestClassificationPipeline:
    """Unit tests for ClassificationPipeline."""

    @pytest.fixture
    def dummy_image_bytes(self):
        """Create a dummy image as bytes."""
        img = Image.new("RGB", (224, 224), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    @pytest.fixture
    def pipeline(self, mock_classification_runtime):
        """Create a classification pipeline with mock runtime."""
        return ClassificationPipeline(
            runtime=mock_classification_runtime,
            class_names={0: "cat", 1: "dog", 2: "bird"},
        )

    def test_preprocess_bytes(self, pipeline, dummy_image_bytes):
        """Test preprocessing from bytes."""
        result = pipeline.preprocess(dummy_image_bytes)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 3, 224, 224)

    def test_preprocess_pil(self, pipeline):
        """Test preprocessing from PIL Image."""
        img = Image.new("RGB", (300, 300), color="blue")
        result = pipeline.preprocess(img)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 3, 224, 224)

    def test_preprocess_numpy(self, pipeline):
        """Test preprocessing from numpy array."""
        arr = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        result = pipeline.preprocess(arr)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 3, 224, 224)

    def test_preprocess_tensor(self, pipeline):
        """Test preprocessing from tensor."""
        tensor = torch.randn(3, 224, 224)
        result = pipeline.preprocess(tensor)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 3, 224, 224)

    def test_postprocess(self, pipeline):
        """Test postprocessing."""
        # Mock model output (10 classes) - batch of 1
        raw = torch.randn(1, 10)

        result = pipeline.postprocess(raw)

        assert 0 <= result.class_id < 10
        assert 0 <= result.confidence <= 1

    def test_end_to_end(self, pipeline, dummy_image_bytes):
        """Test end-to-end pipeline."""
        with pipeline.serve():
            result = pipeline(dummy_image_bytes)

        assert 0 <= result.class_id < 10
        assert 0 <= result.confidence <= 1


__all__ = ["TestClassificationPipeline"]

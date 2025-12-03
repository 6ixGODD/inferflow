from __future__ import annotations

import typing as t
from enum import Enum

import numpy as np
import torch
from PIL import Image

# ============================================================================
# Input Types
# ============================================================================

ImageInput = np.ndarray | Image.Image | torch.Tensor | bytes
"""Supported image input types."""

# ============================================================================
# Generic Type Variables
# ============================================================================

P = t.TypeVar("P")  # Preprocessed type
R = t.TypeVar("R")  # Raw inference result type
O = t.TypeVar("O")  # Final output type

# ============================================================================
# Device Types
# ============================================================================


class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    NPU = "npu"  # Neural Processing Unit


class Device:
    def __init__(self, device: str | DeviceType, index: int = 0):
        if isinstance(device, str):
            device = DeviceType(device)
        self.type = device
        self.index = index

    def __str__(self) -> str:
        if self.type == DeviceType. CPU:
            return "cpu"
        return f"{self.type.value}:{self.index}"

    def __repr__(self) -> str:
        return f"Device({self.type.value}, {self.index})"


# ============================================================================
# Common Output Types
# ============================================================================


class Box:
    """Bounding box representation (center + width/height)."""

    __slots__ = ("xc", "yc", "w", "h")

    def __init__(self, xc: float, yc: float, w: float, h: float):
        self.xc = xc
        self.yc = yc
        self.w = w
        self.h = h

    def to_xyxy(self) -> tuple[float, float, float, float]:
        """Convert to (x1, y1, x2, y2) format."""
        x1 = self.xc - self. w / 2
        y1 = self.yc - self.h / 2
        x2 = self.xc + self.w / 2
        y2 = self.yc + self.h / 2
        return x1, y1, x2, y2

    def to_xywh(self) -> tuple[float, float, float, float]:
        """Convert to (x, y, w, h) format (top-left)."""
        x = self.xc - self.w / 2
        y = self.yc - self.h / 2
        return x, y, self.w, self.h


class ClassificationOutput(t.NamedTuple):
    """Classification result."""

    class_id: int
    """Predicted class ID."""

    confidence: float
    """Confidence score (0-1)."""

    class_name: str | None = None
    """Human-readable class name."""


class DetectionOutput(t.NamedTuple):
    """Single object detection result."""

    box: Box
    """Bounding box."""

    class_id: int
    """Predicted class ID."""

    confidence: float
    """Confidence score (0-1)."""

    class_name: str | None = None
    """Human-readable class name."""


class SegmentationOutput(t.NamedTuple):
    """Single instance segmentation result."""

    mask: np.ndarray
    """Binary mask (H, W)."""

    box: Box
    """Bounding box."""

    class_id: int
    """Predicted class ID."""

    confidence: float
    """Confidence score (0-1)."""

    class_name: str | None = None
    """Human-readable class name."""


# ============================================================================
# Batch Types
# ============================================================================


class BatchConfig(t.NamedTuple):
    """Batch processing configuration."""

    enabled: bool = True
    """Enable batch processing."""

    min_batch_size: int = 1
    """Minimum batch size."""

    max_batch_size: int = 32
    """Maximum batch size."""

    max_wait_ms: float = 50
    """Maximum wait time in milliseconds."""

    queue_size: int = 100
    """Maximum queue size."""


class BatchMetrics(t.NamedTuple):
    """Batch processing metrics."""

    total_requests: int
    """Total number of requests processed."""

    total_batches: int
    """Total number of batches processed."""

    avg_batch_size: float
    """Average batch size."""

    avg_latency_ms: float
    """Average latency in milliseconds."""

    queue_size: int
    """Current queue size."""


# ============================================================================
# Runtime Types
# ============================================================================


class RuntimeType(str, Enum):
    """Supported runtime types."""

    TORCHSCRIPT = "torchscript"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"


class Precision(str, Enum):
    """Model precision."""

    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"

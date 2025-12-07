from __future__ import annotations

import torch

def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from center format to corner format.

    Transforms boxes from (x_center, y_center, width, height) to
    (x_min, y_min, x_max, y_max) format.

    Args:
        x: Input tensor of shape (..., 4) in xywh format

    Returns:
        Output tensor of shape (..., 4) in xyxy format
    """

def xyxy2xywh(x: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from corner format to center format.

    Transforms boxes from (x_min, y_min, x_max, y_max) to
    (x_center, y_center, width, height) format.

    Args:
        x: Input tensor of shape (..., 4) in xyxy format

    Returns:
        Output tensor of shape (..., 4) in xywh format
    """

def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Calculate Intersection over Union (IoU) between two sets of boxes.

    Args:
        box1: First set of boxes, shape (N, 4) in xyxy format
        box2: Second set of boxes, shape (M, 4) in xyxy format
        eps: Small epsilon to avoid division by zero

    Returns:
        IoU matrix of shape (N, M)
    """

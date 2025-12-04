from __future__ import annotations

import time

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchvision
except ImportError as e:
    raise ImportError("Torch is required. Install with: pip install 'inferflow[torch]'") from e


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """Convert box format from (x_center, y_center, w, h) to (x1, y1, x2,
    y2)."""
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y


def xyxy2xywh(x: torch.Tensor) -> torch.Tensor:
    """Convert box format from (x1, y1, x2, y2) to (x_center, y_center, w,
    h)."""
    y = x.clone()
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x_center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y_center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Calculate IoU between two sets of boxes."""
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def nms(
    prediction: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
    nm: int = 0,
) -> list[torch.Tensor]:
    """Non-Maximum Suppression (NMS) on inference results.

    Args:
        prediction: Model predictions (batch, num_pred, 5+nc+nm).
        conf_thres: Confidence threshold.
        iou_thres: IoU threshold for NMS.
        max_det: Maximum detections per image.
        nm: Number of mask coefficients (0 for detection, 32 for segmentation).

    Returns:
        List of detections per image (N, 6+nm) [xyxy, conf, cls, (mask_coeffs)].
    """
    bs = prediction.shape[0]
    nc = prediction.shape[2] - nm - 5
    xc = prediction[..., 4] > conf_thres

    max_wh = 7680
    max_nms = 30000
    time_limit = 0.5 + 0.05 * bs

    t = time.time()
    mi = 5 + nc
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs

    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]

        box = xywh2xyxy(x[:, :4])
        mask = x[:, mi:]

        conf, j = x[:, 5:mi].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        n = x.shape[0]
        if not n:
            continue

        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * max_wh
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_det]

        output[xi] = x[i]

        if (time.time() - t) > time_limit:
            break

    return output


def crop_mask(masks: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Crop masks to bounding boxes."""
    _, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask(
    protos: torch.Tensor,
    masks_in: torch.Tensor,
    bboxes: torch.Tensor,
    shape: tuple[int, int],
    upsample: bool = False,
) -> torch.Tensor:
    """Process prototype masks to create instance masks."""
    c, mh, mw = protos.shape
    ih, iw = shape

    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)

    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih

    masks = crop_mask(masks, downsampled_bboxes)

    if upsample:
        masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[0]

    return masks.gt_(0.5)


def padding_resize(
    image: np.ndarray,
    size: tuple[int, int],
    stride: int = 32,
    full: bool = True,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Resize with padding to maintain aspect ratio."""
    import cv2

    h, w = image.shape[:2]
    scale = min(size[0] / w, size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if full:
        dw = size[0] - new_w
        dh = size[1] - new_h
    else:
        dw = (stride - new_w % stride) % stride
        dh = (stride - new_h % stride) % stride

    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    return padded, (dw, dh)


def scale_bbox(
    bbox: tuple[float, float, float, float],
    input_size: tuple[int, int],
    original_shape: tuple[int, int],
    padding: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Scale bbox from padded coords to original image coords."""
    cx, cy, w, h = bbox
    dw, dh = padding
    input_w, input_h = input_size
    original_w, original_h = original_shape

    scale = min(input_w / original_w, input_h / original_h)

    pad_w = dw / 2
    pad_h = dh / 2

    cx_no_pad = cx - pad_w
    cy_no_pad = cy - pad_h

    cx_original = cx_no_pad / scale
    cy_original = cy_no_pad / scale
    w_original = w / scale
    h_original = h / scale

    return cx_original, cy_original, w_original, h_original


def scale_mask(
    mask: np.ndarray,
    original_shape: tuple[int, int],
    padding: tuple[int, int],
) -> np.ndarray:
    """Scale mask back to original dimensions."""
    import cv2

    dw, dh = padding

    mask_no_padding = mask[
        int(dh // 2) : mask.shape[0] - int(dh // 2 + 0.5),
        int(dw // 2) : mask.shape[1] - int(dw // 2 + 0.5),
    ]

    original_w, original_h = original_shape

    return cv2.resize(mask_no_padding, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

from __future__ import annotations

import time
import typing as t

import cv2
import numpy as np
import numpy.typing as npt


def xywh2xyxy(x: npt.NDArray[t.Any]) -> npt.NDArray[t.Any]:
    """Convert box format from (x_center, y_center, w, h) to (x1, y1, x2,
    y2)."""
    y = x.copy()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y


def xyxy2xywh(x: npt.NDArray[t.Any]) -> npt.NDArray[t.Any]:
    """Convert box format from (x1, y1, x2, y2) to (x_center, y_center, w,
    h)."""
    y = x.copy()
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x_center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y_center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def box_iou(box1: npt.NDArray[t.Any], box2: npt.NDArray[t.Any], eps: float = 1e-7) -> npt.NDArray[t.Any]:
    """Calculate IoU between two sets of boxes.

    Args:
        box1: (N, 4) in xyxy format
        box2: (M, 4) in xyxy format
        eps: Small value to avoid division by zero

    Returns:
        (N, M) IoU matrix
    """
    # Expand dimensions for broadcasting
    box1 = np.expand_dims(box1, axis=1)  # (N, 1, 4)
    box2 = np.expand_dims(box2, axis=0)  # (1, M, 4)

    # Split coordinates
    b1_x1, b1_y1, b1_x2, b1_y2 = np.split(box1, 4, axis=2)
    b2_x1, b2_y1, b2_x2, b2_y2 = np.split(box2, 4, axis=2)

    # Intersection
    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y2 = np.minimum(b1_y2, b2_y2)

    inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
    inter_area = inter_area.squeeze(2)

    # Union
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1)).squeeze(2)
    b2_area = ((b2_x2 - b2_x1) * (b2_y2 - b2_y1)).squeeze(2)
    union_area = b1_area + b2_area - inter_area + eps

    return inter_area / union_area


def nms_numpy(boxes: npt.NDArray[t.Any], scores: npt.NDArray[t.Any], iou_threshold: float) -> npt.NDArray[t.Any]:
    """Pure NumPy NMS implementation (optimized).

    This is faster than naive loop but slower than torchvision.ops.nms.

    Args:
        boxes: (N, 4) in xyxy format
        scores: (N,) confidence scores
        iou_threshold:  IoU threshold

    Returns:
        Indices of boxes to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        # Vectorized IoU calculation
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int64)


def nms(
    prediction: npt.NDArray[t.Any],
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
    nm: int = 0,
) -> list[npt.NDArray[t.Any]]:
    """Non-Maximum Suppression (NumPy version).

    Args:
        prediction: Model predictions (batch, num_pred, 5+nc+nm).
        conf_thres:  Confidence threshold.
        iou_thres: IoU threshold for NMS.
        max_det: Maximum detections per image.
        nm: Number of mask coefficients.

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
    output = [np.zeros((0, 6 + nm), dtype=prediction.dtype)] * bs

    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]

        box = xywh2xyxy(x[:, :4])
        mask = x[:, mi:]

        conf = x[:, 5:mi].max(axis=1, keepdims=True)
        j = x[:, 5:mi].argmax(axis=1, keepdims=True).astype(prediction.dtype)
        x = np.concatenate((box, conf, j, mask), axis=1)[conf.ravel() > conf_thres]

        n = x.shape[0]
        if not n:
            continue

        # Sort by confidence
        x = x[x[:, 4].argsort()[::-1][:max_nms]]

        # NMS with class offset
        c = x[:, 5:6] * max_wh
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = nms_numpy(boxes, scores, iou_thres)
        i = i[:max_det]

        output[xi] = x[i]

        if (time.time() - t) > time_limit:
            break

    return output


def crop_mask(masks: npt.NDArray[t.Any], boxes: npt.NDArray[t.Any]) -> npt.NDArray[t.Any]:
    """Crop masks to bounding boxes (NumPy version).

    Args:
        masks: (N, H, W)
        boxes: (N, 4) in xyxy format

    Returns:
        Cropped masks (N, H, W)
    """
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, axis=1)  # (N, 1, 1) each

    r = np.arange(w, dtype=boxes.dtype)[None, None, :]  # (1, 1, W)
    c = np.arange(h, dtype=boxes.dtype)[None, :, None]  # (1, H, 1)

    return masks * ((r >= x1) & (r < x2) & (c >= y1) & (c < y2))


def process_mask(
    protos: npt.NDArray[t.Any],
    masks_in: npt.NDArray[t.Any],
    bboxes: npt.NDArray[t.Any],
    shape: tuple[int, int],
    upsample: bool = False,
) -> npt.NDArray[t.Any]:
    """Process prototype masks to create instance masks (NumPy version).

    Args:
        protos: (C, Mh, Mw) mask prototypes
        masks_in: (N, C) mask coefficients
        bboxes: (N, 4) in xyxy format
        shape: Target shape (H, W)
        upsample: Whether to upsample

    Returns:
        Instance masks (N, H, W)
    """
    c, mh, mw = protos.shape
    ih, iw = shape

    # Matrix multiplication and sigmoid
    masks = masks_in @ protos.reshape(c, -1)  # (N, Mh*Mw)
    masks = 1 / (1 + np.exp(-masks))  # Sigmoid
    masks = masks.reshape(-1, mh, mw)  # (N, Mh, Mw)

    # Downsample bboxes
    downsampled_bboxes = bboxes.copy()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih

    masks = crop_mask(masks, downsampled_bboxes)

    if upsample:
        # Bilinear interpolation for each mask
        upsampled = []
        for mask in masks:
            resized = cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_LINEAR)
            upsampled.append(resized)
        masks = np.array(upsampled)

    return masks > 0.5


def padding_resize(
    image: np.ndarray,
    size: tuple[int, int],
    stride: int = 32,
    full: bool = True,
) -> tuple[npt.NDArray[t.Any], tuple[int, int]]:
    """Resize with padding to maintain aspect ratio."""
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
    mask: npt.NDArray[t.Any],
    original_shape: tuple[int, int],
    padding: tuple[int, int],
) -> npt.NDArray[t.Any]:
    """Scale mask back to original dimensions."""
    dw, dh = padding

    mask_no_padding = mask[
        int(dh // 2) : mask.shape[0] - int(dh // 2 + 0.5),
        int(dw // 2) : mask.shape[1] - int(dw // 2 + 0.5),
    ]

    original_w, original_h = original_shape

    mask_no_padding = mask_no_padding.astype(np.uint8) * 255
    resized = cv2.resize(mask_no_padding, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    return (resized / 255).astype(mask.dtype)

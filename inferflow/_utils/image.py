from __future__ import annotations

import cv2
import numpy as np
import PIL.Image as Image

from inferflow.types import Box


def crop_by_mask(image: np.ndarray, mask: np.ndarray, bounding: bool = False) -> np.ndarray:
    """Crop image by mask.

    Args:
        image: Input image (H, W, C).
        mask: Binary mask (H, W).
        bounding: If True, crop to bounding box. If False, apply mask and crop.

    Returns:
        Cropped image.
    """
    if bounding:
        # Find bounding box
        coords = np.argwhere(mask > 0)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        return image[y_min : y_max + 1, x_min : x_max + 1]
    # Apply mask and crop to bounding box
    masked = image.copy()
    masked[~mask] = 0

    coords = np.argwhere(mask > 0)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    return masked[y_min : y_max + 1, x_min : x_max + 1]


def draw_boxes(
    image: np.ndarray,
    boxes: list[Box],
    labels: list[str] | None = None,
    confidences: list[float] | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on image.

    Args:
        image: Input image (H, W, C).
        boxes: List of bounding boxes.
        labels: Optional class labels for each box.
        confidences: Optional confidence scores for each box.
        color: Box color in BGR format.
        thickness: Line thickness.

    Returns:
        Image with drawn boxes.
    """
    result = image.copy()

    for i, box in enumerate(boxes):
        # Convert to xyxy format
        x1, y1, x2, y2 = box.to_xyxy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw rectangle
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        # Draw label
        if labels or confidences:
            label_parts = []
            if labels and i < len(labels):
                label_parts.append(labels[i])
            if confidences and i < len(confidences):
                label_parts.append(f"{confidences[i]:.2%}")

            label = " ".join(label_parts)

            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Draw background rectangle
            cv2.rectangle(
                result,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1,
            )

            # Draw text
            cv2.putText(
                result,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

    return result


def draw_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.5,
) -> np.ndarray:
    """Draw segmentation mask on image with transparency.

    Args:
        image: Input image (H, W, C).
        mask: Binary mask (H, W).
        color: Mask color in BGR format.
        alpha: Transparency (0=transparent, 1=opaque).

    Returns:
        Image with drawn mask.
    """
    result = image.copy()

    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask] = color

    # Blend with original image
    return cv2.addWeighted(result, 1 - alpha, colored_mask, alpha, 0)


def save_image(image: np.ndarray, path: str) -> None:
    """Save image to file.

    Args:
        image: Image to save (H, W, C) in BGR format.
        path: Output file path.
    """
    cv2.imwrite(path, image)


def show_image(image: np.ndarray, window_name: str = "Image", wait: bool = True) -> None:
    """Display image in window.

    Args:
        image: Image to display (H, W, C) in BGR format.
        window_name: Window title.
        wait: If True, wait for key press. If False, display for 1ms.
    """
    cv2.imshow(window_name, image)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.waitKey(1)


def resize_keep_aspect(
    image: np.ndarray,
    target_size: tuple[int, int],
    pad_value: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Resize image while keeping aspect ratio by padding.

    Args:
        image: Input image (H, W, C).
        target_size: Target size (height, width).
        pad_value: Padding color in BGR format.

    Returns:
        Tuple of (resized_image, padding) where padding is (top, bottom, left, right).
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # Calculate scale
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Calculate padding
    pad_h = target_h - new_h
    pad_w = target_w - new_w

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    # Apply padding
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=pad_value,
    )

    return padded, (top, bottom, left, right)


def numpy_to_pil(image: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image.

    Args:
        image: Numpy array (H, W, C) in BGR format.

    Returns:
        PIL Image in RGB format.
    """
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array.

    Args:
        image: PIL Image in RGB format.

    Returns:
        Numpy array (H, W, C) in BGR format.
    """
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

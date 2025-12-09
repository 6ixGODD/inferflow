"""Object Detection Example.

Demonstrates:
    - YOLOv5 object detection
    - Bounding box visualization
    - Multi-object detection
"""

import asyncio
import pathlib
import urllib.request

from colorama import Fore
from colorama import Style
from colorama import init
import cv2

from inferflow import Precision
from inferflow.asyncio.pipeline.detection.onnx import YOLOv5DetectionPipeline
from inferflow.asyncio.runtime.onnx import ONNXRuntime

init(autoreset=True)


def print_header(text: str) -> None:
    print(f"\n{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{text.center(60)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}\n")


def print_step(step: int, text: str) -> None:
    print(f"{Fore.GREEN}[{step}]{Style.RESET_ALL} {text}")


def print_info(key: str, value: str) -> None:
    print(f"  {Fore.YELLOW}{key}:{Style.RESET_ALL} {value}")


def print_result(text: str) -> None:
    print(f"{Fore.GREEN}✓{Style.RESET_ALL} {text}")


# COCO class names (subset)
COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    16: "dog",
    17: "cat",
}


def download_file(url: str, dest: pathlib.Path) -> None:
    if dest.exists():
        print_info("Cached", str(dest))
        return

    print_info("Downloading", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print_result(f"Downloaded to {dest}")


def visualize_detections(
    image_path: pathlib.Path,
    detections: list,
    output_path: pathlib.Path,
) -> None:
    """Draw bounding boxes on image."""
    image = cv2.imread(str(image_path))

    for det in detections:
        box = det.box

        # Convert center format to corner format
        x1 = int(box.xc - box.w / 2)
        y1 = int(box.yc - box.h / 2)
        x2 = int(box.xc + box.w / 2)
        y2 = int(box.yc + box.h / 2)

        # Draw box
        color = (0, 0, 255)  # Red color in BGR
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{det.class_name or det.class_id}:  {det.confidence:.2f}"

        # Background for text
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)

        # Text
        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

    cv2.imwrite(str(output_path), image)
    print_result(f"Saved visualization to {output_path}")


async def main():
    print_header("InferFlow Object Detection Example")

    # Setup paths
    data_dir = pathlib.Path("data")
    output_dir = pathlib.Path("output")
    output_dir.mkdir(exist_ok=True)

    model_path = data_dir / "yolov5s.onnx"
    image_path = data_dir / "street.jpg"
    output_path = output_dir / "detected.jpg"

    # Step 1: Download model
    print_step(1, "Downloading YOLOv5s model")
    download_file(
        "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx",
        model_path,
    )

    # Step 2: Download test image
    print_step(2, "Downloading test image")
    download_file(
        "https://ultralytics.com/images/bus.jpg",
        image_path,
    )

    # Step 3: Load model
    print_step(3, "Loading YOLOv5 model")

    runtime = ONNXRuntime(model_path=model_path, device="cpu", precision=Precision.FP16, warmup_shape=(1, 3, 640, 640))

    pipeline = YOLOv5DetectionPipeline(
        runtime=runtime,
        image_size=(640, 640),
        conf_threshold=0.25,
        iou_threshold=0.45,
        class_names=COCO_CLASSES,
    )

    # Step 4: Run detection
    print_step(4, "Running object detection")

    async with pipeline.serve():
        with image_path.open("rb") as f:
            image_bytes = f.read()

        detections = await pipeline(image_bytes)

        print(f"\n{Fore.MAGENTA}{'─' * 60}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}{Style.BRIGHT}Detections:  {len(detections)} objects{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}{'─' * 60}{Style.RESET_ALL}")

        for i, det in enumerate(detections, 1):
            print(f"\n{Fore.CYAN}Object {i}:{Style.RESET_ALL}")
            print_info("  Class", det.class_name or f"ID {det.class_id}")
            print_info("  Confidence", f"{det.confidence:.2%}")
            print_info("  Box", f"({det.box.xc:.0f}, {det.box.yc:.0f}) {det.box.w:.0f}x{det.box.h:.0f}")

    # Step 5: Visualize
    print_step(5, "Creating visualization")
    visualize_detections(image_path, detections, output_path)

    print(f"\n{Fore.GREEN}{Style.BRIGHT}✓ Detection complete!{Style.RESET_ALL}\n")


if __name__ == "__main__":
    asyncio.run(main())

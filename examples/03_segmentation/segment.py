"""Instance Segmentation Example.

Demonstrates:
    - YOLOv5 instance segmentation
    - Mask overlay visualization
    - Multi-instance handling
"""

import asyncio
import pathlib
import urllib.request

from colorama import Fore
from colorama import Style
from colorama import init
import cv2
import numpy as np

from inferflow.asyncio.pipeline.segmentation.torch import YOLOv5SegmentationPipeline
from inferflow.asyncio.runtime.torch import TorchScriptRuntime

init(autoreset=True)


def print_header(text: str) -> None:
    print(f"\n{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{text.center(60)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}\n")


def print_step(step: int | float, text: str) -> None:
    print(f"{Fore.GREEN}[{step}]{Style.RESET_ALL} {text}")


def print_info(key: str, value: str) -> None:
    print(f"  {Fore.YELLOW}{key}:{Style.RESET_ALL} {value}")


def print_result(text: str) -> None:
    print(f"{Fore.GREEN}✓{Style.RESET_ALL} {text}")


COCO_CLASSES = {
    0: "person",
    2: "car",
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


def export_to_torchscript(model_path: pathlib.Path, output_path: pathlib.Path) -> None:
    """Export PyTorch model to TorchScript format."""
    # 1, clone the YOLOv5 repo
    import subprocess
    import sys

    repo_url = "https://github.com/ultralytics/yolov5.git"
    repo_dir = model_path.parent / "yolov5"
    if not repo_dir.exists():
        subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)
        print_result(f"Cloned YOLOv5 repo to {repo_dir}")
    else:
        print_info("Cached", str(repo_dir))

    # 2, create a venv and install dependencies
    venv_dir = repo_dir / "venv"
    if not venv_dir.exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        print_result(f"Created virtual environment at {venv_dir}")

        pip_executable = venv_dir / "Scripts" / "pip" if sys.platform == "win32" else venv_dir / "bin" / "pip"
        subprocess.run(
            [str(pip_executable), "install", "-r", str(repo_dir / "requirements.txt")],
            check=True,
        )
        print_result("Installed YOLOv5 dependencies")
    else:
        # Ensure dependencies are installed
        pip_executable = venv_dir / "Scripts" / "pip" if sys.platform == "win32" else venv_dir / "bin" / "pip"
        subprocess.run(
            [str(pip_executable), "install", "-r", str(repo_dir / "requirements.txt")],
            check=True,
        )
        print_info("Cached", str(venv_dir))

    # 3, run export script
    python_executable = venv_dir / "Scripts" / "python" if sys.platform == "win32" else venv_dir / "bin" / "python"
    subprocess.run(
        [
            str(python_executable),
            str(repo_dir / "export.py"),
            "--weights",
            str(model_path.absolute()),
            "--include",
            "torchscript",
        ],
        check=True,
    )
    # Move exported model to output_path
    exported_model_path = model_path.parent / "yolov5s-seg.torchscript"
    exported_model_path.rename(output_path)

    print_result(f"Exported TorchScript model to {output_path}")


def visualize_segmentation(
    image_path: pathlib.Path,
    segments: list,
    output_path: pathlib.Path,
) -> None:
    """Draw segmentation masks on image."""
    image = cv2.imread(str(image_path))
    overlay = image.copy()

    # Generate colors for each instance
    np.random.seed(42)
    colors = np.random.randint(0, 255, (len(segments), 3), dtype=np.uint8)

    for idx, seg in enumerate(segments):
        mask = seg.mask
        color = colors[idx].tolist()

        # Apply colored mask
        overlay[mask] = overlay[mask] * 0.5 + np.array(color) * 0.5

        # Draw bounding box
        box = seg.box
        x1 = int(box.xc - box.w / 2)
        y1 = int(box.yc - box.h / 2)
        x2 = int(box.xc + box.w / 2)
        y2 = int(box.yc + box.h / 2)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{seg.class_name or seg.class_id}:  {seg.confidence:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
        cv2.putText(
            overlay,
            label,
            (x1, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    # Blend
    result = cv2.addWeighted(image, 0.4, overlay, 0.6, 0)
    cv2.imwrite(str(output_path), result)
    print_result(f"Saved visualization to {output_path}")


async def main():
    print_header("InferFlow Instance Segmentation Example")

    # Setup paths
    data_dir = pathlib.Path("data")
    output_dir = pathlib.Path("output")
    output_dir.mkdir(exist_ok=True)

    model_path = data_dir / "yolov5s-seg.pt"
    image_path = data_dir / "bus.jpg"
    output_path = output_dir / "segmented.jpg"

    # Step 1: Download model
    print_step(1, "Downloading YOLOv5-Seg model")
    download_file(
        "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s-seg.pt",
        model_path,
    )

    # Step 1.5: Export to TorchScript
    torchscript_model_path = data_dir / "yolov5s-seg.torchscript"
    if not torchscript_model_path.exists():
        print_step(1.5, "Exporting to TorchScript format")
        export_to_torchscript(model_path, torchscript_model_path)
        model_path = torchscript_model_path
    else:
        model_path = torchscript_model_path
        print_info("Cached", str(model_path))

    # Step 2: Download test image
    print_step(2, "Downloading test image")
    download_file(
        "https://ultralytics.com/images/bus.jpg",
        image_path,
    )

    # Step 3: Load model
    print_step(3, "Loading YOLOv5-Seg model")

    runtime = TorchScriptRuntime(model_path=model_path, device="cpu", warmup_shape=(1, 3, 640, 640))

    pipeline = YOLOv5SegmentationPipeline(
        runtime=runtime,
        image_size=(640, 640),
        conf_threshold=0.25,
        iou_threshold=0.45,
        class_names=COCO_CLASSES,
    )

    # Step 4: Run segmentation
    print_step(4, "Running instance segmentation")

    async with pipeline.serve():
        with image_path.open("rb") as f:
            image_bytes = f.read()

        segments = await pipeline(image_bytes)

        print(f"\n{Fore.MAGENTA}{'─' * 60}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}{Style.BRIGHT}Segments: {len(segments)} instances{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}{'─' * 60}{Style.RESET_ALL}")

        for i, seg in enumerate(segments, 1):
            mask_area = seg.mask.sum()
            print(f"\n{Fore.CYAN}Instance {i}:{Style.RESET_ALL}")
            print_info("  Class", seg.class_name or f"ID {seg.class_id}")
            print_info("  Confidence", f"{seg.confidence:.2%}")
            print_info("  Mask pixels", f"{mask_area:,}")

    # Step 5: Visualize
    print_step(5, "Creating visualization")
    visualize_segmentation(image_path, segments, output_path)

    print(f"\n{Fore.GREEN}{Style.BRIGHT}✓ Segmentation complete!{Style.RESET_ALL}\n")


if __name__ == "__main__":
    asyncio.run(main())

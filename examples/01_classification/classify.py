"""Image Classification Example.

Demonstrates:
    - Loading a pretrained ResNet model
    - Single image inference
    - Result visualization with confidence scores
"""

import asyncio
import json
import pathlib
import urllib.request

from colorama import Fore
from colorama import Style
from colorama import init
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import torch

from inferflow import Precision
from inferflow.asyncio.pipeline.classification.torch import ClassificationPipeline
from inferflow.asyncio.runtime.torch import TorchScriptRuntime

# Initialize colorama
init(autoreset=True)


def print_header(text: str) -> None:
    """Print a colored header."""
    print(f"\n{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{text.center(60)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}\n")


def print_step(step: int | float, text: str) -> None:
    """Print a step message."""
    print(f"{Fore.GREEN}[{step}]{Style.RESET_ALL} {text}")


def print_info(key: str, value: str) -> None:
    """Print key-value info."""
    print(f"  {Fore.YELLOW}{key}:{Style.RESET_ALL} {value}")


def print_result(text: str) -> None:
    """Print a result message."""
    print(f"{Fore.GREEN}✓{Style.RESET_ALL} {text}")


def print_error(text: str) -> None:
    """Print an error message."""
    print(f"{Fore.RED}✗{Style.RESET_ALL} {text}")


def download_file(url: str, dest: pathlib.Path) -> None:
    """Download a file if it doesn't exist."""
    if dest.exists():
        print_info("Cached", str(dest))
        return

    print_info("Downloading", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print_result(f"Downloaded to {dest}")


def visualize_classification(
    image_path: pathlib.Path,
    class_name: str,
    confidence: float,
    output_path: pathlib.Path,
) -> None:
    """Visualize classification result."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except (OSError, ValueError):
        font = ImageFont.load_default(size=40)

    # Draw result at top
    text = f"{class_name}:  {confidence:.1%}"

    # Background box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    draw.rectangle([(10, 10), (20 + text_width, 20 + text_height)], fill=(0, 0, 0, 128))

    draw.text((15, 15), text, fill=(255, 255, 255), font=font)

    image.save(output_path)
    print_result(f"Saved visualization to {output_path}")


async def main():
    print_header("InferFlow Classification Example")

    # Setup paths
    data_dir = pathlib.Path("data")
    output_dir = pathlib.Path("output")
    output_dir.mkdir(exist_ok=True)

    model_path = data_dir / "resnet18.pt"
    image_path = data_dir / "dog.jpg"
    mapping_path = data_dir / "imagenet_class_index.json"
    output_path = output_dir / "classified.jpg"

    # Step 1: Download model
    print_step(1, "Downloading pretrained model")
    download_file(
        "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        model_path,
    )

    # Step 2: Download test image
    print_step(2, "Downloading test image")
    download_file(
        "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
        image_path,
    )

    # Step 2.5: Download class mapping (not used in this example, but useful for full ImageNet)
    print_step(2.5, "Downloading class mapping")
    download_file(
        "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json",
        mapping_path,
    )
    imagenet_class_index = json.loads(mapping_path.read_text(encoding="utf-8"))
    imagenet_class_index = {int(k): v[1] for k, v in imagenet_class_index.items()}

    # Step 3: Load model
    print_step(3, "Loading model")

    # Convert to TorchScript
    torchscript_path = data_dir / "resnet18_scripted.pt"
    if not torchscript_path.exists():
        print_info("Converting", "PyTorch → TorchScript")
        import torchvision.models as models

        model = models.resnet18(pretrained=False)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        traced = torch.jit.trace(model, dummy_input)
        torch.jit.save(traced, torchscript_path)
        print_result("Model converted")

    runtime = TorchScriptRuntime(
        model_path=torchscript_path,
        device="cpu",
        precision=Precision.FP32,
    )

    pipeline = ClassificationPipeline(
        runtime=runtime,
        image_size=(224, 224),
        class_names=imagenet_class_index,
    )

    # Step 4: Run inference
    print_step(4, "Running inference")

    async with pipeline.serve():
        with image_path.open("rb") as f:
            image_bytes = f.read()

        result = await pipeline(image_bytes)

        print(f"\n{Fore.MAGENTA}{'─' * 60}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}{Style.BRIGHT}Results:{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}{'─' * 60}{Style.RESET_ALL}")
        print_info("Class ID", str(result.class_id))
        print_info("Class Name", result.class_name or "Unknown")
        print_info("Confidence", f"{result.confidence:.2%}")

    # Step 5: Visualize
    print_step(5, "Creating visualization")
    visualize_classification(
        image_path,
        result.class_name or f"Class {result.class_id}",
        result.confidence,
        output_path,
    )

    print(f"\n{Fore.GREEN}{Style.BRIGHT}✓ Classification complete! {Style.RESET_ALL}\n")


if __name__ == "__main__":
    asyncio.run(main())

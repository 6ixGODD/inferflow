"""Custom Workflow Example.

Demonstrates:
    - Multi-stage inference pipeline
    - Conditional task execution
    - Parallel processing
    - Context-based state management
    - Quality control workflow
"""

import asyncio
import dataclasses
import enum
import pathlib
import urllib.request

from colorama import Fore
from colorama import Style
from colorama import init
import cv2
import numpy as np

from inferflow.asyncio.workflow.decorators import Workflow
from inferflow.asyncio.workflow.decorators import parallel
from inferflow.asyncio.workflow.decorators import task

init(autoreset=True)


def print_header(text: str) -> None:
    print(f"\n{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{text.center(70)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}\n")


def print_step(step: int, text: str) -> None:
    print(f"{Fore.GREEN}[{step}]{Style.RESET_ALL} {text}")


def print_task(text: str) -> None:
    print(f"{Fore.YELLOW}  ▸ {text}{Style.RESET_ALL}")


def print_info(key: str, value: str) -> None:
    print(f"    {Fore.CYAN}{key}:{Style.RESET_ALL} {value}")


def print_result(text: str) -> None:
    print(f"{Fore.GREEN}  ✓ {text}{Style.RESET_ALL}")


def print_warning(text: str) -> None:
    print(f"{Fore.YELLOW}  ⚠ {text}{Style.RESET_ALL}")


class QualityStatus(enum.Enum):
    """Quality check result."""

    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclasses.dataclass
class WorkflowContext:
    """Workflow execution context."""

    # Input
    image_path: pathlib.Path
    image_bytes: bytes | None = None

    # Preprocessing
    is_valid: bool = True
    resolution: tuple[int, int] | None = None
    brightness: float = 0.0

    # Detection
    detections: list | None = None
    has_defects: bool = False
    defect_count: int = 0

    # Segmentation
    segments: list | None = None
    defect_areas: list[float] | None = None

    # Classification
    quality_class: str | None = None
    confidence: float = 0.0

    # Results
    quality_status: QualityStatus = QualityStatus.UNKNOWN
    report: dict | None = None
    visualization: None | np.ndarray = None


def download_file(url: str, dest: pathlib.Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


# ============================================================================
# Workflow Tasks
# ============================================================================


@task(name="load_image", description="Load and validate image")
async def load_image(ctx: WorkflowContext) -> WorkflowContext:
    """Load image from disk."""
    print_task("Loading image")

    with ctx.image_path.open("rb") as f:
        ctx.image_bytes = f.read()

    # Decode to check validity
    arr = np.frombuffer(ctx.image_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image is None:
        ctx.is_valid = False
        print_warning("Invalid image")
    else:
        ctx.resolution = (image.shape[1], image.shape[0])
        print_result(f"Loaded {ctx.resolution[0]}x{ctx.resolution[1]} image")

    return ctx


@task(
    name="check_brightness",
    description="Analyze image brightness",
    condition=lambda ctx: ctx.is_valid,
)
async def check_brightness(ctx: WorkflowContext) -> WorkflowContext:
    """Check if image has sufficient brightness."""
    print_task("Checking brightness")

    arr = np.frombuffer(ctx.image_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ctx.brightness = float(gray.mean())
    print_info("Brightness", f"{ctx.brightness:.1f}")

    if ctx.brightness < 50:
        print_warning("Image too dark")
        ctx.is_valid = False
    else:
        print_result("Brightness OK")

    return ctx


@task(
    name="check_resolution",
    description="Validate image resolution",
    condition=lambda ctx: ctx.is_valid,
)
async def check_resolution(ctx: WorkflowContext) -> WorkflowContext:
    """Check if resolution meets requirements."""
    print_task("Checking resolution")

    min_width, min_height = 640, 480

    if ctx.resolution[0] < min_width or ctx.resolution[1] < min_height:
        print_warning(f"Resolution too low (min: {min_width}x{min_height})")
        ctx.is_valid = False
    else:
        print_result("Resolution OK")

    return ctx


@task(
    name="detect_defects",
    description="Run defect detection",
    condition=lambda ctx: ctx.is_valid,
)
async def detect_defects(ctx: WorkflowContext) -> WorkflowContext:
    """Detect defects using object detection."""
    print_task("Running defect detection")

    # Simulate detection (in real scenario, use actual model)
    await asyncio.sleep(0.1)  # Simulate inference

    # Mock results
    ctx.detections = []  # Would be actual detections
    ctx.defect_count = len(ctx.detections)
    ctx.has_defects = ctx.defect_count > 0

    if ctx.has_defects:
        print_info("Defects found", str(ctx.defect_count))
    else:
        print_result("No defects detected")

    return ctx


@task(
    name="segment_defects",
    description="Segment defect regions",
    condition=lambda ctx: ctx.is_valid and ctx.has_defects,
)
async def segment_defects(ctx: WorkflowContext) -> WorkflowContext:
    """Segment defect areas for detailed analysis."""
    print_task("Segmenting defect regions")

    # Simulate segmentation
    await asyncio.sleep(0.1)

    ctx.segments = []
    ctx.defect_areas = []

    print_result("Segmentation complete")

    return ctx


@task(
    name="classify_quality",
    description="Classify overall quality",
    condition=lambda ctx: ctx.is_valid,
)
async def classify_quality(ctx: WorkflowContext) -> WorkflowContext:
    """Classify product quality based on defects."""
    print_task("Classifying quality")

    # Simulate classification
    await asyncio.sleep(0.05)

    if not ctx.has_defects:
        ctx.quality_class = "Grade A"
        ctx.confidence = 0.95
        ctx.quality_status = QualityStatus.PASS
    else:
        ctx.quality_class = "Grade B"
        ctx.confidence = 0.85
        ctx.quality_status = QualityStatus.FAIL

    print_info("Quality", ctx.quality_class)
    print_info("Confidence", f"{ctx.confidence:.1%}")

    return ctx


@task(name="generate_report", description="Create quality report")
async def generate_report(ctx: WorkflowContext) -> WorkflowContext:
    """Generate final quality control report."""
    print_task("Generating report")

    ctx.report = {
        "status": ctx.quality_status.value,
        "quality_class": ctx.quality_class,
        "confidence": ctx.confidence,
        "resolution": f"{ctx.resolution[0]}x{ctx.resolution[1]}" if ctx.resolution else "N/A",
        "brightness": f"{ctx.brightness:.1f}",
        "defect_count": ctx.defect_count,
    }

    print_result("Report generated")

    return ctx


@task(name="create_visualization", description="Create result visualization")
async def create_visualization(ctx: WorkflowContext) -> WorkflowContext:
    """Create visualization of results."""
    print_task("Creating visualization")

    arr = np.frombuffer(ctx.image_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # Draw status badge
    status_color = (0, 255, 0) if ctx.quality_status == QualityStatus.PASS else (0, 0, 255)
    status_text = ctx.quality_status.value.upper()

    cv2.rectangle(image, (10, 10), (200, 60), status_color, -1)
    cv2.putText(
        image,
        status_text,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2,
    )

    # Draw info
    y = 100
    for key, value in ctx.report.items():
        text = f"{key}: {value}"
        cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25

    ctx.visualization = image
    print_result("Visualization created")

    return ctx


# ============================================================================
# Main
# ============================================================================


async def main():
    print_header("InferFlow Custom Workflow Example")
    print(f"{Fore.CYAN}Quality Control Workflow{Style.RESET_ALL}\n")

    # Setup
    data_dir = pathlib.Path("data")
    output_dir = pathlib.Path("output")
    output_dir.mkdir(exist_ok=True)

    image_path = data_dir / "product. jpg"
    output_path = output_dir / "workflow_result.jpg"

    # Download test image
    print_step(1, "Preparing test image")
    download_file(
        "https://ultralytics.com/images/bus.jpg",
        image_path,
    )

    # Step 2: Define workflow
    print_step(2, "Building quality control workflow")

    print(f"\n{Fore.CYAN}Workflow Structure:{Style.RESET_ALL}")
    print("  1. Load & Validate")
    print("  2. Parallel Quality Checks")
    print("     ├─ Brightness check")
    print("     └─ Resolution check")
    print("  3. Defect Detection")
    print("  4. Conditional Segmentation (if defects found)")
    print("  5. Quality Classification")
    print("  6.  Parallel Output Generation")
    print("     ├─ Generate report")
    print("     └─ Create visualization")

    workflow = Workflow[WorkflowContext](
        load_image,
        parallel(
            check_brightness,
            check_resolution,
        ),
        detect_defects,
        segment_defects,  # Conditional
        classify_quality,
        parallel(
            generate_report,
            create_visualization,
        ),
    )

    # Step 3: Execute workflow
    print_step(3, "Executing workflow")
    print()

    context = WorkflowContext(image_path=image_path)

    start = asyncio.get_event_loop().time()
    result = await workflow.run(context)
    elapsed = asyncio.get_event_loop().time() - start

    # Step 4: Display results
    print()
    print_step(4, "Results")

    print(f"\n{Fore.YELLOW}{'─' * 70}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{Style.BRIGHT}Quality Control Report{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'─' * 70}{Style.RESET_ALL}")

    for key, value in result.report.items():
        print(f"  {Fore.CYAN}{key.replace('_', ' ').title().ljust(20)}: {Style.RESET_ALL}{value}")

    print(f"{Fore.YELLOW}{'─' * 70}{Style.RESET_ALL}")
    print(f"  {Fore.GREEN}Processing Time: {elapsed * 1000:.2f} ms{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'─' * 70}{Style.RESET_ALL}\n")

    # Save visualization
    if result.visualization is not None:
        cv2.imwrite(str(output_path), result.visualization)
        print_result(f"Saved visualization to {output_path}")

    # Final status
    if result.quality_status == QualityStatus.PASS:
        print(f"\n{Fore.GREEN}{Style.BRIGHT}✓ QUALITY CHECK PASSED{Style.RESET_ALL}\n")
    else:
        print(f"\n{Fore.RED}{Style.BRIGHT}✗ QUALITY CHECK FAILED{Style.RESET_ALL}\n")


if __name__ == "__main__":
    asyncio.run(main())

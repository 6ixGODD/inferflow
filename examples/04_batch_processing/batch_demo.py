"""Dynamic Batch Processing Example.

Demonstrates:
    - Dynamic batching with adaptive batch size
    - Peak shaving (handling request bursts)
    - Real-time metrics monitoring
    - Throughput optimization
"""

import asyncio
import dataclasses
import pathlib
import time
import urllib.request

from colorama import Fore
from colorama import Style
from colorama import init
import numpy as np
import torch

from inferflow.asyncio.batch.dynamic import DynamicBatchStrategy
from inferflow.asyncio.pipeline.classification.torch import ClassificationPipeline
from inferflow.asyncio.runtime.torch import TorchScriptRuntime

init(autoreset=True)


def print_header(text: str) -> None:
    print(f"\n{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{text.center(70)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}\n")


def print_step(step: int, text: str) -> None:
    print(f"{Fore.GREEN}[{step}]{Style.RESET_ALL} {text}")


def print_info(key: str, value: str) -> None:
    print(f"  {Fore.YELLOW}{key}:{Style.RESET_ALL} {value}")


def print_metric(label: str, value: str, color: str = Fore.CYAN) -> None:
    print(f"{color}  {label.ljust(25)}: {value}{Style.RESET_ALL}")


def print_burst(burst_id: int, size: int, delay: float) -> None:
    print(f"{Fore.MAGENTA}  Burst #{burst_id}:  {size} requests (delay: {delay:.3f}s){Style.RESET_ALL}")


@dataclasses.dataclass
class RequestBurst:
    """Simulated request burst."""

    size: int
    """Number of requests in the burst."""

    delay: float
    """Delay before the burst starts (seconds)."""


# ImageNet subset
IMAGENET_CLASSES = {
    207: "golden_retriever",
    281: "tabby_cat",
    151: "Chihuahua",
}


def download_file(url: str, dest: pathlib.Path) -> None:
    if dest.exists():
        return

    print_info("Downloading", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


async def simulate_client(
    client_id: int,
    pipeline: ClassificationPipeline,
    image_bytes: bytes,
    delay: float,
) -> tuple[int, float]:
    """Simulate a single client request."""
    await asyncio.sleep(delay)

    start = time.time()
    _ = await pipeline(image_bytes)
    latency = time.time() - start

    return client_id, latency


async def run_burst(
    burst_id: int,
    burst: RequestBurst,
    pipeline: ClassificationPipeline,
    image_bytes: bytes,
) -> list[float]:
    """Run a burst of concurrent requests."""
    print_burst(burst_id, burst.size, burst.delay)

    # Create tasks with staggered delays
    tasks = []
    for i in range(burst.size):
        delay = burst.delay + (i * 0.01)  # Slight stagger
        task = simulate_client(i, pipeline, image_bytes, delay)
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    return [lat for _, lat in results]


def print_metrics_table(metrics: dict) -> None:
    """Print metrics in a formatted table."""
    print(f"\n{Fore.YELLOW}{'─' * 70}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{Style.BRIGHT}Batch Processing Metrics{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'─' * 70}{Style.RESET_ALL}")

    print_metric("Total Requests", f"{metrics['total_requests']:,}", Fore.GREEN)
    print_metric("Total Batches", f"{metrics['total_batches']:,}", Fore.GREEN)
    print_metric("Avg Batch Size", f"{metrics['avg_batch_size']:.2f}", Fore.CYAN)
    print_metric("Avg Latency", f"{metrics['avg_latency_ms']:.2f} ms", Fore.CYAN)
    print_metric("Avg Processing Time", f"{metrics['avg_processing_time_ms']:.2f} ms", Fore.CYAN)
    print_metric("Queue Size", f"{metrics['current_queue_size']}", Fore.BLUE)
    print_metric(
        "Rejected Requests",
        f"{metrics['rejected_requests']}",
        Fore.RED if metrics["rejected_requests"] > 0 else Fore.GREEN,
    )

    print(f"{Fore.YELLOW}{'─' * 70}{Style.RESET_ALL}\n")


def print_latency_stats(latencies: list[float]) -> None:
    """Print latency statistics."""
    arr = np.array(latencies) * 1000  # Convert to ms

    print(f"{Fore.CYAN}Latency Statistics:{Style.RESET_ALL}")
    print_metric("  Min", f"{arr.min():.2f} ms", Fore.GREEN)
    print_metric("  Mean", f"{arr.mean():.2f} ms", Fore.CYAN)
    print_metric("  Median", f"{np.median(arr):.2f} ms", Fore.CYAN)
    print_metric("  P95", f"{np.percentile(arr, 95):.2f} ms", Fore.YELLOW)
    print_metric("  Max", f"{arr.max():.2f} ms", Fore.RED)


async def main():
    print_header("InferFlow Dynamic Batch Processing Demo")

    # Setup
    data_dir = pathlib.Path("data")
    data_dir.mkdir(exist_ok=True)

    model_path = data_dir / "resnet18_scripted.pt"
    image_path = data_dir / "dog.jpg"

    # Download resources
    print_step(1, "Preparing resources")
    download_file(
        "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        data_dir / "resnet18.pt",
    )
    download_file(
        "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
        image_path,
    )

    # Convert model
    if not model_path.exists():
        print_info("Converting", "PyTorch → TorchScript")
        import torchvision.models as models

        model = models.resnet18(pretrained=False)
        model.load_state_dict(torch.load(data_dir / "resnet18.pt"))
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        traced = torch.jit.trace(model, dummy_input)
        torch.jit.save(traced, model_path)

    # Load image
    with image_path.open("rb") as f:
        image_bytes = f.read()

    # Step 2: Setup batch strategy
    print_step(2, "Configuring dynamic batch strategy")

    batch_strategy = DynamicBatchStrategy[torch.Tensor, torch.Tensor](
        min_batch_size=1,
        max_batch_size=16,
        max_wait_ms=50,
        queue_size=500,
        block_on_full=True,
    )

    print_info("Min batch size", "1")
    print_info("Max batch size", "16")
    print_info("Max wait time", "50 ms")
    print_info("Queue size", "500")

    # Step 3: Create pipeline with batching
    print_step(3, "Creating pipeline with batching enabled")

    runtime = TorchScriptRuntime(model_path=model_path, device="cpu")

    pipeline = ClassificationPipeline(
        runtime=runtime,
        image_size=(224, 224),
        class_names=IMAGENET_CLASSES,
        batch_strategy=batch_strategy,
    )

    # Step 4: Simulate traffic patterns
    print_step(4, "Simulating request traffic (peak shaving demo)")

    # Define traffic pattern:  bursts and valleys
    traffic_pattern = [
        RequestBurst(size=5, delay=0.0),  # Small burst
        RequestBurst(size=20, delay=0.5),  # Medium burst (peak)
        RequestBurst(size=50, delay=1.0),  # Large burst (PEAK!)
        RequestBurst(size=10, delay=2.0),  # Valley
        RequestBurst(size=30, delay=2.5),  # Medium burst
        RequestBurst(size=5, delay=3.5),  # Small burst (valley)
    ]

    total_requests = sum(b.size for b in traffic_pattern)
    print(f"\n{Fore.CYAN}Traffic Pattern:{Style.RESET_ALL}")
    print_info("Total requests", str(total_requests))
    print_info("Bursts", str(len(traffic_pattern)))
    print()

    async with pipeline.serve():
        all_latencies = []

        for i, burst in enumerate(traffic_pattern, 1):
            latencies = await run_burst(i, burst, pipeline, image_bytes)
            all_latencies.extend(latencies)

            # Print intermediate metrics
            if i % 2 == 0:
                metrics = batch_strategy.get_metrics()
                print(
                    f"\n{Fore.BLUE}  Current queue:  {metrics.current_queue_size}, "
                    f"Avg batch:  {metrics.avg_batch_size:.1f}{Style.RESET_ALL}"
                )

            await asyncio.sleep(0.1)  # Small gap between bursts

    # Step 5: Display results
    print_step(5, "Results & Metrics")

    # Final metrics
    metrics = batch_strategy.get_metrics()
    metrics_dict = {
        "total_requests": metrics.total_requests,
        "total_batches": metrics.total_batches,
        "avg_batch_size": metrics.avg_batch_size,
        "avg_latency_ms": metrics.avg_latency_ms,
        "avg_processing_time_ms": metrics.avg_processing_time_ms,
        "current_queue_size": metrics.current_queue_size,
        "rejected_requests": metrics.rejected_requests,
    }

    print_metrics_table(metrics_dict)

    # Latency statistics
    print_latency_stats(all_latencies)

    # Throughput calculation
    total_time = max(all_latencies) + traffic_pattern[-1].delay
    throughput = total_requests / total_time

    print(f"\n{Fore.GREEN}{'─' * 70}{Style.RESET_ALL}")
    print_metric("Total Time", f"{total_time:.2f} s", Fore.GREEN)
    print_metric("Throughput", f"{throughput:.2f} req/s", Fore.GREEN)
    print_metric("Batching Efficiency", f"{(metrics.avg_batch_size / 16) * 100:.1f}% (of max batch size)", Fore.GREEN)
    print(f"{Fore.GREEN}{'─' * 70}{Style.RESET_ALL}")

    print(f"\n{Fore.GREEN}{Style.BRIGHT}✓ Batch processing demo complete!{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Notice how batching handled {total_requests} requests efficiently! {Style.RESET_ALL}\n")


if __name__ == "__main__":
    asyncio.run(main())

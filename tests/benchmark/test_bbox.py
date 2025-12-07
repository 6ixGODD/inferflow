from __future__ import annotations

import time

import pytest
import torch

try:
    from inferflow import HAS_CPP_EXTENSIONS
    from inferflow._C.ops import bbox
except ImportError:
    HAS_CPP_EXTENSIONS = False
    bbox = None


def timer(func, *args, warmup: int = 3, repeat: int = 10, **kwargs):
    """Measure function execution time.

    Args:
        func: Function to benchmark.
        warmup: Number of warmup iterations.
        repeat: Number of timed iterations.

    Returns:
        Tuple of (avg_time_seconds, last_result).
    """
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    # Synchronize CUDA if available
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed execution
    start = time.perf_counter()
    result = None
    for _ in range(repeat):
        result = func(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    avg_time = elapsed / repeat

    return avg_time, result


def print_performance(
    name: str,
    time_python: float,
    time_cpp: float,
    size: int | None = None,
):
    """Print formatted performance comparison.

    Args:
        name: Operation name.
        time_python: Python execution time (seconds).
        time_cpp: C++ execution time (seconds).
        size: Optional data size for throughput calculation.
    """
    speedup = time_python / time_cpp
    emoji = "🚀" if speedup > 2.0 else "✅" if speedup > 1.2 else "⚠️"

    print(f"\n{'=' * 80}")
    print(f"📊 {name}")
    print(f"{'=' * 80}")
    print(f"  Python:  {time_python * 1000:8.3f} ms")
    print(f"  C++:     {time_cpp * 1000:8.3f} ms")
    print(f"  Speedup: {speedup:8.2f}x  {emoji}")

    if size:
        throughput_py = size / time_python
        throughput_cpp = size / time_cpp
        print("\n  Throughput:")
        print(f"    Python:  {throughput_py:10.0f} boxes/sec")
        print(f"    C++:     {throughput_cpp:10.0f} boxes/sec")


@pytest.mark.skipif(not HAS_CPP_EXTENSIONS, reason="C++ extensions not available")
class TestBBoxConversionPerformance:
    """Benchmark bounding box format conversion."""

    def test_xywh2xyxy_small(self, bbox_samples_small):
        """Test xywh2xyxy on small dataset."""

        def python_impl(x):
            y = x.clone()
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            return y

        time_python, result_py = timer(python_impl, bbox_samples_small, repeat=100)
        time_cpp, result_cpp = timer(bbox.xywh2xyxy, bbox_samples_small, repeat=100)

        # Verify correctness
        assert torch.allclose(result_py, result_cpp, atol=1e-5), "Results differ!"

        print_performance(
            f"xywh2xyxy ({bbox_samples_small.shape[0]} boxes)", time_python, time_cpp, bbox_samples_small.shape[0]
        )

        assert time_cpp <= time_python, "C++ should not be slower than Python"

    def test_xywh2xyxy_medium(self, bbox_samples_medium):
        """Test xywh2xyxy on medium dataset (typical use case)."""

        def python_impl(x):
            y = x.clone()
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            return y

        time_python, result_py = timer(python_impl, bbox_samples_medium, repeat=100)
        time_cpp, result_cpp = timer(bbox.xywh2xyxy, bbox_samples_medium, repeat=100)

        assert torch.allclose(result_py, result_cpp, atol=1e-5)

        print_performance(
            f"xywh2xyxy ({bbox_samples_medium.shape[0]} boxes)", time_python, time_cpp, bbox_samples_medium.shape[0]
        )

        speedup = time_python / time_cpp
        assert speedup > 1.1, f"Expected >1.1x speedup, got {speedup:.2f}x"

    def test_xywh2xyxy_large(self, bbox_samples_large):
        """Test xywh2xyxy on large dataset (stress test)."""

        def python_impl(x):
            y = x.clone()
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            return y

        time_python, result_py = timer(python_impl, bbox_samples_large, repeat=50)
        time_cpp, result_cpp = timer(bbox.xywh2xyxy, bbox_samples_large, repeat=50)

        assert torch.allclose(result_py, result_cpp, atol=1e-5)

        print_performance(
            f"xywh2xyxy ({bbox_samples_large.shape[0]} boxes)", time_python, time_cpp, bbox_samples_large.shape[0]
        )

    def test_xyxy2xywh_medium(self, bbox_samples_medium):
        """Test xyxy2xywh on medium dataset."""

        def python_impl(x):
            y = x.clone()
            y[:, 0] = (x[:, 0] + x[:, 2]) / 2
            y[:, 1] = (x[:, 1] + x[:, 3]) / 2
            y[:, 2] = x[:, 2] - x[:, 0]
            y[:, 3] = x[:, 3] - x[:, 1]
            return y

        time_python, result_py = timer(python_impl, bbox_samples_medium, repeat=100)
        time_cpp, result_cpp = timer(bbox.xyxy2xywh, bbox_samples_medium, repeat=100)

        assert torch.allclose(result_py, result_cpp, atol=1e-5)

        print_performance(
            f"xyxy2xywh ({bbox_samples_medium.shape[0]} boxes)", time_python, time_cpp, bbox_samples_medium.shape[0]
        )

        speedup = time_python / time_cpp
        assert speedup > 1.1, f"Expected >1.1x speedup, got {speedup:.2f}x"

    def test_round_trip_conversion(self, bbox_samples_medium):
        """Test xywh -> xyxy -> xywh round trip."""

        print(f"\n{'=' * 80}")
        print("🔄 Round-Trip Conversion Test")
        print(f"{'=' * 80}")

        # C++ round trip
        start = time.perf_counter()
        xyxy = bbox.xywh2xyxy(bbox_samples_medium)
        xywh_back = bbox.xyxy2xywh(xyxy)
        time_cpp = time.perf_counter() - start

        # Verify correctness
        assert torch.allclose(bbox_samples_medium, xywh_back, atol=1e-4), "Round trip failed!"

        print(f"  Original shape: {bbox_samples_medium.shape}")
        print(f"  C++ round trip: {time_cpp * 1000:.3f} ms")
        print(f"  ✅ Precision: {(bbox_samples_medium - xywh_back).abs().max():.2e}")


@pytest.mark.skipif(not HAS_CPP_EXTENSIONS, reason="C++ extensions not available")
class TestBBoxIoUPerformance:
    """Benchmark IoU calculation."""

    def test_box_iou_small(self, bbox_pair):
        """Test IoU on small box sets."""
        box1, box2 = bbox_pair

        def python_impl(b1, b2, eps=1e-7):
            (a1, a2), (c1, c2) = b1.unsqueeze(1).chunk(2, 2), b2.unsqueeze(0).chunk(2, 2)
            inter = (torch.min(a2, c2) - torch.max(a1, c1)).clamp(0).prod(2)
            return inter / ((a2 - a1).prod(2) + (c2 - c1).prod(2) - inter + eps)

        time_python, result_py = timer(python_impl, box1, box2, repeat=1000)
        time_cpp, result_cpp = timer(bbox.box_iou, box1, box2, repeat=1000)

        assert torch.allclose(result_py, result_cpp, atol=1e-5)

        print_performance(
            f"box_iou ({box1.shape[0]}x{box2.shape[0]} comparisons)",
            time_python,
            time_cpp,
        )

    def test_box_iou_medium(self, bbox_samples_small):
        """Test IoU on medium box sets (NxN comparisons)."""

        def python_impl(b1, b2, eps=1e-7):
            (a1, a2), (c1, c2) = b1.unsqueeze(1).chunk(2, 2), b2.unsqueeze(0).chunk(2, 2)
            inter = (torch.min(a2, c2) - torch.max(a1, c1)).clamp(0).prod(2)
            return inter / ((a2 - a1).prod(2) + (c2 - c1).prod(2) - inter + eps)

        time_python, result_py = timer(python_impl, bbox_samples_small, bbox_samples_small, repeat=50)
        time_cpp, result_cpp = timer(bbox.box_iou, bbox_samples_small, bbox_samples_small, repeat=50)

        assert torch.allclose(result_py, result_cpp, atol=1e-5)

        num_comparisons = bbox_samples_small.shape[0] ** 2
        print_performance(
            f"box_iou ({bbox_samples_small.shape[0]}x{bbox_samples_small.shape[0]} = {num_comparisons:,} comparisons)",
            time_python,
            time_cpp,
        )


@pytest.mark.skipif(not HAS_CPP_EXTENSIONS, reason="C++ extensions not available")
class TestBBoxBatchPerformance:
    """Benchmark batched operations (3D tensors)."""

    def test_xywh2xyxy_batched(self, bbox_batch):
        """Test xywh2xyxy on batched input."""

        def python_impl(x):
            y = x.clone()
            y[..., 0] = x[..., 0] - x[..., 2] / 2
            y[..., 1] = x[..., 1] - x[..., 3] / 2
            y[..., 2] = x[..., 0] + x[..., 2] / 2
            y[..., 3] = x[..., 1] + x[..., 3] / 2
            return y

        time_python, result_py = timer(python_impl, bbox_batch, repeat=100)
        time_cpp, result_cpp = timer(bbox.xywh2xyxy, bbox_batch, repeat=100)

        assert torch.allclose(result_py, result_cpp, atol=1e-5)

        batch_size, num_boxes = bbox_batch.shape[:2]
        total_boxes = batch_size * num_boxes

        print_performance(
            f"xywh2xyxy batched (batch={batch_size}, boxes={num_boxes}, total={total_boxes})",
            time_python,
            time_cpp,
            total_boxes,
        )


def test_summary():
    """Print overall performance summary."""
    if not HAS_CPP_EXTENSIONS:
        pytest.skip("C++ extensions not available")

    print("\n" + "=" * 80)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 80)

    # Test various sizes
    sizes = [100, 1000, 10000]
    results = []

    for size in sizes:
        boxes = torch.rand(size, 4) * 640

        def python_impl(x):
            y = x.clone()
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            return y

        time_py, _ = timer(python_impl, boxes, warmup=5, repeat=100)
        time_cpp, _ = timer(bbox.xywh2xyxy, boxes, warmup=5, repeat=100)

        speedup = time_py / time_cpp
        results.append((size, time_py, time_cpp, speedup))

    print("\n📈 xywh2xyxy Performance Scaling:")
    print(f"{'Size':>10} | {'Python (ms)':>12} | {'C++ (ms)':>12} | {'Speedup':>8}")
    print("-" * 60)

    for size, time_py, time_cpp, speedup in results:
        print(f"{size:10,} | {time_py * 1000:12.3f} | {time_cpp * 1000:12.3f} | {speedup:8.2f}x")

    avg_speedup = sum(r[3] for r in results) / len(results)
    print(f"\n🎯 Average speedup: {avg_speedup:.2f}x")
    print("=" * 80)

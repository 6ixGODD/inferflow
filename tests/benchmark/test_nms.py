import time

import pytest
import torch

from inferflow.utils.yolo import nms as nms_py

try:
    from inferflow import _C
    from inferflow import HAS_CPP_EXTENSIONS
except ImportError:
    HAS_CPP_EXTENSIONS = False
    _C = None


def timer(func, *args, warmup=3, repeat=10, **kwargs):
    for _ in range(warmup):
        func(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    result = None
    for _ in range(repeat):
        result = func(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()

    avg_time = (end - start) / repeat
    return avg_time, result


@pytest.mark.skipif(not HAS_CPP_EXTENSIONS, reason="C++ extensions not available")
class TestNMSPerformance:
    def test_nms_single_image(self, yolo_prediction):
        print("\n" + "=" * 80)
        print("NMS Performance: Single Image (8400 boxes)")
        print("=" * 80)

        time_python, result_python = timer(
            nms_py,
            yolo_prediction,
            conf_thres=0.25,
            iou_thres=0.45,
            max_det=300,
            nm=0,
            warmup=3,
            repeat=50,
        )

        time_cpp, result_cpp = timer(
            _C.nms,
            yolo_prediction,
            0.25,  # conf_thres
            0.45,  # iou_thres
            300,  # max_det
            0,  # nm
            warmup=3,
            repeat=50,
        )

        assert len(result_python) == len(result_cpp), "Length of results differ"
        num_det_python = result_python[0].shape[0]
        num_det_cpp = result_cpp[0].shape[0]

        assert abs(num_det_python - num_det_cpp) <= 2, "Number of detections differ significantly"

        speedup = time_python / time_cpp

        print("\nResults:")
        print(f"  Python:  {time_python * 1000:8.3f} ms  ({num_det_python} detections)")
        print(f"  C++:     {time_cpp * 1000:8.3f} ms  ({num_det_cpp} detections)")
        print(f"  Speedup: {speedup:8.2f}x")
        print("\n  Throughput:")
        print(f"    Python:  {1 / time_python:6.1f} images/sec")
        print(f"    C++:     {1 / time_cpp:6.1f} images/sec")

        assert speedup > 1.5, f"C++ should be faster!  Got {speedup:.2f}x"

    def test_nms_batch(self, large_batch_prediction):
        batch_size = large_batch_prediction.shape[0]

        print("\n" + "=" * 80)
        print(f"NMS Performance: Batch Processing ({batch_size} images)")
        print("=" * 80)

        time_python, result_python = timer(
            nms_py,
            large_batch_prediction,
            conf_thres=0.25,
            iou_thres=0.45,
            max_det=300,
            nm=0,
            warmup=2,
            repeat=20,
        )

        time_cpp, result_cpp = timer(
            _C.nms,
            large_batch_prediction,
            0.25,
            0.45,
            300,
            0,
            warmup=2,
            repeat=20,
        )

        speedup = time_python / time_cpp
        throughput_python = batch_size / time_python
        throughput_cpp = batch_size / time_cpp

        print("\nResults:")
        print(f"  Python:  {time_python * 1000:8.3f} ms  ({throughput_python:6.1f} images/sec)")
        print(f"  C++:     {time_cpp * 1000:8.3f} ms  ({throughput_cpp:6.1f} images/sec)")
        print(f"  Speedup: {speedup:8.2f}x")
        print("  Per-image latency:")
        print(f"    Python:  {time_python / batch_size * 1000:6.2f} ms/image")
        print(f"    C++:     {time_cpp / batch_size * 1000:6.2f} ms/image")

        assert speedup > 1.5

    def test_nms_varying_thresholds(self, yolo_prediction):
        print("\n" + "=" * 80)
        print("NMS Performance: Varying Confidence Thresholds")
        print("=" * 80)

        thresholds = [0.1, 0.25, 0.5, 0.7, 0.9]

        print(f"\n{'Threshold':>10} | {'Python (ms)':>12} | {'C++ (ms)':>12} | {'Speedup':>8} | {'Detections':>12}")
        print("-" * 80)

        for conf_thres in thresholds:
            time_python, result_python = timer(
                nms_py,
                yolo_prediction,
                conf_thres=conf_thres,
                iou_thres=0.45,
                warmup=2,
                repeat=20,
            )

            time_cpp, result_cpp = timer(
                _C.nms,
                yolo_prediction,
                conf_thres,
                0.45,
                300,
                0,
                warmup=2,
                repeat=20,
            )

            speedup = time_python / time_cpp
            num_det = result_cpp[0].shape[0]

            print(
                f"{conf_thres:10.2f} | "
                f"{time_python * 1000:12.3f} | "
                f"{time_cpp * 1000:12.3f} | "
                f"{speedup:8.2f}x | "
                f"{num_det:12d}"
            )


@pytest.mark.skipif(not HAS_CPP_EXTENSIONS, reason="C++ extensions not available")
class TestBBoxOpsPerformance:
    def test_xywh2xyxy(self, bbox_samples):
        print("\n" + "=" * 80)
        print(f"Bbox Conversion: xywh2xyxy ({bbox_samples.shape[0]} boxes)")
        print("=" * 80)

        def python_xywh2xyxy(x):
            y = x.clone()
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            return y

        time_python, _ = timer(python_xywh2xyxy, bbox_samples, repeat=100)
        time_cpp, _ = timer(_C.xywh2xyxy, bbox_samples, repeat=100)

        speedup = time_python / time_cpp

        print("\nResults:")
        print(f"  Python:  {time_python * 1000:8.3f} ms")
        print(f"  C++:     {time_cpp * 1000:8.3f} ms")
        print(f"  Speedup: {speedup:8.2f}x")

        assert speedup > 0.8, "C++ version should not be significantly slower"


def test_summary(yolo_prediction, large_batch_prediction):
    if not HAS_CPP_EXTENSIONS:
        pytest.skip("C++ extensions not available")

    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)

    time_python_single, _ = timer(nms_py, yolo_prediction, repeat=50)
    time_cpp_single, _ = timer(_C.nms, yolo_prediction, 0.25, 0.45, 300, 0, repeat=50)

    time_python_batch, _ = timer(nms_py, large_batch_prediction, repeat=20)
    time_cpp_batch, _ = timer(_C.nms, large_batch_prediction, 0.25, 0.45, 300, 0, repeat=20)

    batch_size = large_batch_prediction.shape[0]

    print("\nSingle Image Performance:")
    print(f"  Python:  {time_python_single * 1000:7.2f} ms  →  {1 / time_python_single:6.1f} FPS")
    print(f"  C++:     {time_cpp_single * 1000:7.2f} ms  →  {1 / time_cpp_single:6.1f} FPS")
    print(f"  Speedup: {time_python_single / time_cpp_single:7.2f}x")

    print(f"\nBatch Processing (batch={batch_size}):")
    print(f"  Python:  {time_python_batch * 1000:7.2f} ms  →  {batch_size / time_python_batch:6.1f} images/sec")
    print(f"  C++:     {time_cpp_batch * 1000:7.2f} ms  →  {batch_size / time_cpp_batch:6.1f} images/sec")
    print(f"  Speedup: {time_python_batch / time_cpp_batch:7.2f}x")

    print("\n" + "=" * 80)

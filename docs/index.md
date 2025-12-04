# InferFlow

**Universal Inference Pipeline Framework**

[![PyPI version](https://badge.fury.io/py/inferflow.svg)](https://pypi.org/project/inferflow/)
[![Python](https://img.shields.io/pypi/pyversions/inferflow.svg)](https://pypi.org/project/inferflow/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](license.md)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://6ixgodd.github.io/inferflow/)


---

## Overview

InferFlow is a production-grade inference pipeline framework designed for computer vision models. It provides a clean abstraction layer that separates model runtime, preprocessing, postprocessing, and batching strategies, enabling seamless deployment across multiple inference backends.

**Key Features:**

- **Runtime Agnostic**: Support for TorchScript, ONNX, TensorRT, and more (in the future :D)
- **Dynamic Batching**: Automatic request batching with adaptive batch sizing
- **Type Safe**: Full type hints with generic pipeline definitions
- **Async First**: Built on asyncio for high concurrency
- **Production Ready**: Comprehensive logging, metrics, and error handling

---

## Installation

```bash
pip install inferflow
```

**Development installation:**

```bash
git clone https://github.com/6ixGODD/inferflow.git
cd inferflow
pip install -e ".[dev]"
```

---

## Quick Start

### Basic Classification

```python
import asyncio
from inferflow.runtime.torch import TorchScriptRuntime
from inferflow.pipeline.classification import ClassificationPipeline

async def main():
    # Setup runtime
    runtime = TorchScriptRuntime(
        model_path="resnet50.pt",
        device="cuda:0",
    )

    # Create pipeline
    pipeline = ClassificationPipeline(
        runtime=runtime,
        class_names={0: "cat", 1: "dog", 2: "bird"},
    )

    # Run inference
    async with pipeline.serve():
        with open("image.jpg", "rb") as f:
            result = await pipeline(f.read())

        print(f"{result.class_name}: {result.confidence:.2%}")

asyncio.run(main())
```

### Object Detection

```python
from inferflow.pipeline.detection import YOLOv5DetectionPipeline

pipeline = YOLOv5DetectionPipeline(
    runtime=runtime,
    conf_threshold=0.5,
    class_names={0: "person", 1: "car"},
)

async with pipeline.serve():
    detections = await pipeline(image_bytes)
    for det in detections:
        print(f"{det.class_name}: {det.confidence:.2%} at {det.box}")
```

### Instance Segmentation

```python
from inferflow.pipeline.segmentation import YOLOv5SegmentationPipeline

pipeline = YOLOv5SegmentationPipeline(
    runtime=runtime,
    conf_threshold=0.5,
    class_names={0: "person"},
)

async with pipeline.serve():
    segments = await pipeline(image_bytes)
    for seg in segments:
        print(f"Mask shape: {seg.mask.shape}, Box: {seg.box}")
```

---

## Dynamic Batching

Enable automatic request batching for higher throughput:

```python
import asyncio

from inferflow.batch.dynamic import DynamicBatchStrategy
from inferflow.pipeline.classification import ClassificationPipeline

batch_strategy = DynamicBatchStrategy(
    min_batch_size=1,
    max_batch_size=32,
    max_wait_ms=50,
)

pipeline = ClassificationPipeline(
    runtime=runtime,
    batch_strategy=batch_strategy,
)

async with pipeline.serve():
    # Submit concurrent requests - automatically batched
    results = await asyncio.gather(*[
        pipeline(img) for img in images
    ])
```

**Batch metrics:**

```python
metrics = batch_strategy.get_metrics()
print(f"Avg batch size: {metrics.avg_batch_size:.2f}")
print(f"Throughput: {metrics.total_requests / elapsed:.2f} req/s")
```

---

## Architecture

### Core Abstractions

```
┌─────────────────────────────────────────────────────┐
│                    Pipeline                         │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────┐   │
│  │ Preprocess  │─▶│   Runtime   │─▶│Postprocess │   │
│  └─────────────┘  └─────────────┘  └────────────┘   │
│         │                │                 │        │
│         └────────────────┴─────────────────┘        │
│                    BatchStrategy                    │
└─────────────────────────────────────────────────────┘
```

**Components:**

- **Runtime**: Model loading and inference execution
- **Pipeline**: End-to-end data transformation
- **BatchStrategy**: Request aggregation and dispatching
---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1
- OpenCV ≥ 4.5

---

## Contributing

Contributions are not currently accepted. This project is maintained for internal use.

---

## License

MIT License. See [LICENSE](license.md) for details.

---

## Citation

```bibtex
@software{inferflow2025,
  title={InferFlow: Universal Inference Pipeline Framework},
  author={6ixGODD},
  year={2025},
  url={https://github.com/6ixGODD/inferflow}
}
```

# InferFlow Examples

Complete examples demonstrating InferFlow's capabilities.

## Examples

### 1. Image Classification

Basic image classification with ResNet18.

```bash
cd 01_classification
python classify.py
```

### 2. Object Detection

YOLOv5 object detection with bounding boxes.

```bash
cd 02_detection
python detect. py
```

### 3. Instance Segmentation

YOLOv5 instance segmentation with masks.

```bash
cd 03_segmentation
python segment.py
```

### 4. Dynamic Batch Processing

High-throughput inference with adaptive batching.

```bash
cd 04_batch_processing
python batch_demo.py
```

### 5. Custom Workflow

Multi-stage quality control pipeline.

```bash
cd 05_custom_workflow
python workflow_demo.py
```

## Requirements

Each example has its own `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

## Features Demonstrated

- ✅ Multiple model types (ResNet, YOLOv5)
- ✅ Preprocessing & postprocessing
- ✅ Batch processing
- ✅ Workflow orchestration
- ✅ Visualization
- ✅ Performance metrics
- ✅ Error handling

## Output

All examples save results to `output/` directory with visualizations.

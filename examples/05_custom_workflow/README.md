# examples/05_custom_workflow/README.md

# Custom Workflow Example

Demonstrates building a multi-stage quality control workflow with conditional logic.

## Workflow Stages

1. **Load & Validate**:  Load image and basic validation
2. **Parallel Quality Checks**:
    - Brightness analysis
    - Resolution validation
3. **Defect Detection**:  Identify defects
4. **Conditional Segmentation**:  Segment defects (if found)
5. **Quality Classification**:  Assign quality grade
6. **Parallel Output**:
    - Generate report
    - Create visualization

## Features

- **Conditional Execution**: Tasks run based on context
- **Parallel Processing**: Independent tasks run concurrently
- **State Management**: Context object tracks all data
- **Quality Control**: Real-world inspection pipeline
- **Visualization**:  Annotated output image

## Usage

```bash
python workflow_demo.py
```

## Output

* `output/workflow_result.jpg` - Annotated image with QC results
* Console report with quality metrics

## Use Cases

* Manufacturing quality control
* Medical image analysis
* Document processing
* Multi-model inference pipelines

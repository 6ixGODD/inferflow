#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "inferflow/ops/bbox_ops.hpp"

namespace py = pybind11;

namespace inferflow {
namespace batch {
void BindBatchModule(py::module_& parent);
}
}  // namespace inferflow

/// @brief Main module entry point with hierarchical submodules.
PYBIND11_MODULE(_C, m) {
  m.doc() = "Inferflow C++ extensions for accelerated computer vision";

  // ========================================================================
  // ops submodule - Computer vision operations
  // ========================================================================
  auto ops = m.def_submodule("ops", "Computer vision operations");

  // ========================================================================
  // ops.bbox submodule - Bounding box operations
  // ========================================================================
  auto bbox = ops.def_submodule("bbox", "Bounding box operations");

  bbox.def("xywh2xyxy", &inferflow::ops::bbox::Xywh2Xyxy, py::arg("x"),
           R"doc(
Convert bounding boxes from center format to corner format.

Transforms boxes from (x_center, y_center, width, height) to
(x_min, y_min, x_max, y_max) format.

Args:
    x: Input tensor of shape (..., 4) in xywh format

Returns:
    Output tensor of shape (..., 4) in xyxy format
)doc");

  bbox.def("xyxy2xywh", &inferflow::ops::bbox::Xyxy2Xywh, py::arg("x"),
           R"doc(
Convert bounding boxes from corner format to center format.

Transforms boxes from (x_min, y_min, x_max, y_max) to
(x_center, y_center, width, height) format.

Args:
    x: Input tensor of shape (..., 4) in xyxy format

Returns:
    Output tensor of shape (..., 4) in xywh format
)doc");

  bbox.def("box_iou", &inferflow::ops::bbox::BoxIou, py::arg("box1"),
           py::arg("box2"), py::arg("eps") = 1e-7,
           R"doc(
Calculate Intersection over Union (IoU) between two sets of boxes.

Args:
    box1: First set of boxes, shape (N, 4) in xyxy format
    box2: Second set of boxes, shape (M, 4) in xyxy format
    eps: Small epsilon to avoid division by zero (default: 1e-7)

Returns:
    IoU matrix of shape (N, M)
)doc");
}

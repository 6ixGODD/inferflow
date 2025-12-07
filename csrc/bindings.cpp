#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "inferflow/nms.hpp"
#include "inferflow/bbox_ops.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
    m.doc() = "inferflow C++ extensions";

    // Bounding box operations
    m.def("xywh2xyxy", &inferflow::xywh2xyxy,
          "Convert boxes from (xc, yc, w, h) to (x1, y1, x2, y2)",
          py::arg("x"));

    m.def("xyxy2xywh", &inferflow::xyxy2xywh,
          "Convert boxes from (x1, y1, x2, y2) to (xc, yc, w, h)",
          py::arg("x"));

    m.def("box_iou", &inferflow::box_iou,
          "Calculate IoU between two sets of boxes",
          py::arg("box1"), py::arg("box2"), py::arg("eps") = 1e-7);

    // NMS
    m.def("nms", &inferflow::nms,
          "Non-Maximum Suppression for object detection",
          py::arg("prediction"),
          py::arg("conf_thres") = 0.25,
          py::arg("iou_thres") = 0.45,
          py::arg("max_det") = 300,
          py::arg("nm") = 0);
}

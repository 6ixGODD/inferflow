#ifndef INFERFLOW_OPS_BBOX_OPS_HPP_
#define INFERFLOW_OPS_BBOX_OPS_HPP_

#include <torch/extension.h>

namespace inferflow {
namespace ops {
namespace bbox {

/// @brief Converts bounding boxes from center format to corner format.
///
/// Transforms boxes from (x_center, y_center, width, height) to
/// (x_min, y_min, x_max, y_max) format.
///
/// @param x Input tensor of shape (..., 4) in xywh format
/// @return Output tensor of shape (..., 4) in xyxy format
///
/// @example
///   auto boxes_xywh = torch::tensor({{100, 100, 50, 50}});
///   auto boxes_xyxy = Xywh2Xyxy(boxes_xywh);
///   // Result: [[75, 75, 125, 125]]
torch::Tensor Xywh2Xyxy(const torch::Tensor& x);

/// @brief Converts bounding boxes from corner format to center format.
///
/// Transforms boxes from (x_min, y_min, x_max, y_max) to
/// (x_center, y_center, width, height) format.
///
/// @param x Input tensor of shape (..., 4) in xyxy format
/// @return Output tensor of shape (..., 4) in xywh format
torch::Tensor Xyxy2Xywh(const torch::Tensor& x);

/// @brief Calculates Intersection over Union (IoU) between two sets of boxes.
///
/// @param box1 First set of boxes, shape (N, 4) in xyxy format
/// @param box2 Second set of boxes, shape (M, 4) in xyxy format
/// @param eps Small epsilon to avoid division by zero (default: 1e-7)
/// @return IoU matrix of shape (N, M)
torch::Tensor BoxIou(const torch::Tensor& box1, const torch::Tensor& box2,
                     double eps = 1e-7);

}  // namespace bbox
}  // namespace ops
}  // namespace inferflow

#endif  // INFERFLOW_OPS_BBOX_OPS_HPP_

#pragma once

#include <torch/extension.h>

namespace inferflow {

// Convert bounding box format from (xc, yc, w, h) to (x1, y1, x2, y2)
torch::Tensor xywh2xyxy(const torch::Tensor& x);

// Convert bounding box format from (x1, y1, x2, y2) to (xc, yc, w, h)
torch::Tensor xyxy2xywh(const torch::Tensor& x);

// Calculate IoU between two sets of boxes
// box1: (N, 4) tensor in xyxy format
// box2: (M, 4) tensor in xyxy format
// Returns: (N, M) tensor of IoU values
torch::Tensor box_iou(const torch::Tensor& box1, const torch::Tensor& box2, double eps = 1e-7);

} // namespace inferflow

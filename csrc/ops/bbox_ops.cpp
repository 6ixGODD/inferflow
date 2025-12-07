// Copyright 2025 BoChenSHEN
// Licensed under the MIT License

#include "inferflow/ops/bbox_ops.hpp"

namespace inferflow {
namespace ops {
namespace bbox {

torch::Tensor Xywh2Xyxy(const torch::Tensor& x) {
  // Clone to avoid modifying input
  auto y = x.clone();

  // Vectorized operations
  using torch::indexing::Ellipsis;

  // x1 = xc - w/2
  y.index_put_({Ellipsis, 0},
               x.index({Ellipsis, 0}) - x.index({Ellipsis, 2}) / 2);
  // y1 = yc - h/2
  y.index_put_({Ellipsis, 1},
               x.index({Ellipsis, 1}) - x.index({Ellipsis, 3}) / 2);
  // x2 = xc + w/2
  y.index_put_({Ellipsis, 2},
               x.index({Ellipsis, 0}) + x.index({Ellipsis, 2}) / 2);
  // y2 = yc + h/2
  y.index_put_({Ellipsis, 3},
               x.index({Ellipsis, 1}) + x.index({Ellipsis, 3}) / 2);

  return y;
}

torch::Tensor Xyxy2Xywh(const torch::Tensor& x) {
  auto y = x.clone();

  using torch::indexing::Ellipsis;

  // xc = (x1 + x2) / 2
  y.index_put_({Ellipsis, 0},
               (x.index({Ellipsis, 0}) + x.index({Ellipsis, 2})) / 2);
  // yc = (y1 + y2) / 2
  y.index_put_({Ellipsis, 1},
               (x.index({Ellipsis, 1}) + x.index({Ellipsis, 3})) / 2);
  // w = x2 - x1
  y.index_put_({Ellipsis, 2}, x.index({Ellipsis, 2}) - x.index({Ellipsis, 0}));
  // h = y2 - y1
  y.index_put_({Ellipsis, 3}, x.index({Ellipsis, 3}) - x.index({Ellipsis, 1}));

  return y;
}

torch::Tensor BoxIou(const torch::Tensor& box1, const torch::Tensor& box2,
                     double eps) {
  // box1: (N, 4), box2: (M, 4) in xyxy format
  // Returns: (N, M) IoU matrix

  using torch::indexing::None;
  using torch::indexing::Slice;

  // Split coordinates
  auto a1 = box1.index({Slice(), Slice(None, 2)}).unsqueeze(1);  // (N, 1, 2)
  auto a2 = box1.index({Slice(), Slice(2, None)}).unsqueeze(1);  // (N, 1, 2)
  auto b1 = box2.index({Slice(), Slice(None, 2)}).unsqueeze(0);  // (1, M, 2)
  auto b2 = box2.index({Slice(), Slice(2, None)}).unsqueeze(0);  // (1, M, 2)

  // Intersection area
  auto inter = (torch::min(a2, b2) - torch::max(a1, b1)).clamp_min(0).prod(2);

  // Union area
  auto area1 = (a2 - a1).prod(2);
  auto area2 = (b2 - b1).prod(2);
  auto union_area = area1 + area2 - inter + eps;

  return inter / union_area;
}

}  // namespace bbox
}  // namespace ops
}  // namespace inferflow

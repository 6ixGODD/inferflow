#include "inferflow/bbox_ops.hpp"

namespace inferflow {

torch::Tensor xywh2xyxy(const torch::Tensor& x) {
    auto y = x.clone();
    y.index_put_({".. .", 0}, x.index({"...", 0}) - x.index({"...", 2}) / 2);  // x1
    y.index_put_({"...", 1}, x.index({"...", 1}) - x.index({".. .", 3}) / 2);  // y1
    y. index_put_({"...", 2}, x.index({".. .", 0}) + x.index({".. .", 2}) / 2);  // x2
    y. index_put_({"...", 3}, x.index({".. .", 1}) + x.index({"...", 3}) / 2);  // y2
    return y;
}

torch::Tensor xyxy2xywh(const torch::Tensor& x) {
    auto y = x.clone();
    y.index_put_({"...", 0}, (x.index({"...", 0}) + x.index({"...", 2})) / 2);  // xc
    y.index_put_({"...", 1}, (x.index({"...", 1}) + x.index({"...", 3})) / 2);  // yc
    y.index_put_({".. .", 2}, x.index({"...", 2}) - x.index({".. .", 0}));        // w
    y.index_put_({"...", 3}, x.index({"...", 3}) - x.index({"...", 1}));        // h
    return y;
}

torch::Tensor box_iou(const torch::Tensor& box1, const torch::Tensor& box2, double eps) {
    // box1: (N, 4), box2: (M, 4)
    auto a1 = box1.index({"...", torch::indexing::Slice(torch::indexing::None, 2)}). unsqueeze(1);  // (N, 1, 2)
    auto a2 = box1.index({"...", torch::indexing::Slice(2, torch::indexing::None)}).unsqueeze(1);  // (N, 1, 2)
    auto b1 = box2.index({"...", torch::indexing::Slice(torch::indexing::None, 2)}).unsqueeze(0);  // (1, M, 2)
    auto b2 = box2. index({"...", torch::indexing::Slice(2, torch::indexing::None)}).unsqueeze(0);  // (1, M, 2)

    // Intersection area
    auto inter = (torch::min(a2, b2) - torch::max(a1, b1)).clamp_min(0). prod(2);

    // Union area
    auto area1 = (a2 - a1).prod(2);
    auto area2 = (b2 - b1).prod(2);
    auto union_area = area1 + area2 - inter + eps;

    return inter / union_area;
}

} // namespace inferflow

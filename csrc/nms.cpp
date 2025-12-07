#include "inferflow/nms.hpp"
#include "inferflow/bbox_ops.hpp"
#include <torch/extension.h>
#include <chrono>

// Include TorchVision NMS (requires torchvision installation)
namespace vision = torch::indexing;

// Forward declaration of torchvision nms (we'll link against it)
torch::Tensor nms_kernel(const torch::Tensor& dets, const torch::Tensor& scores, double iou_threshold);

namespace inferflow {

std::vector<torch::Tensor> nms(
    const torch::Tensor& prediction,
    double conf_thres,
    double iou_thres,
    int max_det,
    int nm
) {
    const int bs = prediction.size(0);
    const int nc = prediction.size(2) - nm - 5;
    const int mi = 5 + nc;

    // Object confidence mask
    auto xc = prediction.index({"...", 4}) > conf_thres;

    // Constants
    const int max_wh = 7680;
    const int max_nms = 30000;
    const double time_limit = 0.5 + 0.05 * bs;

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<torch::Tensor> output;
    output.reserve(bs);

    for (int xi = 0; xi < bs; ++xi) {
        // Filter by confidence
        auto x = prediction[xi].index({xc[xi]});

        if (x.size(0) == 0) {
            output.push_back(torch::zeros({0, 6 + nm}, prediction.options()));
            continue;
        }

        // Multiply class confidence by objectness
        x. index_put_(
            {vision::Slice(), vision::Slice(5, vision::None)},
            x.index({vision::Slice(), vision::Slice(5, vision::None)}) *
            x.index({vision::Slice(), vision::Slice(4, 5)})
        );

        // Convert boxes from xywh to xyxy
        auto box = xywh2xyxy(x. index({vision::Slice(), vision::Slice(vision::None, 4)}));
        auto mask = x.index({vision::Slice(), vision::Slice(mi, vision::None)});

        // Get best class and confidence
        auto conf_class = x.index({vision::Slice(), vision::Slice(5, mi)}). max(1);
        auto conf = std::get<0>(conf_class);
        auto j = std::get<1>(conf_class). to(torch::kFloat32);

        // Concatenate: [xyxy, conf, class, mask_coeffs]
        x = torch::cat({box, conf. unsqueeze(1), j. unsqueeze(1), mask}, 1);

        // Filter by confidence again
        x = x.index({conf. view(-1) > conf_thres});

        int n = x.size(0);
        if (n == 0) {
            output.push_back(torch::zeros({0, 6 + nm}, prediction.options()));
            continue;
        }

        // Sort by confidence and limit to max_nms
        auto sorted_indices = x.index({vision::Slice(), 4}).argsort(/*descending=*/true);
        if (sorted_indices.size(0) > max_nms) {
            sorted_indices = sorted_indices. index({vision::Slice(vision::None, max_nms)});
        }
        x = x.index({sorted_indices});

        // Add offset to boxes by class (to perform class-agnostic NMS)
        auto c = x.index({vision::Slice(), vision::Slice(5, 6)}) * max_wh;
        auto boxes = x.index({vision::Slice(), vision::Slice(vision::None, 4)}) + c;
        auto scores = x.index({vision::Slice(), 4});

        // Call TorchVision NMS
        torch::Tensor keep;
        try {
            // Try to use torchvision C++ NMS if available
            keep = nms_kernel(boxes, scores, iou_thres);
        } catch (...) {
            // Fallback: use our own NMS implementation
            std::vector<int64_t> keep_indices;
            auto boxes_cpu = boxes.cpu();
            auto scores_cpu = scores.cpu();

            std::vector<int64_t> order(scores_cpu.size(0));
            std::iota(order. begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](int64_t i, int64_t j) {
                return scores_cpu[i]. item<float>() > scores_cpu[j].item<float>();
            });

            std::vector<bool> suppressed(order.size(), false);

            for (size_t _i = 0; _i < order.size(); ++_i) {
                auto i = order[_i];
                if (suppressed[i]) continue;

                keep_indices.push_back(i);

                auto box_i = boxes_cpu[i];
                for (size_t _j = _i + 1; _j < order.size(); ++_j) {
                    auto j = order[_j];
                    if (suppressed[j]) continue;

                    auto box_j = boxes_cpu[j];

                    // Calculate IoU
                    auto xx1 = std::max(box_i[0]. item<float>(), box_j[0].item<float>());
                    auto yy1 = std::max(box_i[1].item<float>(), box_j[1].item<float>());
                    auto xx2 = std::min(box_i[2].item<float>(), box_j[2]. item<float>());
                    auto yy2 = std::min(box_i[3]. item<float>(), box_j[3].item<float>());

                    auto w = std::max(0.0f, xx2 - xx1);
                    auto h = std::max(0.0f, yy2 - yy1);
                    auto inter = w * h;

                    auto area_i = (box_i[2] - box_i[0]). item<float>() * (box_i[3] - box_i[1]). item<float>();
                    auto area_j = (box_j[2] - box_j[0]).item<float>() * (box_j[3] - box_j[1]).item<float>();
                    auto union_area = area_i + area_j - inter;

                    if (inter / union_area > iou_thres) {
                        suppressed[j] = true;
                    }
                }
            }

            keep = torch::tensor(keep_indices, torch::kLong). to(prediction.device());
        }

        // Limit to max_det
        if (keep.size(0) > max_det) {
            keep = keep.index({vision::Slice(vision::None, max_det)});
        }

        output.push_back(x.index({keep}));

        // Check time limit
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::high_resolution_clock::now() - start_time
        ). count();

        if (elapsed > time_limit) {
            break;
        }
    }

    return output;
}

} // namespace inferflow

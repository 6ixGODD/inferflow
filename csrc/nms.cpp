#include "inferflow/nms.hpp"
#include "inferflow/bbox_ops.hpp"
#include <chrono>
#include <algorithm>
#include <numeric>

namespace inferflow {

std::vector<torch::Tensor> nms(
    const torch::Tensor& prediction,
    double conf_thres,
    double iou_thres,
    int max_det,
    int nm
) {
    using namespace torch::indexing;

    const int64_t bs = prediction.size(0);
    const int64_t nc = prediction.size(2) - nm - 5;
    const int64_t mi = 5 + nc;

    auto xc = prediction.index({"...", 4}) > conf_thres;

    const int max_wh = 7680;
    const int max_nms = 30000;
    const double time_limit = 0.5 + 0.05 * bs;

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<torch::Tensor> output;
    output.reserve(bs);

    for (int64_t xi = 0; xi < bs; ++xi) {
        auto x = prediction[xi].index({xc[xi]});

        if (x.size(0) == 0) {
            output.push_back(torch::zeros({0, 6 + nm}, prediction.options()));
            continue;
        }

        x.index_put_(
            {Slice(), Slice(5, None)},
            x.index({Slice(), Slice(5, None)}) * x.index({Slice(), Slice(4, 5)})
        );

        auto box = xywh2xyxy(x.index({Slice(), Slice(None, 4)}));
        auto mask = x.index({Slice(), Slice(mi, None)});

        auto conf_class = x.index({Slice(), Slice(5, mi)}).max(1);
        auto conf = std::get<0>(conf_class);
        auto j = std::get<1>(conf_class).to(torch::kFloat32);

        x = torch::cat({box, conf.unsqueeze(1), j.unsqueeze(1), mask}, 1);
        x = x.index({conf.view(-1) > conf_thres});

        if (x.size(0) == 0) {
            output.push_back(torch::zeros({0, 6 + nm}, prediction.options()));
            continue;
        }

        auto sorted_indices = x.index({Slice(), 4}).argsort(/*descending=*/true);
        if (sorted_indices.size(0) > max_nms) {
            sorted_indices = sorted_indices.index({Slice(None, max_nms)});
        }
        x = x.index({sorted_indices});

        auto c = x.index({Slice(), Slice(5, 6)}) * max_wh;
        auto boxes = x.index({Slice(), Slice(None, 4)}) + c;
        auto scores = x.index({Slice(), 4});

        // Simple NMS implementation
        auto boxes_cpu = boxes.cpu();
        auto scores_cpu = scores.cpu();
        int64_t num_boxes = boxes_cpu.size(0);

        std::vector<int64_t> order(num_boxes);
        std::iota(order.begin(), order.end(), 0);

        std::vector<bool> suppressed(num_boxes, false);
        std::vector<int64_t> keep_indices;

        auto boxes_acc = boxes_cpu.accessor<float, 2>();

        for (size_t _i = 0; _i < order.size(); ++_i) {
            auto i = order[_i];
            if (suppressed[i]) continue;

            keep_indices.push_back(i);

            float x1_i = boxes_acc[i][0];
            float y1_i = boxes_acc[i][1];
            float x2_i = boxes_acc[i][2];
            float y2_i = boxes_acc[i][3];
            float area_i = (x2_i - x1_i) * (y2_i - y1_i);

            for (size_t _j = _i + 1; _j < order.size(); ++_j) {
                auto j = order[_j];
                if (suppressed[j]) continue;

                float x1_j = boxes_acc[j][0];
                float y1_j = boxes_acc[j][1];
                float x2_j = boxes_acc[j][2];
                float y2_j = boxes_acc[j][3];

                float xx1 = std::max(x1_i, x1_j);
                float yy1 = std::max(y1_i, y1_j);
                float xx2 = std::min(x2_i, x2_j);
                float yy2 = std::min(y2_i, y2_j);

                float w = std::max(0.0f, xx2 - xx1);
                float h = std::max(0.0f, yy2 - yy1);
                float inter = w * h;

                float area_j = (x2_j - x1_j) * (y2_j - y1_j);
                float iou = inter / (area_i + area_j - inter);

                if (iou > iou_thres) {
                    suppressed[j] = true;
                }
            }
        }

        if (static_cast<int>(keep_indices.size()) > max_det) {
            keep_indices.resize(max_det);
        }

        auto keep = torch::tensor(keep_indices, torch::kLong).to(prediction.device());
        output.push_back(x.index({keep}));

        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::high_resolution_clock::now() - start_time
        ).count();

        if (elapsed > time_limit) break;
    }

    return output;
}

}

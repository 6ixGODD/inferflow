#pragma once

#include <torch/extension. h>
#include <vector>

namespace inferflow {

// Non-Maximum Suppression
// prediction: (batch, num_boxes, 5+nc+nm) where each row is [xc, yc, w, h, obj_conf, class_confs.. ., mask_coeffs...]
// conf_thres: confidence threshold
// iou_thres: IoU threshold for NMS
// max_det: maximum detections per image
// nm: number of mask coefficients
// Returns: vector of tensors, one per batch, each (N, 6+nm) [x1, y1, x2, y2, conf, class, mask_coeffs...]
std::vector<torch::Tensor> nms(
    const torch::Tensor& prediction,
    double conf_thres = 0.25,
    double iou_thres = 0.45,
    int max_det = 300,
    int nm = 0
);

} // namespace inferflow

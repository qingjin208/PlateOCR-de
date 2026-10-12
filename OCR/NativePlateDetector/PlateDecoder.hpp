#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace plate {
struct Box { float x, y, w, h, score; };
inline float iou(const Box &a, const Box &b) {
    float area = std::max(0.f, std::min(a.x + a.w, b.x + b.w) - std::max(a.x, b.x)) *
                 std::max(0.f, std::min(a.y + a.h, b.y + b.h) - std::max(a.y, b.y));
    float total = a.w * a.h + b.w * b.h - area;
    return total > 0 ? area / total : 0;
}
/// Matches VINRecognizeSDK's single-class [1, 5, N] / [1, N, 5] output.
inline std::vector<Box> decode(const float *data, int dim1, int dim2,
                              int width, int height, float scale, int padX, int padY) {
    if (dim1 != 5 && dim2 != 5) throw std::runtime_error("Expected a single-class YOLO tensor.");
    bool channelsFirst = dim1 == 5;
    int count = channelsFirst ? dim2 : dim1;
    std::vector<Box> proposals, kept;
    for (int i = 0; i < count; ++i) {
        auto value = [&](int c) { return data[channelsFirst ? c * count + i : i * 5 + c]; };
        float cx = value(0), cy = value(1), w = value(2), h = value(3), score = value(4);
        if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(w) ||
            !std::isfinite(h) || !std::isfinite(score) || score < .45f || w <= 0 || h <= 0) continue;
        if (std::max({std::abs(cx), std::abs(cy), std::abs(w), std::abs(h)}) <= 2.f) {
            cx *= 640; cy *= 640; w *= 640; h *= 640;
        }
        float x1 = std::clamp((cx - w / 2 - padX) / scale, 0.f, float(width));
        float y1 = std::clamp((cy - h / 2 - padY) / scale, 0.f, float(height));
        float x2 = std::clamp((cx + w / 2 - padX) / scale, 0.f, float(width));
        float y2 = std::clamp((cy + h / 2 - padY) / scale, 0.f, float(height));
        if (x2 > x1 && y2 > y1) proposals.push_back({x1 / width, y1 / height,
            (x2 - x1) / width, (y2 - y1) / height, score});
    }
    std::sort(proposals.begin(), proposals.end(), [](const Box &a, const Box &b) { return a.score > b.score; });
    for (const auto &box : proposals) {
        if (std::none_of(kept.begin(), kept.end(), [&](const Box &other) { return iou(box, other) > .45f; }))
            kept.push_back(box);
    }
    return kept;
}
}

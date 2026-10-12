#include "../NativePlateDetector/PlateDecoder.hpp"
#include <cassert>
#include <limits>
#include <iostream>

int main() {
    // 1920x1080 letterboxing: 640x360 content with 140px vertical padding.
    const float pixels[] = {320, 320, 160, 40, .9f, 322, 320, 160, 40, .8f,
                            120, 280, 80, 30, .7f, 320, 320, 160, 40, .2f};
    auto boxes = plate::decode(pixels, 4, 5, 1920, 1080, 1.f / 3, 0, 140);
    assert(boxes.size() == 2);
    assert(std::abs(boxes[0].x - .375f) < .0001f);
    assert(std::abs(boxes[0].y - 160.f / 360) < .0001f);
    assert(std::abs(boxes[0].w - .25f) < .0001f);
    assert(std::abs(boxes[0].h - 40.f / 360) < .0001f);
    float channels[20];
    for (int p = 0; p < 4; ++p)
        for (int c = 0; c < 5; ++c) channels[c * 4 + p] = pixels[p * 5 + c];
    auto transposed = plate::decode(channels, 5, 4, 1920, 1080, 1.f / 3, 0, 140);
    assert(transposed.size() == boxes.size());
    assert(std::abs(transposed[0].x - boxes[0].x) < .0001f);
    const float normalized[] = {.5f, .5f, .25f, .0625f, .9f};
    auto scaled = plate::decode(normalized, 1, 5, 1920, 1080, 1.f / 3, 0, 140);
    assert(scaled.size() == 1 && std::abs(scaled[0].y - boxes[0].y) < .0001f);
    const float clipped[] = {0, 140, 80, 80, .9f};
    auto edge = plate::decode(clipped, 1, 5, 1920, 1080, 1.f / 3, 0, 140);
    assert(edge.size() == 1 && edge[0].x == 0 && edge[0].y == 0);
    assert(std::abs(edge[0].w - .0625f) < .0001f);
    const float padding[] = {320, 50, 80, 40, .9f};
    assert(plate::decode(padding, 1, 5, 1920, 1080, 1.f / 3, 0, 140).empty());
    const float invalid[] = {320, 320, -80, 40, .9f,
                            320, 320, 80, 40, std::numeric_limits<float>::quiet_NaN()};
    assert(plate::decode(invalid, 2, 5, 640, 640, 1, 0, 0).empty());
    bool threw = false;
    try { plate::decode(pixels, 4, 6, 640, 640, 1, 0, 0); }
    catch (const std::runtime_error &) { threw = true; }
    assert(threw);
    std::cout << "PASS: multi-plate decoding, tensor layouts, normalized coordinates, clipping, NMS and invalid outputs\n";
}

#pragma once
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "paddle_api.h"
#include "utils.h"

using namespace paddle::lite_api; // NOLINT

struct YoloResult {
  float score;
  float x;
  float y;
  float w;
  float h;
};

class YoloPredictor {
public:
    YoloPredictor(const std::string &modelDir, const std::string &cpuPowerMode, const int cpuThreadNum);
    std::vector<YoloResult> Predict(cv::Mat &img,  int width, int height);
    void VisualizeResults(const std::vector<YoloResult> &results, cv::Mat *rgbaImage);
private:
    void Preprocess(const cv::Mat &img, int width, int height);
    std::vector<YoloResult> Postprocess();
    
private:
    std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor;
    
    float scale = 1.0f;
    int dw = 0;
    int dh = 0;
    int origWidth = 0;
    int origHeight = 0;
};

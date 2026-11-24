#include "yolo_process.h"     // NOLINT
#include "timer.h"           // NOLINT
#include <map>               // NOLINT
#include <memory>            // NOLINT
#include <string>            // NOLINT
#include <utility>           // NOLINT
#include <vector>            // NOLINT

void NHWC3ToNC3HW(const float *src, float *dst, const float *mean,
                  const float *std, int width, int height) {
  int size = height * width;
  float32x4_t vmean0 = vdupq_n_f32(mean ? mean[0] : 0.0f);
  float32x4_t vmean1 = vdupq_n_f32(mean ? mean[1] : 0.0f);
  float32x4_t vmean2 = vdupq_n_f32(mean ? mean[2] : 0.0f);
  float scale0 = std ? (1.0f / std[0]) : 1.0f;
  float scale1 = std ? (1.0f / std[1]) : 1.0f;
  float scale2 = std ? (1.0f / std[2]) : 1.0f;
  float32x4_t vscale0 = vdupq_n_f32(scale0);
  float32x4_t vscale1 = vdupq_n_f32(scale1);
  float32x4_t vscale2 = vdupq_n_f32(scale2);
  float *dst_c0 = dst;
  float *dst_c1 = dst + size;
  float *dst_c2 = dst + size * 2;
  int i = 0;
  for (; i < size - 3; i += 4) {
    float32x4x3_t vin3 = vld3q_f32(src);
    float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
    float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
    float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
    float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
    float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
    float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
    vst1q_f32(dst_c0, vs0);
    vst1q_f32(dst_c1, vs1);
    vst1q_f32(dst_c2, vs2);
    src += 12;
    dst_c0 += 4;
    dst_c1 += 4;
    dst_c2 += 4;
  }
  for (; i < size; i++) {
    *(dst_c0++) = (*(src++) - mean[0]) * scale0;
    *(dst_c1++) = (*(src++) - mean[1]) * scale1;
    *(dst_c2++) = (*(src++) - mean[2]) * scale2;
  }
}

int64_t ShapeProduction(const std::vector<int64_t> &shape) {
  int64_t res = 1;
  for (auto i : shape)
    res *= i;
  return res;
}

float IoU(const YoloResult& a, const YoloResult& b) {
    float xx1 = std::max(a.x, b.x);
    float yy1 = std::max(a.y, b.y);
    float xx2 = std::min(a.x + a.w, b.x + b.w);
    float yy2 = std::min(a.y + a.h, b.y + b.h);
    float inter = std::max(0.0f, xx2 - xx1) * std::max(0.0f, yy2 - yy1);
    float union_ = a.w * a.h + b.w * b.h - inter;
    return inter / union_;
}

std::vector<YoloResult> NMS(const std::vector<YoloResult>& boxes, float iou_thresh = 0.45f) {
    std::vector<YoloResult> out;
    std::vector<YoloResult> sorted = boxes;
    std::sort(sorted.begin(), sorted.end(), [](const YoloResult& a, const YoloResult& b) {
        return a.score > b.score;
    });
    std::vector<bool> removed(sorted.size(), false);
    for (size_t i = 0; i < sorted.size(); ++i) {
        if (removed[i]) continue;
        out.push_back(sorted[i]);
        for (size_t j = i + 1; j < sorted.size(); ++j) {
            if (IoU(sorted[i], sorted[j]) > iou_thresh) {
                removed[j] = true;
            }
        }
    }
    return out;
}

YoloPredictor::YoloPredictor(const std::string &modelDir, const std::string &cpuPowerMode, const int cpuThreadNum) {
  paddle::lite_api::MobileConfig config;
  config.set_model_from_file(modelDir);
  config.set_threads(cpuThreadNum);
  config.set_power_mode(ParsePowerMode(cpuPowerMode));
  predictor =
      paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(
          config);
}

void YoloPredictor::Preprocess(const cv::Mat &img_in, int inputWidth, int inputHeight) {
    origWidth = img_in.cols;
    origHeight = img_in.rows;
    
    float r = std::min(static_cast<float>(inputWidth) / origWidth,
                       static_cast<float>(inputHeight) / origHeight);
    int newUnpadW = static_cast<int>(round(origWidth * r));
    int newUnpadH = static_cast<int>(round(origHeight * r));
    
    dw = (inputWidth - newUnpadW) / 2;
    dh = (inputHeight - newUnpadH) / 2;
    scale = r;
    
    // Resize with unchanged aspect ratio
    cv::Mat resized;
    cv::resize(img_in, resized, cv::Size(newUnpadW, newUnpadH));
    
    // Pad to desired size
    cv::Mat padded(inputHeight, inputWidth, img_in.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(dw, dh, resized.cols, resized.rows)));
    
    // BGR -> RGB -> float32
    padded.convertTo(padded, CV_32FC3, 1.0 / 255.0f);
    
    // Normalize (optional, depends on model)
    std::vector<float> mean = {0.485, 0.456, 0.406};
    std::vector<float> std = {0.229, 0.224, 0.225};
    
    auto input_tensor = predictor->GetInput(0);
    std::vector<int64_t> input_shape = {1, 3, inputWidth, inputHeight};  // 替换成你的模型要求的尺寸
    input_tensor->Resize(input_shape);
    float* inputData = input_tensor->mutable_data<float>();
    NHWC3ToNC3HW(reinterpret_cast<const float *>(padded.data),
                 inputData, mean.data(), std.data(),
                 inputWidth, inputHeight);
    
//    std::string tempPath = GetDocumentsPathBridge();
//    std::string savePath = tempPath + "/preprocessed.jpg";
//    imwrite(savePath, padded * 255);
}


std::vector<YoloResult> YoloPredictor::Postprocess() {
    float scoreThreshold = 0.8f;
    std::vector<YoloResult> results;
    
    auto outputTensor = predictor->GetOutput(0);
    auto outputData = outputTensor->data<float>();
    auto outputShape = outputTensor->shape();  // [1, N, 5]
    int numProposals = outputShape[1];
    int num_anchors = outputShape[2];
    
    for (int i = 0; i < num_anchors; ++i) {
        float cx = outputData[0 * num_anchors + i];
        float cy = outputData[1 * num_anchors + i];
        float w  = outputData[2 * num_anchors + i];
        float h  = outputData[3 * num_anchors + i];
        float score = outputData[4 * num_anchors + i];
        
        if (score < scoreThreshold) continue;
        
        // 转为 x1, y1, x2, y2
        float x1 = cx - w / 2.0f;
        float y1 = cy - h / 2.0f;
        float x2 = cx + w / 2.0f;
        float y2 = cy + h / 2.0f;
        
        // 去 padding 和还原比例
        x1 = (x1 - dw) / scale;
        y1 = (y1 - dh) / scale;
        x2 = (x2 - dw) / scale;
        y2 = (y2 - dh) / scale;
        
        x1 = std::max(0.f, std::min(x1, static_cast<float>(origWidth - 1)));
        y1 = std::max(0.f, std::min(y1, static_cast<float>(origHeight - 1)));
        x2 = std::max(0.f, std::min(x2, static_cast<float>(origWidth - 1)));
        y2 = std::max(0.f, std::min(y2, static_cast<float>(origHeight - 1)));
        
        YoloResult obj;
        obj.x = x1;
        obj.y = y1;
        obj.w = x2 - x1;
        obj.h = y2 - y1;
        obj.score = score;
        
        results.push_back(obj);
    }
    
    return NMS(results, 0.45f);
}

std::vector<YoloResult>
YoloPredictor::Predict(cv::Mat &img,
                       int width,
                       int height) {
    cv::Mat srcimg;
    img.copyTo(srcimg);
    
    Preprocess(srcimg, width, height);
    
    predictor->Run();

    auto result = Postprocess();
    
    return result;
}

void YoloPredictor::VisualizeResults(const std::vector<YoloResult> &results, cv::Mat *rgbaImage) {
    for (const auto &object : results) {
        cv::Rect rect(
                      static_cast<int>(object.x),
                      static_cast<int>(object.y),
                      static_cast<int>(object.w),
                      static_cast<int>(object.h)
                      );
        
        rect &= cv::Rect(0, 0, rgbaImage->cols, rgbaImage->rows); // 防止越界
        
        //std::string label = std::to_string(static_cast<int>(object.score * 100)) + "%";
        
//        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
//        double fontScale = 0.5;
//        int thickness = 1;
        
        cv::rectangle(*rgbaImage, rect, cv::Scalar(255, 191, 0, 255), 2);
//        int baseline = 0;
//        cv::Size textSize = cv::getTextSize(label, fontFace, fontScale, thickness, &baseline);
//        cv::rectangle(*rgbaImage,
//                      cv::Point(rect.x, rect.y - textSize.height - 4),
//                      cv::Point(rect.x + textSize.width, rect.y),
//                      cv::Scalar(255, 0, 0, 255), -1);
//        cv::putText(*rgbaImage, label,
//                    cv::Point(rect.x, rect.y - 2),
//                    fontFace, fontScale,
//                    cv::Scalar(255, 255, 255, 255), thickness);
    }
//    std::string tempPath = GetDocumentsPathBridge();
//    std::string savePath = tempPath + "/preprocessed1.jpg";
//    cv::imwrite(savePath, *rgbaImage);
}


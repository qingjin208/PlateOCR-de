// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "cls_process.h"
#include "det_process.h"
#include "paddle_api.h"
#include "rec_process.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

using namespace paddle::lite_api; // NOLINT

void InitModelPath(const std::string& model_path);

struct ResultData {
    std::string stateSource;           // 州名来源（OCR/分类模型（Clas））
    std::string stateName;        // OCR 获取的州名
    double stateConfidence;       // 州名分类/OCR 的置信度/分数
    std::string licensePlate;     // OCR 获取的车牌号
    int plateArea;                // 车牌号对应的面积
};

/// One result returned to the app through the external-frame API. The text is
/// "State Plate" (e.g. "California 8ABC123"); the box is the detected plate region.
struct PlateRecognition {
  std::string text;
  float confidence;
  std::vector<std::vector<int>> box;
};

class Pipeline {
public:
  Pipeline(const std::string &detModelDir,  const std::string &clsModelDir,
           const std::string &recModelDir,  const std::string &classPath,
           const std::string &cPUPowerMode, const int cPUThreadNum,
           const std::string &config_path,  const std::string &dict_path,
           const std::string &label_path);
  
  std::tuple<cv::Mat, std::vector<std::string>> Process(cv::Mat srcimg, std::string output_img_path,
                  std::vector<std::string> &res_txt);
    
  std::tuple<std::string,double, std::string,int> OCRImages(cv::Mat img, std::string output_img_path,
                                                                        std::vector<std::string> &res_txt);

  /// External-frame entry: runs the full iOS pipeline (YOLO plate box is applied
  /// by the caller; this receives the plate ROI) and returns app-facing results.
  /// sampleCount > 1 replicates the original demo's multi-frame voting on a
  /// single still by OCRing several scale variants of the ROI and voting with
  /// the exact original getResult rules (state: Clas-first + confidence window,
  /// plate: longest-then-frequency). Camera passes 1 (already multi-frame).
  std::vector<PlateRecognition> ProcessAll(cv::Mat img, int sampleCount = 1);

private:
  std::map<std::string, double> Config_;
  std::vector<std::string> charactor_dict_;
  std::shared_ptr<ClsPredictor> clsPredictor_;
  std::shared_ptr<DetPredictor> detPredictor_;
  std::shared_ptr<RecPredictor> recPredictor_;
};

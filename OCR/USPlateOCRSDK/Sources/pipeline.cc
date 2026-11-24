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

#include "pipeline.h" // NOLINT
#include "timer.h"    // NOLINT
#include <algorithm>  // NOLINT
#include <iostream>   // NOLINT
#include <map>        // NOLINT
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>
#include "paddle_api.h"
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>



using namespace paddle::lite_api;
using namespace cv;
using namespace std;


// 全局变量声明（与ViewController中的模型共享，确保已初始化）
std::vector<float> g_mean = {0.485f, 0.456f, 0.406f};
std::vector<float> g_scale = {0.229f, 0.224f, 0.225f};
std::vector<std::string> g_labels;
std::shared_ptr<PaddlePredictor> net_mbv1;

// 计算张量（Tensor）中所有元素的平均值。
static double tensor_mean(const Tensor& tin) {
    auto shape = tin.shape();
    int64_t size = 1;
    for (int i = 0; i < shape.size(); i++) {
        size *= shape[i];
    }
    double mean = 0.;
    auto ptr = tin.data<float>();
    for (int i = 0; i < size; i++) {
        mean += ptr[i];
    }
    return mean / size;
}

// 加载标签文件
bool loadLabels(const std::string& label_path) {
    std::ifstream file(label_path);
    if (!file.is_open()) {
        std::cerr << "无法打开标签文件: " << label_path << std::endl;
        return false;
    }
    
    g_labels.clear();
    std::string line;
    while (std::getline(file, line)) {
        g_labels.push_back(line);
    }
    
    std::cout << "成功加载 " << g_labels.size() << " 个标签" << std::endl;
    return true;
}


static const std::string base64_chars =
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len) {
  std::string ret;
  int i = 0;
  int j = 0;
  unsigned char char_array_3[3];
  unsigned char char_array_4[4];

  while (in_len--) {
    char_array_3[i++] = *(bytes_to_encode++);
    if (i == 3) {
      char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
      char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
      char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
      char_array_4[3] = char_array_3[2] & 0x3f;

      for(i = 0; (i <4) ; i++)
        ret += base64_chars[char_array_4[i]];
      i = 0;
    }
  }

  if (i)
  {
    for(j = i; j < 3; j++)
      char_array_3[j] = '\0';

    char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
    char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
    char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);

    for (j = 0; (j < i + 1); j++)
      ret += base64_chars[char_array_4[j]];

    while((i++ < 3))
      ret += '=';
  }

  return ret;
}
std::string matToBase64(const cv::Mat& image, const std::string& format = ".jpg") {
    std::vector<uchar> buffer;
    std::vector<int> params;
    
    // 设置压缩参数
    if (format == ".jpg") {
        params.push_back(cv::IMWRITE_JPEG_QUALITY);
        params.push_back(90); // 质量范围 0-100，数值越大质量越高
    } else if (format == ".png") {
        params.push_back(cv::IMWRITE_PNG_COMPRESSION);
        params.push_back(3); // 压缩级别 0-9，数值越大压缩率越高
    }
    
    // 编码图像
    bool result = cv::imencode(format, image, buffer, params);
    if (!result) {
        return ""; // 编码失败
    }
    
    // 进行 Base64 编码
    return base64_encode(buffer.data(), buffer.size());
}

/**
 * 文本检测前预处理图片
 * 返回值为cv::Mat，处理后的图像
 */
cv::Mat GetRotateCropImage(cv::Mat srcimage, std::vector<std::vector<int>> box) {
    // 处理空图像或无效box（必须是4个点的四边形），返回空矩阵
    if (srcimage.empty() || box.size() != 4) {
        return cv::Mat();
    }

    // 检查box每个点的有效性（必须包含x和y两个坐标）
    for (const auto& point : box) {
        if (point.size() != 2) {
            return cv::Mat();
        }
    }

    // 提取box的四个原始点（按 左上→右上→右下→左下 顺序，与后续扩展逻辑对应）
    cv::Point2f src_pts[4] = {
        cv::Point2f(box[0][0], box[0][1]),  // 原始左上 (TL)
        cv::Point2f(box[1][0], box[1][1]),  // 原始右上 (TR)
        cv::Point2f(box[2][0], box[2][1]),  // 原始右下 (BR)
        cv::Point2f(box[3][0], box[3][1])   // 原始左下 (BL)
    };

    // ---------------------- 1. 定义扩展参数 ----------------------
    float extend_right = 0.20f;  // 右侧扩展比例（20%）
    float extend_left = 0.15f;   // 左侧扩展比例（15%）
    // 边界限制：左侧至少保留原图宽度的12%（与右侧right_margin_limit对应，确保不贴边）
    float left_margin_limit = 0.12f * srcimage.cols;
    // 右侧边界限制：右侧最多到原图宽度的88%（原有逻辑保留）
    float right_margin_limit = 0.88f * srcimage.cols;


    // ---------------------- 2. 左侧扩展计算与边界检查 ----------------------
    // 存储左侧扩展后的两个点（左上TL、左下BL）
    std::vector<std::vector<int>> left_extend_box(2, std::vector<int>(2));
    // 计算左侧扩展方向：基于 左上→右上 的反方向（向左扩展）
    // 公式：扩展后坐标 = 原始点 - 扩展比例×(右上-左上)（反方向扩展）
    left_extend_box[0][0] = box[0][0] - extend_left * (box[1][0] - box[0][0]);  // 扩展后左上x
    left_extend_box[0][1] = box[0][1] - extend_left * (box[1][1] - box[0][1]);  // 扩展后左上y
    // 计算左侧扩展方向：基于 左下→右下 的反方向（向左扩展）
    left_extend_box[1][0] = box[3][0] - extend_left * (box[2][0] - box[3][0]);  // 扩展后左下x
    left_extend_box[1][1] = box[3][1] - extend_left * (box[2][1] - box[3][1]);  // 扩展后左下y

    // 左侧边界检查：确保扩展后不小于left_margin_limit（避免超出原图左边界）
    int minimum_x_coordinate = std::min(left_extend_box[0][0], left_extend_box[1][0]);
    if (minimum_x_coordinate < left_margin_limit) {
        // 计算超出的距离和原始扩展长度，调整扩展比例
        int exceed_left_distance = left_margin_limit - minimum_x_coordinate;  // 左边界超出距离
        float original_left_extend_len = 0.0f;
        // 判断哪个点超出（左上/左下），计算原始扩展长度
        if (minimum_x_coordinate == left_extend_box[0][0]) {
            // 左上点超出：原始扩展长度 = 原始左上x - 扩展后左上x
            original_left_extend_len = box[0][0] - left_extend_box[0][0];
        } else {
            // 左下点超出：原始扩展长度 = 原始左下x - 扩展后左下x
            original_left_extend_len = box[3][0] - left_extend_box[1][0];
        }

        // 避免除零（极端情况），重新计算实际可用的扩展比例
        float adjust_left_ratio = 1.0f;
        if (original_left_extend_len > 1e-6) {  // 浮点数判空
            adjust_left_ratio = 1.0f - (exceed_left_distance / original_left_extend_len);
        }
        adjust_left_ratio = std::max(0.0f, adjust_left_ratio);  // 确保比例非负

        // 重新计算左侧扩展后的坐标（修正后不超出左边界）
        left_extend_box[0][0] = box[0][0] - extend_left * adjust_left_ratio * (box[1][0] - box[0][0]);
        left_extend_box[0][1] = box[0][1] - extend_left * adjust_left_ratio * (box[1][1] - box[0][1]);
        left_extend_box[1][0] = box[3][0] - extend_left * adjust_left_ratio * (box[2][0] - box[3][0]);
        left_extend_box[1][1] = box[3][1] - extend_left * adjust_left_ratio * (box[2][1] - box[3][1]);
    }


    // ---------------------- 3. 右侧扩展计算与边界检查（原有逻辑保留，仅格式调整） ----------------------
    std::vector<std::vector<int>> right_extend_box(2, std::vector<int>(2));
    // 右侧扩展方向：基于 左上→右上、左下→右下 的正方向（向右扩展）
    right_extend_box[0][0] = box[1][0] + extend_right * (box[1][0] - box[0][0]);  // 扩展后右上x
    right_extend_box[0][1] = box[1][1] + extend_right * (box[1][1] - box[0][1]);  // 扩展后右上y
    right_extend_box[1][0] = box[2][0] + extend_right * (box[2][0] - box[3][0]);  // 扩展后右下x
    right_extend_box[1][1] = box[2][1] + extend_right * (box[2][1] - box[3][1]);  // 扩展后右下y

    // 右侧边界检查：确保扩展后不大于right_margin_limit（避免超出原图右边界）
    int maximum_x_coordinate = std::max(right_extend_box[0][0], right_extend_box[1][0]);
    if (maximum_x_coordinate > right_margin_limit) {
        int exceed_right_distance = maximum_x_coordinate - right_margin_limit;  // 右边界超出距离
        float original_right_extend_len = 0.0f;
        // 判断哪个点超出（右上/右下），计算原始扩展长度
        if (maximum_x_coordinate == right_extend_box[0][0]) {
            original_right_extend_len = right_extend_box[0][0] - box[1][0];
        } else {
            original_right_extend_len = right_extend_box[1][0] - box[2][0];
        }

        // 避免除零，重新计算实际可用的扩展比例
        float adjust_right_ratio = 1.0f;
        if (original_right_extend_len > 1e-6) {
            adjust_right_ratio = 1.0f - (exceed_right_distance / original_right_extend_len);
        }
        adjust_right_ratio = std::max(0.0f, adjust_right_ratio);

        // 重新计算右侧扩展后的坐标（修正后不超出右边界）
        right_extend_box[0][0] = box[1][0] + extend_right * adjust_right_ratio * (box[1][0] - box[0][0]);
        right_extend_box[0][1] = box[1][1] + extend_right * adjust_right_ratio * (box[1][1] - box[0][1]);
        right_extend_box[1][0] = box[2][0] + extend_right * adjust_right_ratio * (box[2][0] - box[3][0]);
        right_extend_box[1][1] = box[2][1] + extend_right * adjust_right_ratio * (box[2][1] - box[3][1]);
    }


    // ---------------------- 4. 更新源点坐标（整合左右扩展结果） ----------------------
    src_pts[0] = cv::Point2f(left_extend_box[0][0], left_extend_box[0][1]);    // 左侧扩展后的左上
    src_pts[1] = cv::Point2f(right_extend_box[0][0], right_extend_box[0][1]);  // 右侧扩展后的右上
    src_pts[2] = cv::Point2f(right_extend_box[1][0], right_extend_box[1][1]);  // 右侧扩展后的右下
    src_pts[3] = cv::Point2f(left_extend_box[1][0], left_extend_box[1][1]);    // 左侧扩展后的左下


    // ---------------------- 5. 透视变换与竖排处理（原有逻辑保留） ----------------------
    // 计算目标矩形的宽高（取两组对边的平均长度，确保矩形规则）
    float width = (cv::norm(src_pts[0] - src_pts[1]) + cv::norm(src_pts[2] - src_pts[3])) / 2;
    float height = (cv::norm(src_pts[1] - src_pts[2]) + cv::norm(src_pts[3] - src_pts[0])) / 2;

    // 定义目标矩形的四个顶点（顺序与源点对应：左上→右上→右下→左下）
    cv::Point2f dst_pts[4] = {
        cv::Point2f(0, 0),                // 目标左上
        cv::Point2f(width, 0),            // 目标右上
        cv::Point2f(width, height),       // 目标右下
        cv::Point2f(0, height)            // 目标左下
    };

    // 计算透视变换矩阵（将扩展后的四边形校正为规则矩形）
    cv::Mat M = cv::getPerspectiveTransform(src_pts, dst_pts);

    // 执行透视变换
    cv::Mat result;
    cv::warpPerspective(
        srcimage,   // 输入原始图像
        result,     // 输出校正后的图像
        M,          // 透视变换矩阵
        cv::Size(width, height)  // 输出图像尺寸（与目标矩形一致）
    );

    // 处理竖排情况：若高度明显大于宽度（1.5倍阈值），顺时针旋转90度转为横排
    if (result.rows > result.cols * 1.5) {
        cv::rotate(result, result, cv::ROTATE_90_CLOCKWISE);
    }

    // 仅返回处理后的图像
    return result;
}


std::vector<std::string> ReadDict(std::string path) {
  std::ifstream in(path);
  std::string filename;
  std::string line;
  std::vector<std::string> m_vec;
  if (in) {
    while (getline(in, line)) {
      m_vec.push_back(line);
    }
  } else {
    std::cout << "no such file" << std::endl;
  }
  return m_vec;
}

std::vector<std::string> split(const std::string &str,
                               const std::string &delim) {
  std::vector<std::string> res;
  if ("" == str)
    return res;
  char *strs = new char[str.length() + 1];
  std::strcpy(strs, str.c_str()); // NOLINT

  char *d = new char[delim.length() + 1];
  std::strcpy(d, delim.c_str()); // NOLINT

  char *p = std::strtok(strs, d);
  while (p) {
    std::string s = p;
    res.push_back(s);
    p = std::strtok(NULL, d);
  }

  return res;
}

std::map<std::string, double> LoadConfigTxt(std::string config_path) {
  auto config = ReadDict(config_path);

  std::map<std::string, double> dict;
  for (int i = 0; i < config.size(); i++) {
    std::vector<std::string> res = split(config[i], " ");
    dict[res[0]] = stod(res[1]);
  }
  return dict;
}

cv::Mat Visualization(cv::Mat srcimg,
                      std::vector<std::vector<std::vector<int>>> boxes,
                      std::string output_image_path) {
  cv::Point rook_points[boxes.size()][4];
  for (int n = 0; n < boxes.size(); n++) {
    for (int m = 0; m < boxes[0].size(); m++) {
      rook_points[n][m] = cv::Point(static_cast<int>(boxes[n][m][0]),
                                    static_cast<int>(boxes[n][m][1]));
    }
  }
  cv::Mat img_vis;
  srcimg.copyTo(img_vis);
  for (int n = 0; n < boxes.size(); n++) {
    const cv::Point *ppt[1] = {rook_points[n]};
    int npt[] = {4};
    cv::polylines(img_vis, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, 8, 0);
  }

  cv::imwrite(output_image_path, img_vis);
  std::cout << "The detection visualized image saved in "
            << output_image_path.c_str() << std::endl;
  return img_vis;
}

Pipeline::Pipeline(const std::string &detModelDir,  const std::string &clsModelDir,
         const std::string &recModelDir,  const std::string &classPath,
         const std::string &cPUPowerMode, const int cPUThreadNum,
         const std::string &config_path,  const std::string &dict_path,
         const std::string &label_path) {
  clsPredictor_.reset(
      new ClsPredictor(clsModelDir, cPUThreadNum, cPUPowerMode));
  detPredictor_.reset(
      new DetPredictor(detModelDir, cPUThreadNum, cPUPowerMode));
  recPredictor_.reset(
      new RecPredictor(recModelDir, cPUThreadNum, cPUPowerMode));
  Config_ = LoadConfigTxt(config_path);
  charactor_dict_ = ReadDict(dict_path);
  charactor_dict_.insert(charactor_dict_.begin(), "#"); // NOLINT
  charactor_dict_.push_back(" ");
  if (!loadLabels(label_path)) {
    std::cerr << "警告：标签文件加载失败，后续分类结果可能异常！" << std::endl;
  }
    if (!label_path.empty()) {
        std::cout  << "=》标签文件加路径: " << label_path << std::endl;

    } else {
        std::cout <<"=》未找到标签文件" << label_path << std::endl;
    }
    
    std::string model_path = classPath;
  MobileConfig config;
    if (!model_path.empty()) {
        std::cout  << "=》模型路径: " << model_path << std::endl;
        config.set_model_from_file(model_path);

    } else {
        std::cout <<"=》未找到模型文件" << model_path << std::endl;
    }

  net_mbv1 = CreatePaddlePredictor<MobileConfig>(config);
}


// Calculate the edit distance between two strings
int levenshteinDistance(const std::string& s1, const std::string& s2) {
    int m = s1.length();
    int n = s2.length();
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1));

    for (int i = 0; i <= m; i++) {
        for (int j = 0; j <= n; j++) {
            if (i == 0)
                dp[i][j] = j;
            else if (j == 0)
                dp[i][j] = i;
            else if (s1[i - 1] == s2[j - 1])
                dp[i][j] = dp[i - 1][j - 1];
            else
                dp[i][j] = 1 + std::min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]});
        }
    }
    return dp[m][n];
}

// Calculate the length of the longest common substring between two strings
int longestCommonSubstring(const std::string& s1, const std::string& s2) {
    int m = s1.length();
    int n = s2.length();
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));
    int maxLen = 0;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (s1[i - 1] == s2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
                maxLen = std::max(maxLen, dp[i][j]);
            } else {
                dp[i][j] = 0;
            }
        }
    }
    return maxLen;
}

// Calculate the fuzzy matching degree of two strings
double calculateFuzzySimilarity(const std::string& s1, const std::string& s2, double editWeight = 0.4,
                               double lcsWeight = 0.3, double jaccardWeight = 0.2, double prefixWeight = 0.1) {
    // Preprocessing: convert to lowercase and remove spaces
    std::string processedS1, processedS2;
    for (char c : s1) {
        if (!std::isspace(c)) {
            processedS1 += std::tolower(c);
        }
    }
    for (char c : s2) {
        if (!std::isspace(c)) {
            processedS2 += std::tolower(c);
        }
    }

    // If either string is empty, the similarity is 0
    if (processedS1.empty() || processedS2.empty()) return 0.0;

    // 1. Calculate edit distance similarity (Levenshtein)
    int editDist = levenshteinDistance(processedS1, processedS2);
    double editSim = 1.0 - static_cast<double>(editDist) /
                      std::max(processedS1.length(), processedS2.length());

    // 2. Calculate the longest common substring similarity
    int lcsLen = longestCommonSubstring(processedS1, processedS2);
    double lcsSim = static_cast<double>(lcsLen) /
                    std::max(processedS1.length(), processedS2.length());

    // 3. Calculate Jaccard similarity (based on character set)
    std::unordered_map<char, int> set1, set2;
    for (char c : processedS1) set1[c]++;
    for (char c : processedS2) set2[c]++;

    int intersection = 0;
    for (const auto& pair : set1) {
        intersection += std::min(pair.second, set2[pair.first]);
    }
    int unionSize = processedS1.length() + processedS2.length() - intersection;
    double jaccardSim = unionSize > 0 ? static_cast<double>(intersection) / unionSize : 0.0;

    // 4. Prefix matching similarity
    int prefixLen = 0;
    while (prefixLen < processedS1.length() && prefixLen < processedS2.length() &&
           processedS1[prefixLen] == processedS2[prefixLen]) {
        prefixLen++;
    }
    double prefixSim = static_cast<double>(prefixLen) /
                       std::max(processedS1.length(), processedS2.length());

    // Calculate the final score by combining the four similarities
    return (editSim * editWeight) + (lcsSim * lcsWeight) +
           (jaccardSim * jaccardWeight) + (prefixSim * prefixWeight);
}

std::string compareStateFullName(const std::string& inputStr, double threshold = 0.3,
                                double minLengthRatio = 0.5, double coverThreshold = 0.7) {
    // Preprocess the input string, remove all spaces and convert to lowercase
    std::string processedInput;
    for (char c : inputStr) {
        if (!std::isspace(c)) {
            processedInput += std::tolower(c);
        }
    }
    
    // If the preprocessed string is empty, return an empty result directly
    if (processedInput.empty()) {
        std::cout << inputStr << "==相似度得分: 0 (Empty input)" << std::endl;
        return "";
    }
    

    std::unordered_map<std::string, std::string> fullStates;
    fullStates["alabama"] = "Alabama";
    fullStates["alaska"] = "Alaska";
    fullStates["the last frontier"] = "Alaska";
    fullStates["land of the midnight sun"] = "Alaska";
    fullStates["mainland state"] = "Alaska";

    fullStates["arizona"] = "Arizona";
    fullStates["grand canyon state"] = "Arizona";
    fullStates["copper state"] = "Arizona";
    fullStates["sunset state"] = "Arizona";

    fullStates["arkansas"] = "Arkansas";
    fullStates["land of opportunity"] = "Arkansas";
    fullStates["natural state"] = "Arkansas";
    fullStates["wonder state"] = "Arkansas";

    fullStates["california"] = "California";
    fullStates["golden state"] = "California";
    fullStates["grape state"] = "California";
    fullStates["land of milk and honey"] = "California";

    fullStates["colorado"] = "Colorado";
    fullStates["centennial state"] = "Colorado";
    fullStates["rocky mountain state"] = "Colorado";
    fullStates["mile high state"] = "Colorado";

    fullStates["connecticut"] = "Connecticut";
    fullStates["constitution state"] = "Connecticut";
    fullStates["nutmeg state"] = "Connecticut";
    fullStates["arsenal of the nation"] = "Connecticut";

    fullStates["delaware"] = "Delaware";
    fullStates["first state"] = "Delaware";
    fullStates["diamond state"] = "Delaware";
    fullStates["blue hen state"] = "Delaware";

    fullStates["florida"] = "Florida";
    fullStates["sunshine state"] = "Florida";
    fullStates["everglade state"] = "Florida";
    fullStates["orange state"] = "Florida";

    fullStates["georgia"] = "Georgia";
    fullStates["peach state"] = "Georgia";
    fullStates["empire state of the south"] = "Georgia";
    fullStates["gwinnett"] = "Georgia";
    fullStates["dekalb"] = "Georgia";
    fullStates["forsyth"] = "Georgia";

    fullStates["hawaii"] = "Hawaii";
    fullStates["aloha state"] = "Hawaii";
    fullStates["paradise of the pacific"] = "Hawaii";
    fullStates["pineapple state"] = "Hawaii";

    fullStates["idaho"] = "Idaho";
    fullStates["gem state"] = "Idaho";
    fullStates["potato state"] = "Idaho";

    fullStates["illinois"] = "Illinois";
    fullStates["prairie state"] = "Illinois";
    fullStates["land of lincoln"] = "Illinois";

    fullStates["indiana"] = "Indiana";
    fullStates["hoosier state"] = "Indiana";
    fullStates["crossroads of america"] = "Indiana";

    fullStates["iowa"] = "Iowa";
    fullStates["hawkeye state"] = "Iowa";
    fullStates["tall corn state"] = "Iowa";

    fullStates["kansas"] = "Kansas";
    fullStates["sunflower state"] = "Kansas";
    fullStates["wheat state"] = "Kansas";
    fullStates["cyclone state"] = "Kansas";

    fullStates["kentucky"] = "Kentucky";
    fullStates["bluegrass state"] = "Kentucky";

    fullStates["louisiana"] = "Louisiana";
    fullStates["pelican state"] = "Louisiana";
    fullStates["sugar state"] = "Louisiana";
    fullStates["sportsman's paradise"] = "Louisiana";

    fullStates["maine"] = "Maine";
    fullStates["pine tree state"] = "Maine";
    fullStates["vacationland"] = "Maine";

    fullStates["maryland"] = "Maryland";
    fullStates["old line state"] = "Maryland";
    fullStates["free state"] = "Maryland";
    fullStates["monumental state"] = "Maryland";

    fullStates["massachusetts"] = "Massachusetts";
    fullStates["bay state"] = "Massachusetts";
    fullStates["old colony state"] = "Massachusetts";

    fullStates["michigan"] = "Michigan";
    fullStates["wolverine state"] = "Michigan";
    fullStates["great lakes state"] = "Michigan";
    fullStates["mitten state"] = "Michigan";

    fullStates["minnesota"] = "Minnesota";
    fullStates["land of 10,000 lakes"] = "Minnesota";
    fullStates["north star state"] = "Minnesota";
    fullStates["gopher state"] = "Minnesota";

    fullStates["mississippi"] = "Mississippi";
    fullStates["magnolia state"] = "Mississippi";
    fullStates["hospitality state"] = "Mississippi";

    fullStates["missouri"] = "Missouri";
    fullStates["show - me state"] = "Missouri";
    fullStates["gateway state"] = "Missouri";

    fullStates["montana"] = "Montana";
    fullStates["treasure state"] = "Montana";
    fullStates["big sky country"] = "Montana";

    fullStates["nebraska"] = "Nebraska";
    fullStates["cornhusker state"] = "Nebraska";
    fullStates["beef state"] = "Nebraska";

    fullStates["nevada"] = "Nevada";
    fullStates["silver state"] = "Nevada";
    fullStates["battle - born state"] = "Nevada";
    fullStates["sagebrush state"] = "Nevada";

    fullStates["new hampshire"] = "New Hampshire";
    fullStates["granite state"] = "New Hampshire";
    fullStates["mother of rivers"] = "New Hampshire";
    fullStates["white mountain state"] = "New Hampshire";

    fullStates["new jersey"] = "New Jersey";
    fullStates["garden state"] = "New Jersey";

    fullStates["new mexico"] = "New Mexico";
    fullStates["land of enchantment"] = "New Mexico";
    fullStates["cactus state"] = "New Mexico";

    fullStates["new york"] = "New York";
    fullStates["empire state"] = "New York";
    fullStates["big apple state"] = "New York";

    fullStates["north carolina"] = "North Carolina";
    fullStates["old north state"] = "North Carolina";
    fullStates["tar heel state"] = "North Carolina";

    fullStates["north dakota"] = "North Dakota";
    fullStates["peace garden state"] = "North Dakota";
    fullStates["rough rider state"] = "North Dakota";
    fullStates["flickertail state"] = "North Dakota";

    fullStates["ohio"] = "Ohio";
    fullStates["buckeye state"] = "Ohio";
    fullStates["birthplace of aviation"] = "Ohio";

    fullStates["oklahoma"] = "Oklahoma";
    fullStates["sooner state"] = "Oklahoma";
    fullStates["land of the red man"] = "Oklahoma";

    fullStates["oregon"] = "Oregon";
    fullStates["beaver state"] = "Oregon";
    fullStates["volcano state"] = "Oregon";

    fullStates["pennsylvania"] = "Pennsylvania";
    fullStates["keystone state"] = "Pennsylvania";
    fullStates["quaker state"] = "Pennsylvania";

    fullStates["rhode island"] = "Rhode Island";
    fullStates["ocean state"] = "Rhode Island";
    fullStates["little rhody"] = "Rhode Island";

    fullStates["south carolina"] = "South Carolina";
    fullStates["palmetto state"] = "South Carolina";

    fullStates["south dakota"] = "South Dakota";
    fullStates["coyote state"] = "South Dakota";
    fullStates["great faces, great places"] = "South Dakota";

    fullStates["tennessee"] = "Tennessee";
    fullStates["volunteer state"] = "Tennessee";

    fullStates["texas"] = "Texas";
    fullStates["lone star state"] = "Texas";
    fullStates["cowboy state"] = "Texas";

    fullStates["utah"] = "Utah";
    fullStates["beehive state"] = "Utah";
    fullStates["mormon state"] = "Utah";

    fullStates["vermont"] = "Vermont";
    fullStates["green mountain state"] = "Vermont";
    fullStates["maple sugar state"] = "Vermont";

    fullStates["virginia"] = "Virginia";
    fullStates["old dominion"] = "Virginia";
    fullStates["mother of presidents"] = "Virginia";

    fullStates["washington"] = "Washington";
    fullStates["evergreen state"] = "Washington";
    fullStates["emerald city state"] = "Washington";

    fullStates["west virginia"] = "West Virginia";
    fullStates["mountain state"] = "West Virginia";
    fullStates["wild wonderful"] = "West Virginia";

    fullStates["wisconsin"] = "Wisconsin";
    fullStates["badger state"] = "Wisconsin";
    fullStates["america's dairyland"] = "Wisconsin";

    fullStates["wyoming"] = "Wyoming";
    fullStates["frontier state"] = "Wyoming";
    fullStates["cowboy state"] = "Wyoming";

    // Record the highest matching degree and the corresponding state name
    double maxSimilarity = 0.0;
    std::string bestMatch = "";

    // Calculate the matching degree with each state name
    for (const auto& [stateKey, stateValue] : fullStates) {
        // Basic similarity calculation
        double similarity = calculateFuzzySimilarity(processedInput, stateKey);
        
        // Length ratio (input length / state name length)
        double lengthRatio = static_cast<double>(processedInput.length()) / stateKey.length();
        
        // Penalty for excessively different lengths
        double lengthPenalty = std::max(1.0 - std::abs(lengthRatio - 1.0), 0.3);
        
        // Apply length penalty
        double adjustedSimilarity = similarity * lengthPenalty;
        
        // Special handling: appropriately lower the threshold requirement for longer state names
        if (stateKey.length() >= 8) {
            adjustedSimilarity *= 1.1; // Increase 10% weight
        }
        
        // Exclude overly short matches
        if (processedInput.length() < stateKey.length() * minLengthRatio) {
            adjustedSimilarity = 0.0;
        }
        
        // Update the best match
        if (adjustedSimilarity > maxSimilarity) {
            maxSimilarity = adjustedSimilarity;
            bestMatch = stateValue;
        }
    }

//    // Output the matching result (for debugging)
//    std::cout << inputStr << "==Matched as: " << bestMatch
//              << " Similarity score: " << maxSimilarity
//              << " (Original: " << calculateFuzzySimilarity(processedInput, bestMatch) << ")" << std::endl;

    // Only return the result when the highest matching degree exceeds the threshold
    if (maxSimilarity >= threshold) {
        return bestMatch;
    }

    return "";
}

std::string compareStateAbbrName(std::string inputStr)
{
    std::unordered_map<std::string, std::string> fullStates;
    fullStates["al"] = "Alabama";
    fullStates["ak"] = "Alaska";
    fullStates["az"] = "Arizona";
    fullStates["ar"] = "Arkansas";
    fullStates["ca"] = "California";
    fullStates["co"] = "Colorado";
    fullStates["ct"] = "Connecticut";
    fullStates["de"] = "Delaware";
    fullStates["fl"] = "Florida";
    fullStates["ga"] = "Georgia";
    fullStates["hi"] = "Hawaii";
    fullStates["id"] = "Idaho";
    fullStates["il"] = "Illinois";
    fullStates["in"] = "Indiana";
    fullStates["ia"] = "Iowa";
    fullStates["ks"] = "Kansas";
    fullStates["ky"] = "Kentucky";
    fullStates["la"] = "Louisiana";
    fullStates["me"] = "Maine";
    fullStates["md"] = "Maryland";
    fullStates["ma"] = "Massachusetts";
    fullStates["mi"] = "Michigan";
    fullStates["mn"] = "Minnesota";
    fullStates["ms"] = "Mississippi";
    fullStates["mo"] = "Missouri";
    fullStates["mt"] = "Montana";
    fullStates["ne"] = "Nebraska";
    fullStates["nv"] = "Nevada";
    fullStates["nh"] = "New Hampshire";
    fullStates["nj"] = "New Jersey";
    fullStates["nm"] = "New Mexico";
    fullStates["ny"] = "New York";
    fullStates["nc"] = "North Carolina";
    fullStates["nd"] = "North Dakota";
    fullStates["oh"] = "Ohio";
    fullStates["ok"] = "Oklahoma";
    fullStates["or"] = "Oregon";
    fullStates["pa"] = "Pennsylvania";
    fullStates["ri"] = "Rhode Island";
    fullStates["sc"] = "South Carolina";
    fullStates["sd"] = "South Dakota";
    fullStates["tn"] = "Tennessee";
    fullStates["tx"] = "Texas";
    fullStates["ut"] = "Utah";
    fullStates["vt"] = "Vermont";
    fullStates["va"] = "Virginia";
    fullStates["wa"] = "Washington";
    fullStates["wv"] = "West Virginia";
    fullStates["wi"] = "Wisconsin";
    fullStates["wy"] = "Wyoming";
    
    for (const auto& [stateKey, stateValue] : fullStates) {
        if(stateKey == inputStr)
        {
            return stateValue;
        }
    }
    return "";
}

std::string compareStateSpecialName(std::string inputStr)
{
    std::unordered_map<std::string, std::string> fullStates;
    fullStates["ark"] = "Arkansas";
    fullStates["del"] = "Delaware";
    fullStates["nj"] = "New Jersey";
    fullStates["nc"] = "North Carolina";
    fullStates["northcar"] = "North Carolina";
    fullStates["ndakota"] = "North Dakota";
    fullStates["sdak"] = "South Dakota";
    fullStates["penna"] = "Pennsylvania";
    fullStates["enna"] = "Pennsylvania";
    fullStates["wva"] = "West Virginia";
    fullStates["mass"] = "Massachusetts";
    fullStates["mas"] = "Massachusetts";

    
    
    for (const auto& [stateKey, stateValue] : fullStates) {
        if(stateKey == inputStr
           || inputStr.starts_with(stateKey)
           || inputStr.ends_with(stateKey))
        {
            return stateValue;
        }
    }
    return "";
}

std::string getSimilarStateName(std::string inputStr)
{
    inputStr = cv::toLowerCase(inputStr);
    inputStr.erase(std::remove_if(inputStr.begin(), inputStr.end(),
                               [](char c) {
                                           return c == ' ' || c == '-' || c == '.';
                                       }), inputStr.end());
    //std::cout << "inputStr:\t" << inputStr;
    std::string result = compareStateAbbrName(inputStr);
    if(result == "")
    {
        result = compareStateSpecialName(inputStr);

    }
    
    if(result == "")
    {
        result = compareStateFullName(inputStr);
    }
    return result;
}

int calculateArea(const std::vector<std::vector<int>>& coords) {
    int x1 = coords[0][0], y2 = coords[1][1], x2 = coords[1][0], y3 = coords[2][1];
    return (x2 - x1) * (y3 - y2);
}

std::string removeNonAsciiAlphaNum(const std::string& input) {
    std::string result;
    for (unsigned char c : input) {
        if (std::isalnum(c)) {
            result += c;
        }
    }
    return result;
}

// 基于 ARM Neon 指令集的图像归一化加速方法。
void neon_mean_scale(const float* din, float* dout, int size, std::vector<float> mean, std::vector<float> scale) {
    float32x4_t vmean0 = vdupq_n_f32(mean[0]);
    float32x4_t vmean1 = vdupq_n_f32(mean[1]);
    float32x4_t vmean2 = vdupq_n_f32(mean[2]);
    float32x4_t vscale0 = vdupq_n_f32(1.f / scale[0]);
    float32x4_t vscale1 = vdupq_n_f32(1.f / scale[1]);
    float32x4_t vscale2 = vdupq_n_f32(1.f / scale[2]);
    
    float* dout_c0 = dout;
    float* dout_c1 = dout + size;
    float* dout_c2 = dout + size * 2;
    
    int i = 0;
    for (; i < size - 3; i += 4) {
        float32x4x3_t vin3 = vld3q_f32(din);
        float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
        float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
        float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
        float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
        float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
        float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
        vst1q_f32(dout_c0, vs0);
        vst1q_f32(dout_c1, vs1);
        vst1q_f32(dout_c2, vs2);
        
        din += 12;
        dout_c0 += 4;
        dout_c1 += 4;
        dout_c2 += 4;
    }
    for (; i < size; i++) {
        *(dout_c0++) = (*(din++) - mean[0]) / scale[0];
        *(dout_c1++) = (*(din++) - mean[1]) / scale[1];
        *(dout_c2++) = (*(din++) - mean[2]) / scale[2];
    }
}

// 将 OpenCV 图像（cv::Mat）转换为模型输入张量（Tensor）
void fill_tensor_with_cvmat(const Mat& img_in, Tensor& tout, int width, int height, std::vector<float> mean, std::vector<float> scale) {
    if (img_in.channels() == 4) {
        cv::cvtColor(img_in, img_in, COLOR_BGRA2BGR);
    }
    cv::Mat im;
    cv::resize(img_in, im, cv::Size(width, height), 0.f, 0.f);
    cv::Mat imgf;
    im.convertTo(imgf, CV_32FC3, 1 / 255.f);
    const float* dimg = reinterpret_cast<const float*>(imgf.data);
    float* dout = tout.mutable_data<float>();
    neon_mean_scale(dimg, dout, width * height, mean, scale);
}

// 从模型输出的得分（scores）中筛选出置信度最高的topk个结果，并与标签（labels）匹配后返回字符串。
std::string print_topk(const float* scores, long class_num, long topk, const std::vector<std::string>& labels) {
    if (topk > class_num) topk = class_num;
    // 构建（分数，类别索引）对
    std::vector<std::pair<float, int>> score_idx;
    for (int i = 0; i < class_num; ++i) {
        score_idx.emplace_back(scores[i], i);
    }
    // 按分数降序排序
    std::sort(score_idx.begin(), score_idx.end(),
              [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                  return a.first > b.first;
              });
    // 返回Top1的类别名称
    if (topk >= 1 && !labels.empty() && score_idx[0].second < labels.size()) {
        return labels[score_idx[0].second];
    }
    return "";
}

std::string print_topk(const float* scores, int num_classes, int k, const std::vector<std::string>& labels) {
    if (num_classes <= 0 || k <= 0) return "";
    
    // 创建一个分数和索引的向量
    std::vector<std::pair<float, int>> score_index_pairs;
    for (int i = 0; i < num_classes; ++i) {
        score_index_pairs.push_back({scores[i], i});
    }
    
    // 按分数降序排序
    std::sort(score_index_pairs.begin(), score_index_pairs.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // 取前k个结果，并确保索引在标签范围内
    std::ostringstream oss;
    for (int i = 0; i < std::min(k, static_cast<int>(score_index_pairs.size())); ++i) {
        int label_index = score_index_pairs[i].second;
        
        // 检查标签索引是否越界
        if (label_index >= 0 && label_index < static_cast<int>(labels.size())) {
            oss << labels[label_index] << ", score: " << score_index_pairs[i].first;
            if (i < k - 1) oss << "\n";
        } else {
            oss << "未知类别(" << label_index << "), score: " << score_index_pairs[i].first;
            if (i < k - 1) oss << "\n";
        }
    }
    
    return oss.str();
}


std::tuple<std::vector<std::string>, std::vector<float>> categorizeImage(cv::Mat img) {
    // 存储前五个类别的名称和置信度
    std::vector<std::string> top5_labels;
    std::vector<float> top5_scores;
    
    std::cout << "图像数据有效性检查: " << (img.empty() ? "空图像" : "有效图像") << std::endl;
    if (img.empty()) {
        return std::make_tuple(top5_labels, top5_scores); // 空图像返回空结果
    }
    
    cv::Mat cvimg;
    if (img.channels() == 4) {
        cvtColor(img, cvimg, COLOR_RGBA2RGB);
    }
    
    // 调整图像大小为模型输入尺寸
    cv::resize(img, cvimg, cv::Size(224, 224));
    
    std::unique_ptr<Tensor> input_tensor1(net_mbv1->GetInput(0));
    input_tensor1->Resize({1, 3, 224, 224});
    fill_tensor_with_cvmat(cvimg, *(input_tensor1.get()), 224, 224, g_mean, g_scale);
    net_mbv1->Run();
    
    std::unique_ptr<const Tensor> output_tensor1(net_mbv1->GetOutput(0));
    const float* ptr = output_tensor1->data<float>();  // 使用const版本的data()
    
    // 获取并解析前五个结果（k=5）
    std::string out_classes = print_topk(ptr, 51, 5, g_labels);
    
    // 调试输出完整结果
    std::cout << "=》分类结果：\n" << out_classes << "\n" << std::endl;
    
    // 解析多行结果
    std::istringstream iss(out_classes);
    std::string line;
    
    while (std::getline(iss, line) && top5_labels.size() < 5) {
        // 解析每一行："标签名, score: 0.123"
        size_t comma_pos = line.find(",");
        size_t score_pos = line.find("score: ");
        
        if (comma_pos != std::string::npos && score_pos != std::string::npos) {
            // 提取标签
            std::string label = line.substr(0, comma_pos);
            
            // 提取分数
            float score = 0.0f;
            try {
                score = std::stof(line.substr(score_pos + 7));
            } catch (const std::exception& e) {
                std::cerr << "解析分数失败: " << e.what() << std::endl;
                continue; // 跳过当前行
            }
            
            // 添加到结果向量
            top5_labels.push_back(label);
            top5_scores.push_back(score);
        }
    }
    
    // 验证结果数量
    std::cout << "提取了 " << top5_labels.size() << " 个类别结果" << std::endl;
    
    return std::make_tuple(top5_labels, top5_scores);
}

// OCR 州名后处理
std::pair<std::string, double> postProcess(
    const std::vector<std::string>& rec_text,
    const std::vector<float>& rec_text_score,
    const std::vector<std::vector<std::vector<int>>>& rec_text_box,
    const cv::Mat& img)
{
    std::string stateNameResult;
    double maxScore = 0.0;
    
    for (size_t i = 0; i < rec_text.size(); ++i) {
        std::string currentState = getSimilarStateName(rec_text[i]);
        if (!currentState.empty()) {
            // 存储分数最高的匹配结果
            if (rec_text_score[i] > maxScore) {
                stateNameResult = currentState;
                maxScore = rec_text_score[i];
            }
        }
    }
    
    return {stateNameResult, maxScore};
}

// 处理车牌号字符串，转换为大写并过滤非字母数字字符
std::string processLicensePlate(const std::string& text) {
    std::string cleaned;
    for (char c : text) {
        // 保留字母、数字和部分合法字符（如连字符）
        if (std::isalnum(c)) {
            cleaned += std::toupper(c);
        }
    }
    
    // 移除前后空格
    size_t start = cleaned.find_first_not_of(" \t");
    size_t end = cleaned.find_last_not_of(" \t");
    if (start == std::string::npos || end == std::string::npos) {
        return "";  // 完全由空格组成
    }
    return cleaned.substr(start, end - start + 1);
}

//cv::Mat preprocessForOCR(const cv::Mat& inputMat) {
//    // 1. 检查输入有效性
//    if (inputMat.empty()) {
//        return cv::Mat();
//    }
//
//    cv::Mat grayMat;
//    // 若输入为彩色图，转为灰度图（如果已做灰度化可跳过）
//    if (inputMat.channels() == 3) {
//        cv::cvtColor(inputMat, grayMat, cv::COLOR_BGR2GRAY);
//    } else {
//        grayMat = inputMat.clone();
//    }
//
////    // 3. 轻度高斯模糊：去除高频噪声，保留字符细节（sigma=0.8避免过度模糊）
////    cv::Mat blurMat;
////    cv::GaussianBlur(grayMat, blurMat, cv::Size(3, 3), 0.8, 0.8);
////
////    // 4. 形态学"顶帽"操作：突出字符的细小凸起（Q的尾部）
////    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)); // 3x3矩形核
////    cv::Mat topHatMat;
////    cv::morphologyEx(blurMat, topHatMat, cv::MORPH_TOPHAT, kernel); // 顶帽=原图-开运算，突出亮细节
////
//////    // 5. 叠加顶帽结果到原图，增强细节对比度
////    cv::Mat resultMat;
////    cv::addWeighted(blurMat, 1.0, topHatMat, 1.2, 0, resultMat); // 顶帽权重1.2放大细节
////
//    return grayMat;
//}


// 辅助函数：输出二维向量
void print2DVector(const std::vector<std::vector<int>>& vec) {
    std::cout << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << "[";
        for (size_t j = 0; j < vec[i].size(); ++j) {
            std::cout << vec[i][j];
            if (j != vec[i].size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]";
        if (i != vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]";
}


/**
 * 针对车牌区域截图，仅加深文字颜色，保持背景不变
 * @param plateROI 输入的车牌区域截图（已通过算法裁剪，主要包含文字和底色）
 * @return 处理后文字加深的车牌图像
 */


// 对比度
cv::Mat Contrast(cv::Mat src, int percent)
{
    float alpha = percent / 100.f;
    alpha = max(-1.f, min(1.f, alpha));
    cv::Mat temp = src.clone();
    int row = src.rows;
    int col = src.cols;
    int thresh = 127;
    
    // 对比度调整
    for (int i = 0; i < row; ++i)
    {
        uchar *t = temp.ptr<uchar>(i);
        uchar *s = src.ptr<uchar>(i);
        for (int j = 0; j < col; ++j)
        {
            uchar b = s[3 * j];
            uchar g = s[3 * j + 1];
            uchar r = s[3 * j + 2];
            int newb, newg, newr;
            
            if (alpha == 1)
            {
                t[3 * j + 2] = r > thresh ? 255 : 0;
                t[3 * j + 1] = g > thresh ? 255 : 0;
                t[3 * j] = b > thresh ? 255 : 0;
                continue;
            }
            else if (alpha >= 0)
            {
                newr = static_cast<int>(thresh + (r - thresh) / (1 - alpha));
                newg = static_cast<int>(thresh + (g - thresh) / (1 - alpha));
                newb = static_cast<int>(thresh + (b - thresh) / (1 - alpha));
            }
            else
            {
                newr = static_cast<int>(thresh + (r - thresh) * (1 + alpha));
                newg = static_cast<int>(thresh + (g - thresh) * (1 + alpha));
                newb = static_cast<int>(thresh + (b - thresh) * (1 + alpha));
            }
            
            newr = max(0, min(255, newr));
            newg = max(0, min(255, newg));
            newb = max(0, min(255, newb));
            
            t[3 * j + 2] = static_cast<uchar>(newr);
            t[3 * j + 1] = static_cast<uchar>(newg);
            t[3 * j] = static_cast<uchar>(newb);
        }
    }
    
    // 白平衡处理 - 基于灰度世界假设
    // 替换原白平衡部分，用完美反射法
    if (!temp.empty() && temp.channels() == 3) {
        // 找亮度最大值（RGB 通道最大值的均值）
        double maxLum = 0;
        for (int i = 0; i < row; ++i) {
            uchar* p = temp.ptr<uchar>(i);
            for (int j = 0; j < col; ++j) {
                uchar b = p[3 * j];
                uchar g = p[3 * j + 1];
                uchar r = p[3 * j + 2];
                double lum = (r + g + b) / 3.0;
                if (lum > maxLum) {
                    maxLum = lum;
                }
            }
        }
        // 计算增益（让最亮点趋近255）
        double gain = 255.0 / maxLum;
        for (int i = 0; i < row; ++i) {
            uchar* p = temp.ptr<uchar>(i);
            for (int j = 0; j < col; ++j) {
                int b = static_cast<int>(p[3 * j] * gain);
                int g = static_cast<int>(p[3 * j + 1] * gain);
                int r = static_cast<int>(p[3 * j + 2] * gain);
                // 截断到 0-255
                p[3 * j] = static_cast<uchar>(max(0, min(255, b)));
                p[3 * j + 1] = static_cast<uchar>(max(0, min(255, g)));
                p[3 * j + 2] = static_cast<uchar>(max(0, min(255, r)));
            }
        }
    }
    
    return temp;
}

// 伽马校正函数 - 用于调整曝光
Mat adjustGamma(const Mat& src, float gamma) {
    // 防止gamma为0导致计算错误
    if (gamma <= 0) {
        gamma = 0.1f;
    }
    
    // 创建伽马校正查找表
    Mat lut(1, 256, CV_8U);
    uchar* ptr = lut.ptr();
    for (int i = 0; i < 256; ++i) {
        ptr[i] = saturate_cast<uchar>(pow((float)(i / 255.0), gamma) * 255.0f);
    }
    
    // 应用查找表进行伽马校正
    Mat dst;
    LUT(src, lut, dst);
    return dst;
}

// 动态调整亮度最低、对比度最高、稍加一些曝光、去除噪点并锐化
Mat darkenLicenseText(const Mat& src) {
    Mat img;
    src.copyTo(img);

    // Step 1: 转灰度
    Mat gray;
    if (img.channels() == 3)
        cvtColor(img, gray, COLOR_BGR2GRAY);
    else
        gray = img.clone();

    // Step 2: 计算平均亮度和对比度
    Scalar meanVal, stddevVal;
    meanStdDev(gray, meanVal, stddevVal);
    double meanBrightness = meanVal[0];
    double contrastStd = stddevVal[0];

    // Step 3: 自适应亮度压暗/提亮
    double alpha_brightness = 1.0;
    double beta_brightness = 0.0;

    if (meanBrightness > 150) {
        alpha_brightness = 0.8;
        beta_brightness = -50;
    } else if (meanBrightness < 80) {
        alpha_brightness = 1.2;
        beta_brightness = 20;
    }
    Mat adjusted;
    gray.convertTo(adjusted, -1, alpha_brightness, beta_brightness);

    // Step 4: 自适应对比度
    Mat highContrast;
    equalizeHist(adjusted, highContrast);

    double alpha_contrast = (contrastStd < 40) ? 1.5 : 1.1;
    Mat contrastAdjusted;
    highContrast.convertTo(contrastAdjusted, -1, alpha_contrast, 0);

    // Step 5: 自适应曝光
    double exposureBoost = (meanBrightness < 100) ? 15 : 5;
    Mat exposed;
    contrastAdjusted.convertTo(exposed, -1, 1.05, exposureBoost);

    // Step 6: 去噪
    Mat denoised;
    fastNlMeansDenoising(exposed, denoised, 20, 7, 21);

    // Step 7: 锐化
    Mat sharpened;
    Mat kernel = (Mat_<float>(3,3) <<
                  0, -1,  0,
                 -1,  5, -1,
                  0, -1,  0);
    filter2D(denoised, sharpened, denoised.depth(), kernel);

    // Step 8: 灰度 → BGR
    Mat outputBGR;
    cvtColor(sharpened, outputBGR, COLOR_GRAY2BGR);

    return outputBGR;
}
/**
 * 执行OCR识别的主处理流程
 * @param img 输入的待识别车牌图像
 * @param output_img_path 可视化结果的保存路径
 * @param res_txt 引用传递的识别结果文本向量
 * @return 返回一个元组，包含处理后的图像和结构化的识别结果
 */
std::tuple<std::string, double, std::string, int> Pipeline::OCRImages(cv::Mat img, std::string output_img_path,
                                                                      std::vector<std::string> &res_txt) {
    // 1. 验证图片有效性
    if (img.empty() || img.channels() != 3) {
        return {"", 0.0, "", 0};
    }

    // 从配置获取方向分类器标志
    int use_direction_classify = int(Config_["use_direction_classify"]); // NOLINT

    // 复制图像避免修改原始数据（srcimg用于检测，img_copy用于后续裁剪）
    cv::Mat srcimg, img_copy;
    img.copyTo(srcimg);
    img.copyTo(img_copy);

    // 文本检测：获取所有检测到的文本框（四边形）
    auto boxes = detPredictor_->Predict(srcimg, Config_, nullptr, nullptr, nullptr);
    if (boxes.empty()) {
        return {"", 0.0, "", 0}; // 无检测结果时返回空
    }

    // 预处理参数（后续合并box后识别会复用）
    std::vector<float> mean = {0.5f, 0.5f, 0.5f};
    std::vector<float> scale = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};

    // 存储带核心属性的识别结果
    struct RecognizedText {
        std::string text;       // OCR识别文本
        float score;            // 识别置信度
        std::vector<std::vector<int>> box; // 四边形坐标
        float area;             // 四边形面积
        float height;           // 区域高度（关键匹配条件1）
        cv::Point2f center;     // 区域中心点（关键匹配条件2：判断是否同线）
        bool is_license;        // 是否为车牌号（含主区域和分段区域）
    };
    std::vector<RecognizedText> recognizedTexts;

    // 2. 遍历所有检测框，计算面积、高度、中心点，存储基础识别结果
    for (const auto& box : boxes) { // 遍历每个检测到的四边形
        try {
            // ---------------------- 计算区域高度和中心点 ----------------------
            // 1. 先找box的边界框（用于算高度和中心点）
            int min_x = INT_MAX, max_x = INT_MIN;
            int min_y = INT_MAX, max_y = INT_MIN;
            for (const auto& p : box) {
                min_x = std::min(min_x, p[0]);
                max_x = std::max(max_x, p[0]);
                min_y = std::min(min_y, p[1]);
                max_y = std::max(max_y, p[1]);
            }
            float height = max_y - min_y; // 区域高度
            cv::Point2f center((min_x + max_x) / 2.0f, (min_y + max_y) / 2.0f); // 中心点
            // ----------------------------------------------------------------------

            // 计算四边形面积（原有逻辑保留）
            float area = 0.0f;
            int n = box.size(); // 此处应为4（四边形）
            for (int j = 0; j < n; j++) {
                int k = (j + 1) % n; // 下一个点的索引（闭合多边形）
                area += (box[j][0] * box[k][1]) - (box[k][0] * box[j][1]);
            }
            area = std::abs(area) / 2.0f; // 取绝对值并除以2，得到面积

            // 裁剪并旋转区域（原有逻辑保留）
            cv::Mat crop_img = GetRotateCropImage(img_copy, box);

//            // 转为灰度图→三通道BGR（原有逻辑保留）
//            cv::Mat gray_img;
//            cv::cvtColor(crop_img, gray_img, cv::COLOR_BGR2GRAY);
//            cv::Mat bgr_img;
//            cv::cvtColor(gray_img, bgr_img, cv::COLOR_GRAY2BGR);
            
            
            
            
            // （可选）输出车牌号区域的Base64
            std::string license_base64 = matToBase64(crop_img);
            std::cout << "crop_img的Base64: " << license_base64 << std::endl;
            

            // 高度resize到32的倍数
            int original_height = crop_img.rows;
            int target_height = (original_height / 32) * 32;
            if (target_height <= 0) target_height = 32;
            float ratio = static_cast<float>(target_height) / original_height;
            int target_width = static_cast<int>(crop_img.cols * ratio);
            if (target_width <= 0) target_width = 1;

            cv::Mat resized_img;
            cv::resize(crop_img, resized_img, cv::Size(target_width, target_height));

            // 文本增强（原有逻辑保留）
            cv::Mat enhanced_img = darkenLicenseText(resized_img);

            // 执行基础OCR识别（原有逻辑保留，用于后续州名筛选）
            auto res = recPredictor_->Predict(enhanced_img, nullptr, nullptr, nullptr, charactor_dict_);

            // 存储结果
            recognizedTexts.push_back({
                                              res.first,   // 识别文本
                                              res.second,  // 置信度
                                              box,         // 坐标
                                              area,        // 面积
                                              height,      // 区域高度
                                              center,      // 区域中心点
                                              false        // 先默认不是车牌
                                      });

            // 调试信息：输出每个区域的核心属性
            std::cout << "检测区域 - 面积: " << area << ", 高度: " << height
                      << ", 识别文本: " << res.first << std::endl;
        } catch (...) {
            continue; // 裁剪或识别失败则跳过该区域
        }
    }

    // 3. 核心逻辑：找主车牌区域→找匹配的分段区域→合并box→重新识别完整车牌
    std::string licensePlate = "";
    float total_license_area = 0.0f;
    int max_area_idx = -1;
    RecognizedText* main_license = nullptr; // 主车牌区域（面积最大）

    // 3.1 第一步：找到面积最大的主车牌区域
    for (size_t i = 0; i < recognizedTexts.size(); i++) {
        if (recognizedTexts[i].area > total_license_area) {
            total_license_area = recognizedTexts[i].area;
            max_area_idx = i;
            main_license = &recognizedTexts[i];
        }
    }

    // 3.2 第二步：找与主区域匹配的分段车牌区域（高度接近+大致同线）
    RecognizedText* sub_license = nullptr; // 分段车牌区域
    if (main_license != nullptr) {
        // 标记主区域为车牌
        main_license->is_license = true;
        // 匹配规则：
        // 1. 高度差异≤30%（可根据实际场景调整）
        float height_tolerance = 0.3f;
        float min_match_height = main_license->height * (1 - height_tolerance);
        float max_match_height = main_license->height * (1 + height_tolerance);
        // 2. 垂直方向偏差≤主区域高度的50%（判断是否“大致同线”）
        float vertical_tolerance = main_license->height * 0.5f;
        // 3. 面积≥主区域的20%（过滤过小的干扰区域）
        float area_min_ratio = 0.2f;
        float min_match_area = main_license->area * area_min_ratio;

        // 遍历所有区域，找符合条件的分段区域
        for (auto& text : recognizedTexts) {
            // 跳过主区域自身
            if (&text == main_license) continue;
            // 满足所有匹配规则
            bool is_height_match = (text.height >= min_match_height && text.height <= max_match_height);
            bool is_vertical_align = (std::abs(text.center.y - main_license->center.y) <= vertical_tolerance);
            bool is_area_valid = (text.area >= min_match_area);

            if (is_height_match && is_vertical_align && is_area_valid) {
                sub_license = &text;
                sub_license->is_license = true; // 标记分段区域为车牌
                total_license_area += text.area; // 累计车牌总面积
                std::cout << "\n找到分段车牌区域 - 面积: " << text.area
                          << ", 高度: " << text.height << ", 识别文本: " << text.text << std::endl;
                break; // 假设只有一段分段区域（美国车牌通常最多两段）
            }
        }
    }

    // 3.3 第三步：合并主区域和分段区域的box，重新裁剪+OCR识别（核心！）
    if (main_license != nullptr) {
        std::vector<std::vector<int>> merged_box; // 合并后的完整车牌box
        // 情况1：有分段区域→合并两个box的边界
        if (sub_license != nullptr) {
            // 1. 找两个box的整体边界（min_x, max_x, min_y, max_y）
            int min_x = INT_MAX, max_x = INT_MIN;
            int min_y = INT_MAX, max_y = INT_MIN;
            // 遍历主区域box的所有点
            for (const auto& p : main_license->box) {
                min_x = std::min(min_x, p[0]);
                max_x = std::max(max_x, p[0]);
                min_y = std::min(min_y, p[1]);
                max_y = std::max(max_y, p[1]);
            }
            // 遍历分段区域box的所有点，更新整体边界
            for (const auto& p : sub_license->box) {
                min_x = std::min(min_x, p[0]);
                max_x = std::max(max_x, p[0]);
                min_y = std::min(min_y, p[1]);
                max_y = std::max(max_y, p[1]);
            }
            // 2. 生成合并后的四边形box（按左上、右上、右下、左下顺序，确保是有效四边形）
            merged_box = {
                    {min_x, min_y},     // 左上
                    {max_x, min_y},     // 右上
                    {max_x, max_y},     // 右下
                    {min_x, max_y}      // 左下
            };
            std::cout << "\n合并后车牌box - 左上(" << min_x << "," << min_y
                      << "), 右下(" << max_x << "," << max_y << ")" << std::endl;
        }
            // 情况2：无分段区域→直接用主区域的box
        else {
            merged_box = main_license->box;
            std::cout << "\n无分段车牌，使用主区域box" << std::endl;
        }

        // ---------------------- 关键：用合并后的box重新走OCR流程 ----------------------
        try {
            // 1. 用合并后的box裁剪旋转区域
            cv::Mat merged_crop = GetRotateCropImage(img_copy, merged_box);
            // 2. 灰度化→三通道BGR（与原有预处理逻辑一致）
            cv::Mat merged_gray;
            cv::cvtColor(merged_crop, merged_gray, cv::COLOR_BGR2GRAY);
            cv::Mat merged_bgr;
            cv::cvtColor(merged_gray, merged_bgr, cv::COLOR_GRAY2BGR);
            // 3. 高度resize到32的倍数（与原有逻辑一致）
            int merged_original_h = merged_crop.rows;
            int merged_target_h = (merged_original_h / 32) * 32;
            if (merged_target_h <= 0) merged_target_h = 32;
            float merged_ratio = static_cast<float>(merged_target_h) / merged_original_h;
            int merged_target_w = static_cast<int>(merged_crop.cols * merged_ratio);
            if (merged_target_w <= 0) merged_target_w = 1;
            // 4. 缩放+文本增强
            cv::Mat merged_resized;
            cv::resize(merged_crop, merged_resized, cv::Size(merged_target_w, merged_target_h));
            cv::Mat merged_enhanced = darkenLicenseText(merged_resized);


            // 5. 重新执行OCR识别（得到完整车牌）
            auto merged_res = recPredictor_->Predict(merged_enhanced, nullptr, nullptr, nullptr, charactor_dict_);
            licensePlate = merged_res.first; // 最终车牌结果
            std::cout << "\n合并后车牌识别结果: " << licensePlate
                      << " (置信度: " << merged_res.second << ")" << std::endl;
        } catch (...) {
            // 若合并识别失败， fallback到主区域原始识别结果
            licensePlate = main_license->text;
            std::cout << "\n合并区域识别失败，fallback到主区域结果: " << licensePlate << std::endl;
        }
    }

    // 4. 原有逻辑：其余区域作为州名候选（排除主/分段车牌区域）
    std::vector<std::string> state_texts;
    std::vector<float> state_scores;
    std::vector<std::vector<std::vector<int>>> state_boxes;
    for (const auto& text : recognizedTexts) {
        if (!text.is_license) { // 只排除标记为车牌的区域（主+分段）
            state_texts.push_back(text.text);
            state_scores.push_back(text.score);
            state_boxes.push_back(text.box);
        }
    }

    // 5. 原有逻辑：处理州名结果
    std::string provisionalOCRStateName = "";
    double provisionalOCRStateNameScores = 0.0;
    if (!state_texts.empty()) {
        auto [name, score] = postProcess(state_texts, state_scores, state_boxes, img);
        provisionalOCRStateName = name;
        provisionalOCRStateNameScores = score;
    } else {
        provisionalOCRStateName = "";
    }

    // 输出最终结果
    std::cout << "\n最终识别结果 - 州名: " << provisionalOCRStateName
              << " (置信度: " << provisionalOCRStateNameScores << "), "
              << "车牌号: " << licensePlate << ", 车牌总面积: " << total_license_area << std::endl;

    return std::make_tuple(provisionalOCRStateName, provisionalOCRStateNameScores, licensePlate, (int)total_license_area);
}

/**
 * 程序开始识别
 */
std::tuple<cv::Mat, std::vector<std::string>> Pipeline::Process(cv::Mat img, std::string output_img_path,
                                                                std::vector<std::string> &res_txt) {
    int use_direction_classify = int(Config_["use_direction_classify"]); // NOLINT

    // 存储识别结果
    ResultData data;
    
    data.stateSource = "";
    data.stateName = "";
    data.stateConfidence = 0.0;
    data.licensePlate = "";
    data.plateArea = 0;
    
    // 1. 预处理
    // 在此进行车牌的预处理，如噪音去除、校正等（这里不进行具体实现）
    
    // 2. 车牌州名处理
    // 2.1 分类
    auto [labels, scores] = categorizeImage(img);
    
    // 使用OCR进行二次检测
    auto [provisionalOCRStateName,provisionalOCRStateNameScores, provisionalOCRLicensePlateName,provisionalOCRLincensePlateArea] = OCRImages(img,output_img_path,res_txt);
    
    // 判断分类结果的置信度
    if (scores[0] > 0.1 && scores[0]>(scores[1]*2)) {
        // 选择最优分类结果（州名）
        data.stateName = labels[0];
        data.stateConfidence = scores[0];
        data.stateSource = "Clas";
    } else {
        // 检查OCR结果是否在分类模型的前N个候选中
        bool matched = false;
        if (!provisionalOCRStateName.empty()) {
            for (int i = 0; i < (int)labels.size(); ++i) {
                if (provisionalOCRStateName == labels[i]) {
                    matched = true;
                    data.stateSource = "OCR";
                    data.stateName = provisionalOCRStateName;  // 使用OCR结果
                    data.stateConfidence = provisionalOCRStateNameScores;
                    break;
                }
            }
        }
        
        // 如果没有匹配，使用OCR结果
        if (!matched) {
            if (!provisionalOCRStateName.empty()) {
                data.stateSource = "OCR";
                data.stateName = provisionalOCRStateName;  // 使用OCR结果
                data.stateConfidence = provisionalOCRStateNameScores;
                std::cout << "警告: 分类模型置信度不足，使用OCR州名结果: " << provisionalOCRStateName << std::endl;
            } else {
                std::cout << "警告: OCR州名结果为空!" << std::endl;
            }
        }
    }
    
    // 3. 车牌号处理
    // 判断OCR车牌号是否为空，若不为空则赋值
    if(!provisionalOCRLicensePlateName.empty()){
        data.licensePlate = provisionalOCRLicensePlateName;
        data.plateArea = provisionalOCRLincensePlateArea;
    }


    // 针对Georgia州0/O/Q问题进行处理
    if (data.stateName == "Georgia") {
        // 处理所有O和o替换为0
        for (char& c : data.licensePlate) {
            if (c == 'O' || c == 'o') {
                c = '0';
            }
        }
        
        // 处理第四个字符(索引3)是Q/q的情况，替换为0
        if (data.licensePlate.length() >= 4) {  // 确保车牌长度至少为4，避免越界
            char& fourthChar = data.licensePlate[3];
            if (fourthChar == 'Q' || fourthChar == 'q') {
                fourthChar = '0';
            }
        }
    }
    
    std::vector<std::string> resultText;
    resultText.push_back(data.stateSource);
    resultText.push_back(data.stateName);
    resultText.push_back(std::to_string(data.stateConfidence));
    resultText.push_back(processLicensePlate(data.licensePlate));
    resultText.push_back(std::to_string(data.plateArea));
    
    // 正确输出vector内容的方式
    std::cout << std::endl;
    std::cout<<"---------------------------------------------"<<std::endl;
    std::cout<<"｜州名获取方式｜州名｜州名置信度｜车牌号｜车牌号面积｜"<<std::endl;
    std::cout<<"---------------------------------------------"<<std::endl;

    for (const auto& str : resultText) {
        std::cout <<"|"<< str << " "<<"|"; // 每个元素后加空格分隔
    }
    std::cout << std::endl;
    
    std::cout<<"---------------------------------------------"<<std::endl;


    return std::make_tuple(img, resultText);
}


// ---------------------------------------------------------------------------
// External-frame API adapter (app side). The caller has already localized the
// plate with YOLO and passes a plate ROI. We run the exact iOS pipeline (state
// classifier + region OCR + fusion) and map its 5-field result to PlateRecognition.
// ---------------------------------------------------------------------------
std::vector<PlateRecognition> Pipeline::ProcessAll(cv::Mat img) {
    std::vector<PlateRecognition> results;
    if (img.empty()) return results;

    std::cout << "[OCRSDK-US] pipeline input image: " << img.cols << "x" << img.rows
              << " (if this is the full frame size, plate localization failed)" << std::endl;

    std::vector<std::string> resTxt;
    cv::Mat visualized;
    std::tie(visualized, resTxt) = this->Process(img, "ocr_result.jpg", resTxt);

    // iOS Process emits: [stateSource, stateName, stateConfidence, plateNumber, plateArea]
    if (resTxt.size() < 5) return results;

    std::string state = resTxt[1];
    std::string plate = resTxt[3];
    // Strip any non alphanumeric separators the recognizer may have emitted.
    std::string cleanPlate;
    for (unsigned char c : plate) {
        if (std::isalnum(c)) cleanPlate += static_cast<char>(c);
    }
    std::string text = state.empty() ? cleanPlate : state + " " + cleanPlate;
    if (text.empty()) return results;

    double conf = 0.0;
    try { conf = std::stod(resTxt[2]); } catch (...) { conf = 0.0; }
    if (!std::isfinite(conf)) conf = 0.0;

    PlateRecognition result;
    result.text = text;
    result.confidence = static_cast<float>(std::min(1.0, std::max(0.0, conf)));
    // The box is the whole plate ROI (YOLO-localized by the caller), expressed in
    // ROI pixel coordinates; the SDK adds the ROI offset for the overlay.
    result.box = {{0, 0},
                  {static_cast<int>(img.cols), 0},
                  {static_cast<int>(img.cols), static_cast<int>(img.rows)},
                  {0, static_cast<int>(img.rows)}};
    results.push_back(result);
    return results;
}

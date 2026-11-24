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
#include <cctype>     // NOLINT
#include <cmath>      // NOLINT
#include <iostream>   // NOLINT
#include <map>        // NOLINT

cv::Mat GetRotateCropImage(cv::Mat srcimage,
                           std::vector<std::vector<int>> box) {
  cv::Mat image;
  srcimage.copyTo(image);
  std::vector<std::vector<int>> points = box;

  int x_collect[4] = {box[0][0], box[1][0], box[2][0], box[3][0]};
  int y_collect[4] = {box[0][1], box[1][1], box[2][1], box[3][1]};
  int left = int(*std::min_element(x_collect, x_collect + 4));   // NOLINT
  int right = int(*std::max_element(x_collect, x_collect + 4));  // NOLINT
  int top = int(*std::min_element(y_collect, y_collect + 4));    // NOLINT
  int bottom = int(*std::max_element(y_collect, y_collect + 4)); // NOLINT

  cv::Mat img_crop;
  image(cv::Rect(left, top, right - left, bottom - top)).copyTo(img_crop);

  for (int i = 0; i < points.size(); i++) {
    points[i][0] -= left;
    points[i][1] -= top;
  }

  int img_crop_width =
      static_cast<int>(sqrt(pow(points[0][0] - points[1][0], 2) +
                            pow(points[0][1] - points[1][1], 2)));
  int img_crop_height =
      static_cast<int>(sqrt(pow(points[0][0] - points[3][0], 2) +
                            pow(points[0][1] - points[3][1], 2)));

  cv::Point2f pts_std[4];
  pts_std[0] = cv::Point2f(0., 0.);
  pts_std[1] = cv::Point2f(img_crop_width, 0.);
  pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
  pts_std[3] = cv::Point2f(0.f, img_crop_height);

  cv::Point2f pointsf[4];
  pointsf[0] = cv::Point2f(points[0][0], points[0][1]);
  pointsf[1] = cv::Point2f(points[1][0], points[1][1]);
  pointsf[2] = cv::Point2f(points[2][0], points[2][1]);
  pointsf[3] = cv::Point2f(points[3][0], points[3][1]);

  cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);

  cv::Mat dst_img;
  cv::warpPerspective(img_crop, dst_img, M,
                      cv::Size(img_crop_width, img_crop_height),
                      cv::BORDER_REPLICATE);

  const float ratio = 1.5;
  if (static_cast<float>(dst_img.rows) >=
      static_cast<float>(dst_img.cols) * ratio) {
    cv::Mat srcCopy = cv::Mat(dst_img.rows, dst_img.cols, dst_img.depth());
    cv::transpose(dst_img, srcCopy);
    cv::flip(srcCopy, srcCopy, 0);
    return srcCopy;
  } else {
    return dst_img;
  }
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

Pipeline::Pipeline(const std::string &detModelDir,
                   const std::string &clsModelDir,
                   const std::string &recModelDir,
                   const std::string &cPUPowerMode, const int cPUThreadNum,
                   const std::string &config_path,
                   const std::string &dict_path) {
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
}

int levenshteinDistance(const std::string &str1, const std::string &str2) {
    long len1 = str1.length();
    long len2 = str2.length();
    std::vector<std::vector<int>> dp(len1 + 1, std::vector<int>(len2 + 1));

    for (int i = 0; i <= len1; i++) {
        for (int j = 0; j <= len2; j++) {
            if (i == 0) dp[i][j] = j;
            else if (j == 0) dp[i][j] = i;
            else if (str1[i - 1] == str2[j - 1]) dp[i][j] = dp[i - 1][j - 1];
            else dp[i][j] = std::min({dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]}) + 1;
        }
    }
    return dp[len1][len2];
}

bool hasCommonSubstring(const std::string& s1, const std::string& s2, int minLen = 4) {
    int len1 = s1.length();
    int len2 = s2.length();

    for (int l = minLen; l <= std::min(len1, len2); ++l) { // l 是子串长度
        for (int i = 0; i <= len1 - l; ++i) {
            std::string sub = s1.substr(i, l);
            if (s2.find(sub) != std::string::npos) {
                return true;
            }
        }
    }
    return false;
}

std::string compareStateFullName(std::string inputStr)
{
    int threshold = 2;
    std::unordered_map<std::string, std::string> fullStates;
    fullStates["alabama"] = "Alabama";
    fullStates["alaska"] = "Alaska";
    fullStates["arizona"] = "Arizona";
    fullStates["arkansas"] = "Arkansas";
    fullStates["california"] = "California";
    fullStates["colorado"] = "Colorado";
    fullStates["connecticut"] = "Connecticut";
    fullStates["delaware"] = "Delaware";
    fullStates["florida"] = "Florida";
    fullStates["georgia"] = "Georgia";
    fullStates["hawaii"] = "Hawaii";
    fullStates["idaho"] = "Idaho";
    fullStates["illinois"] = "Illinois";
    fullStates["indiana"] = "Indiana";
    fullStates["iowa"] = "Iowa";
    fullStates["kansas"] = "Kansas";
    fullStates["kentucky"] = "Kentucky";
    fullStates["louisiana"] = "Louisiana";
    fullStates["maine"] = "Maine";
    fullStates["maryland"] = "Maryland";
    fullStates["massachusetts"] = "Massachusetts";
    fullStates["michigan"] = "Michigan";
    fullStates["minnesota"] = "Minnesota";
    fullStates["mississippi"] = "Mississippi";
    fullStates["missouri"] = "Missouri";
    fullStates["montana"] = "Montana";
    fullStates["nebraska"] = "Nebraska";
    fullStates["nevada"] = "Nevada";
    fullStates["newhampshire"] = "New Hampshire";
    fullStates["newjersey"] = "New Jersey";
    fullStates["newmexico"] = "New Mexico";
    fullStates["newyork"] = "New York";
    fullStates["northcarolina"] = "North Carolina";
    fullStates["northdakota"] = "North Dakota";
    fullStates["ohio"] = "Ohio";
    fullStates["oklahoma"] = "Oklahoma";
    fullStates["oregon"] = "Oregon";
    fullStates["pennsylvania"] = "Pennsylvania";
    fullStates["rhodeisland"] = "Rhode Island";
    fullStates["southcarolina"] = "South Carolina";
    fullStates["southdakota"] = "South Dakota";
    fullStates["tennessee"] = "Tennessee";
    fullStates["texas"] = "Texas";
    fullStates["utah"] = "Utah";
    fullStates["vermont"] = "Vermont";
    fullStates["virginia"] = "Virginia";
    fullStates["washington"] = "Washington";
    fullStates["westvirginia"] = "West Virginia";
    fullStates["wisconsin"] = "Wisconsin";
    fullStates["wyoming"] = "Wyoming";
    
    for (const auto& [stateKey, stateValue] : fullStates) {
        if(stateKey == inputStr)
        {
            return stateValue;
        }
        
        if(stateKey != "virginia")
        {
            if(inputStr.starts_with(stateKey))
            {
                return stateValue;
            }
            
            if (inputStr.ends_with(stateKey))
            {
                return stateValue;
            }
            
            int minSimilarChacter = 4;
            if(stateKey == "maryland" || stateKey == "alaska" || stateKey == "nebraska" || stateKey == "louisiana")
            {
                minSimilarChacter = 5;
            }
            else if (stateKey == "northcarolina" || stateKey == "southcarolina")
            {
                minSimilarChacter = 11;
            }
            else if (stateKey == "northdakota" || stateKey == "southdakota")
            {
                minSimilarChacter = 9;
            }
            else if (stateKey == "westvirginia")
            {
                minSimilarChacter = 9;
            }
            
            if (hasCommonSubstring(stateKey, inputStr, minSimilarChacter))
            {
                return stateValue;
            }
        }
        
        if(stateKey != "northcarolina" && stateKey != "southcarolina"
           && stateKey != "northdakota" && stateKey != "southdakota")
        {
            int dist = levenshteinDistance(inputStr, stateKey);
            if (dist <= threshold) {
                return stateValue;
            }
        }
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
    std::string result = compareStateFullName(inputStr);
    if(result == "")
    {
        result = compareStateAbbrName(inputStr);
    }
    
    if(result == "")
    {
        result = compareStateSpecialName(inputStr);
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

namespace {

struct USPlateRegion {
    size_t index;
    std::string raw;
    std::string alnumLower;
    std::string plateUpper;
    std::string state;
    float score;
    cv::Rect rect;
    bool plateLike;
};

std::string lowerAlnum(const std::string& input) {
    std::string result;
    for (unsigned char c : input) {
        if (std::isalnum(c)) result += (char)std::tolower(c);
    }
    return result;
}

std::string upperAlnum(const std::string& input) {
    std::string result;
    for (unsigned char c : input) {
        if (std::isalnum(c)) result += (char)std::toupper(c);
    }
    return result;
}

bool allAlpha(const std::string& s) {
    if (s.empty()) return false;
    for (unsigned char c : s) if (!std::isalpha(c)) return false;
    return true;
}

// US plates vary by state, but every readable candidate must contain both a
// letter and a digit. Pure words (state names, signs, labels) are rejected.
bool looksLikePlateNumber(const std::string& s) {
    if (s.size() < 3 || s.size() > 8) return false;
    int letters = 0, digits = 0;
    for (unsigned char c : s) {
        if (std::isalpha(c)) ++letters;
        else if (std::isdigit(c)) ++digits;
        else return false;
    }
    return letters > 0 && digits > 0 && (letters >= 2 || digits >= 3);
}

const std::pair<const char*, const char*>* usStateNames() {
    static const std::pair<const char*, const char*> kStates[] = {
        {"alabama","Alabama"}, {"alaska","Alaska"}, {"arizona","Arizona"},
        {"arkansas","Arkansas"}, {"california","California"}, {"colorado","Colorado"},
        {"connecticut","Connecticut"}, {"delaware","Delaware"}, {"florida","Florida"},
        {"georgia","Georgia"}, {"hawaii","Hawaii"}, {"idaho","Idaho"},
        {"illinois","Illinois"}, {"indiana","Indiana"}, {"iowa","Iowa"},
        {"kansas","Kansas"}, {"kentucky","Kentucky"}, {"louisiana","Louisiana"},
        {"maine","Maine"}, {"maryland","Maryland"}, {"massachusetts","Massachusetts"},
        {"michigan","Michigan"}, {"minnesota","Minnesota"}, {"mississippi","Mississippi"},
        {"missouri","Missouri"}, {"montana","Montana"}, {"nebraska","Nebraska"},
        {"nevada","Nevada"}, {"newhampshire","New Hampshire"}, {"newjersey","New Jersey"},
        {"newmexico","New Mexico"}, {"newyork","New York"}, {"northcarolina","North Carolina"},
        {"northdakota","North Dakota"}, {"ohio","Ohio"}, {"oklahoma","Oklahoma"},
        {"oregon","Oregon"}, {"pennsylvania","Pennsylvania"}, {"rhodeisland","Rhode Island"},
        {"southcarolina","South Carolina"}, {"southdakota","South Dakota"},
        {"tennessee","Tennessee"}, {"texas","Texas"}, {"utah","Utah"},
        {"vermont","Vermont"}, {"virginia","Virginia"}, {"washington","Washington"},
        {"westvirginia","West Virginia"}, {"wisconsin","Wisconsin"}, {"wyoming","Wyoming"}
    };
    return kStates;
}
constexpr size_t kUSStateCount = 50;

const std::pair<const char*, const char*>* usStateAbbrs() {
    static const std::pair<const char*, const char*> kAbbrs[] = {
        {"al","Alabama"}, {"ak","Alaska"}, {"az","Arizona"}, {"ar","Arkansas"},
        {"ca","California"}, {"co","Colorado"}, {"ct","Connecticut"}, {"de","Delaware"},
        {"fl","Florida"}, {"ga","Georgia"}, {"hi","Hawaii"}, {"id","Idaho"},
        {"il","Illinois"}, {"in","Indiana"}, {"ia","Iowa"}, {"ks","Kansas"},
        {"ky","Kentucky"}, {"la","Louisiana"}, {"me","Maine"}, {"md","Maryland"},
        {"ma","Massachusetts"}, {"mi","Michigan"}, {"mn","Minnesota"},
        {"ms","Mississippi"}, {"mo","Missouri"}, {"mt","Montana"}, {"ne","Nebraska"},
        {"nv","Nevada"}, {"nh","New Hampshire"}, {"nj","New Jersey"},
        {"nm","New Mexico"}, {"ny","New York"}, {"nc","North Carolina"},
        {"nd","North Dakota"}, {"oh","Ohio"}, {"ok","Oklahoma"}, {"or","Oregon"},
        {"pa","Pennsylvania"}, {"ri","Rhode Island"}, {"sc","South Carolina"},
        {"sd","South Dakota"}, {"tn","Tennessee"}, {"tx","Texas"}, {"ut","Utah"},
        {"vt","Vermont"}, {"va","Virginia"}, {"wa","Washington"},
        {"wv","West Virginia"}, {"wi","Wisconsin"}, {"wy","Wyoming"}
    };
    return kAbbrs;
}

// Strict state decision: exact full name / exact abbreviation, plus a very
// conservative typo tolerance. The old prefix/suffix/common-substring rules
// matched arbitrary plate numbers and background words.
std::string strictStateName(const std::string& s) {
    if (!allAlpha(s)) return "";
    const auto* states = usStateNames();
    for (size_t i = 0; i < kUSStateCount; ++i) {
        const std::string key(states[i].first);
        if (s == key) return states[i].second;
        if (s.size() >= 8 && levenshteinDistance(s, key) <= 2) return states[i].second;
        if (s.size() >= 5 && levenshteinDistance(s, key) <= 1) return states[i].second;
    }
    const auto* abbrs = usStateAbbrs();
    for (size_t i = 0; i < kUSStateCount; ++i) {
        if (s == abbrs[i].first) return abbrs[i].second;
    }
    return "";
}

// Handles the common case where detection merges the state title and the plate
// number into one OCR region, e.g. "CALIFORNIA8ABC123".
bool splitMergedStatePlate(const std::string& s, std::string& state, std::string& plate) {
    const auto* states = usStateNames();
    size_t bestLength = 0;
    for (size_t i = 0; i < kUSStateCount; ++i) {
        const std::string key(states[i].first);
        if (s.size() <= key.size() || !s.starts_with(key)) continue;
        const std::string rest = s.substr(key.size());
        if (looksLikePlateNumber(rest) && key.size() > bestLength) {
            bestLength = key.size();
            state = states[i].second;
            plate = upperAlnum(rest);
        }
    }
    if (bestLength > 0) return true;

    const auto* abbrs = usStateAbbrs();
    for (size_t i = 0; i < kUSStateCount; ++i) {
        const std::string key(abbrs[i].first);
        if (s.size() <= key.size() || !s.starts_with(key)) continue;
        const std::string rest = s.substr(key.size());
        if (looksLikePlateNumber(rest)) {
            state = abbrs[i].second;
            plate = upperAlnum(rest);
            return true;
        }
    }
    return false;
}

cv::Rect rectFromPlateBox(const std::vector<std::vector<int>>& box) {
    int minX = INT_MAX, minY = INT_MAX, maxX = INT_MIN, maxY = INT_MIN;
    for (const auto& point : box) {
        if (point.size() < 2) continue;
        minX = std::min(minX, point[0]);
        minY = std::min(minY, point[1]);
        maxX = std::max(maxX, point[0]);
        maxY = std::max(maxY, point[1]);
    }
    if (minX == INT_MAX) return cv::Rect();
    return cv::Rect(minX, minY, maxX - minX, maxY - minY);
}

std::vector<std::vector<int>> boxFromRect(const cv::Rect& r) {
    return {{r.x, r.y}, {r.x + r.width, r.y},
            {r.x + r.width, r.y + r.height}, {r.x, r.y + r.height}};
}

// The state title sits ABOVE the plate number and their horizontal spans
// overlap. Without this check a state word elsewhere in the frame can be
// paired with an unrelated plate-like region.
bool stateAbovePlate(const cv::Rect& state, const cv::Rect& plate) {
    if (state.empty() || plate.empty()) return false;
    const double stateCenterY = state.y + state.height / 2.0;
    const double plateCenterY = plate.y + plate.height / 2.0;
    if (plateCenterY <= stateCenterY) return false;
    const double overlap = std::min(state.x + state.width, plate.x + plate.width)
                         - std::max(state.x, plate.x);
    if (overlap < 0.25 * std::min(state.width, plate.width)) return false;
    const double gap = plate.y - (state.y + state.height);
    return gap <= 3.0 * std::max(state.height, plate.height);
}

}  // namespace

std::vector<std::string> postProcess(std::vector<std::string> rec_text,std::vector<float> rec_text_score,std::vector<std::vector<std::vector<int>>> rec_text_box)
{
    std::vector<std::string> resultVector;
    for (int i = 0; i < rec_text.size(); i++) {
        std::string result = getSimilarStateName(rec_text[i]);
        if(result != "")
        {
            resultVector.push_back(result);
            int maxArea = INT_MIN;
            int maxRegionIndex = -1;
            for (int m = 0; m < rec_text_box.size(); m++) {
                for (int n = 0; n < rec_text_box[m].size(); n++) {
                    int area = calculateArea(rec_text_box[m]);
                    if (area > maxArea) {
                        maxArea = area;
                        maxRegionIndex = m; // 保存最大区域的索引
                    }
                }
            }
            
            if(maxRegionIndex != -1)
            {
                std::string tempCarNumber = removeNonAsciiAlphaNum(rec_text[maxRegionIndex].c_str());
                if(getSimilarStateName(tempCarNumber) != result)
                {
                    resultVector.push_back(tempCarNumber);
                }
            }
        }
    }
    
    return  resultVector;
}

std::tuple<cv::Mat, std::vector<std::string>> Pipeline::Process(cv::Mat img, std::string output_img_path,
                                                                std::vector<std::string> &res_txt) {
    //  Timer tic;
    //  tic.start();
    int use_direction_classify = int(Config_["use_direction_classify"]); // NOLINT
    cv::Mat srcimg;
    img.copyTo(srcimg);
    // det predict
    auto boxes =
    detPredictor_->Predict(srcimg, Config_, nullptr, nullptr, nullptr);
    
    std::vector<float> mean = {0.5f, 0.5f, 0.5f};
    std::vector<float> scale = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
    
    cv::Mat img_copy;
    img.copyTo(img_copy);
    cv::Mat crop_img;
    
    std::vector<std::string> rec_text;
    std::vector<float> rec_text_score;
    std::vector<std::vector<std::vector<int>>> rec_text_box;
    //printf("box = %d\n", boxes.size() - 1);
    for (int i = boxes.size() - 1; i >= 0; i--) {
        try
        {
            crop_img = GetRotateCropImage(img_copy, boxes[i]);
        }
        catch(...)
        {
            continue;
        }
        
        if (use_direction_classify >= 1) {
            try {
                crop_img = clsPredictor_->Predict(crop_img, nullptr, nullptr, nullptr, 0.9);
            } catch (...) {
            }
        }
        auto res = recPredictor_->Predict(crop_img, nullptr, nullptr, nullptr,
                                          charactor_dict_);
        //printf("result:%s\n",res.first.c_str());
        rec_text.push_back(res.first);
        rec_text_score.push_back(res.second);
        rec_text_box.push_back(boxes[i]);
    }
    // tic.end();
    // *processTime = tic.get_average_ms();
    // std::cout << "pipeline predict costs" <<  *processTime;
    
    //// visualization comment out for not need green line(allen)
    //auto img_vis = Visualization(img, boxes, output_img_path);
    
    // print recognized text
    //  res_txt.resize(rec_text.size() * 2);
    //  for (int i = 0; i < rec_text.size(); i++) {
    //        std::cout << i << "\t" << rec_text[i] << "\t" <x< rec_text_score[i]
    //                  << std::endl;
    //    res_txt[2 * i] = rec_text[i];
    //    res_txt[2 * i + 1] = rec_text_score[i];
    //  }
    std::vector<std::string> resultVector = postProcess(rec_text,rec_text_score,rec_text_box);
    //return std::make_tuple(img_vis, resultVector);
    return std::make_tuple(img, resultVector);
}

std::vector<PlateRecognition> Pipeline::ProcessAll(cv::Mat img) {
    std::vector<PlateRecognition> results;
    int use_direction_classify = int(Config_["use_direction_classify"]);

    cv::Mat srcimg;
    img.copyTo(srcimg);
    auto boxes = detPredictor_->Predict(srcimg, Config_, nullptr, nullptr, nullptr);

    cv::Mat img_copy;
    img.copyTo(img_copy);

    std::vector<std::string> rec_text;
    std::vector<float> rec_text_score;
    std::vector<std::vector<std::vector<int>>> rec_text_box;

    for (int i = (int)boxes.size() - 1; i >= 0; i--) {
        try {
            cv::Mat crop_img = GetRotateCropImage(img_copy, boxes[i]);

            if (use_direction_classify >= 1) {
                try {
                    crop_img = clsPredictor_->Predict(crop_img, nullptr, nullptr, nullptr, 0.9);
                } catch (...) {
                    // Keep the unclassified crop when direction classification fails.
                }
            }

            auto rec = recPredictor_->Predict(crop_img, nullptr, nullptr, nullptr,
                                              charactor_dict_);
            rec_text.push_back(rec.first);
            rec_text_score.push_back(rec.second);
            rec_text_box.push_back(boxes[i]);
        } catch (...) {
            // A malformed region must not discard the rest of the frame.
            continue;
        }
    }

    std::vector<USPlateRegion> regions;
    regions.reserve(rec_text.size());
    for (size_t i = 0; i < rec_text.size(); ++i) {
        USPlateRegion region;
        region.index = i;
        region.raw = rec_text[i];
        region.alnumLower = lowerAlnum(region.raw);
        region.plateUpper = upperAlnum(region.raw);
        const float score = (i < rec_text_score.size()) ? rec_text_score[i] : 0.0f;
        region.score = std::isfinite(score) ? score : 0.0f;
        region.rect = (i < rec_text_box.size()) ? rectFromPlateBox(rec_text_box[i]) : cv::Rect();
        region.state = strictStateName(region.alnumLower);
        region.plateLike = region.state.empty() && looksLikePlateNumber(region.alnumLower);
        regions.push_back(region);
    }

    // Case 1: the detector merged "STATE" and the number into one region.
    for (const auto& region : regions) {
        std::string state, plate;
        if (!splitMergedStatePlate(region.alnumLower, state, plate)) continue;
        PlateRecognition result;
        result.text = state + " " + plate;
        result.confidence = region.score;
        result.box = boxFromRect(region.rect);
        results.push_back(result);
        return results;
    }

    // Case 2: state title and plate number are separate regions. Require the
    // state strip to sit above the number with horizontal overlap.
    int bestState = -1, bestPlate = -1;
    float bestScore = -1.0f;
    for (size_t s = 0; s < regions.size(); ++s) {
        if (regions[s].state.empty()) continue;
        for (size_t p = 0; p < regions.size(); ++p) {
            if (s == p || !regions[p].plateLike) continue;
            if (!stateAbovePlate(regions[s].rect, regions[p].rect)) continue;
            const float pairScore = std::min(regions[s].score, regions[p].score);
            if (pairScore > bestScore) {
                bestScore = pairScore;
                bestState = (int)s;
                bestPlate = (int)p;
            }
        }
    }
    if (bestState >= 0 && bestPlate >= 0) {
        PlateRecognition combined;
        combined.text = regions[bestState].state + " " + regions[bestPlate].plateUpper;
        combined.confidence = bestScore;
        cv::Rect unionRect = regions[bestState].rect | regions[bestPlate].rect;
        combined.box = boxFromRect(unionRect);
        results.push_back(combined);
        return results;
    }

    // Case 3: no trustworthy state reading. Return only plate-shaped candidates,
    // never arbitrary OCR strings. Better no result than a fabricated one.
    std::vector<size_t> order;
    for (size_t i = 0; i < regions.size(); ++i) {
        if (regions[i].plateLike) order.push_back(i);
    }
    std::stable_sort(order.begin(), order.end(), [&regions](size_t a, size_t b) {
        return regions[a].score > regions[b].score;
    });
    for (size_t index : order) {
        PlateRecognition result;
        result.text = regions[index].plateUpper;
        result.confidence = regions[index].score;
        result.box = boxFromRect(regions[index].rect);
        results.push_back(result);
        if (results.size() >= 2) break;
    }
    return results;
}



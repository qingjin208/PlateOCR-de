#!/usr/bin/env bash
#
# 德国车牌关键点检测 + ONNX Runtime 一键接入脚本 (macOS)
#
# 重要事实：新版 ONNX Runtime 已不在 GitHub Release 提供 iOS 包。
# iOS 版 onnxruntime-mobile-objc 由微软通过 CocoaPods 分发（二进制在
# download.onnxruntime.ai，且会自动依赖 onnxruntime-mobile-c）。
# 因此这里用 CocoaPods 引入，而不是手动下载 xcframework。
#
# 用法：
#   1. 把整个 OCR 文件夹拷到 Mac
#   2. 进入 OCR 目录：cd <你存放的>/OCR
#   3. 运行：bash setup_mac.sh
#   4. 脚本会安装 CocoaPods（如未装）、生成 Podfile、pod install，
#      并打开 LicensePlateOCR.xcworkspace
#   5. 在 Xcode 里选设备 → 点击 ▶ 运行（连真机即可）
#
set -euo pipefail

if [ "$(uname)" != "Darwin" ]; then
  echo "!! 这个脚本只能在 macOS 上运行。" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. 检查 / 安装 CocoaPods
if ! command -v pod >/dev/null 2>&1; then
  echo "==> 未检测到 CocoaPods，尝试安装（macOS 自带 Ruby）"
  if command -v brew >/dev/null 2>&1; then
    brew install cocoapods || true
  fi
  if ! command -v pod >/dev/null 2>&1; then
    sudo gem install cocoapods || {
      echo "!! 自动安装失败。请手动安装其一后再运行本脚本："
      echo "     sudo gem install cocoapods"
      echo "     或  brew install cocoapods"
      exit 1
    }
  fi
fi
echo "==> CocoaPods 版本: $(pod --version)"

# 2. 生成 Podfile（已存在则跳过）
PODFILE="$SCRIPT_DIR/Podfile"
if [ ! -f "$PODFILE" ]; then
  cat > "$PODFILE" <<'EOF'
platform :ios, '18.0'
target 'LicensePlateOCR' do
  pod 'onnxruntime-mobile-objc', '1.18.0'
end
EOF
  echo "==> 已生成 Podfile"
else
  echo "==> 已存在 Podfile，跳过生成"
fi

# 3. pod install（会从 download.onnxruntime.ai 下载框架，需要联网）
echo "==> 执行 pod install（首次会拉取 ONNX Runtime iOS 框架，请保持联网）"
pod install --repo-update

echo ""
echo "============================================================"
echo " 完成 ✅"
echo "============================================================"
echo " 关键：以后用 Xcode 打开 LICENSEPLATEOCR.XCWORKSPACE"
echo "       （不是 .xcodeproj！pod install 后只能用 workspace）"
echo ""
echo "   • onnxruntime 模块由 CocoaPods 提供 → #if canImport(onnxruntime) 生效 → 新管线激活"
echo "   • De-plate-pose-500.onnx 在『文件夹同步』目录内，自动打进 App 包，无需手动处理"
echo "   • 选设备 → ▶ 运行；OCR 引擎先选 Apple Vision，验证检测框对齐"
echo ""
echo " ⚠️ 若 pod install 报网络/下载错误：确认 Mac 能访问 download.onnxruntime.ai"
echo "   （与 GitHub 不同主机）。连不上就告诉我，我给你『Windows 下载后 scp』的离线方案。"
echo ""
echo " ℹ️ PaddleOCR：App 内可选择，但当前自动回退 Apple Vision；"
echo "   真正启用需接入 PaddleOCR-iOS 框架并补全 runPaddleOCR(on:)，见 POSE_INTEGRATION.md。"
echo "============================================================"

if command -v open >/dev/null 2>&1; then
  open "$SCRIPT_DIR/LicensePlateOCR.xcworkspace" || true
fi

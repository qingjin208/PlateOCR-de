#!/usr/bin/env bash
#
# download_model.sh — 一键下载官方 YOLO 占位模型并导出为 Core ML
#
# 用途：当前工程用官方 Ultralytics yolov8n（COCO，检测汽车）作占位模型，
#       等你有自己的德国车牌模型再替换。本脚本在你的 Mac 上联网执行一次即可：
#         bash /Users/zhenzhipeng/Downloads/OCR/Scripts/download_model.sh
#       它会：建 venv → 装 ultralytics（会顺带下载 PyTorch，约需几分钟）→
#       下载 yolov8n.pt → 导出 yolov8n.mlpackage → 复制到 LicensePlateSDK/Resources/。
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESOURCES="$PROJECT_ROOT/LicensePlateOCR/LicensePlateSDK/Resources"

echo "==> 工程根目录: $PROJECT_ROOT"

# 1) 建虚拟环境（若没有）
VENV="$PROJECT_ROOT/.venv"
if [ ! -d "$VENV" ]; then
  echo "==> 创建 Python 虚拟环境..."
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# 2) 安装 ultralytics（会下载 PyTorch，耗时几分钟，请耐心等待）
echo "==> 安装 ultralytics ..."
pip install --quiet --upgrade pip
pip install ultralytics

# 3) 下载 yolov8n.pt 并导出为 Core ML（imgsz=640，关闭 NMS，由 App 自己做 NMS）
echo "==> 导出 yolov8n -> Core ML（imgsz=640, nms=False）..."
cd "$PROJECT_ROOT"
python3 - <<'PY'
from ultralytics import YOLO
m = YOLO('yolov8n.pt')            # 自动下载 COCO 权重（约 6 MB）
out = m.export(format='coreml', imgsz=640, nms=False, int8=False)
print('EXPORTED:', out)
PY

# 4) 复制到 SDK 的 Resources 目录
SRC=$(find "$PROJECT_ROOT" -name 'yolov8n.mlpackage' -maxdepth 3 -type d | head -n1)
if [ -z "$SRC" ]; then
  echo "!! 未找到导出的 yolov8n.mlpackage，请检查上方 ultralytics 导出日志。" >&2
  exit 1
fi
echo "==> 复制 $SRC -> $RESOURCES/"
cp -R "$SRC" "$RESOURCES/"

echo ""
echo "==> 完成！yolov8n 的 Core ML 模型已放入 Resources。"
echo "    回到 Xcode：重新编译（⌘R）即可，缺失模型的报错会消失。"
echo "    提示：若 Xcode 没自动识别新文件，Clean Build Folder（⌘⇧K）后再编译。"

"""P0 自检脚本：配置、模型文件与 OpenCV 算子是否可用。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402

from config.settings import ConfigError, Settings, ensure_models  # noqa: E402


def main() -> int:
    try:
        settings = Settings.load()
        ensure_models(settings)
    except ConfigError as exc:
        print(f"[失败] {exc}", file=sys.stderr)
        return 2

    cv2.FaceDetectorYN.create(
        settings.models.yunet_path,
        "",
        (320, 320),
        settings.detector.score_threshold,
        settings.detector.nms_threshold,
    )
    if settings.tracker.appearance_enabled:
        cv2.FaceRecognizerSF.create(settings.models.sface_path, "")

    print("[通过] 配置加载成功")
    print(f"[通过] YuNet 加载成功：{settings.models.yunet_path}")
    if settings.tracker.appearance_enabled:
        print(f"[通过] SFace 加载成功：{settings.models.sface_path}")
    else:
        print("[信息] 外观兜底已关闭，跳过 SFace 检查")
    print(f"[信息] OpenCV {cv2.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

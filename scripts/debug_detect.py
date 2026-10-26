"""P1 调试脚本：检测框、关键点、目标区域可视化。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from config.settings import (  # noqa: E402
    OUTPUT_DEBUG_DIR,
    ConfigError,
    Settings,
    ensure_models,
    ensure_runtime_dirs,
)
from pipeline.detector.yunet_detector import YuNetDetector  # noqa: E402
from pipeline.target.face_provider import FaceTargetProvider  # noqa: E402
from sources.base import SourceError  # noqa: E402
from sources.image_source import ImageSource  # noqa: E402
from utils.image import write_image  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="人脸检测与目标区域调试")
    parser.add_argument("--input", required=True, help="输入图片路径")
    args = parser.parse_args()

    try:
        settings = Settings.load()
        ensure_models(settings)
        ensure_runtime_dirs()
    except ConfigError as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        return 2

    detector = YuNetDetector(
        model_path=settings.models.yunet_path,
        score_threshold=settings.detector.score_threshold,
        nms_threshold=settings.detector.nms_threshold,
        max_input_size=settings.detector.max_input_size,
    )
    provider = FaceTargetProvider(settings.target.box_expand_ratio)

    started = time.perf_counter()
    try:
        with ImageSource(args.input) as source:
            frame = source.read()
            if frame is None:
                print("[失败] 未能读取到画面", file=sys.stderr)
                return 2
            detections = detector.detect(frame.image, frame.frame_id)
            regions = provider.from_detections(
                detections, (frame.width, frame.height)
            )
    except (SourceError, RuntimeError, ValueError) as exc:
        print(f"[失败] {exc}", file=sys.stderr)
        return 2

    canvas = frame.image.copy()
    for index, detection in enumerate(detections, start=1):
        x, y, w, h = (int(round(value)) for value in detection.bbox)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for point in detection.landmarks:
            cv2.circle(
                canvas,
                (int(round(point[0])), int(round(point[1]))),
                2,
                (0, 0, 255),
                -1,
            )

        print(
            f"[{index}] conf={detection.confidence:.4f} "
            f"bbox=({x}, {y}, {w}, {h})"
        )

    for region in regions:
        x, y, w, h = region.bbox
        if settings.render.mask_shape == "ellipse":
            center = (int(round(x + w / 2)), int(round(y + h / 2)))
            axes = (max(1, int(round(w / 2))), max(1, int(round(h / 2))))
            cv2.ellipse(canvas, center, axes, 0, 0, 360, (255, 255, 0), 2)
        elif region.polygon:
            polygon = np.asarray(region.polygon, dtype=np.int32)
            cv2.polylines(canvas, [polygon], True, (255, 255, 0), 2)

    elapsed = time.perf_counter() - started
    output_path = OUTPUT_DEBUG_DIR / f"{Path(args.input).stem}_detect.jpg"
    write_image(output_path, canvas)
    print(f"[完成] 检出 {len(detections)} 张人脸，耗时 {elapsed:.3f}s")
    print(
        f"[信息] 形状 {settings.render.mask_shape}，"
        f"缩放 {settings.target.box_expand_ratio:+.2f}"
    )
    print(f"[输出] {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

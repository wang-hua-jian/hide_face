"""独立的摄像头截图脚本：预览画面，按 Space/S 截图，按 Q/Esc 退出。"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from config.settings import IMAGES_DIR, ConfigError, Settings, ensure_runtime_dirs
from sources.base import SourceError
from sources.camera_source import CameraSource
from utils.image import write_image
from utils.timefmt import unique_camera_path

WINDOW_NAME = "hide_face - snapshot"


def _scale_for_display(image, scale: float):
    if scale == 1.0:
        return image
    height, width = image.shape[:2]
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=interpolation,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="摄像头截图：Space/S 截图，Q/Esc 退出",
    )
    parser.parse_args()

    try:
        settings = Settings.load()
        ensure_runtime_dirs()
    except ConfigError as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        return 2

    source = None
    captured = 0
    try:
        source = CameraSource(
            index=settings.camera.index,
            jpeg_quality=settings.render.jpeg_quality,
        )
        source.open()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        print("[提示] Space/S 截图，Q/Esc 退出")

        interval = 1.0 / max(1, settings.camera.max_fps)
        next_deadline = time.perf_counter()

        while True:
            frame = source.read()
            if frame is None:
                break

            display = _scale_for_display(
                frame.image, settings.camera.display_scale
            )
            cv2.putText(
                display,
                "SPACE/S: capture   Q/ESC: quit",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                break

            if key in (32, ord("s"), ord("S")):
                path = unique_camera_path(IMAGES_DIR)
                write_image(
                    path,
                    frame.image,
                    [cv2.IMWRITE_JPEG_QUALITY, settings.render.jpeg_quality],
                )
                captured += 1
                print(f"[截图 {captured}] {path}")

            try:
                visible = cv2.getWindowProperty(
                    WINDOW_NAME, cv2.WND_PROP_VISIBLE
                )
            except cv2.error:
                break
            if visible < 1:
                break

            next_deadline += interval
            delay = next_deadline - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_deadline = time.perf_counter()
    except KeyboardInterrupt:
        print("\n[退出] 已停止，摄像头已释放。")
    except (cv2.error, SourceError, OSError, RuntimeError, ValueError) as exc:
        print(f"[摄像头失败] {exc}", file=sys.stderr)
        return 2
    finally:
        if source is not None:
            source.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    print(f"[完成] 共截图 {captured} 张")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

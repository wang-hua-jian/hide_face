"""独立的摄像头实时遮挡脚本：只显示遮挡后的画面，不录像、不写分析。"""

from __future__ import annotations

import argparse
import sys

from config.settings import (
    ConfigError,
    Settings,
    ensure_models,
    ensure_runtime_dirs,
)
from main import build_tracking_stack
from pipeline.camera_pipeline import CameraPipeline
from pipeline.render.mask_renderer import MaskRenderer
from sources.camera_source import CameraSource


def main() -> int:
    parser = argparse.ArgumentParser(
        description="摄像头实时遮挡：Q/Esc 退出",
    )
    parser.add_argument(
        "--mask",
        default=None,
        help="自定义遮挡图路径（指定后使用 custom_image 样式）",
    )
    args = parser.parse_args()

    try:
        settings = Settings.load()
        ensure_models(settings)
        ensure_runtime_dirs()
    except ConfigError as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        return 2

    try:
        style = "custom_image" if args.mask else settings.render.style
        (
            detector,
            tracker,
            smoother,
            target_provider,
            _,
        ) = build_tracking_stack(
            settings,
            settings.target.box_expand_ratio,
        )
        pipeline = CameraPipeline(
            detector=detector,
            tracker=tracker,
            smoother=smoother,
            target_provider=target_provider,
            renderer=MaskRenderer(
                style=style,
                opacity=settings.render.opacity,
                mosaic_block=settings.render.mosaic_block,
                blur_kernel=settings.render.blur_kernel,
                mask_path=args.mask,
                mask_shape=settings.render.mask_shape,
            ),
            max_fps=settings.camera.max_fps,
            display_scale=settings.camera.display_scale,
            analysis_flush_seconds=settings.video.chunk_seconds,
        )

        source = CameraSource(
            index=settings.camera.index,
            jpeg_quality=settings.render.jpeg_quality,
        )
        print("[提示] 实时遮挡运行中，按 Q/Esc 退出")
        with source:
            result = pipeline.run(source, show=True)
    except KeyboardInterrupt:
        print("\n[退出] 已停止，摄像头已释放。")
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[摄像头失败] {exc}", file=sys.stderr)
        return 2

    print(
        f"[完成] 处理 {result.processed_frames} 帧，"
        f"{result.track_count} 个人脸 Track，"
        f"耗时 {result.elapsed_seconds:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

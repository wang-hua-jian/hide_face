"""统一入口：4 种处理模式 + reference 档案命令。

P0 阶段只完成参数解析、配置加载与模型自检，业务逻辑在后续阶段接入。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from config.settings import (
    IMAGES_DIR,
    OUTPUT_IMAGES_DIR,
    OUTPUT_JSON_DIR,
    OUTPUT_LIVE_DIR,
    OUTPUT_VIDEOS_DIR,
    ConfigError,
    Settings,
    ensure_models,
    ensure_runtime_dirs,
)

STYLE_CHOICES = ("mosaic", "blur", "solid", "custom_image")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hide_face",
        description="社恐/高隐私人士面部自动脱敏器（第一版纯后端，路径即处理目标）",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    image = subparsers.add_parser("image", help="处理单张本地图片")
    image.add_argument("--input", required=True, help="输入图片路径")
    _add_render_args(image)
    _add_analysis_args(image)
    _add_identity_args(image)
    image.set_defaults(handler="image")

    video = subparsers.add_parser("video", help="处理本地视频")
    video.add_argument("--input", required=True, help="输入视频路径")
    _add_render_args(video)
    _add_analysis_args(video)
    _add_identity_args(video)
    video.set_defaults(handler="video")

    camera = subparsers.add_parser("camera", help="摄像头截图或实时遮挡")
    camera_sub = camera.add_subparsers(dest="camera_command", required=True)

    snap = camera_sub.add_parser("snap", help="从摄像头截图")
    snap.add_argument("--count", type=int, default=None, help="截图张数（默认取配置）")
    snap.add_argument("--interval", type=float, default=None, help="间隔秒数（默认取配置）")
    snap.add_argument("--process", action="store_true", help="截图后立即走图片脱敏链路")
    _add_render_args(snap)
    _add_analysis_args(snap)
    _add_identity_args(snap)
    snap.set_defaults(handler="camera_snap")

    live = camera_sub.add_parser("live", help="摄像头实时遮挡")
    live.add_argument("--max-fps", type=int, default=None, help="处理帧率上限（默认取配置）")
    live.add_argument("--display-scale", type=float, default=None, help="显示缩放（默认取配置）")
    live.add_argument("--record", action="store_true", help="录制到 output/live/")
    live.add_argument("--save-analysis", action="store_true", help="实时模式写 analysis.json")
    _add_render_args(live)
    _add_identity_args(live)
    live.set_defaults(handler="camera_live")

    reference = subparsers.add_parser("reference", help="管理参考照片档案")
    reference_sub = reference.add_subparsers(dest="reference_command", required=True)

    build = reference_sub.add_parser("build", help="从参考照片生成档案")
    build.add_argument("--dir", default=None, help="参考照片目录")
    build.add_argument("--reference", nargs="+", default=None, metavar="PATH", help="参考照片路径")
    build.add_argument("--name", required=True, help="档案名称，如 me")
    build.set_defaults(handler="reference_build")

    return parser


def _add_render_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mask", default=None, help="自定义遮挡图路径（指定后等价 custom_image）")
    parser.add_argument(
        "--style",
        choices=STYLE_CHOICES,
        default=None,
        help="遮挡样式（默认取配置）",
    )
    parser.add_argument(
        "--shape",
        choices=("rect", "ellipse"),
        default=None,
        help="遮挡形状：矩形或椭圆（默认取配置）",
    )
    parser.add_argument(
        "--expand",
        type=float,
        default=None,
        help="遮挡区域缩放比例 (-0.5, 1]：负值向内收缩，0 贴合检测框，正值向外扩张",
    )


def _add_analysis_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-analysis", action="store_true", help="不写 analysis.json")


def _add_identity_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", default=None, help="参考档案路径，如 data/profiles/me.json")
    parser.add_argument(
        "--reference",
        nargs="+",
        default=None,
        metavar="PATH",
        help="临时参考照片（1~N 张），可替代 --profile",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "reference" and not args.dir and not args.reference:
        parser.error("reference build 需要 --dir 或 --reference 之一")

    try:
        settings = Settings.load()
        ensure_models(settings)
        ensure_runtime_dirs()
    except ConfigError as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        return 2

    if args.command == "image":
        return _run_image(args, settings)
    if args.command == "video":
        return _run_video(args, settings)
    if args.command == "camera":
        if args.camera_command == "snap":
            return _run_camera_snap(args, settings)
        return _run_camera_live(args, settings)

    print(f"[未实现] {_describe_command(args)} 将在后续阶段接入。", file=sys.stderr)
    return 1


def _resolve_render_options(
    args: argparse.Namespace,
    settings: Settings,
):
    style = "custom_image" if args.mask else (args.style or settings.render.style)
    if style == "custom_image" and not args.mask:
        raise ValueError("style=custom_image 需要配合 --mask 使用")

    expand_ratio = (
        settings.target.box_expand_ratio if args.expand is None else args.expand
    )
    if not -0.5 < expand_ratio <= 1.0:
        raise ValueError("--expand 必须大于 -0.5 且不超过 1")

    mask_shape = args.shape or settings.render.mask_shape
    return style, mask_shape, expand_ratio


def build_tracking_stack(settings: Settings, expand_ratio: float):
    from pipeline.analysis.temporal import TemporalSmoother
    from pipeline.detector.yunet_detector import YuNetDetector
    from pipeline.embedder.sface_embedder import SFaceEmbedder
    from pipeline.target.face_provider import FaceTargetProvider
    from pipeline.tracker.appearance_matcher import AppearanceMatcher
    from pipeline.tracker.bytetrack_tracker import ByteTrackTracker
    from pipeline.tracker.face_geometry import FaceGeometryPrior

    appearance_matcher = None
    if settings.tracker.appearance_enabled:
        appearance_matcher = AppearanceMatcher(
            embedder=SFaceEmbedder(settings.models.sface_path),
            similarity_threshold=settings.tracker.appearance_similarity,
            ema_alpha=settings.tracker.ema_alpha,
            min_face_size=settings.tracker.appearance_min_face_size,
            max_lost_frames=settings.tracker.appearance_max_lost_frames,
            min_eye_ratio=settings.tracker.appearance_min_eye_ratio,
        )

    geometry_prior = FaceGeometryPrior(
        window_size=settings.tracker.eye_ratio_window,
        min_samples=settings.tracker.eye_ratio_min_samples,
        lower_factor=settings.tracker.eye_ratio_lower_factor,
        upper_factor=settings.tracker.eye_ratio_upper_factor,
        fallback_min_ratio=settings.tracker.eye_ratio_fallback,
    )

    detector = YuNetDetector(
        model_path=settings.models.yunet_path,
        score_threshold=settings.tracker.low_score_threshold,
        nms_threshold=settings.detector.nms_threshold,
        max_input_size=settings.detector.max_input_size,
    )
    tracker = ByteTrackTracker(
        high_score_threshold=settings.tracker.high_score_threshold,
        low_score_threshold=settings.tracker.low_score_threshold,
        iou_threshold=settings.tracker.iou_threshold,
        second_iou_threshold=settings.tracker.second_iou_threshold,
        max_lost_frames=settings.tracker.max_lost_frames,
        mask_lost_frames=settings.tracker.mask_lost_frames,
        lost_mask_distance_ratio=settings.tracker.lost_mask_distance_ratio,
        frame_step=settings.detector.detect_interval,
        appearance_matcher=appearance_matcher,
        geometry_prior=geometry_prior,
    )
    smoother = TemporalSmoother(
        alpha=settings.tracker.smooth_alpha,
        max_scale_step=settings.tracker.max_scale_step,
    )
    target_provider = FaceTargetProvider(expand_ratio)
    return detector, tracker, smoother, target_provider, appearance_matcher


def _run_image(args: argparse.Namespace, settings: Settings) -> int:
    if args.profile or args.reference:
        print("[提示] 参考照片增强将在 P7 接入，本次忽略。", file=sys.stderr)

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"[处理失败] 输入图片不存在：{input_path}", file=sys.stderr)
        return 2

    try:
        style, mask_shape, expand_ratio = _resolve_render_options(args, settings)
    except ValueError as exc:
        print(f"[参数错误] {exc}", file=sys.stderr)
        return 2

    output_path = OUTPUT_IMAGES_DIR / f"{input_path.stem}_masked.jpg"
    analysis_path = (
        None
        if args.no_analysis
        else OUTPUT_JSON_DIR / f"{input_path.stem}_analysis.json"
    )

    try:
        from pipeline.detector.yunet_detector import YuNetDetector
        from pipeline.image_pipeline import ImagePipeline
        from pipeline.render.mask_renderer import MaskRenderer
        from pipeline.target.face_provider import FaceTargetProvider

        detector = YuNetDetector(
            model_path=settings.models.yunet_path,
            score_threshold=settings.detector.score_threshold,
            nms_threshold=settings.detector.nms_threshold,
            max_input_size=settings.detector.max_input_size,
        )
        pipeline = ImagePipeline(
            detector=detector,
            target_provider=FaceTargetProvider(expand_ratio),
            renderer=MaskRenderer(
                style=style,
                opacity=settings.render.opacity,
                mosaic_block=settings.render.mosaic_block,
                blur_kernel=settings.render.blur_kernel,
                mask_path=args.mask,
                mask_shape=mask_shape,
            ),
            jpeg_quality=settings.render.jpeg_quality,
        )
        result = pipeline.run(input_path, output_path, analysis_path)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[处理失败] {exc}", file=sys.stderr)
        return 2

    print(
        f"[完成] 检出 {result.face_count} 张人脸，样式 {style}/{mask_shape}，"
        f"缩放 {expand_ratio:+.2f}，耗时 {result.elapsed_seconds:.2f}s"
    )
    print(f"[输出] {result.output_path}")
    if result.analysis_path is not None:
        print(f"[分析] {result.analysis_path}")
    return 0


def _run_video(args: argparse.Namespace, settings: Settings) -> int:
    if args.profile or args.reference:
        print("[提示] 参考照片增强将在 P7 接入，本次忽略。", file=sys.stderr)

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"[处理失败] 输入视频不存在：{input_path}", file=sys.stderr)
        return 2

    try:
        style, mask_shape, expand_ratio = _resolve_render_options(args, settings)
    except ValueError as exc:
        print(f"[参数错误] {exc}", file=sys.stderr)
        return 2

    output_path = OUTPUT_VIDEOS_DIR / f"{input_path.stem}_masked.mp4"
    analysis_path = (
        None
        if args.no_analysis
        else OUTPUT_JSON_DIR / f"{input_path.stem}_analysis.json"
    )

    try:
        from pipeline.render.mask_renderer import MaskRenderer
        from pipeline.video_pipeline import VideoPipeline

        (
            detector,
            tracker,
            smoother,
            target_provider,
            appearance_matcher,
        ) = build_tracking_stack(settings, expand_ratio)
        pipeline = VideoPipeline(
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
                mask_shape=mask_shape,
            ),
            detect_interval=settings.detector.detect_interval,
            chunk_seconds=settings.video.chunk_seconds,
            chunk_overlap_seconds=settings.video.chunk_overlap_seconds,
        )

        last_percent = -1

        def report(done: int, total: int) -> None:
            nonlocal last_percent
            if total <= 0:
                if done % 50 == 0:
                    print(f"[进度] 已处理 {done} 帧")
                return
            percent = done * 100 // total
            if percent >= last_percent + 5 or done == total:
                last_percent = percent
                print(f"[进度] {done}/{total} ({percent}%)")

        result = pipeline.run(
            input_path,
            output_path,
            report,
            analysis_path=analysis_path,
        )
    except KeyboardInterrupt:
        print("\n[中断] 已停止，保留已生成的视频。", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[处理失败] {exc}", file=sys.stderr)
        return 2

    print(
        f"[完成] {result.processed_frames} 帧，{result.track_count} 个人脸 Track，"
        f"样式 {style}/{mask_shape}，缩放 {expand_ratio:+.2f}，"
        f"耗时 {result.elapsed_seconds:.2f}s"
    )
    print(f"[输出] {result.output_path}")
    if result.analysis_path is not None:
        print(f"[分析] {result.analysis_path}")
    if appearance_matcher is not None:
        print(
            f"[外观] SFace 调用 {appearance_matcher.calls} 次，"
            f"认回 {appearance_matcher.matches} 次"
        )
    return 0


def _run_camera_snap(args: argparse.Namespace, settings: Settings) -> int:
    if args.profile or args.reference:
        print("[提示] 参考照片增强将在 P7 接入，本次忽略。", file=sys.stderr)

    try:
        style, mask_shape, expand_ratio = _resolve_render_options(args, settings)
    except ValueError as exc:
        print(f"[参数错误] {exc}", file=sys.stderr)
        return 2

    count = settings.camera.snap_count if args.count is None else args.count
    interval = (
        settings.camera.snap_interval_seconds
        if args.interval is None
        else args.interval
    )
    if count < 1:
        print(f"[参数错误] --count 必须 >= 1：{count}", file=sys.stderr)
        return 2
    if interval < 0:
        print(f"[参数错误] --interval 不能为负：{interval}", file=sys.stderr)
        return 2

    try:
        from pipeline.camera_pipeline import capture_snapshots
        from sources.camera_source import CameraSource

        source = CameraSource(
            index=settings.camera.index,
            jpeg_quality=settings.render.jpeg_quality,
        )
        with source:
            snapshot_paths = capture_snapshots(
                source,
                IMAGES_DIR,
                count,
                interval,
            )
    except KeyboardInterrupt:
        print("\n[中断] 已停止，摄像头已释放。", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[摄像头失败] {exc}", file=sys.stderr)
        return 2

    for path in snapshot_paths:
        print(f"[截图] {path}")

    if not args.process:
        return 0

    try:
        from pipeline.detector.yunet_detector import YuNetDetector
        from pipeline.image_pipeline import ImagePipeline
        from pipeline.render.mask_renderer import MaskRenderer
        from pipeline.target.face_provider import FaceTargetProvider

        detector = YuNetDetector(
            model_path=settings.models.yunet_path,
            score_threshold=settings.detector.score_threshold,
            nms_threshold=settings.detector.nms_threshold,
            max_input_size=settings.detector.max_input_size,
        )
        pipeline = ImagePipeline(
            detector=detector,
            target_provider=FaceTargetProvider(expand_ratio),
            renderer=MaskRenderer(
                style=style,
                opacity=settings.render.opacity,
                mosaic_block=settings.render.mosaic_block,
                blur_kernel=settings.render.blur_kernel,
                mask_path=args.mask,
                mask_shape=mask_shape,
            ),
            jpeg_quality=settings.render.jpeg_quality,
        )

        for raw_path in snapshot_paths:
            output_path = OUTPUT_IMAGES_DIR / f"{raw_path.stem}_masked.jpg"
            analysis_path = (
                None
                if args.no_analysis
                else OUTPUT_JSON_DIR / f"{raw_path.stem}_analysis.json"
            )
            result = pipeline.run(raw_path, output_path, analysis_path)
            print(
                f"[完成] {raw_path.name} 检出 {result.face_count} 张人脸，"
                f"耗时 {result.elapsed_seconds:.2f}s"
            )
            print(f"[输出] {result.output_path}")
            if result.analysis_path is not None:
                print(f"[分析] {result.analysis_path}")
    except KeyboardInterrupt:
        print("\n[中断] 截图已保留，处理已停止。", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[处理失败] {exc}", file=sys.stderr)
        return 2

    return 0


def _run_camera_live(args: argparse.Namespace, settings: Settings) -> int:
    if args.profile or args.reference:
        print("[提示] 参考照片增强将在 P7 接入，本次忽略。", file=sys.stderr)

    try:
        style, mask_shape, expand_ratio = _resolve_render_options(args, settings)
    except ValueError as exc:
        print(f"[参数错误] {exc}", file=sys.stderr)
        return 2

    max_fps = settings.camera.max_fps if args.max_fps is None else args.max_fps
    display_scale = (
        settings.camera.display_scale
        if args.display_scale is None
        else args.display_scale
    )
    if max_fps <= 0:
        print(f"[参数错误] --max-fps 必须大于 0：{max_fps}", file=sys.stderr)
        return 2
    if not 0.0 < display_scale <= 3.0:
        print(
            f"[参数错误] --display-scale 必须在 (0, 3]：{display_scale}",
            file=sys.stderr,
        )
        return 2

    try:
        from pipeline.camera_pipeline import CameraPipeline
        from pipeline.render.mask_renderer import MaskRenderer
        from sources.camera_source import CameraSource
        from utils.timefmt import format_timestamp

        (
            detector,
            tracker,
            smoother,
            target_provider,
            appearance_matcher,
        ) = build_tracking_stack(settings, expand_ratio)
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
                mask_shape=mask_shape,
            ),
            max_fps=max_fps,
            display_scale=display_scale,
            analysis_flush_seconds=settings.video.chunk_seconds,
        )

        timestamp = format_timestamp()
        record_path = (
            OUTPUT_LIVE_DIR / f"live_{timestamp}.mp4"
            if args.record
            else None
        )
        analysis_path = (
            OUTPUT_JSON_DIR / f"live_{timestamp}_analysis.json"
            if args.save_analysis
            else None
        )

        source = CameraSource(
            index=settings.camera.index,
            jpeg_quality=settings.render.jpeg_quality,
        )
        with source:
            result = pipeline.run(
                source,
                record_path=record_path,
                analysis_path=analysis_path,
                show=True,
            )
    except KeyboardInterrupt:
        print("\n[中断] 已停止，摄像头已释放。", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[摄像头失败] {exc}", file=sys.stderr)
        return 2

    print(
        f"[完成] 处理 {result.processed_frames} 帧，"
        f"{result.track_count} 个人脸 Track，"
        f"样式 {style}/{mask_shape}，耗时 {result.elapsed_seconds:.2f}s"
    )
    if result.record_path is not None:
        print(f"[录像] {result.record_path}")
    if result.analysis_path is not None:
        print(f"[分析] {result.analysis_path}")
    if appearance_matcher is not None:
        print(
            f"[外观] SFace 调用 {appearance_matcher.calls} 次，"
            f"认回 {appearance_matcher.matches} 次"
        )
    return 0


def _describe_command(args: argparse.Namespace) -> str:
    if args.command == "camera":
        return f"camera {args.camera_command}"
    if args.command == "reference":
        return f"reference {args.reference_command}"
    return str(args.command)


if __name__ == "__main__":
    sys.exit(main())

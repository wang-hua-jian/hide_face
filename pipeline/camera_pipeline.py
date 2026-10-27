"""摄像头截图与实时遮挡管线。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import cv2

from pipeline.analysis.recorder import AnalysisRecorder
from pipeline.analysis.temporal import TemporalSmoother
from pipeline.detector.base import FaceDetector
from pipeline.regions import build_track_regions
from pipeline.render.base import Renderer
from pipeline.target.base import TargetProvider
from pipeline.tracker.base import Tracker
from pipeline.video_writer import VideoWriter
from sources.base import Source
from utils.timefmt import unique_camera_path

WINDOW_NAME = "hide_face - camera"


class CameraPipelineError(RuntimeError):
    """摄像头显示或处理失败。"""


@dataclass(frozen=True)
class CameraResult:
    processed_frames: int
    track_count: int
    elapsed_seconds: float
    record_path: Optional[Path]
    analysis_path: Optional[Path]


def capture_snapshots(
    source: Source,
    directory: Union[str, Path],
    count: int,
    interval_seconds: float,
) -> List[Path]:
    if count < 1:
        raise ValueError(f"截图张数必须 >= 1：{count}")
    if interval_seconds < 0:
        raise ValueError(f"截图间隔不能为负：{interval_seconds}")

    target_dir = Path(directory)
    paths: List[Path] = []
    for index in range(count):
        path = unique_camera_path(target_dir)
        source.snapshot(path)
        paths.append(path)
        if index < count - 1 and interval_seconds > 0:
            time.sleep(interval_seconds)
    return paths


class CameraPipeline:
    def __init__(
        self,
        detector: FaceDetector,
        tracker: Tracker,
        smoother: TemporalSmoother,
        target_provider: TargetProvider,
        renderer: Renderer,
        max_fps: int,
        display_scale: float,
        analysis_flush_seconds: int,
    ) -> None:
        if max_fps <= 0:
            raise ValueError(f"max_fps 必须大于 0：{max_fps}")
        if not 0.0 < display_scale <= 3.0:
            raise ValueError(f"display_scale 必须在 (0, 3]：{display_scale}")
        if analysis_flush_seconds <= 0:
            raise ValueError(
                f"analysis_flush_seconds 必须大于 0：{analysis_flush_seconds}"
            )

        self._detector = detector
        self._tracker = tracker
        self._smoother = smoother
        self._target_provider = target_provider
        self._renderer = renderer
        self._max_fps = max_fps
        self._display_scale = display_scale
        self._analysis_flush_seconds = analysis_flush_seconds

    def run(
        self,
        source: Source,
        record_path: Optional[Union[str, Path]] = None,
        analysis_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ) -> CameraResult:
        started = time.perf_counter()
        meta = source.meta
        width = int(meta.get("width", 0))
        height = int(meta.get("height", 0))
        if width <= 0 or height <= 0:
            raise CameraPipelineError(
                f"无法获取摄像头画面尺寸：{width}x{height}"
            )
        if width % 2 != 0 or height % 2 != 0:
            raise CameraPipelineError(
                f"当前编码器要求宽高为偶数：{width}x{height}"
            )

        record_path = Path(record_path) if record_path is not None else None
        analysis_path = Path(analysis_path) if analysis_path is not None else None

        writer: Optional[VideoWriter] = None
        recorder: Optional[AnalysisRecorder] = None
        if record_path is not None:
            writer = VideoWriter(
                record_path,
                width,
                height,
                self._max_fps,
                input_path=None,
            )

        try:
            if analysis_path is not None:
                recorder = AnalysisRecorder(
                    analysis_path,
                    {
                        "type": "camera",
                        "index": meta.get("index"),
                        "width": width,
                        "height": height,
                        "fps": self._max_fps,
                        "camera_fps": meta.get("camera_fps", 0.0),
                        "started_at": datetime.now().isoformat(timespec="seconds"),
                    },
                )

            if show:
                try:
                    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                except cv2.error as exc:
                    raise CameraPipelineError(
                        f"无法创建摄像头显示窗口：{exc}"
                    ) from exc

            frame_interval = 1.0 / self._max_fps
            next_deadline = time.perf_counter()
            next_flush = float(self._analysis_flush_seconds)
            processed = 0
            track_ids = set()

            while True:
                frame = source.read()
                if frame is None:
                    break

                detections = self._detector.detect(
                    frame.image, frame.frame_id
                )
                tracks = self._tracker.update(
                    detections, frame.frame_id, frame.image
                )
                boxes = self._smoother.update(tracks)
                regions = build_track_regions(
                    tracks,
                    boxes,
                    self._target_provider,
                    frame.frame_id,
                    frame.width,
                    frame.height,
                )
                rendered = self._renderer.draw(frame.image, regions)

                if writer is not None:
                    writer.write(rendered, frame.timestamp)

                if recorder is not None:
                    recorder.add_frame(frame.frame_id, regions)
                    if frame.timestamp >= next_flush:
                        recorder.flush()
                        next_flush += self._analysis_flush_seconds

                processed += 1
                track_ids.update(track.track_id for track in tracks)

                if show:
                    display = self._scale_for_display(rendered)
                    try:
                        cv2.imshow(WINDOW_NAME, display)
                        key = cv2.waitKey(1) & 0xFF
                        visible = cv2.getWindowProperty(
                            WINDOW_NAME, cv2.WND_PROP_VISIBLE
                        )
                    except cv2.error:
                        # 用户用窗口关闭按钮退出时，部分 Windows 后端会抛错
                        break
                    if key in (27, ord("q")):
                        break
                    if visible < 1:
                        break

                next_deadline += frame_interval
                delay = next_deadline - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_deadline = time.perf_counter()
        finally:
            try:
                if writer is not None:
                    writer.close()
            finally:
                if recorder is not None:
                    recorder.close()
                if show:
                    try:
                        cv2.destroyAllWindows()
                    except cv2.error:
                        pass

        return CameraResult(
            processed_frames=processed,
            track_count=len(track_ids),
            elapsed_seconds=time.perf_counter() - started,
            record_path=record_path,
            analysis_path=analysis_path,
        )

    def _scale_for_display(self, image):
        if self._display_scale == 1.0:
            return image
        height, width = image.shape[:2]
        target_width = max(1, int(width * self._display_scale))
        target_height = max(1, int(height * self._display_scale))
        interpolation = (
            cv2.INTER_AREA
            if self._display_scale < 1.0
            else cv2.INTER_LINEAR
        )
        return cv2.resize(
            image,
            (target_width, target_height),
            interpolation=interpolation,
        )

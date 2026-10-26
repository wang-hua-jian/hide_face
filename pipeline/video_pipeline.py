"""视频管线：流式读帧 → 检测 → 跟踪 → 平滑 → 遮挡 → 编码。"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import av
import numpy as np

from pipeline.analysis.temporal import TemporalSmoother
from pipeline.detector.base import FaceDetector
from pipeline.render.base import Renderer
from pipeline.schema import FaceDetection, TargetRegion
from pipeline.target.base import TargetProvider
from pipeline.tracker.base import Tracker
from pipeline.tracker.state import TrackState
from sources.video_source import VideoSource

ProgressCallback = Callable[[int, int], None]


class VideoPipelineError(RuntimeError):
    """视频处理流程失败。"""


@dataclass(frozen=True)
class VideoResult:
    output_path: Path
    processed_frames: int
    track_count: int
    elapsed_seconds: float


class VideoPipeline:
    def __init__(
        self,
        detector: FaceDetector,
        tracker: Tracker,
        smoother: TemporalSmoother,
        target_provider: TargetProvider,
        renderer: Renderer,
        detect_interval: int,
    ) -> None:
        if detect_interval < 1:
            raise ValueError(f"detect_interval 必须 >= 1：{detect_interval}")
        self._detector = detector
        self._tracker = tracker
        self._smoother = smoother
        self._target_provider = target_provider
        self._renderer = renderer
        self._detect_interval = detect_interval

    def run(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        progress: Optional[ProgressCallback] = None,
    ) -> VideoResult:
        started = time.perf_counter()
        input_path = Path(input_path)
        output_path = Path(output_path)

        with VideoSource(input_path) as source:
            meta = source.meta
            width = int(meta["width"])
            height = int(meta["height"])
            total_frames = int(meta.get("total_frames", 0))
            rate = meta.get("rate") or meta.get("fps") or 30

            if width % 2 != 0 or height % 2 != 0:
                raise VideoPipelineError(
                    f"当前编码器要求宽高为偶数：{width}x{height}"
                )

            writer = _VideoWriter(output_path, width, height, rate)
            track_ids = set()
            processed = 0

            try:
                while True:
                    frame = source.read()
                    if frame is None:
                        break

                    is_detection_frame = (
                        frame.frame_id == 0
                        or frame.frame_id % self._detect_interval == 0
                    )
                    if is_detection_frame:
                        detections = self._detector.detect(
                            frame.image, frame.frame_id
                        )
                        tracks = self._tracker.update(
                            detections, frame.frame_id
                        )
                    else:
                        tracks = self._tracker.predict(frame.frame_id)

                    boxes = self._smoother.update(tracks)
                    regions = self._build_regions(
                        tracks,
                        boxes,
                        frame.frame_id,
                        frame.width,
                        frame.height,
                    )

                    rendered = self._renderer.draw(frame.image, regions)
                    writer.write(rendered)

                    processed += 1
                    track_ids.update(track.track_id for track in tracks)
                    if progress is not None:
                        progress(processed, total_frames)
            finally:
                writer.close()

        return VideoResult(
            output_path=output_path,
            processed_frames=processed,
            track_count=len(track_ids),
            elapsed_seconds=time.perf_counter() - started,
        )

    def _build_regions(
        self,
        tracks: Sequence[TrackState],
        boxes: dict,
        frame_id: int,
        width: int,
        height: int,
    ) -> List[TargetRegion]:
        regions: List[TargetRegion] = []

        for track in tracks:
            if not track.mask_state:
                continue
            if track.updated and track.measurement_bbox is not None:
                # 检测帧直接用检测框，避免滤波/平滑带来的视觉延迟
                box = track.measurement_bbox
            else:
                box = boxes.get(track.track_id)
            if box is None:
                continue

            detection = FaceDetection(
                bbox=box,
                landmarks=(),
                confidence=track.score,
                frame_id=frame_id,
            )
            for region in self._target_provider.from_detections(
                [detection], (width, height)
            ):
                regions.append(
                    replace(
                        region,
                        track_id=track.track_id,
                        attributes={
                            "frame_id": frame_id,
                            "detection_bbox": box,
                            "lost_frames": track.lost_frames,
                            "status": track.status,
                        },
                    )
                )

        return regions


class _VideoWriter:
    def __init__(self, path: Path, width: int, height: int, rate: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._container = av.open(str(path), mode="w")
            try:
                stream = self._container.add_stream("libx264", rate=rate)
                stream.options = {"crf": "23", "preset": "veryfast"}
            except (av.error.FFmpegError, ValueError):
                stream = self._container.add_stream("mpeg4", rate=rate)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            self._stream = stream
        except (av.error.FFmpegError, ValueError) as exc:
            raise VideoPipelineError(f"无法创建视频编码器：{exc}") from exc

    def write(self, image: np.ndarray) -> None:
        try:
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            for packet in self._stream.encode(frame):
                self._container.mux(packet)
        except (av.error.FFmpegError, ValueError) as exc:
            raise VideoPipelineError(f"视频编码失败：{exc}") from exc

    def close(self) -> None:
        try:
            for packet in self._stream.encode():
                self._container.mux(packet)
        finally:
            self._container.close()

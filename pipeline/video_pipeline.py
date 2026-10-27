"""视频管线：流式读帧 → 检测 → 跟踪 → 平滑 → 遮挡 → 编码 / 分析记录。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

from pipeline.analysis.chunks import plan_chunks
from pipeline.analysis.recorder import AnalysisRecorder
from pipeline.analysis.temporal import TemporalSmoother
from pipeline.detector.base import FaceDetector
from pipeline.regions import build_track_regions
from pipeline.render.base import Renderer
from pipeline.target.base import TargetProvider
from pipeline.tracker.base import Tracker
from pipeline.video_writer import VideoWriter
from sources.video_source import VideoSource

ProgressCallback = Callable[[int, int], None]


class VideoPipelineError(RuntimeError):
    """视频处理流程失败。"""


@dataclass(frozen=True)
class VideoResult:
    output_path: Path
    analysis_path: Optional[Path]
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
        chunk_seconds: int,
        chunk_overlap_seconds: int,
    ) -> None:
        if detect_interval < 1:
            raise ValueError(f"detect_interval 必须 >= 1：{detect_interval}")
        if chunk_seconds <= 0:
            raise ValueError(f"chunk_seconds 必须大于 0：{chunk_seconds}")
        if not 0 <= chunk_overlap_seconds < chunk_seconds:
            raise ValueError(
                "chunk_overlap_seconds 必须满足 0 <= overlap < chunk_seconds"
            )
        self._detector = detector
        self._tracker = tracker
        self._smoother = smoother
        self._target_provider = target_provider
        self._renderer = renderer
        self._detect_interval = detect_interval
        self._chunk_seconds = chunk_seconds
        self._chunk_overlap_seconds = chunk_overlap_seconds

    def run(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        progress: Optional[ProgressCallback] = None,
        analysis_path: Optional[Union[str, Path]] = None,
    ) -> VideoResult:
        started = time.perf_counter()
        input_path = Path(input_path)
        output_path = Path(output_path)
        analysis_path = Path(analysis_path) if analysis_path is not None else None

        with VideoSource(input_path) as source:
            meta = source.meta
            width = int(meta["width"])
            height = int(meta["height"])
            total_frames = int(meta.get("total_frames", 0))
            duration = float(meta.get("duration", 0.0))
            rate = meta.get("rate") or meta.get("fps") or 30

            if width % 2 != 0 or height % 2 != 0:
                raise VideoPipelineError(
                    f"当前编码器要求宽高为偶数：{width}x{height}"
                )

            chunks = plan_chunks(
                duration,
                self._chunk_seconds,
                self._chunk_overlap_seconds,
            )
            chunk_index = 0

            recorder: Optional[AnalysisRecorder] = None
            writer = VideoWriter(
                output_path, width, height, rate, input_path
            )
            track_ids = set()
            processed = 0

            try:
                if analysis_path is not None:
                    recorder = AnalysisRecorder(
                        analysis_path,
                        {
                            "type": "video",
                            "input": str(input_path),
                            "width": width,
                            "height": height,
                            "fps": float(meta.get("fps", 0.0)),
                            "total_frames": total_frames,
                            "duration": round(duration, 3),
                            "chunk_seconds": self._chunk_seconds,
                            "chunk_overlap_seconds": self._chunk_overlap_seconds,
                        },
                    )

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
                            detections, frame.frame_id, frame.image
                        )
                    else:
                        tracks = self._tracker.predict(frame.frame_id)

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
                    writer.write(rendered, frame.timestamp)

                    if recorder is not None:
                        recorder.add_frame(frame.frame_id, regions)
                        while (
                            chunk_index < len(chunks)
                            and frame.timestamp >= chunks[chunk_index].end
                        ):
                            recorder.flush()
                            chunk_index += 1

                    processed += 1
                    track_ids.update(track.track_id for track in tracks)
                    if progress is not None:
                        progress(processed, total_frames)
            finally:
                try:
                    writer.close()
                finally:
                    if recorder is not None:
                        recorder.close()

        return VideoResult(
            output_path=output_path,
            analysis_path=analysis_path,
            processed_frames=processed,
            track_count=len(track_ids),
            elapsed_seconds=time.perf_counter() - started,
        )

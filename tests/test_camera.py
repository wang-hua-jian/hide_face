"""P6 摄像头截图与实时遮挡测试（使用假摄像头，不依赖硬件）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pipeline.camera_pipeline import CameraPipeline, capture_snapshots
from pipeline.analysis.temporal import TemporalSmoother
from pipeline.render.mask_renderer import MaskRenderer
from pipeline.schema import FaceDetection
from pipeline.target.face_provider import FaceTargetProvider
from pipeline.tracker.bytetrack_tracker import ByteTrackTracker
from sources.base import SourceError
from sources.frame import FramePacket
from utils.image import write_image


class _FakeCamera:
    def __init__(self, frames: int = 5, width: int = 64, height: int = 64) -> None:
        self._frames = frames
        self._width = width
        self._height = height
        self._index = 0
        self.opened = False
        self.closed = False
        self.meta = {
            "index": 0,
            "width": width,
            "height": height,
            "camera_fps": 30.0,
        }

    def open(self) -> None:
        self.opened = True

    def read(self):
        if self._index >= self._frames:
            return None
        image = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        image[:, :, 1] = self._index * 10
        packet = FramePacket(
            frame_id=self._index,
            timestamp=self._index / 30.0,
            image=image,
        )
        self._index += 1
        return packet

    def close(self) -> None:
        self.closed = True

    def snapshot(self, path) -> FramePacket:
        frame = self.read()
        if frame is None:
            raise SourceError("没有可用画面")
        write_image(path, frame.image)
        return frame


class _FakeDetector:
    def detect(self, image, frame_id: int = 0):
        return [
            FaceDetection(
                bbox=(16.0, 16.0, 24.0, 24.0),
                landmarks=(
                    (22.0, 24.0),
                    (34.0, 24.0),
                    (28.0, 30.0),
                    (23.0, 36.0),
                    (33.0, 36.0),
                ),
                confidence=0.9,
                frame_id=frame_id,
            )
        ]


def _pipeline(
    max_fps: int = 1000,
    display_scale: float = 1.0,
) -> CameraPipeline:
    return CameraPipeline(
        detector=_FakeDetector(),
        tracker=ByteTrackTracker(
            high_score_threshold=0.7,
            low_score_threshold=0.1,
            iou_threshold=0.1,
            second_iou_threshold=0.2,
            max_lost_frames=30,
        ),
        smoother=TemporalSmoother(alpha=0.85, max_scale_step=0.3),
        target_provider=FaceTargetProvider(box_expand_ratio=0.0),
        renderer=MaskRenderer(
            style="solid",
            opacity=1.0,
            mosaic_block=8,
            blur_kernel=9,
        ),
        max_fps=max_fps,
        display_scale=display_scale,
        analysis_flush_seconds=125,
    )


def test_capture_snapshots_writes_unique_files(tmp_path: Path) -> None:
    source = _FakeCamera(frames=3)
    source.open()

    paths = capture_snapshots(source, tmp_path, count=3, interval_seconds=0)

    assert len(paths) == 3
    assert len(set(paths)) == 3
    assert all(path.is_file() for path in paths)
    source.close()


def test_camera_pipeline_records_video_and_analysis(tmp_path: Path) -> None:
    source = _FakeCamera(frames=5)
    source.open()
    record_path = tmp_path / "live.mp4"
    analysis_path = tmp_path / "live.json"

    result = _pipeline().run(
        source,
        record_path=record_path,
        analysis_path=analysis_path,
        show=False,
    )
    source.close()

    assert result.processed_frames == 5
    assert result.track_count == 1
    assert result.record_path == record_path
    assert result.analysis_path == analysis_path
    assert record_path.is_file()

    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert payload["media"]["type"] == "camera"
    assert len(payload["frames"]) == 5
    assert payload["frames"][0]["targets"][0]["track_id"] == 1
    assert source.closed is True


def test_display_scale_can_upscale_window() -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    scaled = _pipeline(display_scale=1.5)._scale_for_display(image)

    assert scaled.shape == (60, 90, 3)


def test_camera_pipeline_without_recording_or_analysis(tmp_path: Path) -> None:
    source = _FakeCamera(frames=3)
    source.open()

    result = _pipeline().run(source, show=False)
    source.close()

    assert result.processed_frames == 3
    assert result.record_path is None
    assert result.analysis_path is None
    assert not list(tmp_path.glob("*"))

"""P5 视频管线端到端测试（无音频合成视频）。"""

from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np

from pipeline.analysis.temporal import TemporalSmoother
from pipeline.render.mask_renderer import MaskRenderer
from pipeline.schema import FaceDetection
from pipeline.target.face_provider import FaceTargetProvider
from pipeline.tracker.bytetrack_tracker import ByteTrackTracker
from pipeline.video_pipeline import VideoPipeline


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


def _make_video(
    path: Path,
    width: int = 64,
    height: int = 64,
    fps: int = 10,
    frames: int = 12,
) -> None:
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    for index in range(frames):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, :, 1] = index * 10
        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _pipeline() -> VideoPipeline:
    return VideoPipeline(
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
        detect_interval=1,
        chunk_seconds=5,
        chunk_overlap_seconds=1,
    )


def test_video_pipeline_writes_video_and_analysis(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    analysis_path = tmp_path / "analysis.json"
    _make_video(input_path)

    result = _pipeline().run(
        input_path,
        output_path,
        analysis_path=analysis_path,
    )

    assert result.processed_frames == 12
    assert result.track_count == 1
    assert result.analysis_path == analysis_path
    assert output_path.is_file()

    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert payload["media"]["type"] == "video"
    assert payload["media"]["width"] == 64
    assert payload["media"]["height"] == 64
    assert len(payload["frames"]) == 12
    assert payload["frames"][0]["frame_id"] == 0
    assert payload["frames"][0]["targets"][0]["track_id"] == 1


def test_video_pipeline_without_analysis_writes_no_json(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    _make_video(input_path, frames=6)

    result = _pipeline().run(input_path, output_path, analysis_path=None)

    assert result.analysis_path is None
    assert output_path.is_file()
    assert not list(tmp_path.glob("*.json"))

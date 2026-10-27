"""P5 analysis.json 流式记录测试。"""

from __future__ import annotations

import json

import pytest

from pipeline.analysis.recorder import AnalysisRecorder, build_frame_record
from pipeline.schema import TargetRegion


def _region(track_id: int = 1, identity=None) -> TargetRegion:
    return TargetRegion(
        target_type="face",
        bbox=(10.0, 10.0, 20.0, 20.0),
        polygon=((10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)),
        confidence=0.9123,
        track_id=track_id,
        attributes={
            "detection_bbox": (12.0, 12.0, 16.0, 16.0),
            "identity": identity,
        },
    )


def test_frame_record_fields() -> None:
    record = build_frame_record(7, [_region(track_id=3, identity="self")])

    assert record["frame_id"] == 7
    target = record["targets"][0]
    assert target["track_id"] == 3
    assert target["bbox"] == [12.0, 12.0, 16.0, 16.0]
    assert target["confidence"] == pytest.approx(0.9123)
    assert target["identity"] == "self"
    assert target["target_region"]["bbox"] == [10.0, 10.0, 20.0, 20.0]


def test_recorder_writes_valid_streaming_json(tmp_path) -> None:
    path = tmp_path / "analysis.json"
    recorder = AnalysisRecorder(
        path,
        {"type": "video", "width": 64, "height": 64},
    )

    recorder.add_frame(0, [_region(track_id=1)])
    recorder.add_frame(1, [_region(track_id=2, identity="self")])
    recorder.flush()
    recorder.close()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["media"]["type"] == "video"
    assert [frame["frame_id"] for frame in payload["frames"]] == [0, 1]
    assert payload["frames"][0]["targets"][0]["identity"] is None
    assert payload["frames"][1]["targets"][0]["identity"] == "self"


def test_empty_recorder_is_valid_json(tmp_path) -> None:
    path = tmp_path / "empty.json"
    recorder = AnalysisRecorder(path, {"type": "video"})
    recorder.close()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["frames"] == []


def test_recorder_rejects_write_after_close(tmp_path) -> None:
    recorder = AnalysisRecorder(tmp_path / "closed.json", {"type": "video"})
    recorder.close()

    with pytest.raises(RuntimeError):
        recorder.add_frame(0, [])

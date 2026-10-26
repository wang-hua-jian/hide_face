"""P3 视频输入测试。"""

from __future__ import annotations

import pytest

from config.settings import PROJECT_ROOT
from sources.base import SourceError
from sources.video_source import VideoSource

VIDEO_PATH = PROJECT_ROOT / "data" / "videos" / "dance.mp4"


@pytest.fixture()
def video_path():
    if not VIDEO_PATH.is_file():
        pytest.skip(f"缺少测试视频：{VIDEO_PATH}")
    return VIDEO_PATH


def test_video_meta_and_first_frames(video_path) -> None:
    with VideoSource(video_path) as source:
        meta = source.meta

        assert meta["width"] > 0
        assert meta["height"] > 0
        assert meta["fps"] > 0
        assert meta["total_frames"] > 0
        assert meta["has_audio"] is True

        first = source.read()
        second = source.read()

        assert first is not None
        assert second is not None
        assert first.frame_id == 0
        assert second.frame_id == 1
        assert first.image.shape == (meta["height"], meta["width"], 3)
        assert first.timestamp >= 0.0


def test_missing_video_raises(tmp_path) -> None:
    with pytest.raises(SourceError, match="视频不存在"):
        VideoSource(tmp_path / "missing.mp4").open()


def test_read_before_open_raises(video_path) -> None:
    with pytest.raises(SourceError, match="请先调用 open"):
        VideoSource(video_path).read()

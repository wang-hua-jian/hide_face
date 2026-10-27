"""P5 长视频分块规划测试。"""

from __future__ import annotations

import pytest

from pipeline.analysis.chunks import plan_chunks


def test_plan_chunks_with_overlap() -> None:
    chunks = plan_chunks(300, 125, 5)

    assert [(c.start, c.end, c.overlap_start) for c in chunks] == pytest.approx(
        [
            (0.0, 125.0, 0.0),
            (125.0, 250.0, 120.0),
            (250.0, 300.0, 245.0),
        ]
    )


def test_short_video_single_chunk() -> None:
    chunks = plan_chunks(16.5, 125, 5)

    assert len(chunks) == 1
    assert chunks[0].start == 0.0
    assert chunks[0].end == pytest.approx(16.5)


def test_zero_duration_has_no_chunks() -> None:
    assert plan_chunks(0, 125, 5) == []


def test_overlap_must_be_smaller_than_chunk() -> None:
    with pytest.raises(ValueError):
        plan_chunks(300, 125, 125)


def test_invalid_chunk_values() -> None:
    with pytest.raises(ValueError):
        plan_chunks(300, 0, 0)
    with pytest.raises(ValueError):
        plan_chunks(300, 125, -1)

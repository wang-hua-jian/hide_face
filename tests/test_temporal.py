"""P3 时序平滑测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipeline.analysis.temporal import TemporalSmoother


def _track(
    track_id: int = 1,
    bbox: tuple = (0.0, 0.0, 100.0, 100.0),
    lost_frames: int = 0,
):
    return SimpleNamespace(track_id=track_id, bbox=bbox, lost_frames=lost_frames)


def test_first_observation_is_kept() -> None:
    smoother = TemporalSmoother(alpha=0.7, max_scale_step=0.2)

    boxes = smoother.update([_track(bbox=(10, 20, 30, 40))])

    assert boxes[1] == pytest.approx((10, 20, 30, 40))


def test_ema_smooths_position() -> None:
    smoother = TemporalSmoother(alpha=0.5, max_scale_step=1.0)
    smoother.update([_track(bbox=(0, 0, 100, 100))])

    boxes = smoother.update([_track(bbox=(20, 0, 100, 100))])

    assert boxes[1] == pytest.approx((10, 0, 100, 100))


def test_scale_change_is_clamped() -> None:
    smoother = TemporalSmoother(alpha=1.0, max_scale_step=0.2)
    smoother.update([_track(bbox=(0, 0, 100, 100))])

    boxes = smoother.update([_track(bbox=(0, 0, 200, 100))])

    assert boxes[1][2] == pytest.approx(120.0)
    assert boxes[1][0] == pytest.approx(40.0)


def test_lost_track_follows_predicted_motion() -> None:
    smoother = TemporalSmoother(alpha=0.7, max_scale_step=0.2)
    smoother.update([_track(bbox=(10, 10, 50, 50))])

    boxes = smoother.update(
        [_track(bbox=(500, 500, 50, 50), lost_frames=1)]
    )

    assert boxes[1] == pytest.approx((353, 353, 50, 50))


def test_invalid_alpha_rejected() -> None:
    with pytest.raises(ValueError):
        TemporalSmoother(alpha=1.5, max_scale_step=0.2)


def test_invalid_scale_step_rejected() -> None:
    with pytest.raises(ValueError):
        TemporalSmoother(alpha=0.7, max_scale_step=2.0)

"""动态眼距先验测试：中位数区间、冷启动、新 Track 准入与观测时机。"""

from __future__ import annotations

import pytest

from pipeline.schema import FaceDetection
from pipeline.tracker.bytetrack_tracker import ByteTrackTracker
from pipeline.tracker.face_geometry import FaceGeometryPrior, eye_ratio


def _det_with_ratio(
    bbox: tuple,
    ratio: float,
    confidence: float = 0.9,
) -> FaceDetection:
    x, y, width, height = bbox
    eye_distance = ratio * max(width, height)
    center_x = x + width / 2.0
    half = eye_distance / 2.0
    landmarks = (
        (center_x - half, y + height * 0.35),
        (center_x + half, y + height * 0.35),
        (center_x, y + height * 0.55),
        (center_x - half * 0.6, y + height * 0.75),
        (center_x + half * 0.6, y + height * 0.75),
    )
    return FaceDetection(
        bbox=bbox,
        landmarks=landmarks,
        confidence=confidence,
    )


def test_eye_ratio_is_scale_invariant() -> None:
    small = _det_with_ratio((0, 0, 40, 40), 0.3)
    large = _det_with_ratio((0, 0, 400, 400), 0.3)

    assert eye_ratio(small) == pytest.approx(0.3)
    assert eye_ratio(large) == pytest.approx(0.3)


def test_cold_start_uses_fallback() -> None:
    prior = FaceGeometryPrior(min_samples=20, fallback_min_ratio=0.12)

    assert prior.bounds == pytest.approx((0.12, 1.0))
    assert prior.is_plausible(_det_with_ratio((0, 0, 100, 100), 0.3)) is True
    assert prior.is_plausible(_det_with_ratio((0, 0, 100, 100), 0.05)) is False


def test_dynamic_bounds_from_confirmed_samples() -> None:
    prior = FaceGeometryPrior(
        min_samples=5,
        lower_factor=0.4,
        upper_factor=2.0,
    )
    for _ in range(10):
        prior.observe(_det_with_ratio((0, 0, 100, 100), 0.3))

    assert prior.sample_count == 10
    assert prior.median == pytest.approx(0.3)
    assert prior.bounds == pytest.approx((0.12, 0.6))
    assert prior.is_plausible(_det_with_ratio((0, 0, 100, 100), 0.093)) is False
    assert prior.is_plausible(_det_with_ratio((0, 0, 100, 100), 0.3)) is True
    assert prior.is_plausible(_det_with_ratio((0, 0, 100, 100), 0.8)) is False


def test_new_track_rejected_when_outside_dynamic_bounds() -> None:
    prior = FaceGeometryPrior(min_samples=5, lower_factor=0.4, upper_factor=2.0)
    for _ in range(10):
        prior.observe(_det_with_ratio((0, 0, 100, 100), 0.3))

    tracker = ByteTrackTracker(
        high_score_threshold=0.7,
        low_score_threshold=0.1,
        iou_threshold=0.1,
        second_iou_threshold=0.2,
        max_lost_frames=30,
        geometry_prior=prior,
    )

    rejected = tracker.update(
        [_det_with_ratio((100, 100, 100, 100), 0.05)],
        frame_id=0,
    )
    accepted = tracker.update(
        [_det_with_ratio((100, 100, 100, 100), 0.3)],
        frame_id=1,
    )

    assert rejected == []
    assert len(accepted) == 1


def test_prior_observes_only_position_matched_detections() -> None:
    prior = FaceGeometryPrior(min_samples=100, fallback_min_ratio=0.0)
    tracker = ByteTrackTracker(
        high_score_threshold=0.7,
        low_score_threshold=0.1,
        iou_threshold=0.1,
        second_iou_threshold=0.2,
        max_lost_frames=30,
        geometry_prior=prior,
    )

    tracker.update(
        [_det_with_ratio((10, 10, 40, 40), 0.3)],
        frame_id=0,
    )
    assert prior.sample_count == 0

    tracker.update(
        [_det_with_ratio((10, 10, 40, 40), 0.3)],
        frame_id=1,
    )
    assert prior.sample_count == 1

"""P1 目标区域生成测试：不加载任何检测模型。"""

from __future__ import annotations

import pytest

from pipeline.schema import FaceDetection
from pipeline.target.face_provider import FaceTargetProvider


def _detection(
    bbox: tuple = (40.0, 40.0, 20.0, 20.0),
    confidence: float = 0.9,
    frame_id: int = 5,
) -> FaceDetection:
    return FaceDetection(
        bbox=bbox,
        landmarks=((50.0, 55.0),) * 5,
        confidence=confidence,
        frame_id=frame_id,
    )


def test_expands_bbox_and_builds_polygon() -> None:
    provider = FaceTargetProvider(box_expand_ratio=0.15)

    regions = provider.from_detections([_detection()], (100, 100))

    assert len(regions) == 1
    region = regions[0]
    assert region.target_type == "face"
    assert region.bbox == pytest.approx((37, 37, 26, 26))
    assert region.polygon == ((37, 37), (63, 37), (63, 63), (37, 63))
    assert region.confidence == pytest.approx(0.9)
    assert region.track_id is None
    assert region.attributes["frame_id"] == 5


def test_expansion_is_clipped_at_border() -> None:
    provider = FaceTargetProvider(box_expand_ratio=0.5)

    regions = provider.from_detections(
        [_detection(bbox=(0.0, 0.0, 20.0, 20.0))],
        (100, 100),
    )

    assert regions[0].bbox == pytest.approx((0, 0, 30, 30))


def test_zero_size_detection_is_skipped() -> None:
    provider = FaceTargetProvider(box_expand_ratio=0.15)

    regions = provider.from_detections(
        [_detection(bbox=(10.0, 10.0, 0.0, 0.0))],
        (100, 100),
    )

    assert regions == []


def test_multiple_detections_keep_order() -> None:
    provider = FaceTargetProvider(box_expand_ratio=0.0)
    detections = [
        _detection(bbox=(10.0, 10.0, 10.0, 10.0), confidence=0.8),
        _detection(bbox=(50.0, 50.0, 10.0, 10.0), confidence=0.7),
    ]

    regions = provider.from_detections(detections, (100, 100))

    assert [region.confidence for region in regions] == pytest.approx([0.8, 0.7])
    assert regions[0].bbox == pytest.approx((10, 10, 10, 10))


def test_negative_ratio_shrinks_bbox() -> None:
    provider = FaceTargetProvider(box_expand_ratio=-0.2)

    regions = provider.from_detections([_detection()], (100, 100))

    assert regions[0].bbox == pytest.approx((44, 44, 12, 12))


def test_ratio_at_minus_half_rejected() -> None:
    with pytest.raises(ValueError):
        FaceTargetProvider(box_expand_ratio=-0.5)

"""P3 跟踪测试：卡尔曼预测、匈牙利匹配、ByteTrack 两阶段关联与生命周期。"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.schema import FaceDetection
from pipeline.tracker.bytetrack_tracker import ByteTrackTracker
from pipeline.tracker.matching import iou_matrix, linear_assignment


def _det(
    bbox: tuple,
    confidence: float = 0.9,
    frame_id: int = 0,
) -> FaceDetection:
    return FaceDetection(
        bbox=bbox,
        landmarks=(),
        confidence=confidence,
        frame_id=frame_id,
    )


def _tracker(
    max_lost_frames: int = 30,
    mask_lost_frames: int = 3,
) -> ByteTrackTracker:
    return ByteTrackTracker(
        high_score_threshold=0.6,
        low_score_threshold=0.1,
        iou_threshold=0.3,
        second_iou_threshold=0.5,
        max_lost_frames=max_lost_frames,
        mask_lost_frames=min(mask_lost_frames, max_lost_frames),
    )


def test_iou_matrix() -> None:
    matrix = iou_matrix(
        [(0, 0, 10, 10)],
        [(0, 0, 10, 10), (20, 20, 10, 10)],
    )

    assert matrix.shape == (1, 2)
    assert matrix[0, 0] == pytest.approx(1.0)
    assert matrix[0, 1] == pytest.approx(0.0)


def test_linear_assignment_respects_threshold() -> None:
    cost = np.array([[0.1, 0.9], [0.9, 0.2]])

    matches, unmatched_rows, unmatched_cols = linear_assignment(cost, 0.5)

    assert matches == [(0, 0), (1, 1)]
    assert unmatched_rows == []
    assert unmatched_cols == []


def test_linear_assignment_rejects_expensive_pairs() -> None:
    cost = np.array([[0.9]])

    matches, unmatched_rows, unmatched_cols = linear_assignment(cost, 0.5)

    assert matches == []
    assert unmatched_rows == [0]
    assert unmatched_cols == [0]


def test_first_association_keeps_track_id() -> None:
    tracker = _tracker()

    first = tracker.update([_det((10, 10, 40, 40))], frame_id=0)
    second = tracker.update([_det((12, 12, 40, 40))], frame_id=3)

    assert len(first) == 1
    assert len(second) == 1
    assert second[0].track_id == first[0].track_id
    assert second[0].status == "tracked"
    assert second[0].updated is True


def test_second_stage_uses_low_score_detection() -> None:
    tracker = _tracker()
    track_id = tracker.update([_det((10, 10, 40, 40))], frame_id=0)[0].track_id

    tracks = tracker.update(
        [_det((12, 12, 40, 40), confidence=0.3)],
        frame_id=3,
    )

    assert len(tracks) == 1
    assert tracks[0].track_id == track_id
    assert tracks[0].lost_frames == 0


def test_low_score_detection_alone_does_not_create_track() -> None:
    tracker = _tracker()

    tracks = tracker.update(
        [_det((10, 10, 40, 40), confidence=0.3)],
        frame_id=0,
    )

    assert tracks == []


def test_new_detections_get_unique_ids() -> None:
    tracker = _tracker()

    tracks = tracker.update(
        [
            _det((10, 10, 40, 40)),
            _det((200, 200, 40, 40)),
        ],
        frame_id=0,
    )

    assert len(tracks) == 2
    assert tracks[0].track_id != tracks[1].track_id


def test_track_expires_after_max_lost_frames() -> None:
    tracker = _tracker(max_lost_frames=2)
    tracker.update([_det((10, 10, 40, 40))], frame_id=0)

    assert len(tracker.update([], frame_id=1)) == 1
    assert len(tracker.update([], frame_id=2)) == 1
    assert tracker.update([], frame_id=3) == []


def test_frame_step_counts_lost_in_video_frames() -> None:
    tracker = ByteTrackTracker(
        high_score_threshold=0.6,
        low_score_threshold=0.1,
        iou_threshold=0.3,
        second_iou_threshold=0.5,
        max_lost_frames=30,
        frame_step=3,
    )
    tracker.update([_det((10, 10, 40, 40))], frame_id=0)

    for step in range(1, 11):
        tracks = tracker.update([], frame_id=step * 3)
        assert len(tracks) == 1
        assert tracks[0].lost_frames == step * 3

    assert tracker.update([], frame_id=33) == []


def test_no_detection_drops_mask_immediately() -> None:
    tracker = _tracker(max_lost_frames=30, mask_lost_frames=3)
    tracker.update([_det((10, 10, 40, 40))], frame_id=0)

    tracks = tracker.update([], frame_id=1)

    assert len(tracks) == 1
    assert tracks[0].mask_state is False
    assert tracks[0].lost_frames == 1


def test_nearby_low_detection_keeps_mask_during_grace() -> None:
    tracker = _tracker(max_lost_frames=30, mask_lost_frames=3)
    tracker.update([_det((10, 10, 40, 40))], frame_id=0)

    nearby = _det((60, 10, 20, 20), confidence=0.3)
    for frame_id in range(1, 4):
        tracks = tracker.update([nearby], frame_id=frame_id)
        assert tracks[0].mask_state is True
        assert tracks[0].lost_frames == frame_id

    tracks = tracker.update([nearby], frame_id=4)
    assert tracks[0].mask_state is False


def test_predict_does_not_reenable_expired_mask() -> None:
    tracker = _tracker(max_lost_frames=30, mask_lost_frames=1)
    tracker.update([_det((10, 10, 40, 40))], frame_id=0)
    tracker.update([], frame_id=1)
    tracker.update([], frame_id=2)

    for frame_id in range(3, 8):
        tracks = tracker.predict(frame_id)
        assert tracks[0].mask_state is False


def test_mask_lost_longer_than_track_life_rejected() -> None:
    with pytest.raises(ValueError):
        ByteTrackTracker(
            high_score_threshold=0.6,
            low_score_threshold=0.1,
            iou_threshold=0.3,
            second_iou_threshold=0.5,
            max_lost_frames=2,
            mask_lost_frames=5,
        )


def test_predict_does_not_increment_lost_frames() -> None:
    tracker = _tracker(max_lost_frames=1)
    tracker.update([_det((10, 10, 40, 40))], frame_id=0)

    for frame_id in range(1, 10):
        tracks = tracker.predict(frame_id)
        assert len(tracks) == 1
        assert tracks[0].lost_frames == 0


def test_measurement_bbox_matches_detection_on_update() -> None:
    tracker = _tracker()

    tracks = tracker.update([_det((10, 10, 40, 40))], frame_id=0)

    assert tracks[0].measurement_bbox == pytest.approx((10, 10, 40, 40))
    assert tracks[0].updated is True


def test_low_iou_threshold_keeps_id_on_vertical_jump() -> None:
    tracker = ByteTrackTracker(
        high_score_threshold=0.6,
        low_score_threshold=0.1,
        iou_threshold=0.1,
        second_iou_threshold=0.2,
        max_lost_frames=30,
    )
    track_id = tracker.update(
        [_det((10, 10, 40, 80))],
        frame_id=0,
    )[0].track_id

    tracks = tracker.update(
        [_det((10, 70, 40, 80))],
        frame_id=1,
    )

    assert len(tracks) == 1
    assert tracks[0].track_id == track_id


def test_brief_loss_keeps_same_id() -> None:
    tracker = _tracker()
    track_id = tracker.update([_det((10, 10, 40, 40))], frame_id=0)[0].track_id

    tracker.update([], frame_id=1)
    tracks = tracker.update([_det((14, 12, 40, 40))], frame_id=2)

    assert len(tracks) == 1
    assert tracks[0].track_id == track_id
    assert tracks[0].lost_frames == 0


def test_kalman_predicts_continuing_motion() -> None:
    tracker = _tracker()
    for index, frame_id in enumerate((0, 3, 6, 9)):
        x = 10.0 + index * 10.0
        tracks = tracker.update([_det((x, 10, 40, 40))], frame_id=frame_id)

    last_x = tracks[0].bbox[0]
    predicted = tracker.predict(frame_id=12)[0]

    assert predicted.bbox[0] > last_x

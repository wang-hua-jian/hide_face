"""P4 外观兜底测试：触发条件、匈牙利匹配、EMA 与质量门槛。"""

from __future__ import annotations

import numpy as np
import pytest

from config.settings import (
    PROJECT_ROOT,
    ConfigError,
    Settings,
    ensure_models,
)
from pipeline.detector.yunet_detector import YuNetDetector
from pipeline.embedder.base import FaceEmbedder
from pipeline.embedder.sface_embedder import SFaceEmbedder
from pipeline.schema import FaceDetection
from pipeline.tracker.appearance_matcher import AppearanceMatcher
from pipeline.tracker.bytetrack_tracker import ByteTrackTracker
from utils.image import read_image
from utils.math import l2_normalize


class _FakeEmbedder(FaceEmbedder):
    def __init__(self, vectors, fail: bool = False) -> None:
        self._vectors = list(vectors)
        self._fail = fail
        self.calls = 0

    def embed(
        self,
        image: np.ndarray,
        detection: FaceDetection,
    ) -> np.ndarray:
        self.calls += 1
        if self._fail:
            raise RuntimeError("fake embedder failure")
        vector = self._vectors.pop(0)
        return l2_normalize(np.asarray(vector, dtype=np.float32))


def _det(bbox: tuple, confidence: float = 0.9, frame_id: int = 0) -> FaceDetection:
    x, y, width, height = bbox
    landmarks = (
        (x + width * 0.3, y + height * 0.35),
        (x + width * 0.7, y + height * 0.35),
        (x + width * 0.5, y + height * 0.55),
        (x + width * 0.35, y + height * 0.75),
        (x + width * 0.65, y + height * 0.75),
    )
    return FaceDetection(
        bbox=bbox,
        landmarks=landmarks,
        confidence=confidence,
        frame_id=frame_id,
    )


def _matcher(
    embedder: FaceEmbedder,
    threshold: float = 0.6,
    min_face_size: int = 10,
) -> AppearanceMatcher:
    return AppearanceMatcher(
        embedder=embedder,
        similarity_threshold=threshold,
        ema_alpha=0.9,
        min_face_size=min_face_size,
    )


def _tracker(matcher: AppearanceMatcher) -> ByteTrackTracker:
    return ByteTrackTracker(
        high_score_threshold=0.7,
        low_score_threshold=0.1,
        iou_threshold=0.1,
        second_iou_threshold=0.2,
        max_lost_frames=30,
        mask_lost_frames=4,
        appearance_matcher=matcher,
    )


def _image() -> np.ndarray:
    return np.zeros((200, 200, 3), dtype=np.uint8)


def test_position_match_does_not_call_sface() -> None:
    embedder = _FakeEmbedder([[1.0, 0.0, 0.0]])
    matcher = _matcher(embedder)
    tracker = _tracker(matcher)
    image = _image()

    tracker.update([_det((10, 10, 40, 40))], frame_id=0, image=image)
    assert matcher.calls == 1
    assert embedder.calls == 1

    tracker.update([_det((12, 12, 40, 40))], frame_id=1, image=image)
    tracker.predict(frame_id=2)

    assert matcher.calls == 1
    assert embedder.calls == 1


def test_no_image_never_calls_sface() -> None:
    embedder = _FakeEmbedder([[1.0, 0.0, 0.0]])
    matcher = _matcher(embedder)
    tracker = _tracker(matcher)

    tracks = tracker.update([_det((10, 10, 40, 40))], frame_id=0)

    assert len(tracks) == 1
    assert matcher.calls == 0
    assert tracks[0].appearance_ema is None


def test_appearance_reassociates_failed_position_match() -> None:
    embedder = _FakeEmbedder(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
        ]
    )
    matcher = _matcher(embedder)
    tracker = _tracker(matcher)
    image = _image()

    first = tracker.update([_det((10, 10, 40, 40))], frame_id=0, image=image)
    track_id = first[0].track_id

    tracks = tracker.update(
        [_det((150, 150, 40, 40))],
        frame_id=1,
        image=image,
    )

    assert len(tracks) == 1
    assert tracks[0].track_id == track_id
    assert tracks[0].appearance_updates == 2
    assert np.linalg.norm(tracks[0].appearance_ema) == pytest.approx(1.0)
    assert matcher.matches == 1


def test_dissimilar_embedding_creates_new_track() -> None:
    embedder = _FakeEmbedder(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    tracker = _tracker(_matcher(embedder))
    image = _image()

    first = tracker.update([_det((10, 10, 40, 40))], frame_id=0, image=image)
    tracks = tracker.update(
        [_det((150, 150, 40, 40))],
        frame_id=1,
        image=image,
    )

    assert len(tracks) == 2
    assert tracks[0].track_id == first[0].track_id
    assert tracks[1].track_id != first[0].track_id


def test_one_to_one_assignment_uses_embedding() -> None:
    embedder = _FakeEmbedder(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    matcher = _matcher(embedder)
    tracker = _tracker(matcher)
    image = _image()

    initial = tracker.update(
        [
            _det((10, 10, 40, 40)),
            _det((150, 10, 40, 40)),
        ],
        frame_id=0,
        image=image,
    )
    assert [track.track_id for track in initial] == [1, 2]

    tracks = tracker.update(
        [
            _det((150, 150, 40, 40)),
            _det((10, 150, 40, 40)),
        ],
        frame_id=1,
        image=image,
    )

    assert len(tracks) == 2
    assert tracks[0].measurement_bbox == pytest.approx((10, 150, 40, 40))
    assert tracks[1].measurement_bbox == pytest.approx((150, 150, 40, 40))
    assert matcher.matches == 2


def test_small_face_skips_sface() -> None:
    embedder = _FakeEmbedder([[1.0, 0.0, 0.0]])
    matcher = _matcher(embedder, min_face_size=24)
    tracker = _tracker(matcher)

    tracks = tracker.update(
        [_det((10, 10, 10, 10))],
        frame_id=0,
        image=_image(),
    )

    assert len(tracks) == 1
    assert matcher.calls == 0
    assert tracks[0].appearance_ema is None


def test_embedder_failure_is_contained() -> None:
    embedder = _FakeEmbedder([[1.0, 0.0, 0.0]], fail=True)
    matcher = _matcher(embedder)
    tracker = _tracker(matcher)

    tracks = tracker.update(
        [_det((10, 10, 40, 40))],
        frame_id=0,
        image=_image(),
    )

    assert len(tracks) == 1
    assert matcher.calls == 1
    assert tracks[0].appearance_ema is None


def test_low_score_detection_does_not_trigger_sface() -> None:
    embedder = _FakeEmbedder([[1.0, 0.0, 0.0]])
    matcher = _matcher(embedder)
    tracker = _tracker(matcher)

    tracks = tracker.update(
        [_det((10, 10, 40, 40), confidence=0.4)],
        frame_id=0,
        image=_image(),
    )

    assert tracks == []
    assert matcher.calls == 0


def test_ema_update_keeps_unit_norm() -> None:
    embedder = _FakeEmbedder(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
        ]
    )
    matcher = _matcher(embedder)
    tracker = _tracker(matcher)
    image = _image()

    tracker.update([_det((10, 10, 40, 40))], frame_id=0, image=image)
    tracks = tracker.update(
        [_det((150, 150, 40, 40))],
        frame_id=1,
        image=image,
    )

    assert np.linalg.norm(tracks[0].appearance_ema) == pytest.approx(1.0)
    assert tracks[0].appearance_ema[0] > tracks[0].appearance_ema[1]


def test_long_lost_track_is_not_used_for_appearance() -> None:
    embedder = _FakeEmbedder(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    tracker = _tracker(_matcher(embedder))
    image = _image()

    first = tracker.update([_det((10, 10, 40, 40))], frame_id=0, image=image)
    for frame_id in range(1, 7):
        tracker.update([], frame_id=frame_id)

    tracks = tracker.update(
        [_det((150, 150, 40, 40))],
        frame_id=7,
        image=image,
    )

    assert len(tracks) == 2
    assert tracks[0].track_id == first[0].track_id
    assert tracks[1].track_id != first[0].track_id


def test_implausible_eye_distance_skips_sface() -> None:
    embedder = _FakeEmbedder([[1.0, 0.0, 0.0]])
    matcher = _matcher(embedder)
    tracker = _tracker(matcher)
    detection = FaceDetection(
        bbox=(10, 10, 40, 80),
        landmarks=(
            (20.0, 30.0),
            (22.0, 30.0),
            (21.0, 50.0),
            (20.0, 65.0),
            (22.0, 65.0),
        ),
        confidence=0.9,
    )

    tracks = tracker.update([detection], frame_id=0, image=_image())

    assert len(tracks) == 1
    assert matcher.calls == 0
    assert tracks[0].appearance_ema is None


def test_sface_embedding_on_reference_photo() -> None:
    try:
        settings = Settings.load()
        ensure_models(settings)
    except ConfigError as exc:
        pytest.skip(f"配置或模型不可用：{exc}")

    reference = PROJECT_ROOT / "data" / "references" / "dance_01.png"
    if not reference.is_file():
        pytest.skip(f"缺少参考图：{reference}")

    image = read_image(reference)
    detector = YuNetDetector(
        model_path=settings.models.yunet_path,
        score_threshold=settings.detector.score_threshold,
        nms_threshold=settings.detector.nms_threshold,
        max_input_size=settings.detector.max_input_size,
    )
    detections = detector.detect(image)
    if not detections:
        pytest.skip("参考图中未检出人脸")

    embedder = SFaceEmbedder(settings.models.sface_path)
    embedding = embedder.embed(image, detections[0])

    assert embedding.shape == (128,)
    assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-4)

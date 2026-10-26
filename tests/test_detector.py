"""P1 检测与图片输入测试。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from config.settings import PROJECT_ROOT, ConfigError, Settings, ensure_models
from pipeline.detector.yunet_detector import YuNetDetector, _parse_rows
from sources.base import SourceError
from sources.image_source import ImageSource
from utils.image import write_image


@pytest.fixture(scope="module")
def settings() -> Settings:
    try:
        loaded = Settings.load()
        ensure_models(loaded)
    except ConfigError as exc:
        pytest.skip(f"配置或模型不可用：{exc}")
    return loaded


@pytest.fixture(scope="module")
def detector(settings: Settings) -> YuNetDetector:
    return YuNetDetector(
        model_path=settings.models.yunet_path,
        score_threshold=settings.detector.score_threshold,
        nms_threshold=settings.detector.nms_threshold,
        max_input_size=settings.detector.max_input_size,
    )


@pytest.fixture()
def sample_path() -> Path:
    path = PROJECT_ROOT / "data" / "images" / "test.png"
    if not path.is_file():
        pytest.skip(f"缺少测试图片：{path}")
    return path


def test_parse_rows_maps_to_original_and_filters() -> None:
    rows = np.array(
        [
            [10, 20, 30, 40, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0.91],
            [0, 0, 5, 5, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0.20],
        ],
        dtype=np.float32,
    )

    detections = _parse_rows(
        rows,
        src_w=1280,
        src_h=960,
        det_w=640,
        det_h=480,
        frame_id=7,
        score_threshold=0.6,
    )

    assert len(detections) == 1
    assert detections[0].bbox == pytest.approx((20, 40, 60, 80))
    assert detections[0].landmarks[0] == pytest.approx((2, 4))
    assert len(detections[0].landmarks) == 5
    assert detections[0].confidence == pytest.approx(0.91)
    assert detections[0].frame_id == 7


def test_parse_rows_empty_input() -> None:
    assert _parse_rows(None, 100, 100, 50, 50, 0, 0.6) == []
    assert _parse_rows(np.empty((0, 15)), 100, 100, 50, 50, 0, 0.6) == []


def test_detect_finds_face(
    sample_path: Path, detector: YuNetDetector, settings: Settings
) -> None:
    with ImageSource(sample_path) as source:
        frame = source.read()

    assert frame is not None
    detections = detector.detect(frame.image, frame_id=3)

    assert len(detections) >= 1
    for detection in detections:
        x, y, w, h = detection.bbox
        assert w > 0 and h > 0
        assert -2 <= x <= frame.width + 2
        assert -2 <= y <= frame.height + 2
        assert len(detection.landmarks) == 5
        assert detection.confidence >= settings.detector.score_threshold
        assert detection.frame_id == 3


def test_detect_accepts_grayscale(
    sample_path: Path, detector: YuNetDetector
) -> None:
    image = cv2.imread(str(sample_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detections = detector.detect(gray)

    assert len(detections) >= 1


def test_detect_blank_image_returns_empty(detector: YuNetDetector) -> None:
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    assert detector.detect(blank) == []


def test_detect_tiny_image_returns_empty(detector: YuNetDetector) -> None:
    tiny = np.zeros((8, 8, 3), dtype=np.uint8)
    assert detector.detect(tiny) == []


def test_detect_alpha_image_does_not_crash(detector: YuNetDetector) -> None:
    bgra = np.zeros((64, 64, 4), dtype=np.uint8)
    bgra[..., 3] = 255
    assert detector.detect(bgra) == []


def test_image_source_reads_once(tmp_path: Path) -> None:
    image = np.zeros((30, 40, 3), dtype=np.uint8)
    path = tmp_path / "single.png"
    write_image(path, image)

    with ImageSource(path) as source:
        frame = source.read()
        assert frame is not None
        assert frame.image.shape == (30, 40, 3)
        assert frame.frame_id == 0
        assert frame.eof is False
        assert frame.width == 40 and frame.height == 30
        assert source.meta["width"] == 40
        assert source.read() is None


def test_image_source_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SourceError, match="图片不存在"):
        ImageSource(tmp_path / "missing.png").open()


def test_image_source_rejects_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"not-an-image")
    with pytest.raises(SourceError):
        ImageSource(path).open()


def test_image_source_converts_grayscale(tmp_path: Path) -> None:
    gray = np.zeros((20, 30), dtype=np.uint8)
    path = tmp_path / "gray.png"
    write_image(path, gray)

    with ImageSource(path) as source:
        frame = source.read()

    assert frame is not None
    assert frame.image.shape == (20, 30, 3)


def test_image_source_converts_bgra(tmp_path: Path) -> None:
    bgra = np.zeros((20, 30, 4), dtype=np.uint8)
    bgra[..., 3] = 255
    path = tmp_path / "alpha.png"
    write_image(path, bgra)

    with ImageSource(path) as source:
        frame = source.read()

    assert frame is not None
    assert frame.image.shape == (20, 30, 3)


def test_image_source_supports_unicode_path(tmp_path: Path) -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    path = tmp_path / "中文图片.png"
    write_image(path, image)

    with ImageSource(path) as source:
        frame = source.read()

    assert frame is not None
    assert frame.image.shape == (20, 30, 3)

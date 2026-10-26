"""P0 工具测试：坐标映射、bbox 处理、向量运算、图像裁剪、时间戳。"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from utils.geometry import (
    clip_bbox,
    expand_bbox,
    fit_size,
    map_bbox_to_original,
    map_points_to_original,
)
from utils.image import crop, paste_with_alpha, resize_max_side
from utils.math import cosine_similarity, ema_update, l2_normalize
from utils.timefmt import camera_filename, format_timestamp, live_filename


def test_fit_size_keeps_aspect_ratio() -> None:
    assert fit_size(4000, 3000, 1280) == (1280, 960)
    assert fit_size(3000, 4000, 1280) == (960, 1280)


def test_fit_size_does_not_upscale() -> None:
    assert fit_size(100, 80, 1280) == (100, 80)


def test_fit_size_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        fit_size(0, 100, 1280)
    with pytest.raises(ValueError):
        fit_size(100, 100, 0)


def test_map_bbox_back_to_original() -> None:
    bbox = map_bbox_to_original((10, 20, 30, 40), 1280, 960, 640, 480)
    assert bbox == pytest.approx((20, 40, 60, 80))


def test_map_points_back_to_original() -> None:
    points = map_points_to_original([(10, 20), (30, 40)], 1280, 960, 640, 480)
    assert points == pytest.approx([(20, 40), (60, 80)])


def test_expand_bbox() -> None:
    assert expand_bbox((100, 100, 50, 50), 0.2, 500, 500) == pytest.approx(
        (90, 90, 70, 70)
    )


def test_expand_bbox_is_clipped() -> None:
    assert expand_bbox((100, 100, 50, 50), 0.2, 120, 120) == pytest.approx(
        (90, 90, 30, 30)
    )


def test_expand_bbox_supports_shrinking() -> None:
    assert expand_bbox((10, 10, 40, 40), -0.25, 100, 100) == pytest.approx(
        (20, 20, 20, 20)
    )


def test_expand_bbox_rejects_ratio_at_minus_half() -> None:
    with pytest.raises(ValueError):
        expand_bbox((10, 10, 10, 10), -0.5, 100, 100)


def test_clip_bbox() -> None:
    assert clip_bbox((-10, -10, 30, 30), 100, 100) == pytest.approx((0, 0, 20, 20))
    assert clip_bbox((90, 90, 30, 30), 100, 100) == pytest.approx((90, 90, 10, 10))


def test_l2_normalize() -> None:
    vector = l2_normalize(np.array([3.0, 4.0]))
    assert np.linalg.norm(vector) == pytest.approx(1.0)
    assert vector == pytest.approx(np.array([0.6, 0.8]))


def test_l2_normalize_zero_vector_is_safe() -> None:
    vector = l2_normalize(np.zeros(5))
    assert np.all(vector == 0)
    assert not np.isnan(vector).any()


def test_cosine_similarity_range() -> None:
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector() -> None:
    assert cosine_similarity([0, 0], [1, 0]) == pytest.approx(0.0)


def test_cosine_similarity_dimension_mismatch() -> None:
    with pytest.raises(ValueError):
        cosine_similarity([1, 0], [1, 0, 0])


def test_ema_update_is_normalized() -> None:
    old = np.array([1.0, 0.0], dtype=np.float32)
    new = np.array([0.0, 1.0], dtype=np.float32)

    updated = ema_update(old, new, 0.9)

    assert np.linalg.norm(updated) == pytest.approx(1.0)
    assert updated[0] > updated[1]


def test_ema_update_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError):
        ema_update([1, 0], [0, 1], 1.5)


def test_crop_uses_padding() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cropped = crop(image, (10, 10, 20, 20), padding_ratio=0.1)
    assert cropped.shape == (24, 24, 3)


def test_crop_clips_out_of_bounds() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cropped = crop(image, (95, 95, 20, 20))
    assert cropped.shape == (5, 5, 3)


def test_crop_rejects_empty_region() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        crop(image, (0, 0, 0, 0))


def test_resize_max_side() -> None:
    image = np.zeros((3000, 4000, 3), dtype=np.uint8)
    resized = resize_max_side(image, 1280)
    assert resized.shape == (960, 1280, 3)


def test_resize_max_side_keeps_small_image() -> None:
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    resized = resize_max_side(image, 1280)
    assert resized.shape == image.shape


def test_paste_with_alpha() -> None:
    background = np.full((10, 10, 3), 255, dtype=np.uint8)
    overlay = np.zeros((4, 4, 4), dtype=np.uint8)
    overlay[..., 0] = 0
    overlay[..., 1] = 0
    overlay[..., 2] = 255
    overlay[..., 3] = 255

    result = paste_with_alpha(background, overlay, 2, 2)

    assert result.shape == background.shape
    assert result[2, 2].tolist() == [0, 0, 255]
    assert result[0, 0].tolist() == [255, 255, 255]


def test_paste_with_alpha_out_of_bounds_is_noop() -> None:
    background = np.full((10, 10, 3), 255, dtype=np.uint8)
    overlay = np.zeros((4, 4, 4), dtype=np.uint8)

    result = paste_with_alpha(background, overlay, 100, 100)

    assert np.array_equal(result, background)


def test_format_timestamp_uses_milliseconds() -> None:
    moment = datetime(2026, 10, 25, 15, 30, 12, 871000)
    assert format_timestamp(moment) == "20261025_153012_871"


def test_camera_and_live_filenames() -> None:
    moment = datetime(2026, 10, 25, 15, 30, 12, 871000)
    assert camera_filename(moment) == "camera_20261025_153012_871.jpg"
    assert live_filename(moment) == "live_20261025_153012_871.mp4"

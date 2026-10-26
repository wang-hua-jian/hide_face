"""P2 渲染与图片管线测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.analysis.recorder import build_frame_record
from pipeline.image_pipeline import ImagePipeline
from pipeline.render.mask_renderer import MaskRenderer
from pipeline.render.styles import mosaic
from pipeline.schema import FaceDetection, TargetRegion
from pipeline.target.face_provider import FaceTargetProvider
from sources.base import SourceError
from utils.image import read_image, write_image


def _frame(width: int = 160, height: int = 120) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def _region(
    bbox: tuple = (40.0, 30.0, 50.0, 40.0),
    polygon: tuple = None,
) -> TargetRegion:
    return TargetRegion(
        target_type="face",
        bbox=bbox,
        polygon=polygon,
        confidence=0.9,
        track_id=1,
    )


def _inside_mask(shape: tuple, bbox: tuple) -> np.ndarray:
    x, y, w, h = (int(round(value)) for value in bbox)
    mask = np.zeros(shape[:2], dtype=bool)
    mask[y:y + h, x:x + w] = True
    return mask


@pytest.mark.parametrize("style", ["mosaic", "blur", "solid"])
def test_styles_change_only_inside_region(style: str) -> None:
    frame = _frame()
    before = frame.copy()
    renderer = MaskRenderer(
        style=style,
        opacity=1.0,
        mosaic_block=8,
        blur_kernel=9,
    )

    result = renderer.draw(frame, [_region()])

    assert result is frame
    inside = _inside_mask(before.shape, (40, 30, 50, 40))
    assert not np.array_equal(result[inside], before[inside])
    assert np.array_equal(result[~inside], before[~inside])


@pytest.mark.parametrize("style", ["mosaic", "blur", "solid"])
def test_opacity_zero_keeps_original(style: str) -> None:
    frame = _frame()
    before = frame.copy()
    renderer = MaskRenderer(
        style=style,
        opacity=0.0,
        mosaic_block=8,
        blur_kernel=9,
    )

    result = renderer.draw(frame, [_region()])

    assert np.array_equal(result, before)


def test_opacity_blends_partially() -> None:
    frame = np.full((60, 60, 3), 200, dtype=np.uint8)
    renderer = MaskRenderer(
        style="solid",
        opacity=0.5,
        mosaic_block=8,
        blur_kernel=9,
    )

    result = renderer.draw(frame, [_region(bbox=(10, 10, 40, 40))])

    assert result[30, 30].tolist() == [100, 100, 100]


def test_solid_fills_region_with_color() -> None:
    frame = _frame()
    renderer = MaskRenderer(
        style="solid",
        opacity=1.0,
        mosaic_block=8,
        blur_kernel=9,
        solid_color=(0, 0, 255),
    )

    result = renderer.draw(frame, [_region()])

    region = result[30:70, 40:90]
    assert np.all(region == np.array([0, 0, 255], dtype=np.uint8))


def test_custom_image_is_scaled_to_region(tmp_path: Path) -> None:
    mask = np.zeros((20, 20, 4), dtype=np.uint8)
    mask[..., 2] = 255
    mask[..., 3] = 255
    mask_path = tmp_path / "mask.png"
    write_image(mask_path, mask)

    frame = _frame()
    renderer = MaskRenderer(
        style="custom_image",
        opacity=1.0,
        mosaic_block=8,
        blur_kernel=9,
        mask_path=str(mask_path),
    )

    result = renderer.draw(frame, [_region()])

    assert result[50, 65].tolist() == [0, 0, 255]


def test_custom_image_alpha_zero_keeps_original(tmp_path: Path) -> None:
    mask = np.zeros((20, 20, 4), dtype=np.uint8)
    mask[..., 0] = 255
    mask[..., 3] = 0
    mask_path = tmp_path / "transparent.png"
    write_image(mask_path, mask)

    frame = _frame()
    before = frame.copy()
    renderer = MaskRenderer(
        style="custom_image",
        opacity=1.0,
        mosaic_block=8,
        blur_kernel=9,
        mask_path=str(mask_path),
    )

    result = renderer.draw(frame, [_region()])

    assert np.array_equal(result, before)


def test_polygon_limits_masking() -> None:
    frame = _frame(width=200, height=200)
    before = frame.copy()
    polygon = ((40.0, 30.0), (100.0, 30.0), (40.0, 90.0))
    renderer = MaskRenderer(
        style="solid",
        opacity=1.0,
        mosaic_block=8,
        blur_kernel=9,
    )

    result = renderer.draw(
        frame,
        [_region(bbox=(40, 30, 60, 60), polygon=polygon)],
    )

    assert result[35, 45].tolist() == [0, 0, 0]
    assert np.array_equal(result[80, 95], before[80, 95])


def test_ellipse_shape_skips_bbox_corners() -> None:
    frame = _frame(width=200, height=200)
    before = frame.copy()
    renderer = MaskRenderer(
        style="solid",
        opacity=1.0,
        mosaic_block=8,
        blur_kernel=9,
        mask_shape="ellipse",
    )

    result = renderer.draw(frame, [_region(bbox=(40, 40, 100, 100))])

    assert result[90, 90].tolist() == [0, 0, 0]
    assert np.array_equal(result[45, 45], before[45, 45])
    assert np.array_equal(result[40, 40], before[40, 40])


def test_invalid_mask_shape_raises() -> None:
    with pytest.raises(ValueError, match="未知遮挡形状"):
        MaskRenderer(
            style="solid",
            opacity=1.0,
            mosaic_block=8,
            blur_kernel=9,
            mask_shape="triangle",
        )


def test_zero_size_region_is_ignored() -> None:
    frame = _frame()
    before = frame.copy()
    renderer = MaskRenderer(
        style="solid",
        opacity=1.0,
        mosaic_block=8,
        blur_kernel=9,
    )

    result = renderer.draw(frame, [_region(bbox=(10, 10, 0, 0))])

    assert np.array_equal(result, before)


def test_missing_mask_path_raises() -> None:
    with pytest.raises(ValueError, match="custom_image"):
        MaskRenderer(
            style="custom_image",
            opacity=1.0,
            mosaic_block=8,
            blur_kernel=9,
        )


def test_invalid_opacity_raises() -> None:
    with pytest.raises(ValueError, match="opacity"):
        MaskRenderer(
            style="solid",
            opacity=1.2,
            mosaic_block=8,
            blur_kernel=9,
        )


def test_invalid_style_raises() -> None:
    with pytest.raises(ValueError, match="未知遮挡样式"):
        MaskRenderer(
            style="pixelate",
            opacity=1.0,
            mosaic_block=8,
            blur_kernel=9,
        )


def test_mosaic_rejects_invalid_block() -> None:
    with pytest.raises(ValueError):
        mosaic(_frame(), 1)


class _FakeDetector:
    def detect(self, image: np.ndarray, frame_id: int = 0):
        return [
            FaceDetection(
                bbox=(20.0, 20.0, 30.0, 30.0),
                landmarks=((30.0, 30.0),) * 5,
                confidence=0.9123,
                frame_id=frame_id,
            )
        ]


def _pipeline() -> ImagePipeline:
    return ImagePipeline(
        detector=_FakeDetector(),
        target_provider=FaceTargetProvider(box_expand_ratio=0.1),
        renderer=MaskRenderer(
            style="solid",
            opacity=1.0,
            mosaic_block=8,
            blur_kernel=9,
        ),
        jpeg_quality=95,
    )


def test_image_pipeline_writes_output_and_analysis(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.jpg"
    analysis_path = tmp_path / "analysis.json"
    write_image(input_path, np.zeros((100, 120, 3), dtype=np.uint8))

    result = _pipeline().run(input_path, output_path, analysis_path)

    assert result.output_path == output_path
    assert result.face_count == 1
    assert result.analysis_path == analysis_path
    assert output_path.is_file()
    assert read_image(output_path).shape == (100, 120, 3)
    assert read_image(output_path)[35, 35].max() <= 12

    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert payload["media"]["width"] == 120
    assert payload["media"]["height"] == 100
    target = payload["frames"][0]["targets"][0]
    assert target["track_id"] is None
    assert target["bbox"] == pytest.approx([20.0, 20.0, 30.0, 30.0])
    assert target["confidence"] == pytest.approx(0.9123)
    assert target["identity"] is None
    assert target["target_region"]["bbox"] == pytest.approx([17.0, 17.0, 36.0, 36.0])
    assert len(target["target_region"]["polygon"]) == 4


def test_image_pipeline_without_analysis(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.jpg"
    write_image(input_path, np.zeros((60, 80, 3), dtype=np.uint8))

    result = _pipeline().run(input_path, output_path, analysis_path=None)

    assert result.analysis_path is None
    assert output_path.is_file()
    assert not list(tmp_path.glob("*.json"))


def test_image_pipeline_missing_input(tmp_path: Path) -> None:
    with pytest.raises(SourceError):
        _pipeline().run(
            tmp_path / "missing.png",
            tmp_path / "output.jpg",
            None,
        )


def test_frame_record_uses_detection_bbox() -> None:
    region = TargetRegion(
        target_type="face",
        bbox=(17.0, 17.0, 36.0, 36.0),
        polygon=((17.0, 17.0), (53.0, 17.0), (53.0, 53.0), (17.0, 53.0)),
        confidence=0.9,
        track_id=None,
        attributes={"detection_bbox": (20.0, 20.0, 30.0, 30.0)},
    )

    record = build_frame_record(frame_id=0, regions=[region])

    assert record["frame_id"] == 0
    assert record["targets"][0]["bbox"] == [20.0, 20.0, 30.0, 30.0]
    assert record["targets"][0]["target_region"]["bbox"] == [17.0, 17.0, 36.0, 36.0]

"""几何工具：缩放尺寸、坐标回映射、bbox 扩张与裁边。"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

BBox = Tuple[float, float, float, float]


def fit_size(src_w: int, src_h: int, max_size: int) -> Tuple[int, int]:
    """等比缩放到最长边不超过 max_size；小图不放大。"""

    _require_positive(src_w=src_w, src_h=src_h, max_size=max_size)
    if max(src_w, src_h) <= max_size:
        return src_w, src_h
    scale = max_size / float(max(src_w, src_h))
    return max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))


def map_bbox_to_original(
    bbox: Sequence[float],
    src_w: int,
    src_h: int,
    det_w: int,
    det_h: int,
) -> BBox:
    """把检测图上的 bbox 映射回原图坐标。"""

    _require_positive(src_w=src_w, src_h=src_h, det_w=det_w, det_h=det_h)
    _require_bbox(bbox)
    scale_x = src_w / float(det_w)
    scale_y = src_h / float(det_h)
    x, y, w, h = bbox
    return (x * scale_x, y * scale_y, w * scale_x, h * scale_y)


def map_points_to_original(
    points: Iterable[Sequence[float]],
    src_w: int,
    src_h: int,
    det_w: int,
    det_h: int,
) -> List[Tuple[float, float]]:
    """把检测图上的关键点映射回原图坐标。"""

    _require_positive(src_w=src_w, src_h=src_h, det_w=det_w, det_h=det_h)
    scale_x = src_w / float(det_w)
    scale_y = src_h / float(det_h)
    mapped: List[Tuple[float, float]] = []
    for point in points:
        if len(point) != 2:
            raise ValueError(f"关键点必须是 (x, y)，收到：{point}")
        mapped.append((float(point[0]) * scale_x, float(point[1]) * scale_y))
    return mapped


def clip_bbox(bbox: Sequence[float], image_w: int, image_h: int) -> BBox:
    """把 bbox 裁到图像范围内。"""

    _require_positive(image_w=image_w, image_h=image_h)
    _require_bbox(bbox)
    x, y, w, h = (float(value) for value in bbox)
    x0 = min(max(x, 0.0), float(image_w))
    y0 = min(max(y, 0.0), float(image_h))
    x1 = min(max(x + w, 0.0), float(image_w))
    y1 = min(max(y + h, 0.0), float(image_h))
    return (x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))


def expand_bbox(
    bbox: Sequence[float],
    ratio: float,
    image_w: int,
    image_h: int,
) -> BBox:
    """按比例扩张（ratio > 0）或收缩（ratio < 0）bbox，并裁到图像范围内。

    ratio = -0.5 会把区域缩到 0，因此要求 ratio > -0.5。
    """

    _require_bbox(bbox)
    if ratio <= -0.5:
        raise ValueError(f"扩张/收缩比例必须大于 -0.5：{ratio}")
    x, y, w, h = (float(value) for value in bbox)
    pad_x = w * ratio
    pad_y = h * ratio
    adjusted = (x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y)
    return clip_bbox(adjusted, image_w, image_h)


def _require_positive(**values: float) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} 必须大于 0：{value}")


def _require_bbox(bbox: Sequence[float]) -> None:
    if len(bbox) != 4:
        raise ValueError(f"bbox 必须是 (x, y, w, h)，收到：{bbox}")
    if bbox[2] < 0 or bbox[3] < 0:
        raise ValueError(f"bbox 宽高不能为负：{bbox}")

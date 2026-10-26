"""图像工具：读写、裁剪、等比缩放、带 alpha 贴图。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np

from utils.geometry import expand_bbox, fit_size


def read_image(
    path: Union[str, Path],
    flags: int = cv2.IMREAD_COLOR,
) -> np.ndarray:
    """读取图片，兼容 Windows 中文路径。"""

    image_path = Path(path)
    if not image_path.is_file():
        raise ValueError(f"图片不存在：{image_path}")

    data = np.fromfile(str(image_path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"图片为空：{image_path}")

    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"无法解码图片：{image_path}")
    return image


def write_image(
    path: Union[str, Path],
    image: np.ndarray,
    params: Optional[Sequence[int]] = None,
) -> None:
    """写出图片，兼容 Windows 中文路径。"""

    image_path = Path(path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = image_path.suffix or ".jpg"

    ok, buffer = cv2.imencode(suffix, image, list(params or []))
    if not ok:
        raise ValueError(f"无法编码图片：{image_path}")
    buffer.tofile(str(image_path))


def crop(
    image: np.ndarray,
    bbox: Sequence[float],
    padding_ratio: float = 0.0,
) -> np.ndarray:
    """按 bbox 裁剪，可选外扩；越界自动裁边。"""

    if image.ndim < 2:
        raise ValueError("输入必须是图像数组")
    height, width = image.shape[:2]
    x, y, w, h = expand_bbox(bbox, padding_ratio, width, height)

    x0 = max(0, min(int(round(x)), width))
    y0 = max(0, min(int(round(y)), height))
    x1 = max(0, min(int(round(x + w)), width))
    y1 = max(0, min(int(round(y + h)), height))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"裁剪区域为空：{bbox}")

    return image[y0:y1, x0:x1].copy()


def resize_max_side(image: np.ndarray, max_size: int) -> np.ndarray:
    """等比缩放使最长边不超过 max_size；小图原样返回副本。"""

    height, width = image.shape[:2]
    new_w, new_h = fit_size(width, height, max_size)
    if (new_w, new_h) == (width, height):
        return image.copy()

    shrinking = new_w * new_h < width * height
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def paste_with_alpha(
    background: np.ndarray,
    overlay: np.ndarray,
    x: int,
    y: int,
) -> np.ndarray:
    """把带 alpha 的 overlay 贴到 background 的 (x, y)，返回新图。"""

    if background.ndim != 3 or background.shape[2] != 3:
        raise ValueError("background 必须是 BGR 三通道图像")
    if overlay.ndim != 3 or overlay.shape[2] not in (3, 4):
        raise ValueError("overlay 必须是 BGR 或 BGRA 图像")

    canvas = background.copy()
    bg_h, bg_w = canvas.shape[:2]
    ov_h, ov_w = overlay.shape[:2]

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(bg_w, x + ov_w)
    y1 = min(bg_h, y + ov_h)
    if x0 >= x1 or y0 >= y1:
        return canvas

    ox0 = x0 - x
    oy0 = y0 - y
    region = overlay[oy0:oy0 + (y1 - y0), ox0:ox0 + (x1 - x0)]

    rgb = region[..., :3].astype(np.float32)
    if region.shape[2] == 4:
        alpha = region[..., 3:4].astype(np.float32) / 255.0
    else:
        alpha = np.ones((region.shape[0], region.shape[1], 1), dtype=np.float32)

    target = canvas[y0:y1, x0:x1].astype(np.float32)
    blended = rgb * alpha + target * (1.0 - alpha)
    canvas[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return canvas

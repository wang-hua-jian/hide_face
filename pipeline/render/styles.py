"""遮挡样式：mosaic / blur / solid / custom_image。

样式函数只负责把一块 ROI 变成遮挡后的像素，不做区域裁剪与透明度合成。
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np

from utils.image import read_image


def mosaic(image: np.ndarray, block_size: int) -> np.ndarray:
    """像素块马赛克。"""

    if block_size < 2:
        raise ValueError(f"mosaic_block 必须 >= 2：{block_size}")

    height, width = image.shape[:2]
    small_w = max(1, width // block_size)
    small_h = max(1, height // block_size)
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


def blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """高斯模糊，核大小自动修正为奇数。"""

    if kernel_size < 3:
        raise ValueError(f"blur_kernel 必须 >= 3：{kernel_size}")

    kernel = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(image, (kernel, kernel), 0)


def solid(image: np.ndarray, color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """纯色覆盖。"""

    return np.full_like(image, color)


def load_mask_image(path: Union[str, Path]) -> np.ndarray:
    """读取自定义遮挡图，统一返回 BGRA。"""

    image = read_image(path, cv2.IMREAD_UNCHANGED)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    if image.ndim == 3 and image.shape[2] == 4:
        return image
    raise ValueError(f"不支持的遮挡图通道结构：{image.shape}")


def fit_mask(
    mask_bgra: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """把遮挡图拉伸到 ROI 尺寸，返回 (BGR, alpha 0~1)。

    为保证隐私，拉伸铺满整个区域，不做留白。
    """

    if width <= 0 or height <= 0:
        raise ValueError(f"目标区域尺寸非法：{width}x{height}")

    resized = cv2.resize(mask_bgra, (width, height), interpolation=cv2.INTER_LINEAR)
    alpha = resized[..., 3].astype(np.float32) / 255.0
    return resized[..., :3], alpha


def ellipse_alpha(width: int, height: int) -> np.ndarray:
    """生成内切椭圆的 alpha 蒙版（0~1），用于椭圆遮挡。"""

    if width <= 0 or height <= 0:
        raise ValueError(f"目标区域尺寸非法：{width}x{height}")

    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    axes = (max(1, width // 2), max(1, height // 2))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask.astype(np.float32) / 255.0

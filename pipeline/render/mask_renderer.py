"""MaskRenderer：把 TargetRegion 落成实际遮挡像素。"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from pipeline.render.base import Renderer
from pipeline.render.styles import (
    blur,
    ellipse_alpha,
    fit_mask,
    load_mask_image,
    mosaic,
    solid,
)
from pipeline.schema import TargetRegion

STYLES = ("mosaic", "blur", "solid", "custom_image")
MASK_SHAPES = ("rect", "ellipse")


class MaskRenderer(Renderer):
    def __init__(
        self,
        style: str,
        opacity: float,
        mosaic_block: int,
        blur_kernel: int,
        mask_path: Optional[str] = None,
        mask_shape: str = "rect",
        solid_color: Tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        if style not in STYLES:
            raise ValueError(f"未知遮挡样式：{style}")
        if mask_shape not in MASK_SHAPES:
            raise ValueError(f"未知遮挡形状：{mask_shape}")
        if not 0.0 <= opacity <= 1.0:
            raise ValueError(f"opacity 必须在 [0, 1] 内：{opacity}")

        self._style = style
        self._mask_shape = mask_shape
        self._opacity = opacity
        self._mosaic_block = mosaic_block
        self._blur_kernel = blur_kernel
        self._solid_color = solid_color
        self._mask: Optional[np.ndarray] = None

        if style == "custom_image":
            if not mask_path:
                raise ValueError("custom_image 样式需要提供遮挡图路径")
            self._mask = load_mask_image(mask_path)

    def draw(
        self,
        frame: np.ndarray,
        regions: Sequence[TargetRegion],
    ) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"frame 必须是 BGR 三通道图像：{frame.shape}")

        height, width = frame.shape[:2]
        for region in regions:
            self._draw_region(frame, region, width, height)
        return frame

    def _draw_region(
        self,
        frame: np.ndarray,
        region: TargetRegion,
        width: int,
        height: int,
    ) -> None:
        x, y, box_w, box_h = region.bbox
        x0 = max(0, int(math.floor(x)))
        y0 = max(0, int(math.floor(y)))
        x1 = min(width, int(math.ceil(x + box_w)))
        y1 = min(height, int(math.ceil(y + box_h)))
        if x1 <= x0 or y1 <= y0:
            return

        roi = frame[y0:y1, x0:x1]
        roi_h, roi_w = roi.shape[:2]
        styled, alpha = self._style_roi(roi, roi_w, roi_h)

        if self._mask_shape == "ellipse":
            alpha = alpha * ellipse_alpha(roi_w, roi_h)
        elif region.polygon and len(region.polygon) >= 3:
            polygon_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            points = np.asarray(
                [(px - x0, py - y0) for px, py in region.polygon],
                dtype=np.int32,
            )
            cv2.fillPoly(polygon_mask, [points], 255)
            alpha = alpha * (polygon_mask.astype(np.float32) / 255.0)

        alpha = (alpha * self._opacity)[..., None]
        blended = (
            styled.astype(np.float32) * alpha
            + roi.astype(np.float32) * (1.0 - alpha)
        )
        roi[:] = np.clip(blended, 0, 255).astype(np.uint8)

    def _style_roi(
        self,
        roi: np.ndarray,
        roi_w: int,
        roi_h: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._style == "mosaic":
            return mosaic(roi, self._mosaic_block), np.ones(
                (roi_h, roi_w), dtype=np.float32
            )
        if self._style == "blur":
            return blur(roi, self._blur_kernel), np.ones(
                (roi_h, roi_w), dtype=np.float32
            )
        if self._style == "solid":
            return solid(roi, self._solid_color), np.ones(
                (roi_h, roi_w), dtype=np.float32
            )
        if self._mask is None:
            raise ValueError("custom_image 样式缺少遮挡图")
        return fit_mask(self._mask, roi_w, roi_h)

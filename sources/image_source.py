"""单张图片输入源。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np

from sources.base import Source, SourceError
from sources.frame import FramePacket
from utils.image import read_image


class ImageSource(Source):
    def __init__(self, path: Union[str, Path]) -> None:
        self._path = Path(path)
        self._image: Optional[np.ndarray] = None
        self._delivered = False

    def open(self) -> None:
        if not self._path.is_file():
            raise SourceError(f"图片不存在：{self._path}")

        try:
            image = read_image(self._path, cv2.IMREAD_UNCHANGED)
        except ValueError as exc:
            raise SourceError(str(exc)) from exc

        self._image = _to_bgr(image)
        self._delivered = False

    def read(self) -> Optional[FramePacket]:
        if self._image is None:
            raise SourceError("请先调用 open()")
        if self._delivered:
            return None
        self._delivered = True
        return FramePacket(frame_id=0, timestamp=0.0, image=self._image, eof=False)

    def close(self) -> None:
        self._image = None
        self._delivered = False

    @property
    def meta(self) -> Dict[str, Any]:
        if self._image is None:
            return {"path": str(self._path)}
        return {
            "path": str(self._path),
            "width": int(self._image.shape[1]),
            "height": int(self._image.shape[0]),
        }


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 3:
            return image
        if channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise SourceError(f"不支持的图片通道结构：{image.shape}")

"""摄像头输入源：只负责打开、读帧、截图、释放。"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np

from sources.base import Source, SourceError
from sources.frame import FramePacket
from utils.image import write_image


class CameraSource(Source):
    def __init__(
        self,
        index: int = 0,
        jpeg_quality: int = 95,
    ) -> None:
        if index < 0:
            raise ValueError(f"摄像头索引不能为负：{index}")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError(f"jpeg_quality 必须在 [1, 100] 内：{jpeg_quality}")
        self._index = index
        self._jpeg_quality = jpeg_quality
        self._capture: Optional[cv2.VideoCapture] = None
        self._pending: Optional[np.ndarray] = None
        self._frame_id = 0
        self._started = 0.0
        self._meta: Dict[str, Any] = {"index": index}

    def open(self) -> None:
        if self._capture is not None:
            self.close()

        capture, frame = self._open_capture()
        self._capture = capture
        self._pending = frame
        self._frame_id = 0
        self._started = time.perf_counter()

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or frame.shape[1]
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frame.shape[0]
        camera_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self._meta = {
            "index": self._index,
            "width": width,
            "height": height,
            "camera_fps": camera_fps,
        }

    def _open_capture(self):
        backends = []
        if sys.platform == "win32":
            backends.append(cv2.CAP_DSHOW)
        backends.append(cv2.CAP_ANY)

        for backend in backends:
            capture = cv2.VideoCapture(self._index, backend)
            if not capture.isOpened():
                capture.release()
                continue

            try:
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            ok, frame = capture.read()
            if ok and frame is not None:
                return capture, frame
            capture.release()

        raise SourceError(
            f"无法打开摄像头 index={self._index}，"
            "请检查设备是否存在、是否被其他程序占用、系统是否允许访问摄像头"
        )

    def read(self) -> Optional[FramePacket]:
        if self._capture is None:
            raise SourceError("摄像头尚未打开")

        if self._pending is not None:
            image = self._pending
            self._pending = None
            timestamp = 0.0
        else:
            ok, image = self._capture.read()
            if not ok or image is None:
                raise SourceError(
                    f"摄像头 index={self._index} 读帧失败，"
                    "设备可能已断开或被其他程序占用"
                )
            timestamp = time.perf_counter() - self._started

        packet = FramePacket(
            frame_id=self._frame_id,
            timestamp=timestamp,
            image=image,
            eof=False,
        )
        self._frame_id += 1
        return packet

    def snapshot(self, path: Union[str, Path]) -> FramePacket:
        frame = self.read()
        if frame is None:
            raise SourceError("摄像头没有可用的画面")
        write_image(
            path,
            frame.image,
            [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
        )
        return frame

    def close(self) -> None:
        self._pending = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    @property
    def meta(self) -> Dict[str, Any]:
        return dict(self._meta)

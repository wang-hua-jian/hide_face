"""统一的帧数据结构。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FramePacket:
    frame_id: int
    timestamp: float
    image: np.ndarray
    eof: bool = False

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

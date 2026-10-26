"""人脸检测器抽象。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from pipeline.schema import FaceDetection


class FaceDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray, frame_id: int = 0) -> List[FaceDetection]:
        """输入 BGR 原图，返回原图坐标系下的人脸。"""

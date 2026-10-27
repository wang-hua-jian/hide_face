"""人脸特征提取抽象。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from pipeline.schema import FaceDetection


class FaceEmbedder(ABC):
    @abstractmethod
    def embed(self, image: np.ndarray, detection: FaceDetection) -> np.ndarray:
        """输入原图与人脸检测，输出 L2 归一化特征向量。"""

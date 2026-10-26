"""遮挡渲染抽象。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from pipeline.schema import TargetRegion


class Renderer(ABC):
    @abstractmethod
    def draw(
        self,
        frame: np.ndarray,
        regions: Sequence[TargetRegion],
    ) -> np.ndarray:
        """按目标区域绘制遮挡，原地修改并返回同一张图。"""

"""目标区域提供者抽象。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple

from pipeline.schema import FaceDetection, TargetRegion


class TargetProvider(ABC):
    @abstractmethod
    def from_detections(
        self,
        detections: Sequence[FaceDetection],
        image_size: Tuple[int, int],
    ) -> List[TargetRegion]:
        """输入检测结果与图像尺寸，输出遮挡目标区域。"""

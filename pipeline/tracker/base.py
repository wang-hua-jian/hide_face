"""跟踪器抽象。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

import numpy as np

from pipeline.schema import FaceDetection
from pipeline.tracker.state import TrackState


class Tracker(ABC):
    @abstractmethod
    def update(
        self,
        detections: Sequence[FaceDetection],
        frame_id: int,
        image: Optional[np.ndarray] = None,
    ) -> List[TrackState]:
        """检测帧：预测 + 关联 + 生命周期更新。

        image 仅在外观兜底需要裁剪人脸时使用。
        """

    @abstractmethod
    def predict(self, frame_id: int) -> List[TrackState]:
        """非检测帧：只做卡尔曼预测，不增加丢失计数。"""

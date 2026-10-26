"""跟踪器抽象。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from pipeline.schema import FaceDetection
from pipeline.tracker.state import TrackState


class Tracker(ABC):
    @abstractmethod
    def update(
        self,
        detections: Sequence[FaceDetection],
        frame_id: int,
    ) -> List[TrackState]:
        """检测帧：预测 + 关联 + 生命周期更新。"""

    @abstractmethod
    def predict(self, frame_id: int) -> List[TrackState]:
        """非检测帧：只做卡尔曼预测，不增加丢失计数。"""

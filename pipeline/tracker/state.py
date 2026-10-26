"""单个 Track 的状态。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pipeline.tracker.kalman import BBox, KalmanBoxFilter


@dataclass
class TrackState:
    track_id: int
    kalman: KalmanBoxFilter
    score: float
    last_bbox: BBox
    last_seen_frame: int
    measurement_bbox: Optional[BBox] = None
    lost_frames: int = 0
    status: str = "tracked"
    updated: bool = True
    mask_state: bool = True

    # P4 外观兜底使用，P3 只预留字段
    appearance_ema: Optional[np.ndarray] = None
    appearance_updates: int = 0
    appearance_frame: int = -1

    @property
    def bbox(self) -> BBox:
        return self.last_bbox

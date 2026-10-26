"""跨模块标准中间对象。

detector 只产出 FaceDetection；target 只消费 FaceDetection 并产出 TargetRegion；
tracker 只读写 track_id；renderer 只消费 TargetRegion。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

from utils.geometry import BBox

Point = Tuple[float, float]


@dataclass(frozen=True)
class FaceDetection:
    bbox: BBox
    landmarks: Tuple[Point, ...]
    confidence: float
    frame_id: int = 0


@dataclass(frozen=True)
class TargetRegion:
    target_type: str
    bbox: BBox
    polygon: Optional[Tuple[Point, ...]]
    confidence: float
    track_id: Optional[int] = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

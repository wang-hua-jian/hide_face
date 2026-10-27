"""基于已确认人脸的眼距比例先验。

眼距比例 = 双眼中点距离 / max(脸框宽, 高)，本身与人物远近无关。
这里用同视频内“位置匹配成功”的高置信检测做滚动统计，
为新建 Track 提供一个随场景自适应的合理区间。
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

from pipeline.schema import FaceDetection


def eye_ratio(detection: FaceDetection) -> Optional[float]:
    if len(detection.landmarks) != 5:
        return None
    left_eye, right_eye = detection.landmarks[0], detection.landmarks[1]
    distance = math.hypot(
        left_eye[0] - right_eye[0],
        left_eye[1] - right_eye[1],
    )
    _, _, width, height = detection.bbox
    scale = max(width, height, 1.0)
    return distance / scale


class FaceGeometryPrior:
    def __init__(
        self,
        window_size: int = 200,
        min_samples: int = 20,
        lower_factor: float = 0.4,
        upper_factor: float = 2.0,
        fallback_min_ratio: float = 0.12,
    ) -> None:
        if window_size < 1:
            raise ValueError(f"eye_ratio_window 非法：{window_size}")
        if min_samples < 1:
            raise ValueError(f"eye_ratio_min_samples 非法：{min_samples}")
        if not 0.0 < lower_factor <= 1.0:
            raise ValueError(f"eye_ratio_lower_factor 非法：{lower_factor}")
        if upper_factor < 1.0:
            raise ValueError(f"eye_ratio_upper_factor 非法：{upper_factor}")
        if not 0.0 <= fallback_min_ratio <= 1.0:
            raise ValueError(f"eye_ratio_fallback 非法：{fallback_min_ratio}")

        self._window_size = window_size
        self._min_samples = min_samples
        self._lower_factor = lower_factor
        self._upper_factor = upper_factor
        self._fallback_min_ratio = fallback_min_ratio
        self._ratios: Deque[float] = deque(maxlen=window_size)

    @property
    def sample_count(self) -> int:
        return len(self._ratios)

    @property
    def median(self) -> Optional[float]:
        if not self._ratios:
            return None
        return float(np.median(np.fromiter(self._ratios, dtype=np.float64)))

    @property
    def mad(self) -> Optional[float]:
        if not self._ratios:
            return None
        values = np.fromiter(self._ratios, dtype=np.float64)
        median = float(np.median(values))
        return float(np.median(np.abs(values - median)))

    @property
    def bounds(self) -> Tuple[float, float]:
        if len(self._ratios) < self._min_samples:
            return (self._fallback_min_ratio, 1.0)

        median = float(np.median(np.fromiter(self._ratios, dtype=np.float64)))
        lower = max(0.02, median * self._lower_factor)
        upper = min(1.0, median * self._upper_factor)
        return (lower, upper)

    def observe(self, detection: FaceDetection) -> None:
        ratio = eye_ratio(detection)
        if ratio is not None and 0.0 < ratio <= 1.0:
            self._ratios.append(ratio)

    def is_plausible(self, detection: FaceDetection) -> bool:
        ratio = eye_ratio(detection)
        if ratio is None:
            return False
        lower, upper = self.bounds
        return lower <= ratio <= upper

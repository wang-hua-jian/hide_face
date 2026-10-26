"""时序平滑：抑制遮挡框抖动，短暂丢失时保持上一帧位置。"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

from pipeline.tracker.state import TrackState

BBox = Tuple[float, float, float, float]


class TemporalSmoother:
    def __init__(self, alpha: float, max_scale_step: float) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"smooth_alpha 必须在 (0, 1] 内：{alpha}")
        if not 0.0 <= max_scale_step <= 1.0:
            raise ValueError(
                f"max_scale_step 必须在 [0, 1] 内：{max_scale_step}"
            )
        self._alpha = alpha
        self._max_scale_step = max_scale_step
        self._boxes: Dict[int, BBox] = {}

    def update(self, tracks: Sequence[TrackState]) -> Dict[int, BBox]:
        smoothed: Dict[int, BBox] = {}

        for track in tracks:
            previous = self._boxes.get(track.track_id)
            raw = track.bbox

            if previous is None:
                box = raw
            else:
                # 丢失期间也跟随卡尔曼预测轨迹，避免遮挡框冻结在原地
                box = _smooth_box(
                    previous,
                    raw,
                    self._alpha,
                    self._max_scale_step,
                )

            smoothed[track.track_id] = box

        self._boxes = smoothed
        return smoothed


def _smooth_box(
    previous: BBox,
    current: BBox,
    alpha: float,
    max_scale_step: float,
) -> BBox:
    prev_x, prev_y, prev_w, prev_h = previous
    cur_x, cur_y, cur_w, cur_h = current

    min_w = prev_w * (1.0 - max_scale_step)
    max_w = prev_w * (1.0 + max_scale_step)
    min_h = prev_h * (1.0 - max_scale_step)
    max_h = prev_h * (1.0 + max_scale_step)
    width = min(max(cur_w, min_w), max_w)
    height = min(max(cur_h, min_h), max_h)

    prev_cx = prev_x + prev_w / 2.0
    prev_cy = prev_y + prev_h / 2.0
    cur_cx = cur_x + cur_w / 2.0
    cur_cy = cur_y + cur_h / 2.0
    center_x = alpha * cur_cx + (1.0 - alpha) * prev_cx
    center_y = alpha * cur_cy + (1.0 - alpha) * prev_cy

    return (
        center_x - width / 2.0,
        center_y - height / 2.0,
        width,
        height,
    )

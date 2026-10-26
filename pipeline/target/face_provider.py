"""人脸目标区域生成：FaceDetection[] -> TargetRegion[]。

本模块不调用任何检测模型，只做区域扩张与 polygon 生成。
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from pipeline.schema import FaceDetection, TargetRegion
from pipeline.target.base import TargetProvider
from utils.geometry import expand_bbox


class FaceTargetProvider(TargetProvider):
    def __init__(self, box_expand_ratio: float) -> None:
        if box_expand_ratio <= -0.5:
            raise ValueError(
                f"box_expand_ratio 必须大于 -0.5：{box_expand_ratio}"
            )
        self._box_expand_ratio = box_expand_ratio

    def from_detections(
        self,
        detections: Sequence[FaceDetection],
        image_size: Tuple[int, int],
    ) -> List[TargetRegion]:
        image_w, image_h = image_size
        regions: List[TargetRegion] = []

        for detection in detections:
            bbox = expand_bbox(
                detection.bbox,
                self._box_expand_ratio,
                image_w,
                image_h,
            )
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue

            regions.append(
                TargetRegion(
                    target_type="face",
                    bbox=bbox,
                    polygon=_bbox_to_polygon(bbox),
                    confidence=detection.confidence,
                    track_id=None,
                    attributes={
                        "frame_id": detection.frame_id,
                        "detection_bbox": detection.bbox,
                    },
                )
            )

        return regions


def _bbox_to_polygon(
    bbox: Tuple[float, float, float, float]
) -> Tuple[Tuple[float, float], ...]:
    x, y, w, h = bbox
    return ((x, y), (x + w, y), (x + w, y + h), (x, y + h))

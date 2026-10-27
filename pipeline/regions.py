"""把 Track 状态转换为统一的 TargetRegion。"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Sequence

from pipeline.schema import FaceDetection, TargetRegion
from pipeline.target.base import TargetProvider
from pipeline.tracker.state import TrackState


def build_track_regions(
    tracks: Sequence[TrackState],
    boxes: Dict[int, Sequence[float]],
    target_provider: TargetProvider,
    frame_id: int,
    width: int,
    height: int,
) -> List[TargetRegion]:
    regions: List[TargetRegion] = []

    for track in tracks:
        if not track.mask_state:
            continue
        if track.updated and track.measurement_bbox is not None:
            # 检测帧直接用检测框，避免滤波/平滑带来的视觉延迟
            box = track.measurement_bbox
        else:
            box = boxes.get(track.track_id)
        if box is None:
            continue

        detection = FaceDetection(
            bbox=box,
            landmarks=(),
            confidence=track.score,
            frame_id=frame_id,
        )
        for region in target_provider.from_detections(
            [detection], (width, height)
        ):
            regions.append(
                replace(
                    region,
                    track_id=track.track_id,
                    attributes={
                        "frame_id": frame_id,
                        "detection_bbox": box,
                        "lost_frames": track.lost_frames,
                        "status": track.status,
                    },
                )
            )

    return regions

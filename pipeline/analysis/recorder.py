"""分析结果记录：生成 analysis.json 的通用片段。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from pipeline.schema import TargetRegion


def build_frame_record(
    frame_id: int,
    regions: Sequence[TargetRegion],
) -> Dict[str, Any]:
    targets = []
    for region in regions:
        attributes = region.attributes or {}
        detection_bbox = attributes.get("detection_bbox", region.bbox)
        targets.append(
            {
                "track_id": region.track_id,
                "bbox": _numbers(detection_bbox),
                "confidence": round(float(region.confidence), 4),
                "identity": None,
                "target_type": region.target_type,
                "target_region": {
                    "bbox": _numbers(region.bbox),
                    "polygon": [
                        _numbers(point) for point in (region.polygon or ())
                    ],
                },
            }
        )
    return {"frame_id": frame_id, "targets": targets}


def build_image_payload(
    input_path: Path,
    width: int,
    height: int,
    frame_id: int,
    regions: Sequence[TargetRegion],
) -> Dict[str, Any]:
    return {
        "media": {
            "type": "image",
            "input": str(input_path),
            "width": width,
            "height": height,
        },
        "frames": [build_frame_record(frame_id, regions)],
    }


def write_analysis(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _numbers(values: Sequence[float]) -> list:
    return [round(float(value), 2) for value in values]

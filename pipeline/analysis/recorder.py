"""分析结果记录：生成 analysis.json 的通用片段。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

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
                "identity": attributes.get("identity"),
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


class AnalysisRecorder:
    """逐帧流式写 analysis.json，避免长视频把全部记录留在内存。"""

    def __init__(self, path: Path, media: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._handle = open(path, "w", encoding="utf-8")
        self._first = True
        self._closed = False
        self._handle.write('{\n  "media": ')
        json.dump(media, self._handle, ensure_ascii=False)
        self._handle.write(',\n  "frames": [')

    @property
    def path(self) -> Path:
        return self._path

    def add_frame(
        self,
        frame_id: int,
        regions: Sequence[TargetRegion],
    ) -> None:
        if self._closed:
            raise RuntimeError("AnalysisRecorder 已关闭")
        record = build_frame_record(frame_id, regions)
        if not self._first:
            self._handle.write(",")
        self._handle.write("\n    ")
        json.dump(record, self._handle, ensure_ascii=False)
        self._first = False

    def flush(self) -> None:
        if not self._closed:
            self._handle.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._handle.write("\n  ]\n}\n")
        self._handle.close()
        self._closed = True


def _numbers(values: Sequence[float]) -> list:
    return [round(float(value), 2) for value in values]

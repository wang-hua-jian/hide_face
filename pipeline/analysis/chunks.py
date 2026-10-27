"""长视频分块规划。

当前版本是单进程连续流式处理，所以不会真的切出多个文件；
分块信息用于分析文件定期落盘，并为未来并行处理保留边界定义。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ChunkRange:
    index: int
    start: float
    end: float
    overlap_start: float


def plan_chunks(
    duration: float,
    chunk_seconds: int,
    overlap_seconds: int,
) -> List[ChunkRange]:
    if duration <= 0:
        return []
    if chunk_seconds <= 0:
        raise ValueError(f"chunk_seconds 必须大于 0：{chunk_seconds}")
    if overlap_seconds < 0:
        raise ValueError(f"chunk_overlap_seconds 不能为负：{overlap_seconds}")
    if overlap_seconds >= chunk_seconds:
        raise ValueError(
            "chunk_overlap_seconds 必须小于 chunk_seconds："
            f"{overlap_seconds} >= {chunk_seconds}"
        )

    chunks: List[ChunkRange] = []
    start = 0.0
    index = 0
    while start < duration:
        end = min(start + chunk_seconds, duration)
        overlap_start = max(0.0, start - overlap_seconds)
        chunks.append(
            ChunkRange(
                index=index,
                start=start,
                end=end,
                overlap_start=overlap_start,
            )
        )
        start = end
        index += 1
    return chunks

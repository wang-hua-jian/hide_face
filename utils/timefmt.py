"""时间戳与文件名工具。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Union

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S_%f"


def format_timestamp(moment: Optional[datetime] = None) -> str:
    """格式化为 YYYYMMDD_HHMMSS_mmm（毫秒精度）。"""

    value = moment if moment is not None else datetime.now()
    return value.strftime(TIMESTAMP_FORMAT)[:-3]


def camera_filename(moment: Optional[datetime] = None) -> str:
    return f"camera_{format_timestamp(moment)}.jpg"


def live_filename(moment: Optional[datetime] = None) -> str:
    return f"live_{format_timestamp(moment)}.mp4"


def unique_camera_path(
    directory: Union[str, Path],
    moment: Optional[datetime] = None,
) -> Path:
    """生成不重名的摄像头截图路径。"""

    target_dir = Path(directory)
    path = target_dir / camera_filename(moment)
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    index = 1
    while True:
        candidate = target_dir / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1

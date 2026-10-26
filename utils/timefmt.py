"""时间戳与文件名工具。"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S_%f"


def format_timestamp(moment: Optional[datetime] = None) -> str:
    """格式化为 YYYYMMDD_HHMMSS_mmm（毫秒精度）。"""

    value = moment if moment is not None else datetime.now()
    return value.strftime(TIMESTAMP_FORMAT)[:-3]


def camera_filename(moment: Optional[datetime] = None) -> str:
    return f"camera_{format_timestamp(moment)}.jpg"


def live_filename(moment: Optional[datetime] = None) -> str:
    return f"live_{format_timestamp(moment)}.mp4"

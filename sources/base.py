"""输入源抽象：只负责拿帧。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from sources.frame import FramePacket


class SourceError(RuntimeError):
    """输入源打开或读取失败。"""


class Source(ABC):
    @abstractmethod
    def open(self) -> None:
        """打开输入源。"""

    @abstractmethod
    def read(self) -> Optional[FramePacket]:
        """返回下一帧；结束后返回 None。"""

    @abstractmethod
    def close(self) -> None:
        """释放资源。"""

    @property
    @abstractmethod
    def meta(self) -> Dict[str, Any]:
        """输入源元信息（宽高、路径、fps 等）。"""

    def __enter__(self) -> "Source":
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

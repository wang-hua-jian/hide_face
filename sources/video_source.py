"""视频输入源：PyAV 逐帧解码，不缓存整段。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import av

from sources.base import Source, SourceError
from sources.frame import FramePacket


class VideoSource(Source):
    def __init__(self, path: Union[str, Path]) -> None:
        self._path = Path(path)
        self._container: Optional[av.container.InputContainer] = None
        self._decoder = None
        self._stream = None
        self._frame_id = 0
        self._meta: Dict[str, Any] = {"path": str(self._path)}

    def open(self) -> None:
        if not self._path.is_file():
            raise SourceError(f"视频不存在：{self._path}")

        try:
            container = av.open(str(self._path))
        except av.error.FFmpegError as exc:
            raise SourceError(f"无法打开视频：{self._path}\n{exc}") from exc

        if not container.streams.video:
            container.close()
            raise SourceError(f"视频中没有视频流：{self._path}")

        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rate = stream.average_rate
        fps = float(rate) if rate else 0.0
        duration = (
            float(container.duration) / av.time_base
            if container.duration is not None
            else 0.0
        )
        total_frames = int(stream.frames) if stream.frames else 0
        if total_frames == 0 and fps > 0 and duration > 0:
            total_frames = int(round(duration * fps))

        self._container = container
        self._stream = stream
        self._decoder = container.decode(video=0)
        self._frame_id = 0
        self._meta = {
            "path": str(self._path),
            "width": int(stream.codec_context.width),
            "height": int(stream.codec_context.height),
            "fps": fps,
            "rate": rate,
            "total_frames": total_frames,
            "duration": duration,
            "codec": stream.codec_context.name,
            "has_audio": bool(container.streams.audio),
        }

    def read(self) -> Optional[FramePacket]:
        if self._decoder is None:
            raise SourceError("请先调用 open()")

        try:
            frame = next(self._decoder)
        except StopIteration:
            return None
        except av.error.FFmpegError as exc:
            raise SourceError(f"解码失败：{self._path}\n{exc}") from exc

        image = frame.to_ndarray(format="bgr24")
        timestamp = float(frame.time) if frame.time is not None else 0.0
        packet = FramePacket(
            frame_id=self._frame_id,
            timestamp=timestamp,
            image=image,
            eof=False,
        )
        self._frame_id += 1
        return packet

    def close(self) -> None:
        self._decoder = None
        self._stream = None
        if self._container is not None:
            self._container.close()
            self._container = None

    @property
    def meta(self) -> Dict[str, Any]:
        return dict(self._meta)

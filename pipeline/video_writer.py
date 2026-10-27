"""视频写出：H.264 重编码，可选复制输入音轨。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import av
import numpy as np


class VideoWriterError(RuntimeError):
    """视频编码或音频复制失败。"""


class VideoWriter:
    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        rate: object,
        input_path: Optional[Path] = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._audio_input = None
        self._audio_packets = None
        self._audio_stream = None
        self._audio_pending = None

        try:
            self._container = av.open(str(path), mode="w")
            try:
                stream = self._container.add_stream("libx264", rate=rate)
                stream.options = {"crf": "23", "preset": "veryfast"}
            except (av.error.FFmpegError, ValueError):
                stream = self._container.add_stream("mpeg4", rate=rate)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            self._stream = stream
            if input_path is not None:
                self._setup_audio(input_path)
        except VideoWriterError:
            self._close_resources()
            raise
        except (av.error.FFmpegError, OSError, ValueError) as exc:
            self._close_resources()
            raise VideoWriterError(f"无法创建视频输出：{exc}") from exc

    def _setup_audio(self, input_path: Path) -> None:
        try:
            audio_input = av.open(str(input_path))
        except (av.error.FFmpegError, OSError) as exc:
            raise VideoWriterError(f"无法读取音频轨道：{exc}") from exc

        if not audio_input.streams.audio:
            audio_input.close()
            return

        try:
            audio_stream = self._container.add_stream_from_template(
                audio_input.streams.audio[0]
            )
        except (av.error.FFmpegError, ValueError) as exc:
            audio_input.close()
            raise VideoWriterError(f"无法复制音频轨道：{exc}") from exc

        self._audio_input = audio_input
        self._audio_stream = audio_stream
        self._audio_packets = audio_input.demux(audio=0)

    def write(self, image: np.ndarray, timestamp: float) -> None:
        try:
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            for packet in self._stream.encode(frame):
                self._container.mux(packet)
            self._mux_audio_until(timestamp)
        except av.error.FFmpegError as exc:
            raise VideoWriterError(f"视频编码失败：{exc}") from exc

    def _mux_audio_until(self, timestamp: float) -> None:
        if self._audio_stream is None or self._audio_packets is None:
            return

        while True:
            if self._audio_pending is not None:
                packet = self._audio_pending
            else:
                try:
                    packet = next(self._audio_packets)
                except StopIteration:
                    return

            if packet.dts is None:
                self._audio_pending = None
                continue

            pts = packet.pts if packet.pts is not None else packet.dts
            packet_time = (
                float(pts * packet.time_base) if packet.time_base else 0.0
            )
            if packet_time > timestamp:
                self._audio_pending = packet
                return

            packet.stream = self._audio_stream
            self._container.mux(packet)
            self._audio_pending = None

    def close(self) -> None:
        try:
            try:
                for packet in self._stream.encode():
                    self._container.mux(packet)
                self._mux_audio_until(float("inf"))
            except av.error.FFmpegError as exc:
                raise VideoWriterError(f"视频收尾失败：{exc}") from exc
        finally:
            self._close_resources()

    def _close_resources(self) -> None:
        if self._audio_input is not None:
            self._audio_input.close()
            self._audio_input = None
        if getattr(self, "_container", None) is not None:
            self._container.close()
            self._container = None

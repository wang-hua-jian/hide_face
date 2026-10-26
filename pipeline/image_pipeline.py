"""图片处理管线：读 → 检测 → 区域 → 渲染 → 写图（可选写分析）。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2

from pipeline.analysis.recorder import build_image_payload, write_analysis
from pipeline.detector.base import FaceDetector
from pipeline.render.base import Renderer
from pipeline.target.base import TargetProvider
from sources.image_source import ImageSource
from utils.image import write_image


class ImagePipelineError(RuntimeError):
    """图片处理流程失败。"""


@dataclass(frozen=True)
class ImageResult:
    output_path: Path
    face_count: int
    analysis_path: Optional[Path]
    elapsed_seconds: float


class ImagePipeline:
    def __init__(
        self,
        detector: FaceDetector,
        target_provider: TargetProvider,
        renderer: Renderer,
        jpeg_quality: int,
    ) -> None:
        if not 1 <= jpeg_quality <= 100:
            raise ValueError(f"jpeg_quality 必须在 [1, 100] 内：{jpeg_quality}")
        self._detector = detector
        self._target_provider = target_provider
        self._renderer = renderer
        self._jpeg_quality = jpeg_quality

    def run(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        analysis_path: Optional[Union[str, Path]] = None,
    ) -> ImageResult:
        started = time.perf_counter()
        input_path = Path(input_path)
        output_path = Path(output_path)

        with ImageSource(input_path) as source:
            frame = source.read()
            if frame is None:
                raise ImagePipelineError(f"未能读取图片：{input_path}")

            detections = self._detector.detect(frame.image, frame.frame_id)
            regions = self._target_provider.from_detections(
                detections, (frame.width, frame.height)
            )
            rendered = self._renderer.draw(frame.image, regions)
            write_image(
                output_path,
                rendered,
                [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
            )

            if analysis_path is not None:
                payload = build_image_payload(
                    input_path=input_path,
                    width=frame.width,
                    height=frame.height,
                    frame_id=frame.frame_id,
                    regions=regions,
                )
                write_analysis(Path(analysis_path), payload)

        return ImageResult(
            output_path=output_path,
            face_count=len(detections),
            analysis_path=Path(analysis_path) if analysis_path is not None else None,
            elapsed_seconds=time.perf_counter() - started,
        )

"""YuNet 人脸检测适配。"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from pipeline.detector.base import FaceDetector
from pipeline.schema import FaceDetection
from utils.geometry import fit_size, map_bbox_to_original, map_points_to_original

LANDMARK_COUNT = 5


class YuNetDetector(FaceDetector):
    def __init__(
        self,
        model_path: str,
        score_threshold: float,
        nms_threshold: float,
        max_input_size: int,
    ) -> None:
        self._score_threshold = score_threshold
        self._max_input_size = max_input_size
        self._detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),
            score_threshold,
            nms_threshold,
        )

    def detect(self, image: np.ndarray, frame_id: int = 0) -> List[FaceDetection]:
        bgr = _to_bgr(image)
        src_h, src_w = bgr.shape[:2]
        if src_w < 2 or src_h < 2:
            return []

        det_w, det_h = fit_size(src_w, src_h, self._max_input_size)
        if (det_w, det_h) == (src_w, src_h):
            det_image = bgr
        else:
            det_image = cv2.resize(bgr, (det_w, det_h), interpolation=cv2.INTER_AREA)

        try:
            self._detector.setInputSize((det_w, det_h))
            _, faces = self._detector.detect(det_image)
        except cv2.error as exc:
            raise RuntimeError(f"YuNet 推理失败：{exc}") from exc

        return _parse_rows(
            faces,
            src_w=src_w,
            src_h=src_h,
            det_w=det_w,
            det_h=det_h,
            frame_id=frame_id,
            score_threshold=self._score_threshold,
        )


def _parse_rows(
    rows: Optional[np.ndarray],
    src_w: int,
    src_h: int,
    det_w: int,
    det_h: int,
    frame_id: int,
    score_threshold: float,
) -> List[FaceDetection]:
    """把 YuNet 输出行（检测图坐标）转换为原图坐标的 FaceDetection。"""

    if rows is None or len(rows) == 0:
        return []

    detections: List[FaceDetection] = []
    for row in rows:
        confidence = float(row[-1])
        if confidence < score_threshold:
            continue

        bbox = map_bbox_to_original(row[:4], src_w, src_h, det_w, det_h)
        raw_landmarks = np.asarray(row[4:4 + LANDMARK_COUNT * 2], dtype=np.float32)
        mapped = map_points_to_original(
            raw_landmarks.reshape(LANDMARK_COUNT, 2),
            src_w,
            src_h,
            det_w,
            det_h,
        )
        detections.append(
            FaceDetection(
                bbox=_as_bbox(bbox),
                landmarks=tuple(mapped),
                confidence=confidence,
                frame_id=frame_id,
            )
        )
    return detections


def _as_bbox(values: Sequence[float]) -> Tuple[float, float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"不支持的图像通道结构：{image.shape}")

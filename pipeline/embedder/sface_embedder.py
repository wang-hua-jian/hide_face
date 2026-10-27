"""SFace 特征提取：使用 YuNet 的 5 点关键点对齐后提特征。"""

from __future__ import annotations

import cv2
import numpy as np

from pipeline.embedder.base import FaceEmbedder
from pipeline.schema import FaceDetection
from utils.math import l2_normalize


class SFaceEmbedder(FaceEmbedder):
    def __init__(self, model_path: str) -> None:
        try:
            self._recognizer = cv2.FaceRecognizerSF.create(model_path, "")
        except cv2.error as exc:
            raise RuntimeError(f"无法加载 SFace 模型：{model_path}\n{exc}") from exc

    def embed(self, image: np.ndarray, detection: FaceDetection) -> np.ndarray:
        if len(detection.landmarks) != 5:
            raise ValueError(
                f"需要 5 点关键点才能对齐，收到 {len(detection.landmarks)} 个"
            )

        face_row = np.array(
            [
                *detection.bbox,
                *(
                    coordinate
                    for point in detection.landmarks
                    for coordinate in point
                ),
                detection.confidence,
            ],
            dtype=np.float32,
        )
        if face_row.size != 15:
            raise ValueError(f"人脸行应为 15 个值，实际 {face_row.size}")

        try:
            aligned = self._recognizer.alignCrop(image, face_row)
            feature = self._recognizer.feature(aligned)
        except cv2.error as exc:
            raise RuntimeError(f"SFace 特征提取失败：{exc}") from exc

        return l2_normalize(np.asarray(feature, dtype=np.float32).reshape(-1))

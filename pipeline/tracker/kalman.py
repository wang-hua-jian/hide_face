"""卡尔曼滤波：8 维状态 [cx, cy, aspect, h, vx, vy, va, vh]。

参考 SORT / ByteTrack 的常量速度模型实现，用于每个 Track 的位置预测。
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]


class KalmanBoxFilter:
    _STD_WEIGHT_POSITION = 1.0 / 20.0
    _STD_WEIGHT_VELOCITY = 1.0 / 160.0

    def __init__(self, bbox: Sequence[float]) -> None:
        self.mean = np.zeros(8, dtype=np.float64)
        self.covariance = np.eye(8, dtype=np.float64)

        self._motion_mat = np.eye(8, dtype=np.float64)
        for index in range(4):
            self._motion_mat[index, index + 4] = 1.0
        self._update_mat = np.eye(4, 8, dtype=np.float64)

        self.initiate(bbox)

    def initiate(self, bbox: Sequence[float]) -> None:
        cx, cy, aspect, height = _bbox_to_measurement(bbox)
        self.mean[:4] = (cx, cy, aspect, height)

        std = [
            2 * self._STD_WEIGHT_POSITION * height,
            2 * self._STD_WEIGHT_POSITION * height,
            1e-2,
            2 * self._STD_WEIGHT_POSITION * height,
            10 * self._STD_WEIGHT_VELOCITY * height,
            10 * self._STD_WEIGHT_VELOCITY * height,
            1e-5,
            10 * self._STD_WEIGHT_VELOCITY * height,
        ]
        self.covariance = np.diag(np.square(std))

    def predict(self) -> None:
        height = max(float(self.mean[3]), 1e-6)
        std_position = [
            self._STD_WEIGHT_POSITION * height,
            self._STD_WEIGHT_POSITION * height,
            1e-2,
            self._STD_WEIGHT_POSITION * height,
        ]
        std_velocity = [
            self._STD_WEIGHT_VELOCITY * height,
            self._STD_WEIGHT_VELOCITY * height,
            1e-5,
            self._STD_WEIGHT_VELOCITY * height,
        ]
        process_noise = np.diag(np.square(std_position + std_velocity))

        self.mean = self._motion_mat @ self.mean
        self.covariance = (
            self._motion_mat @ self.covariance @ self._motion_mat.T + process_noise
        )

    def update(self, bbox: Sequence[float]) -> None:
        measurement = np.asarray(
            _bbox_to_measurement(bbox), dtype=np.float64
        )
        height = max(float(self.mean[3]), 1e-6)
        std = [
            self._STD_WEIGHT_POSITION * height,
            self._STD_WEIGHT_POSITION * height,
            1e-1,
            self._STD_WEIGHT_POSITION * height,
        ]
        measurement_noise = np.diag(np.square(std))

        projected_cov = self._update_mat @ self.covariance @ self._update_mat.T
        innovation_cov = projected_cov + measurement_noise

        gain = np.linalg.solve(
            innovation_cov,
            (self.covariance @ self._update_mat.T).T,
        ).T

        innovation = measurement - self._update_mat @ self.mean
        self.mean = self.mean + gain @ innovation
        self.covariance = (
            np.eye(8, dtype=np.float64) - gain @ self._update_mat
        ) @ self.covariance

    @property
    def bbox(self) -> BBox:
        cx, cy = float(self.mean[0]), float(self.mean[1])
        aspect = max(float(self.mean[2]), 1e-6)
        height = max(float(self.mean[3]), 1e-6)
        width = aspect * height
        return (cx - width / 2.0, cy - height / 2.0, width, height)


def _bbox_to_measurement(
    bbox: Sequence[float]
) -> Tuple[float, float, float, float]:
    x, y, width, height = (float(value) for value in bbox)
    height = max(height, 1e-6)
    width = max(width, 1e-6)
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    aspect = width / height
    return (center_x, center_y, aspect, height)

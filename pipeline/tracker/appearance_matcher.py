"""外观兜底匹配：只在位置匹配失败时，用 SFace 特征认回原 Track。"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from pipeline.embedder.base import FaceEmbedder
from pipeline.schema import FaceDetection
from pipeline.tracker.face_geometry import eye_ratio
from pipeline.tracker.state import TrackState
from utils.math import cosine_similarity, ema_update, l2_normalize

Match = Tuple[int, TrackState]


class AppearanceMatcher:
    def __init__(
        self,
        embedder: FaceEmbedder,
        similarity_threshold: float,
        ema_alpha: float,
        min_face_size: int,
        max_lost_frames: int = 5,
        min_eye_ratio: float = 0.12,
    ) -> None:
        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                f"appearance_similarity 非法：{similarity_threshold}"
            )
        if not 0.0 <= ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha 非法：{ema_alpha}")
        if min_face_size <= 0:
            raise ValueError(f"appearance_min_face_size 非法：{min_face_size}")
        if max_lost_frames < 0:
            raise ValueError(
                f"appearance_max_lost_frames 非法：{max_lost_frames}"
            )
        if not 0.0 <= min_eye_ratio <= 1.0:
            raise ValueError(
                f"appearance_min_eye_ratio 非法：{min_eye_ratio}"
            )

        self._embedder = embedder
        self._similarity_threshold = similarity_threshold
        self._ema_alpha = ema_alpha
        self._min_face_size = min_face_size
        self._max_lost_frames = max_lost_frames
        self._min_eye_ratio = min_eye_ratio

        self.calls = 0
        self.matches = 0

    def embed_detections(
        self,
        image: np.ndarray,
        detections: Sequence[FaceDetection],
    ) -> List[Optional[np.ndarray]]:
        """批量提取特征；质量不达标或提取失败时返回 None。"""

        return [self._embed_one(image, detection) for detection in detections]

    def associate(
        self,
        embeddings: Sequence[Optional[np.ndarray]],
        tracks: Sequence[TrackState],
    ) -> List[Match]:
        """用余弦相似度做一对一匹配，返回 (检测下标, Track)。"""

        candidates = [
            (index, embedding)
            for index, embedding in enumerate(embeddings)
            if embedding is not None
        ]
        candidates_with_ema = [
            track
            for track in tracks
            if track.appearance_ema is not None
            and track.lost_frames <= self._max_lost_frames
        ]
        if not candidates or not candidates_with_ema:
            return []

        similarity = np.full(
            (len(candidates), len(candidates_with_ema)),
            -1.0,
            dtype=np.float64,
        )
        for row, (_, embedding) in enumerate(candidates):
            for col, track in enumerate(candidates_with_ema):
                try:
                    similarity[row, col] = cosine_similarity(
                        embedding, track.appearance_ema
                    )
                except ValueError:
                    similarity[row, col] = -1.0

        rows, cols = linear_sum_assignment(-similarity)
        matches: List[Match] = []
        for row, col in zip(rows, cols):
            if similarity[row, col] < self._similarity_threshold:
                continue
            matches.append((candidates[row][0], candidates_with_ema[col]))

        self.matches += len(matches)
        return matches

    def update(
        self,
        track: TrackState,
        embedding: np.ndarray,
        frame_id: int,
    ) -> None:
        """更新 Track 的滑动平均特征；首次直接初始化。"""

        normalized = l2_normalize(embedding)
        if (
            track.appearance_ema is None
            or track.appearance_ema.size != normalized.size
        ):
            track.appearance_ema = normalized
            track.appearance_updates = 1
        else:
            track.appearance_ema = ema_update(
                track.appearance_ema,
                normalized,
                self._ema_alpha,
            )
            track.appearance_updates += 1
        track.appearance_frame = frame_id

    def _embed_one(
        self,
        image: np.ndarray,
        detection: FaceDetection,
    ) -> Optional[np.ndarray]:
        _, _, width, height = detection.bbox
        if min(width, height) < self._min_face_size:
            return None
        if len(detection.landmarks) != 5:
            return None
        if not self._plausible_landmarks(detection):
            return None

        self.calls += 1
        try:
            return self._embedder.embed(image, detection)
        except (RuntimeError, ValueError):
            return None

    def _plausible_landmarks(self, detection: FaceDetection) -> bool:
        """过滤眼距过小的手部/背景误检，避免污染外观匹配。"""

        ratio = eye_ratio(detection)
        return ratio is not None and ratio >= self._min_eye_ratio

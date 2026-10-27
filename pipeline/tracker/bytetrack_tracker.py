"""ByteTrack 风格的两阶段关联跟踪器。

主路径是卡尔曼预测 + IoU 匈牙利匹配（不是最近邻）：
1. 高置信检测与全部 Track 做第一次匹配；
2. 低置信检测与剩余 Track 做第二次匹配；
3. 仍未匹配的高置信检测才新建 Track；
4. 连续丢失超过 max_lost_frames 才删除。

遮挡不再无条件保持：丢失后只有附近仍存在检测证据时才继续遮挡，
避免旧位置已经没有脸、遮挡框却还残留。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from pipeline.schema import FaceDetection
from pipeline.tracker.appearance_matcher import AppearanceMatcher
from pipeline.tracker.base import Tracker
from pipeline.tracker.face_geometry import FaceGeometryPrior
from pipeline.tracker.kalman import KalmanBoxFilter
from pipeline.tracker.matching import iou_matrix, linear_assignment
from pipeline.tracker.state import TrackState


class ByteTrackTracker(Tracker):
    def __init__(
        self,
        high_score_threshold: float,
        low_score_threshold: float,
        iou_threshold: float,
        second_iou_threshold: float,
        max_lost_frames: int,
        mask_lost_frames: int = 3,
        lost_mask_distance_ratio: float = 1.5,
        frame_step: int = 1,
        appearance_matcher: Optional[AppearanceMatcher] = None,
        geometry_prior: Optional[FaceGeometryPrior] = None,
    ) -> None:
        if not 0.0 <= low_score_threshold < high_score_threshold <= 1.0:
            raise ValueError(
                "需要满足 0 <= low_score_threshold < high_score_threshold <= 1"
            )
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold 非法：{iou_threshold}")
        if not 0.0 <= second_iou_threshold <= 1.0:
            raise ValueError(
                f"second_iou_threshold 非法：{second_iou_threshold}"
            )
        if max_lost_frames < 0:
            raise ValueError(f"max_lost_frames 不能为负：{max_lost_frames}")
        if not 0 <= mask_lost_frames <= max_lost_frames:
            raise ValueError(
                "mask_lost_frames 必须满足 0 <= mask_lost_frames <= max_lost_frames"
            )
        if lost_mask_distance_ratio <= 0:
            raise ValueError(
                "lost_mask_distance_ratio 必须大于 0"
            )
        if frame_step < 1:
            raise ValueError(f"frame_step 必须 >= 1：{frame_step}")

        self._high_score_threshold = high_score_threshold
        self._low_score_threshold = low_score_threshold
        self._first_cost_threshold = 1.0 - iou_threshold
        self._second_cost_threshold = 1.0 - second_iou_threshold
        self._max_lost_frames = max_lost_frames
        self._mask_lost_frames = mask_lost_frames
        self._lost_mask_distance_ratio = lost_mask_distance_ratio
        self._frame_step = frame_step
        self._appearance_matcher = appearance_matcher
        self._geometry_prior = geometry_prior

        self._tracks: Dict[int, TrackState] = {}
        self._next_id = 1

    @property
    def tracks(self) -> List[TrackState]:
        return self._active_tracks()

    def update(
        self,
        detections: Sequence[FaceDetection],
        frame_id: int,
        image: Optional[np.ndarray] = None,
    ) -> List[TrackState]:
        self._predict_all()

        high_detections = [
            detection
            for detection in detections
            if detection.confidence >= self._high_score_threshold
        ]
        low_detections = [
            detection
            for detection in detections
            if self._low_score_threshold
            <= detection.confidence
            < self._high_score_threshold
        ]

        tracks = list(self._tracks.values())

        # 第一次关联：高置信检测 vs 全部 Track
        cost = 1.0 - iou_matrix(
            [track.bbox for track in tracks],
            [detection.bbox for detection in high_detections],
        )
        matches, unmatched_track_indices, unmatched_detection_indices = (
            linear_assignment(cost, self._first_cost_threshold)
        )
        for track_index, detection_index in matches:
            detection = high_detections[detection_index]
            self._apply_detection(
                tracks[track_index],
                detection,
                frame_id,
            )
            if self._geometry_prior is not None:
                self._geometry_prior.observe(detection)

        for track_index in unmatched_track_indices:
            track = tracks[track_index]
            track.status = "lost"
            track.updated = False
            track.lost_frames += self._frame_step
            track.mask_state = (
                track.lost_frames <= self._mask_lost_frames
                and self._has_nearby_detection(track, detections)
            )

        # 外观兜底：只处理第一次位置匹配失败的高置信检测
        remaining_track_indices = list(unmatched_track_indices)
        pending_high_indices = list(unmatched_detection_indices)
        embeddings: Dict[int, Optional[np.ndarray]] = {}
        appearance_matched_detections = set()
        appearance_matched_tracks = set()

        if (
            self._appearance_matcher is not None
            and image is not None
            and pending_high_indices
        ):
            pending_detections = [
                high_detections[index] for index in pending_high_indices
            ]
            embedding_list = self._appearance_matcher.embed_detections(
                image, pending_detections
            )
            for position, original_index in enumerate(pending_high_indices):
                embeddings[original_index] = embedding_list[position]

            remaining_tracks = [
                tracks[index] for index in remaining_track_indices
            ]
            if remaining_tracks:
                pairs = self._appearance_matcher.associate(
                    embedding_list, remaining_tracks
                )
                for position, track in pairs:
                    original_index = pending_high_indices[position]
                    self._apply_detection(
                        track,
                        high_detections[original_index],
                        frame_id,
                    )
                    embedding = embedding_list[position]
                    if embedding is not None:
                        self._appearance_matcher.update(
                            track, embedding, frame_id
                        )
                    appearance_matched_detections.add(original_index)
                    appearance_matched_tracks.add(track.track_id)

        # 第二次关联：低置信检测 vs 第一次未匹配且未被外观认回的 Track
        remaining_tracks = [
            tracks[index]
            for index in remaining_track_indices
            if tracks[index].track_id not in appearance_matched_tracks
        ]
        cost_second = 1.0 - iou_matrix(
            [track.bbox for track in remaining_tracks],
            [detection.bbox for detection in low_detections],
        )
        matches_second, _, _ = linear_assignment(
            cost_second, self._second_cost_threshold
        )
        for track_index, detection_index in matches_second:
            self._apply_detection(
                remaining_tracks[track_index],
                low_detections[detection_index],
                frame_id,
            )

        # 仍未匹配的高置信检测：几何先验通过的才新建 Track
        for detection_index in pending_high_indices:
            if detection_index in appearance_matched_detections:
                continue
            detection = high_detections[detection_index]
            if (
                self._geometry_prior is not None
                and not self._geometry_prior.is_plausible(detection)
            ):
                continue
            self._create_track(
                detection,
                frame_id,
                embeddings.get(detection_index),
            )

        self._remove_expired_tracks()
        return self._active_tracks()

    def predict(self, frame_id: int) -> List[TrackState]:
        self._predict_all()
        return self._active_tracks()

    def _predict_all(self) -> None:
        for track in self._tracks.values():
            track.kalman.predict()
            track.last_bbox = track.kalman.bbox
            track.updated = False

    def _apply_detection(
        self,
        track: TrackState,
        detection: FaceDetection,
        frame_id: int,
    ) -> None:
        track.kalman.update(detection.bbox)
        track.last_bbox = track.kalman.bbox
        track.measurement_bbox = detection.bbox
        track.score = detection.confidence
        track.last_seen_frame = frame_id
        track.lost_frames = 0
        track.status = "tracked"
        track.updated = True
        track.mask_state = True

    def _create_track(
        self,
        detection: FaceDetection,
        frame_id: int,
        embedding: Optional[np.ndarray] = None,
    ) -> None:
        kalman = KalmanBoxFilter(detection.bbox)
        track = TrackState(
            track_id=self._next_id,
            kalman=kalman,
            score=detection.confidence,
            last_bbox=kalman.bbox,
            last_seen_frame=frame_id,
            measurement_bbox=detection.bbox,
            status="tracked",
            updated=True,
            mask_state=True,
        )
        if (
            embedding is not None
            and self._appearance_matcher is not None
        ):
            self._appearance_matcher.update(track, embedding, frame_id)
        self._tracks[self._next_id] = track
        self._next_id += 1

    def _has_nearby_detection(
        self,
        track: TrackState,
        detections: Sequence[FaceDetection],
    ) -> bool:
        """旧位置附近是否还有任意检测，作为“脸仍在”的证据。"""

        x, y, width, height = track.bbox
        center_x = x + width / 2.0
        center_y = y + height / 2.0
        scale = max(width, height, 1.0)

        for detection in detections:
            dx, dy, dw, dh = detection.bbox
            distance = math.hypot(
                center_x - (dx + dw / 2.0),
                center_y - (dy + dh / 2.0),
            ) / scale
            if distance <= self._lost_mask_distance_ratio:
                return True
        return False

    def _remove_expired_tracks(self) -> None:
        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if track.lost_frames > self._max_lost_frames
        ]
        for track_id in expired:
            del self._tracks[track_id]

    def _active_tracks(self) -> List[TrackState]:
        return sorted(self._tracks.values(), key=lambda track: track.track_id)

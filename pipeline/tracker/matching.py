"""检测框关联：IoU 矩阵与匈牙利匹配（对应 ByteTrack matching 的逻辑）。"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

Match = Tuple[int, int]


def iou_matrix(
    boxes_a: Sequence[Sequence[float]],
    boxes_b: Sequence[Sequence[float]],
) -> np.ndarray:
    if not boxes_a or not boxes_b:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)

    a = np.asarray(boxes_a, dtype=np.float64)
    b = np.asarray(boxes_b, dtype=np.float64)

    a_x1 = a[:, 0]
    a_y1 = a[:, 1]
    a_x2 = a[:, 0] + a[:, 2]
    a_y2 = a[:, 1] + a[:, 3]
    b_x1 = b[:, 0]
    b_y1 = b[:, 1]
    b_x2 = b[:, 0] + b[:, 2]
    b_y2 = b[:, 1] + b[:, 3]

    inter_width = np.clip(
        np.minimum(a_x2[:, None], b_x2[None, :])
        - np.maximum(a_x1[:, None], b_x1[None, :]),
        0.0,
        None,
    )
    inter_height = np.clip(
        np.minimum(a_y2[:, None], b_y2[None, :])
        - np.maximum(a_y1[:, None], b_y1[None, :]),
        0.0,
        None,
    )
    intersection = inter_width * inter_height

    area_a = np.clip(a_x2 - a_x1, 0.0, None) * np.clip(a_y2 - a_y1, 0.0, None)
    area_b = np.clip(b_x2 - b_x1, 0.0, None) * np.clip(b_y2 - b_y1, 0.0, None)
    union = area_a[:, None] + area_b[None, :] - intersection

    return np.where(union > 0, intersection / union, 0.0)


def linear_assignment(
    cost_matrix: np.ndarray,
    threshold: float,
) -> Tuple[List[Match], List[int], List[int]]:
    """匈牙利匹配，并丢弃代价超过 threshold 的配对。"""

    rows, cols = cost_matrix.shape
    if rows == 0 or cols == 0:
        return [], list(range(rows)), list(range(cols))

    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    matches: List[Match] = []
    unmatched_rows = set(range(rows))
    unmatched_cols = set(range(cols))

    for row, col in zip(row_indices, col_indices):
        if cost_matrix[row, col] > threshold:
            continue
        matches.append((int(row), int(col)))
        unmatched_rows.discard(int(row))
        unmatched_cols.discard(int(col))

    return matches, sorted(unmatched_rows), sorted(unmatched_cols)

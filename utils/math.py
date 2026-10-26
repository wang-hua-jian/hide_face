"""向量工具：L2 归一化、余弦相似度、EMA 更新。"""

from __future__ import annotations

import numpy as np


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """返回 L2 归一化后的向量；零向量原样返回，避免 NaN。"""

    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        return array.copy()
    return array / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度，范围 [-1, 1]；零向量与任何向量相似度为 0。"""

    vector_a = np.asarray(a, dtype=np.float32).reshape(-1)
    vector_b = np.asarray(b, dtype=np.float32).reshape(-1)
    if vector_a.size != vector_b.size:
        raise ValueError(
            f"向量维度不一致：{vector_a.size} vs {vector_b.size}"
        )

    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    similarity = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
    return float(np.clip(similarity, -1.0, 1.0))


def ema_update(old: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
    """滑动平均：normalize(alpha * old + (1 - alpha) * new)。"""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha 必须在 [0, 1] 内：{alpha}")

    vector_old = np.asarray(old, dtype=np.float32).reshape(-1)
    vector_new = np.asarray(new, dtype=np.float32).reshape(-1)
    if vector_old.size != vector_new.size:
        raise ValueError(
            f"向量维度不一致：{vector_old.size} vs {vector_new.size}"
        )

    return l2_normalize(alpha * vector_old + (1.0 - alpha) * vector_new)

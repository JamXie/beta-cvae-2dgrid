"""径向基函数插值（RBFI），用于在 (时间步, 角度) 上连续生成潜变量。"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RBFInterpolator


def fit_rbfi(
    points: np.ndarray,
    values: np.ndarray,
    *,
    kernel: str = "multiquadric",
    epsilon: float = 1.8,
    smoothing: float = 0.0,
) -> RBFInterpolator:
    """拟合 `(step, degree) -> latent` 的 RBF 插值器。"""
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points 应为 [N, 2]，实际为 {points.shape}")
    if values.shape[0] != points.shape[0]:
        raise ValueError("points 与 values 的样本数不一致")

    return RBFInterpolator(
        points,
        values,
        kernel=kernel,
        epsilon=epsilon,
        smoothing=smoothing,
    )


def query_rbfi(rbf: RBFInterpolator, steps, angles) -> np.ndarray:
    """在给定 (时间步, 角度) 上查询潜变量，返回 `[N, latent_dim]`。"""
    steps = np.atleast_1d(np.asarray(steps, dtype=np.float64)).reshape(-1)
    angles = np.atleast_1d(np.asarray(angles, dtype=np.float64)).reshape(-1)

    if steps.size == 1 and angles.size > 1:
        steps = np.repeat(steps, angles.size)
    elif angles.size == 1 and steps.size > 1:
        angles = np.repeat(angles, steps.size)
    elif steps.size != angles.size:
        raise ValueError("steps 与 angles 长度不一致，且都不是单值")

    query = np.stack([steps, angles], axis=1)
    return np.asarray(rbf(query), dtype=np.float64)

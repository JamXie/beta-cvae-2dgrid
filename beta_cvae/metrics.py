"""评价指标：位移/应力 MAE、RMSE 与欧式位移误差。

输入网格为归一化后的 `[4, 80, 84]`（或批量的 `[N, 4, 80, 84]`），
计算前按论文尺度反归一化：位移 ×300，应力 ×200。
"""

from __future__ import annotations

import numpy as np

from .constants import DISP_SCALE, STRESS_SCALE

AXES = ("x", "y", "z")


def denormalize(grid: np.ndarray):
    """返回 `(displacement[N,3,H,W], stress[N,H,W])`，单位为 mm 与 MPa。"""
    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim == 3:
        grid = grid[None]
    if grid.ndim != 4 or grid.shape[1] != 4:
        raise ValueError(f"期望 [N,4,H,W] 网格，实际为 {grid.shape}")

    displacement = grid[:, :3] * DISP_SCALE
    stress = grid[:, 3] * STRESS_SCALE
    return displacement, stress


def grid_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """计算单帧或批量的误差指标。"""
    pred_disp, pred_stress = denormalize(pred)
    true_disp, true_stress = denormalize(target)

    if pred_disp.shape != true_disp.shape:
        raise ValueError(f"预测与目标形状不一致：{pred_disp.shape} vs {true_disp.shape}")

    diff = pred_disp - true_disp
    stress_diff = pred_stress - true_stress

    metrics: dict[str, float] = {}
    for idx, axis in enumerate(AXES):
        channel = diff[:, idx]
        metrics[f"mae_{axis}_mm"] = float(np.mean(np.abs(channel)))
        metrics[f"rmse_{axis}_mm"] = float(np.sqrt(np.mean(channel ** 2)))

    # 每个节点的三维位移误差模长
    euclidean = np.sqrt(np.sum(diff ** 2, axis=1))
    metrics["euclidean_mae_mm"] = float(np.mean(euclidean))
    metrics["euclidean_rmse_mm"] = float(np.sqrt(np.mean(euclidean ** 2)))

    metrics["stress_mae_mpa"] = float(np.mean(np.abs(stress_diff)))
    metrics["stress_rmse_mpa"] = float(np.sqrt(np.mean(stress_diff ** 2)))
    return metrics

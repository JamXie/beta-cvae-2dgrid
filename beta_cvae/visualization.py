"""可视化：损失曲线、2D 通道对比图与三维点云图。

所有图都保存为 PNG，不调用 `plt.show()`；后备端在模块导入时设为 Agg。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import DISP_SCALE, STRESS_SCALE
from .utils import use_agg_backend

use_agg_backend()

import matplotlib.pyplot as plt  # noqa: E402

CHANNEL_NAMES = ("dx", "dy", "dz", "Mises")


def save_loss_curve(history: dict, path) -> Path:
    """保存训练/验证损失曲线。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, history["train_loss"], "r-o", label="train total")
    ax.plot(epochs, history["train_mse"], "b-s", label="train MSE")
    ax.plot(epochs, history["train_kl"], "g-^", label="train KL")
    ax.plot(epochs, history["val_loss"], "y--d", label="val total")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Beta-CVAE training")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_channel_comparison(pred, target, path, title: str = "Prediction vs target") -> Path:
    """两行四列：上行为预测，下行为目标。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.ndim == 4:
        pred = pred[0]
    if target.ndim == 4:
        target = target[0]

    fig, axes = plt.subplots(2, 4, figsize=(13, 5.5))
    for channel in range(4):
        for row, (data, label) in enumerate(((pred, "pred"), (target, "true"))):
            ax = axes[row, channel]
            image = ax.imshow(data[channel], cmap="viridis", aspect="auto")
            ax.set_title(f"{label} {CHANNEL_NAMES[channel]}")
            ax.set_xlabel("layer index")
            ax.set_ylabel("ring index")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_prediction_montage(grids, steps, path, *, angle: int) -> Path:
    """按时间步逐行展示四个通道的预测结果。"""
    grids = np.asarray(grids)
    if grids.ndim == 3:
        grids = grids[None]

    num_steps = grids.shape[0]
    steps = list(steps)

    fig, axes = plt.subplots(num_steps, 4, figsize=(12, 2.4 * num_steps), squeeze=False)
    for row in range(num_steps):
        for channel in range(4):
            ax = axes[row, channel]
            image = ax.imshow(grids[row, channel], cmap="viridis", aspect="auto")
            ax.set_title(f"step={steps[row]}  {CHANNEL_NAMES[channel]}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(f"Predicted 2D grids at angle={angle} deg")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_3d_prediction(grid, geometry_path, path, *, angle: int, step: int, max_points: int = 4000) -> Path | None:
    """按参考节点坐标与预测位移绘制三维点云；geometry 缺失时返回 None。"""
    geometry_path = Path(geometry_path)
    if not geometry_path.exists():
        return None

    data = np.load(geometry_path, allow_pickle=False)
    reference_nodes = data["reference_nodes"]
    grid_index = data["grid_index"]

    grid = np.asarray(grid)
    if grid.ndim == 4:
        grid = grid[0]

    height, width = grid_index.shape
    rows = min(height, grid.shape[1])
    cols = min(width, grid.shape[2])

    node_ids = grid_index[:rows, :cols].reshape(-1)
    flat = grid[:, :rows, :cols].reshape(4, -1)
    valid = node_ids >= 0
    node_ids = node_ids[valid]
    flat = flat[:, valid]

    points = reference_nodes[node_ids] + flat[:3].T * DISP_SCALE
    stress = flat[3] * STRESS_SCALE

    if points.shape[0] > max_points:
        rng = np.random.RandomState(0)
        selection = rng.choice(points.shape[0], max_points, replace=False)
        points = points[selection]
        stress = stress[selection]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=stress, cmap="jet", s=2, vmin=0.0,
    )
    ax.set_title(f"Predicted shape  angle={angle} deg  step={step}")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    fig.colorbar(scatter, ax=ax, shrink=0.6, label="Mises (MPa)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

"""从训练数据中提取结构潜变量（编码器均值 mu）。"""

from __future__ import annotations

import numpy as np
import torch

from ..models.registry import get_spec_for
from ..training.trainer import normalize_conditions, pad_to_network


@torch.no_grad()
def extract_latents(model, loader, device, max_batches: int | None = None):
    """返回 `(points, values)`。

    - `points`：`[N, 2]`，每行为 `[step, degree]` 的原始数值；
    - `values`：`[N, latent_dim]`，编码器均值。
    """
    spec = get_spec_for(model)
    model.eval()

    points: list[np.ndarray] = []
    values: list[np.ndarray] = []

    for batch_idx, (grid, degree, step) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        grid = grid.to(device)
        degree = degree.to(device)
        step = step.to(device)

        padded = pad_to_network(grid)
        time_step, angle = normalize_conditions(degree, step)

        if spec.encode_takes_condition:
            mu, _ = model.encode(padded, time_step, angle)
        else:
            mu, _ = model.encode(padded)

        points.append(torch.stack([step, degree], dim=1).cpu().numpy().astype(np.float64))
        values.append(mu.cpu().numpy().astype(np.float64))

    if not points:
        raise ValueError("没有得到任何潜变量，请检查数据加载器与 max_batches")

    return np.concatenate(points, axis=0), np.concatenate(values, axis=0)

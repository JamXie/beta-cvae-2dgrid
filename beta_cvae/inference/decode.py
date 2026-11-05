"""统一解码入口：按模型注册表声明的 `decode_style` 分派。"""

from __future__ import annotations

import numpy as np
import torch

from ..constants import ANGLE_SCALE, STEP_SCALE
from ..models.registry import get_spec_for


def _as_batch_tensor(value, device, dtype=torch.float32) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device).reshape(-1)
    return tensor


@torch.no_grad()
def decode_from_latent(model, z, step, angle, device=None) -> torch.Tensor:
    """由结构潜变量与条件解码出 `[N, 4, 96, 96]` 网格。

    - `z`：`[N, latent_dim]` 或 `[latent_dim]`；
    - `step` / `angle`：标量或长度 N 的数组（原始数值，内部归一化）。

    条件采样模型（`decode_style="z_zt_za"`）在评估时使用条件嵌入的均值，
    不添加随机噪声，保证连续预测可重复。
    """
    if device is None:
        device = next(model.parameters()).device

    z = torch.as_tensor(z, dtype=torch.float32, device=device)
    if z.ndim == 1:
        z = z.unsqueeze(0)
    num = z.size(0)

    steps = _as_batch_tensor(step, device)
    angles = _as_batch_tensor(angle, device)
    if steps.numel() == 1:
        steps = steps.expand(num)
    if angles.numel() == 1:
        angles = angles.expand(num)
    if steps.numel() != num or angles.numel() != num:
        raise ValueError("step / angle 的数量必须与 z 的 batch 大小一致")

    time_step = steps / STEP_SCALE
    angle_norm = angles / ANGLE_SCALE

    spec = get_spec_for(model)
    model.eval()

    if spec.decode_style == "z":
        return model.decode(z)
    if spec.decode_style == "z_t_a":
        return model.decode(z, time_step, angle_norm)
    if spec.decode_style == "z_zt_za":
        z_time = model.time_embed(time_step.view(-1, 1))
        z_angle = model.angle_embed(angle_norm.view(-1, 1))
        return model.decode(z, z_time, z_angle)

    raise ValueError(f"未知 decode_style: {spec.decode_style!r}")


def crop_prediction(pred: torch.Tensor) -> torch.Tensor:
    """把网络输出 `[N, 4, 96, 96]` 裁回 `[N, 4, 80, 84]`。"""
    return pred[:, :, 8:88, 6:90]

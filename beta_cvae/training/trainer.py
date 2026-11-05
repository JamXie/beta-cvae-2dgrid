"""训练循环：填充、掩码、Beta-CVAE 损失与验证。"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from ..constants import ANGLE_SCALE, PAD, STEP_SCALE
from ..models.registry import get_spec_for
from .losses import kl_divergence, masked_mse_loss


def pad_to_network(grid: torch.Tensor) -> torch.Tensor:
    """`[B, 4, 80, 84]` -> `[B, 4, 96, 96]`，边缘补零。"""
    return F.pad(grid, pad=PAD, mode="constant", value=0)


def make_mask(padded: torch.Tensor) -> torch.Tensor:
    """构造与填充后张量同形状的掩码，只保留有效网格区域。"""
    mask = torch.zeros_like(padded)
    mask[:, :, 8:88, 6:90] = 1.0
    return mask


def normalize_conditions(degree: torch.Tensor, step: torch.Tensor):
    """条件归一化：时间步 / 500，角度 / 45。"""
    return step.float() / STEP_SCALE, degree.float() / ANGLE_SCALE


def _forward(model, x, time_step, angle, takes_condition: bool):
    if takes_condition:
        return model(x, time_step, angle)
    return model(x)


def run_epoch(
    model,
    loader,
    *,
    beta: float,
    device,
    optimizer=None,
    max_batches: int | None = None,
) -> dict[str, float]:
    """跑一个 epoch；optimizer 为 None 时只做验证。"""
    training = optimizer is not None
    spec = get_spec_for(model)
    model.train(training)

    totals = {"loss": 0.0, "mse": 0.0, "kl": 0.0}
    num_batches = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_idx, (grid, degree, step) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            grid = grid.to(device)
            degree = degree.to(device)
            step = step.to(device)

            padded = pad_to_network(grid)
            mask = make_mask(padded)
            time_step, angle = normalize_conditions(degree, step)

            if training:
                optimizer.zero_grad(set_to_none=True)

            pred, mu, logvar = _forward(model, padded, time_step, angle, spec.forward_takes_condition)
            mse = masked_mse_loss(pred, padded, mask)
            kl = kl_divergence(mu, logvar, beta)
            loss = mse + kl

            if training:
                loss.backward()
                optimizer.step()

            totals["loss"] += loss.item()
            totals["mse"] += mse.item()
            totals["kl"] += kl.item()
            num_batches += 1

    num_batches = max(num_batches, 1)
    return {key: value / num_batches for key, value in totals.items()}


def train_model(
    model,
    train_loader,
    val_loader,
    *,
    beta: float = 1e-6,
    epochs: int = 1,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    device=None,
    max_batches: int | None = None,
    log_fn=print,
) -> dict[str, list[float]]:
    """训练模型并返回逐 epoch 的损失历史。"""
    if device is None:
        device = torch.device("cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "train_mse": [],
        "train_kl": [],
        "val_loss": [],
        "val_mse": [],
        "val_kl": [],
    }

    for epoch in range(epochs):
        start = time.time()
        train_stats = run_epoch(
            model,
            train_loader,
            beta=beta,
            device=device,
            optimizer=optimizer,
            max_batches=max_batches,
        )
        val_stats = run_epoch(
            model,
            val_loader,
            beta=beta,
            device=device,
            optimizer=None,
            max_batches=max_batches,
        )

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_stats["loss"])
        history["train_mse"].append(train_stats["mse"])
        history["train_kl"].append(train_stats["kl"])
        history["val_loss"].append(val_stats["loss"])
        history["val_mse"].append(val_stats["mse"])
        history["val_kl"].append(val_stats["kl"])

        log_fn(
            f"epoch {epoch + 1}/{epochs} "
            f"train_loss={train_stats['loss']:.6f} mse={train_stats['mse']:.6f} "
            f"kl={train_stats['kl']:.6f} | "
            f"val_loss={val_stats['loss']:.6f} ({time.time() - start:.1f}s)"
        )

        # 只在 CUDA 上清理缓存，CPU 训练不调用
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return history

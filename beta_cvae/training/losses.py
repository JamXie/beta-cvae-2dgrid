"""损失函数。

保持原始工程形式：

```text
loss = masked_MSE + beta * 0.5 * sum(mu^2 + logvar^2 - log(1e-8 + logvar^2) - 1)
                     / (batch_size * latent_dim)
```
"""

from __future__ import annotations

import torch


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """只在 mask == 1 的区域计算 MSE（含填充边界之外被忽略）。"""
    diff = (pred - target) * mask
    return torch.sum(diff ** 2) / torch.sum(mask)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor, beta: float) -> torch.Tensor:
    """Beta 加权 KL 项，保留原始工程的写法和数值细节。"""
    return (
        beta
        * 0.5
        * torch.sum(
            torch.pow(mu, 2)
            + torch.pow(logvar, 2)
            - torch.log(1e-8 + torch.pow(logvar, 2))
            - 1
        )
        / (mu.size(0) * mu.size(-1))
    )


def vae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
):
    """返回 (总损失, MSE, KL)。"""
    mse = masked_mse_loss(pred, target, mask)
    kl = kl_divergence(mu, logvar, beta)
    return mse + kl, mse, kl

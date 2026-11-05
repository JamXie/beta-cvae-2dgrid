"""训练相关测试：掩码 MSE、KL 数值与一步优化。"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from beta_cvae.models.registry import build_model
from beta_cvae.training.losses import kl_divergence, masked_mse_loss
from beta_cvae.training.trainer import make_mask, run_epoch


def test_masked_mse_ignores_padding():
    """掩码外的数值改动不影响损失。"""
    torch.manual_seed(0)
    pred = torch.randn(2, 4, 96, 96)
    target = torch.randn(2, 4, 96, 96)
    mask = make_mask(pred)

    pred_perturbed = pred.clone()
    target_perturbed = target.clone()
    pred_perturbed[mask == 0] = 123.0
    target_perturbed[mask == 0] = -456.0

    base = masked_mse_loss(pred, target, mask)
    perturbed = masked_mse_loss(pred_perturbed, target_perturbed, mask)
    assert torch.equal(base, perturbed)

    # 期望值等于有效区域上的手算 MSE
    diff = (pred - target)[:, :, 8:88, 6:90]
    expected = torch.sum(diff ** 2) / diff.numel()
    assert torch.allclose(base, expected)


def test_kl_matches_manual_formula():
    """KL 项与论文公式逐项一致。"""
    torch.manual_seed(0)
    mu = torch.randn(4, 16) * 0.1
    logvar = torch.randn(4, 16) * 0.1
    beta = 1e-3

    manual = (
        beta
        * 0.5
        * torch.sum(mu ** 2 + logvar ** 2 - torch.log(1e-8 + logvar ** 2) - 1)
        / (mu.size(0) * mu.size(-1))
    )
    assert torch.allclose(kl_divergence(mu, logvar, beta), manual)


def test_one_step_training_changes_parameters():
    """一步优化后参数发生变化，损失保持有限。"""
    torch.manual_seed(0)
    model = build_model("VAE")
    before = {name: tensor.detach().clone() for name, tensor in model.named_parameters()}

    grids = torch.randn(4, 4, 80, 84) * 0.1
    dataset = TensorDataset(grids, torch.tensor([0, 5, 10, 15]), torch.tensor([0, 50, 100, 150]))
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    stats = run_epoch(model, loader, beta=1e-6, device=torch.device("cpu"), optimizer=optimizer)

    assert np.isfinite(stats["loss"])
    assert stats["loss"] >= 0.0
    changed = any(
        not torch.equal(before[name], tensor.detach())
        for name, tensor in model.named_parameters()
    )
    assert changed


def test_val_epoch_does_not_require_optimizer():
    """验证模式（optimizer=None）可正常计算损失。"""
    torch.manual_seed(0)
    model = build_model("VAE").eval()
    grids = torch.randn(2, 4, 80, 84) * 0.1
    dataset = TensorDataset(grids, torch.tensor([0, 5]), torch.tensor([0, 50]))
    loader = DataLoader(dataset, batch_size=2)

    stats = run_epoch(model, loader, beta=1e-6, device=torch.device("cpu"), optimizer=None)
    assert np.isfinite(stats["mse"])
    assert stats["kl"] >= 0.0

"""模型测试：注册表内全部模型的前向输出形状与统一解码。"""

from __future__ import annotations

import torch
import pytest

from beta_cvae.inference.decode import crop_prediction, decode_from_latent
from beta_cvae.models.registry import (
    DEFAULT_MODEL,
    MODEL_NAMES,
    build_model,
    get_spec,
)


def _condition_batch(batch: int = 2):
    """构造归一化后的时间步与角度条件。"""
    steps = torch.tensor([0.0, 250.0, 500.0])[:batch] / 500.0
    angles = torch.tensor([7.0, 23.0, 45.0])[:batch] / 45.0
    return steps, angles


def test_default_model_is_final():
    """默认模型为论文第 5 章最终结构。"""
    assert DEFAULT_MODEL == "M_Rm_CVAE02"
    assert DEFAULT_MODEL in MODEL_NAMES


def test_unknown_model_raises():
    """未注册的模型名给出明确错误。"""
    with pytest.raises(KeyError):
        build_model("NotAModel")


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_all_registered_models_forward(name):
    """全部模型：输出 `[2,4,96,96]`，mu/logvar 为 `[2,16]`。"""
    torch.manual_seed(0)
    model = build_model(name).eval()
    spec = get_spec(name)
    x = torch.randn(2, 4, 96, 96)
    time_step, angle = _condition_batch(2)

    with torch.no_grad():
        if spec.forward_takes_condition:
            pred, mu, logvar = model(x, time_step, angle)
        else:
            pred, mu, logvar = model(x)

    assert pred.shape == (2, 4, 96, 96)
    assert mu.shape == (2, 16)
    assert logvar.shape == (2, 16)
    assert torch.isfinite(pred).all()


@pytest.mark.parametrize(
    ("name", "style"),
    (("VAE", "z"), ("CVAE03", "z_t_a"), ("M_Rm_CVAE02", "z_zt_za")),
)
def test_decode_dispatch_by_style(name, style):
    """三种 decode_style 都能由统一入口解码并裁回 `[3,4,80,84]`。"""
    assert get_spec(name).decode_style == style
    model = build_model(name).eval()
    z = torch.randn(3, 16)
    steps = torch.tensor([0.0, 250.0, 500.0])
    angles = torch.full((3,), 23.0)

    pred = decode_from_latent(model, z, steps, angles)
    cropped = crop_prediction(pred)

    assert cropped.shape == (3, 4, 80, 84)
    assert torch.isfinite(cropped).all()

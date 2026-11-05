"""模型注册表：统一构造入口与推理所需的元数据。

推理时不能靠 `decode` 的形参数量猜测调用方式，因此每个模型显式声明：

- `decode_style`：
  - `"z"`：`decode(z)`；
  - `"z_t_a"`：`decode(z, t, a)`，条件嵌入在解码器内部完成；
  - `"z_zt_za"`：先由 `time_embed` / `angle_embed` 得到条件潜变量均值，
    再 `decode(z, z_t, z_a)`。
- `encode_takes_condition`：编码时是否需要时间步与角度。
- `forward_takes_condition`：训练时是否需要向 `forward` 传入条件。
"""

from __future__ import annotations

from dataclasses import dataclass

from .basic import CVAE01, CVAE02, CVAE03, CVAE04, VAE
from .final import (
    M_R_CVAE01,
    M_R_CVAE02,
    M_R_CVAE03,
    M_R_CVAE04,
    M_Rm_CVAE01,
    M_Rm_CVAE02,
    M_Rm_CVAE03,
    M_Rm_CVAE04,
)
from .multibranch import MultiBranchVAE01, MultiBranchVAE02, MultiBranchVAE03
from .residual import ResidualVAE01, ResidualVAE02, ResidualVAE03


@dataclass(frozen=True)
class ModelSpec:
    """单个模型的构造信息与调用约定。"""

    name: str
    cls: type
    family: str  # plain / residual / cvae / final
    decode_style: str
    encode_takes_condition: bool
    forward_takes_condition: bool


_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("VAE", VAE, "plain", "z", False, False),
    ModelSpec("CVAE01", CVAE01, "cvae", "z", True, True),
    ModelSpec("CVAE02", CVAE02, "cvae", "z_zt_za", False, True),
    ModelSpec("CVAE03", CVAE03, "cvae", "z_t_a", False, True),
    ModelSpec("CVAE04", CVAE04, "cvae", "z_t_a", True, True),
    ModelSpec("MultiBranchVAE01", MultiBranchVAE01, "plain", "z", False, False),
    ModelSpec("MultiBranchVAE02", MultiBranchVAE02, "plain", "z", False, False),
    ModelSpec("MultiBranchVAE03", MultiBranchVAE03, "plain", "z", False, False),
    ModelSpec("ResidualVAE01", ResidualVAE01, "residual", "z", False, False),
    ModelSpec("ResidualVAE02", ResidualVAE02, "residual", "z", False, False),
    ModelSpec("ResidualVAE03", ResidualVAE03, "residual", "z", False, False),
    ModelSpec("M_R_CVAE01", M_R_CVAE01, "final", "z", True, True),
    ModelSpec("M_R_CVAE02", M_R_CVAE02, "final", "z_zt_za", False, True),
    ModelSpec("M_R_CVAE03", M_R_CVAE03, "final", "z_t_a", False, True),
    ModelSpec("M_R_CVAE04", M_R_CVAE04, "final", "z_t_a", True, True),
    ModelSpec("M_Rm_CVAE01", M_Rm_CVAE01, "final", "z", True, True),
    ModelSpec("M_Rm_CVAE02", M_Rm_CVAE02, "final", "z_zt_za", False, True),
    ModelSpec("M_Rm_CVAE03", M_Rm_CVAE03, "final", "z_t_a", False, True),
    ModelSpec("M_Rm_CVAE04", M_Rm_CVAE04, "final", "z_t_a", True, True),
)

MODEL_NAMES: tuple[str, ...] = tuple(spec.name for spec in _SPECS)
_BY_NAME = {spec.name: spec for spec in _SPECS}
_BY_CLASS = {spec.cls.__name__: spec for spec in _SPECS}

# 默认模型：论文第 5 章最终版本
DEFAULT_MODEL = "M_Rm_CVAE02"


def get_spec(name: str) -> ModelSpec:
    """按名字取模型元数据，未注册时给出清晰错误。"""
    try:
        return _BY_NAME[name]
    except KeyError as exc:
        raise KeyError(f"未知模型 {name!r}，可选：{', '.join(MODEL_NAMES)}") from exc


def get_spec_for(model) -> ModelSpec:
    """按模型实例取元数据。"""
    try:
        return _BY_CLASS[type(model).__name__]
    except KeyError as exc:
        raise KeyError(f"模型类 {type(model).__name__!r} 未注册") from exc


def build_model(
    name: str = DEFAULT_MODEL,
    latent_num: int = 6,
    latent_value: int = 16,
    in_channels: int = 4,
    condition_dim: int = 1,
    img_size: int = 96,
):
    """按注册表构造模型实例。"""
    spec = get_spec(name)
    cls = spec.cls

    if spec.family == "plain":
        return cls(latent_Num=latent_num, latent_value=latent_value)
    if spec.family == "residual":
        return cls(
            latent_Num=latent_num,
            latent_value=latent_value,
            img_size=img_size,
            in_channels=in_channels,
        )
    if spec.family == "cvae":
        return cls(latent_Num=latent_num, latent_value=latent_value, condition_dim=condition_dim)
    if spec.family == "final":
        return cls(
            latent_Num=latent_num,
            latent_value=latent_value,
            in_channels=in_channels,
            condition_dim=condition_dim,
        )
    raise ValueError(f"未知模型族 {spec.family!r}")

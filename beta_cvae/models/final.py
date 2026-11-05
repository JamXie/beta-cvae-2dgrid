"""最终模型族：残差 + 多分支 + 条件 VAE。

- `M_R_CVAE01..04`：共享残差编码器 + 多通道独立解码器；
- `M_Rm_CVAE01..04`：多通道独立残差编码器 + 多通道独立解码器；
- 后缀 01..04 对应 CVAE01..04 的四种条件耦合方式。

论文最终模型为 `M_Rm_CVAE02`：四通道独立编码器、四通道独立解码器、
残差块，条件按 N(embedding, 1) 采样后与结构潜变量拼接。
"""

from __future__ import annotations

import torch
from torch import nn

from .basic import ResidualBlock


def _shared_residual_encoder(in_channels: int) -> nn.Sequential:
    """共享的四通道残差编码器，输出 [B, 256, 1, 1]。"""
    return nn.Sequential(
        ResidualBlock(in_channels, 32, downsample=True),
        ResidualBlock(32, 64, downsample=True),
        ResidualBlock(64, 128, downsample=True),
        ResidualBlock(128, 256, downsample=True),
        nn.AdaptiveAvgPool2d(1),
    )


def _residual_encoder_branch() -> nn.Sequential:
    """单通道残差编码分支，输出 [B, 256, 1, 1]。"""
    return nn.Sequential(
        ResidualBlock(1, 32, downsample=True),
        ResidualBlock(32, 64, downsample=True),
        ResidualBlock(64, 128, downsample=True),
        ResidualBlock(128, 256, downsample=True),
        nn.AdaptiveAvgPool2d(1),
    )


def _shared_decoder(latent_Num: int, in_features: int) -> nn.Sequential:
    """共享上采样头：latent -> 6×6 特征图 -> 12×12。"""
    return nn.Sequential(
        nn.Linear(in_features, 256 * latent_Num * latent_Num),
        nn.ReLU(),
        nn.Unflatten(1, (256, latent_Num, latent_Num)),
        nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
        nn.ReLU(),
    )


def _decoder_branch(out_channels: int = 1) -> nn.Sequential:
    """单通道解码分支：12×12 -> 24 -> 48 -> 96。"""
    return nn.Sequential(
        nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1),
        nn.Tanh(),
    )


class _FinalModelBase(nn.Module):
    """公共的潜变量重参数化与多分支解码逻辑。"""

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def _decode_feat(self, feat: torch.Tensor) -> torch.Tensor:
        disp_outs = [decoder(feat) for decoder in self.disp_decoders]
        stress_out = self.stress_decoder(feat)
        return torch.cat(disp_outs + [stress_out], dim=1)


class M_R_CVAE01(_FinalModelBase):
    """共享残差编码器；条件与结构耦合；条件不进入解码器。"""

    def __init__(self, latent_Num: int = 6, latent_value: int = 16, in_channels: int = 4, condition_dim: int = 1):
        super().__init__()
        self.time_embed = nn.Linear(1, condition_dim)
        self.angle_embed = nn.Linear(1, condition_dim)

        self.encoder = _shared_residual_encoder(in_channels)

        self.fc_mu = nn.Linear(256 + 2 * condition_dim, latent_value)
        self.fc_logvar = nn.Linear(256 + 2 * condition_dim, latent_value)

        self.shared_decoder = _shared_decoder(latent_Num, latent_value)
        self.disp_decoders = nn.ModuleList([_decoder_branch(1) for _ in range(3)])
        self.stress_decoder = _decoder_branch(1)

    def encode(self, x: torch.Tensor, time_step: torch.Tensor, angle: torch.Tensor):
        time_emb = self.time_embed(time_step.view(-1, 1))
        angle_emb = self.angle_embed(angle.view(-1, 1))

        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, time_emb, angle_emb], dim=1)
        return self.fc_mu(x), self.fc_logvar(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._decode_feat(self.shared_decoder(z))

    def forward(self, x: torch.Tensor, time_step: torch.Tensor, angle: torch.Tensor):
        mu, logvar = self.encode(x, time_step, angle)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class M_R_CVAE02(_FinalModelBase):
    """共享残差编码器；条件解耦且按高斯采样后拼接解码。"""

    def __init__(self, latent_Num: int = 6, latent_value: int = 16, in_channels: int = 4, condition_dim: int = 1):
        super().__init__()
        self.time_embed = nn.Linear(1, condition_dim)
        self.angle_embed = nn.Linear(1, condition_dim)

        self.encoder = _shared_residual_encoder(in_channels)

        self.fc_mu = nn.Linear(256, latent_value)
        self.fc_logvar = nn.Linear(256, latent_value)

        self.shared_decoder = _shared_decoder(latent_Num, latent_value + 2 * condition_dim)
        self.disp_decoders = nn.ModuleList([_decoder_branch(1) for _ in range(3)])
        self.stress_decoder = _decoder_branch(1)

    def encode(self, x: torch.Tensor):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def sample_condition_latent(self, time_step: torch.Tensor, angle: torch.Tensor):
        time_emb = self.time_embed(time_step.view(-1, 1))
        angle_emb = self.angle_embed(angle.view(-1, 1))
        z_time = time_emb + torch.randn_like(time_emb)
        z_angle = angle_emb + torch.randn_like(angle_emb)
        return z_time, z_angle

    def decode(self, z_structure: torch.Tensor, z_time: torch.Tensor, z_angle: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z_structure, z_time, z_angle], dim=1)
        return self._decode_feat(self.shared_decoder(z))

    def forward(self, x: torch.Tensor, time_step: torch.Tensor, angle: torch.Tensor):
        mu, logvar = self.encode(x)
        z_x = self.reparameterize(mu, logvar)
        z_time, z_angle = self.sample_condition_latent(time_step, angle)
        return self.decode(z_x, z_time, z_angle), mu, logvar


class M_R_CVAE03(_FinalModelBase):
    """共享残差编码器；条件解耦；条件嵌入直接进入解码器。"""

    def __init__(self, latent_Num: int = 6, latent_value: int = 16, in_channels: int = 4, condition_dim: int = 1):
        super().__init__()
        self.time_embed = nn.Linear(1, condition_dim)
        self.angle_embed = nn.Linear(1, condition_dim)

        self.encoder = _shared_residual_encoder(in_channels)

        self.fc_mu = nn.Linear(256, latent_value)
        self.fc_logvar = nn.Linear(256, latent_value)

        self.shared_decoder = _shared_decoder(latent_Num, latent_value + 2 * condition_dim)
        self.disp_decoders = nn.ModuleList([_decoder_branch(1) for _ in range(3)])
        self.stress_decoder = _decoder_branch(1)

    def encode(self, x: torch.Tensor):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def decode(self, z: torch.Tensor, time_step: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_embed(time_step.view(-1, 1))
        angle_emb = self.angle_embed(angle.view(-1, 1))
        z = torch.cat([z, time_emb, angle_emb], dim=1)
        return self._decode_feat(self.shared_decoder(z))

    def forward(self, x: torch.Tensor, time_step: torch.Tensor, angle: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, time_step, angle), mu, logvar


class M_R_CVAE04(_FinalModelBase):
    """共享残差编码器；条件与结构耦合；条件嵌入直接进入解码器。"""

    def __init__(self, latent_Num: int = 6, latent_value: int = 16, in_channels: int = 4, condition_dim: int = 1):
        super().__init__()
        self.time_embed = nn.Linear(1, condition_dim)
        self.angle_embed = nn.Linear(1, condition_dim)

        self.encoder = _shared_residual_encoder(in_channels)

        self.fc_mu = nn.Linear(256 + 2 * condition_dim, latent_value)
        self.fc_logvar = nn.Linear(256 + 2 * condition_dim, latent_value)

        self.shared_decoder = _shared_decoder(latent_Num, latent_value + 2 * condition_dim)
        self.disp_decoders = nn.ModuleList([_decoder_branch(1) for _ in range(3)])
        self.stress_decoder = _decoder_branch(1)

    def encode(self, x: torch.Tensor, time_step: torch.Tensor, angle: torch.Tensor):
        time_emb = self.time_embed(time_step.view(-1, 1))
        angle_emb = self.angle_embed(angle.view(-1, 1))

        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, time_emb, angle_emb], dim=1)
        return self.fc_mu(x), self.fc_logvar(x)

    def decode(self, z: torch.Tensor, time_step: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_embed(time_step.view(-1, 1))
        angle_emb = self.angle_embed(angle.view(-1, 1))
        z = torch.cat([z, time_emb, angle_emb], dim=1)
        return self._decode_feat(self.shared_decoder(z))

    def forward(self, x: torch.Tensor, time_step: torch.Tensor, angle: torch.Tensor):
        mu, logvar = self.encode(x, time_step, angle)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, time_step, angle), mu, logvar


class M_Rm_CVAE01(_FinalModelBase):
    """四通道独立残差编码器；条件与结构耦合；条件不进入解码器。"""

    def __init__(self, latent_Num: int = 6, latent_value: int = 16, in_channels: int = 4, condition_dim: int = 1):
        super().__init__()
        self.time_embed = nn.Linear(1, condition_dim)
        self.angle_embed = nn.Linear(1, condition_dim)

        self.encoder_branches = nn.ModuleList([_residual_encoder_branch() for _ in range(4)])

        self.fc_mu = nn.Linear(256 * 4 + 2 * condition_dim, latent_value)
        self.fc_logvar = nn.Linear(256 * 4 + 2 * condition_dim, latent_value)

        self.shared_decoder = _shared_decoder(latent_Num, latent_value)
        self.disp_decoders = nn.ModuleList([_decoder_branch(1) for _ in range(3)])
        self.stress_decoder = _decoder_branch(1)

    def encode(self, x: torch.Tensor, time_step: torch.Tensor, angle: torch.Tensor):
        time_emb = self.time_embed(time_step.view(-1, 1))
        angle_emb = self.angle_embed(angle.view(-1, 1))

        branches_out = [branch(x[:, i:i + 1, :, :]) for i, branch in enumerate(self.encoder_branches)]
        x_flat = torch.cat(branches_out, dim=1).view(x.size(0), -1)
        x_flat = torch.cat([x_flat, time_emb, angle_emb], dim=1)
        return self.fc_mu(x_flat), self.fc_logvar(x_flat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._decode_feat(self.shared_decoder(z))

    def forward(self, x: torch.Tensor, time_step: torch.Tensor, angle: torch.Tensor):
        mu, logvar = self.encode(x, time_step, angle)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class M_Rm_CVAE02(_FinalModelBase):
    """论文最终模型：四通道独立残差编码器 + 独立解码器 + CVAE02 条件模式。"""

    def __init__(self, latent_Num: int = 6, latent_value: int = 16, in_channels: int = 4, condition_dim: int = 1):
        super().__init__()
        self.time_embed = nn.Linear(1, condition_dim)
        self.angle_embed = nn.Linear(1, condition_dim)

        self.encoder_branches = nn.ModuleList([_residual_encoder_branch() for _ in range(4)])

        self.fc_mu = nn.Linear(256 * 4, latent_value)
        self.fc_logvar = nn.Linear(256 * 4, latent_value)

        self.shared_decoder = _shared_decoder(latent_Num, latent_value + 2 * condition_dim)
        self.disp_decoders = nn.ModuleList([_decoder_branch(1) for _ in range(3)])
        self.stress_decoder = _decoder_branch(1)

    def encode(self, x: torch.Tensor):
        branches_out = [branch(x[:, i:i + 1, :, :]) for i, branch in enumerate(self.encoder_branches)]
        x_flat = torch.cat(branches_out, dim=1).view(x.size(0), -1)
        return self.fc_mu(x_flat), self.fc_logvar(x_flat)

    def sample_condition_latent(self, time_step: torch.Tensor, angle: torch.Tensor):
        time_emb = self.time_embed(time_step.view(-1, 1))
        angle_emb = self.angle_embed(angle.view(-1, 1))
        z_time = time_emb + torch.randn_like(time_emb)
        z_angle = angle_emb + torch.randn_like(angle_emb)
        return z_time, z_angle

    def decode(self, z_structure: torch.Tensor, z_time: torch.Tensor, z_angle: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z_structure, z_time, z_angle], dim=1)
        return self._decode_feat(self.shared_decoder(z))

    def forward(self, x: torch.Tensor, time_step: torch.Tensor, angle: torch.Tensor):
        mu, logvar = self.encode(x)
        z_x = self.reparameterize(mu, logvar)
        z_time, z_angle = self.sample_condition_latent(time_step, angle)
        return self.decode(z_x, z_time, z_angle), mu, logvar


class M_Rm_CVAE03(_FinalModelBase):
    """四通道独立残差编码器；条件解耦；条件嵌入直接进入解码器。"""

    def __init__(self, latent_Num: int = 6, latent_value: int = 16, in_channels: int = 4, condition_dim: int = 1):
        super().__init__()
        self.time_embed = nn.Linear(1, condition_dim)
        self.angle_embed = nn.Linear(1, condition_dim)

        self.encoder_branches = nn.ModuleList([_residual_encoder_branch() for _ in range(4)])

        self.fc_mu = nn.Linear(256 * 4, latent_value)
        self.fc_logvar = nn.Linear(256 * 4, latent_value)

        self.shared_decoder = _shared_decoder(latent_Num, latent_value + 2 * condition_dim)
        self.disp_decoders = nn.ModuleList([_decoder_branch(1) for _ in range(3)])
        self.stress_decoder = _decoder_branch(1)

    def encode(self, x: torch.Tensor):
        branches_out = [branch(x[:, i:i + 1, :, :]) for i, branch in enumerate(self.encoder_branches)]
        x_flat = torch.cat(branches_out, dim=1).view(x.size(0), -1)
        return self.fc_mu(x_flat), self.fc_logvar(x_flat)

    def decode(self, z: torch.Tensor, time_step: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_embed(time_step.view(-1, 1))
        angle_emb = self.angle_embed(angle.view(-1, 1))
        z = torch.cat([z, time_emb, angle_emb], dim=1)
        return self._decode_feat(self.shared_decoder(z))

    def forward(self, x: torch.Tensor, time_step: torch.Tensor, angle: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, time_step, angle), mu, logvar


class M_Rm_CVAE04(_FinalModelBase):
    """四通道独立残差编码器；条件与结构耦合；条件嵌入直接进入解码器。"""

    def __init__(self, latent_Num: int = 6, latent_value: int = 16, in_channels: int = 4, condition_dim: int = 1):
        super().__init__()
        self.time_embed = nn.Linear(1, condition_dim)
        self.angle_embed = nn.Linear(1, condition_dim)

        self.encoder_branches = nn.ModuleList([_residual_encoder_branch() for _ in range(4)])

        self.fc_mu = nn.Linear(256 * 4 + 2 * condition_dim, latent_value)
        self.fc_logvar = nn.Linear(256 * 4 + 2 * condition_dim, latent_value)

        self.shared_decoder = _shared_decoder(latent_Num, latent_value + 2 * condition_dim)
        self.disp_decoders = nn.ModuleList([_decoder_branch(1) for _ in range(3)])
        self.stress_decoder = _decoder_branch(1)

    def encode(self, x: torch.Tensor, time_step: torch.Tensor, angle: torch.Tensor):
        time_emb = self.time_embed(time_step.view(-1, 1))
        angle_emb = self.angle_embed(angle.view(-1, 1))

        branches_out = [branch(x[:, i:i + 1, :, :]) for i, branch in enumerate(self.encoder_branches)]
        x_flat = torch.cat(branches_out, dim=1).view(x.size(0), -1)
        x_flat = torch.cat([x_flat, time_emb, angle_emb], dim=1)
        return self.fc_mu(x_flat), self.fc_logvar(x_flat)

    def decode(self, z: torch.Tensor, time_step: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_embed(time_step.view(-1, 1))
        angle_emb = self.angle_embed(angle.view(-1, 1))
        z = torch.cat([z, time_emb, angle_emb], dim=1)
        return self._decode_feat(self.shared_decoder(z))

    def forward(self, x: torch.Tensor, time_step: torch.Tensor, angle: torch.Tensor):
        mu, logvar = self.encode(x, time_step, angle)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, time_step, angle), mu, logvar

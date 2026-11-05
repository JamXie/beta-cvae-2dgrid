"""基础网络：残差块、VAE 与四种条件 VAE（CVAE01-CVAE04）。

结构、层顺序、激活与输出范围与原始工程 `model.py` 完全一致，
只补充了中文注释与类型标注。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    """支持下采样/上采样的残差块。

    主路径为 conv(或转置卷积) -> BN -> LeakyReLU -> conv -> BN，
    捷径在通道数或空间尺寸变化时使用 1×1 卷积（或转置卷积）对齐。
    """

    def __init__(self, in_ch: int, out_ch: int, downsample: bool = False, upsample: bool = False):
        super().__init__()
        self.downsample = downsample
        self.upsample = upsample

        # =========== 主路径 ===========
        if upsample:
            self.conv1 = nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1
            )
        else:
            self.conv1 = nn.Conv2d(
                in_ch, out_ch, kernel_size=3, padding=1, stride=2 if downsample else 1
            )

        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.LeakyReLU(0.2)

        # =========== 捷径路径 ===========
        self.shortcut = nn.Sequential()
        if in_ch != out_ch or downsample or upsample:
            if downsample:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2),
                    nn.BatchNorm2d(out_ch),
                )
            elif upsample:
                self.shortcut = nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=1, stride=2, output_padding=1),
                    nn.BatchNorm2d(out_ch),
                )
            else:  # 仅通道数调整
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1),
                    nn.BatchNorm2d(out_ch),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        # 尺寸兜底对齐（正常输入不会触发）
        if x.shape != residual.shape:
            residual = F.interpolate(residual, size=x.shape[-2:], mode="bilinear", align_corners=True)

        return self.relu(x + residual)


class VAE(nn.Module):
    """四通道基础 VAE，无任何条件输入。"""

    def __init__(self, latent_Num: int = 6, latent_value: int = 16):
        # latent_Num 为编码器最后一层的空间尺寸，96×96 输入经 4 次下采样为 6×6
        super().__init__()

        # 编码器：4 -> 32 -> 64 -> 128 -> 256，空间尺寸每次减半
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        # 均值与对数方差
        self.fc_mu = nn.Linear(256 * latent_Num * latent_Num, latent_value)
        self.fc_logvar = nn.Linear(256 * latent_Num * latent_Num, latent_value)

        # 解码器：latent -> 特征图 -> 4 次上采样 -> 4 通道 Tanh 输出
        self.decoder = nn.Sequential(
            nn.Linear(latent_value, 256 * latent_Num * latent_Num),
            nn.ReLU(),
            nn.Unflatten(1, (256, latent_Num, latent_Num)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(32, 4, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class CVAE01(nn.Module):
    """条件与结构信息相互耦合：条件嵌入拼接到编码器特征后再求 mu/logvar。"""

    def __init__(self, latent_Num: int = 6, latent_value: int = 16, condition_dim: int = 1):
        super().__init__()
        self.latent_Num = latent_Num
        self.latent_value = latent_value
        self.condition_dim = condition_dim

        # 条件嵌入层：时间步与角度各一个线性层
        self.time_embed = nn.Linear(1, condition_dim)
        self.angle_embed = nn.Linear(1, condition_dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        # 全连接层输入包含 2 个条件嵌入
        encoder_output_dim = 256 * latent_Num * latent_Num
        self.fc_mu = nn.Linear(encoder_output_dim + 2 * condition_dim, latent_value)
        self.fc_logvar = nn.Linear(encoder_output_dim + 2 * condition_dim, latent_value)

        self.decoder = nn.Sequential(
            nn.Linear(latent_value, 256 * latent_Num * latent_Num),
            nn.ReLU(),
            nn.Unflatten(1, (256, latent_Num, latent_Num)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(32, 4, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor, time_step: torch.Tensor, angle: torch.Tensor):
        time_emb = self.time_embed(time_step.view(-1, 1))
        angle_emb = self.angle_embed(angle.view(-1, 1))

        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, time_emb, angle_emb], dim=1)

        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor, time_step: torch.Tensor, angle: torch.Tensor):
        mu, logvar = self.encode(x, time_step, angle)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class CVAE02(nn.Module):
    """条件与结构信息解耦：条件按 N(embedding, 1) 采样后与 z 拼接解码。

    这是论文最终模型采用的条件模式。
    """

    def __init__(self, latent_Num: int = 6, latent_value: int = 16, condition_dim: int = 1):
        super().__init__()
        self.latent_Num = latent_Num
        self.latent_value = latent_value
        self.condition_dim = condition_dim

        self.time_embed = nn.Linear(1, condition_dim)
        self.angle_embed = nn.Linear(1, condition_dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        encoder_output_dim = 256 * latent_Num * latent_Num
        self.fc_mu = nn.Linear(encoder_output_dim, latent_value)
        self.fc_logvar = nn.Linear(encoder_output_dim, latent_value)

        # 解码器输入为 z 与两个条件潜变量
        self.decoder = nn.Sequential(
            nn.Linear(latent_value + 2 * condition_dim, 256 * latent_Num * latent_Num),
            nn.ReLU(),
            nn.Unflatten(1, (256, latent_Num, latent_Num)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(32, 4, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def sample_condition_latent(self, time_step: torch.Tensor, angle: torch.Tensor):
        """条件潜变量：均值由嵌入层给出，方差固定为单位值。"""
        time_emb = self.time_embed(time_step.view(-1, 1))
        angle_emb = self.angle_embed(angle.view(-1, 1))
        z_time = time_emb + torch.randn_like(time_emb)
        z_angle = angle_emb + torch.randn_like(angle_emb)
        return z_time, z_angle

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z_structure: torch.Tensor, z_time: torch.Tensor, z_angle: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z_structure, z_time, z_angle], dim=1)
        return self.decoder(z)

    def forward(self, x: torch.Tensor, time_step: torch.Tensor, angle: torch.Tensor):
        mu, logvar = self.encode(x)
        z_x = self.reparameterize(mu, logvar)
        z_time, z_angle = self.sample_condition_latent(time_step, angle)
        return self.decode(z_x, z_time, z_angle), mu, logvar


class CVAE03(nn.Module):
    """条件与结构信息解耦：条件嵌入直接作为解码器输入（不在潜空间采样）。"""

    def __init__(self, latent_Num: int = 6, latent_value: int = 16, condition_dim: int = 1):
        super().__init__()
        self.latent_Num = latent_Num
        self.latent_value = latent_value
        self.condition_dim = condition_dim

        self.time_embed = nn.Linear(1, condition_dim)
        self.angle_embed = nn.Linear(1, condition_dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        encoder_output_dim = 256 * latent_Num * latent_Num
        self.fc_mu = nn.Linear(encoder_output_dim, latent_value)
        self.fc_logvar = nn.Linear(encoder_output_dim, latent_value)

        self.decoder = nn.Sequential(
            nn.Linear(latent_value + 2 * condition_dim, 256 * latent_Num * latent_Num),
            nn.ReLU(),
            nn.Unflatten(1, (256, latent_Num, latent_Num)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(32, 4, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, time_step: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_embed(time_step.view(-1, 1))
        angle_emb = self.angle_embed(angle.view(-1, 1))
        z = torch.cat([z, time_emb, angle_emb], dim=1)
        return self.decoder(z)

    def forward(self, x: torch.Tensor, time_step: torch.Tensor, angle: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, time_step, angle), mu, logvar


class CVAE04(nn.Module):
    """条件与结构信息相互耦合，且条件嵌入直接作为解码器输入。"""

    def __init__(self, latent_Num: int = 6, latent_value: int = 16, condition_dim: int = 1):
        super().__init__()
        self.latent_Num = latent_Num
        self.latent_value = latent_value
        self.condition_dim = condition_dim

        self.time_embed = nn.Linear(1, condition_dim)
        self.angle_embed = nn.Linear(1, condition_dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        encoder_output_dim = 256 * latent_Num * latent_Num
        self.fc_mu = nn.Linear(encoder_output_dim + 2 * condition_dim, latent_value)
        self.fc_logvar = nn.Linear(encoder_output_dim + 2 * condition_dim, latent_value)

        self.decoder = nn.Sequential(
            nn.Linear(latent_value + 2 * condition_dim, 256 * latent_Num * latent_Num),
            nn.ReLU(),
            nn.Unflatten(1, (256, latent_Num, latent_Num)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(32, 4, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor, time_step: torch.Tensor, angle: torch.Tensor):
        time_emb = self.time_embed(time_step.view(-1, 1))
        angle_emb = self.angle_embed(angle.view(-1, 1))

        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, time_emb, angle_emb], dim=1)

        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, time_step: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_embed(time_step.view(-1, 1))
        angle_emb = self.angle_embed(angle.view(-1, 1))
        z = torch.cat([z, time_emb, angle_emb], dim=1)
        return self.decoder(z)

    def forward(self, x: torch.Tensor, time_step: torch.Tensor, angle: torch.Tensor):
        mu, logvar = self.encode(x, time_step, angle)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, time_step, angle), mu, logvar

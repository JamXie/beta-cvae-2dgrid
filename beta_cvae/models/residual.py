"""残差 VAE（论文中的 R-VAE01、R-VAE02、R-VAE03）。

- R-VAE01（ResidualVAE01）：编码器与解码器均使用残差块；
- R-VAE02（ResidualVAE02）：仅解码器使用残差块；
- R-VAE03（ResidualVAE03）：仅编码器使用残差块。
"""

from __future__ import annotations

import torch
from torch import nn

from .basic import ResidualBlock


class ResidualVAE01(nn.Module):
    """编码器与解码器都为残差结构。"""

    def __init__(self, latent_Num: int = 6, latent_value: int = 16, img_size: int = 96, in_channels: int = 4):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels

        # 4 个下采样残差块，最后全局平均池化到 [B, 256, 1, 1]
        self.encoder = nn.Sequential(
            ResidualBlock(in_channels, 32, downsample=True),
            ResidualBlock(32, 64, downsample=True),
            ResidualBlock(64, 128, downsample=True),
            ResidualBlock(128, 256, downsample=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fc_mu = nn.Linear(256, latent_value)
        self.fc_logvar = nn.Linear(256, latent_value)

        # 4 个上采样残差块，回到 96×96
        self.decoder = nn.Sequential(
            nn.Linear(latent_value, 256 * latent_Num * latent_Num),
            nn.Unflatten(1, (256, latent_Num, latent_Num)),
            ResidualBlock(256, 128, upsample=True),
            ResidualBlock(128, 64, upsample=True),
            ResidualBlock(64, 32, upsample=True),
            ResidualBlock(32, 16, upsample=True),
            nn.Conv2d(16, in_channels, 3, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # 评估模式直接取均值，减少推理随机性
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std if self.training else mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class ResidualVAE02(nn.Module):
    """仅解码器为残差结构。"""

    def __init__(self, latent_Num: int = 6, latent_value: int = 16, img_size: int = 96, in_channels: int = 4):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels

        # 普通卷积编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(256 * latent_Num * latent_Num, latent_value)
        self.fc_logvar = nn.Linear(256 * latent_Num * latent_Num, latent_value)

        self.decoder = nn.Sequential(
            nn.Linear(latent_value, 256 * latent_Num * latent_Num),
            nn.Unflatten(1, (256, latent_Num, latent_Num)),
            ResidualBlock(256, 128, upsample=True),
            ResidualBlock(128, 64, upsample=True),
            ResidualBlock(64, 32, upsample=True),
            ResidualBlock(32, 16, upsample=True),
            nn.Conv2d(16, in_channels, 3, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std if self.training else mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class ResidualVAE03(nn.Module):
    """仅编码器为残差结构。"""

    def __init__(self, latent_value: int = 16, latent_Num: int = 6, img_size: int = 96, in_channels: int = 4):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels

        self.encoder = nn.Sequential(
            ResidualBlock(in_channels, 32, downsample=True),
            ResidualBlock(32, 64, downsample=True),
            ResidualBlock(64, 128, downsample=True),
            ResidualBlock(128, 256, downsample=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fc_mu = nn.Linear(256, latent_value)
        self.fc_logvar = nn.Linear(256, latent_value)

        # 普通转置卷积解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_value, 256 * latent_Num * latent_Num),
            nn.ReLU(),
            nn.Unflatten(1, (256, latent_Num, latent_Num)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, 4, 2, 1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std if self.training else mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

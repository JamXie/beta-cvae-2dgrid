"""多分支 VAE（论文中的 M-VAE01、M-VAE02、M-VAE03）。

- M-VAE01（MultiBranchVAE01）：编码器与解码器均按通道独立；
- M-VAE02（MultiBranchVAE02）：仅解码器按通道独立；
- M-VAE03（MultiBranchVAE03）：仅编码器按通道独立。

类名沿用原始工程，论文名称与代码的映射见 docs/METHOD.md。
"""

from __future__ import annotations

import torch
from torch import nn


class MultiBranchVAE01(nn.Module):
    """编码器与解码器都按通道独立。"""

    def __init__(self, latent_Num: int = 6, latent_value: int = 16):
        super().__init__()

        # 每个通道独立编码器，单个分支 1 -> 8 -> 16 -> 32 -> 64
        def make_encoder_branch() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(8),
                nn.LeakyReLU(0.2),
                nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.2),
                nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
            )

        self.encoder_branches = nn.ModuleList([make_encoder_branch() for _ in range(4)])
        self.fc_mu = nn.Linear(64 * 4 * latent_Num * latent_Num, latent_value)
        self.fc_logvar = nn.Linear(64 * 4 * latent_Num * latent_Num, latent_value)

        # 共享上采样层，之后按通道分成 4 个解码分支
        self.shared_decoder = nn.Sequential(
            nn.Linear(latent_value, 256 * latent_Num * latent_Num),
            nn.ReLU(),
            nn.Unflatten(1, (256, latent_Num, latent_Num)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
        )

        def make_decoder_branch(out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1),
                nn.Tanh(),
            )

        # 3 个位移通道 + 1 个应力通道
        self.disp_decoders = nn.ModuleList([make_decoder_branch(1) for _ in range(3)])
        self.stress_decoder = make_decoder_branch(1)

    def encode(self, x: torch.Tensor):
        branches_out = [branch(x[:, i:i + 1, :, :]) for i, branch in enumerate(self.encoder_branches)]
        x_cat = torch.cat(branches_out, dim=1)
        x_flat = x_cat.view(x_cat.size(0), -1)
        return self.fc_mu(x_flat), self.fc_logvar(x_flat)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        feat = self.shared_decoder(z)
        disp_outs = [decoder(feat) for decoder in self.disp_decoders]
        stress_out = self.stress_decoder(feat)
        return torch.cat(disp_outs + [stress_out], dim=1)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class MultiBranchVAE02(nn.Module):
    """仅解码器按通道独立。"""

    def __init__(self, latent_Num: int = 6, latent_value: int = 16):
        super().__init__()

        # 共享编码器
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

        self.shared_decoder = nn.Sequential(
            nn.Linear(latent_value, 256 * latent_Num * latent_Num),
            nn.ReLU(),
            nn.Unflatten(1, (256, latent_Num, latent_Num)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
        )

        def make_decoder_branch(out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, out_channels, 4, 2, 1),
                nn.Tanh(),
            )

        self.disp_decoders = nn.ModuleList([make_decoder_branch(1) for _ in range(3)])
        self.stress_decoder = make_decoder_branch(1)

    def encode(self, x: torch.Tensor):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        feat = self.shared_decoder(z)
        disp_outs = [decoder(feat) for decoder in self.disp_decoders]
        stress_out = self.stress_decoder(feat)
        return torch.cat(disp_outs + [stress_out], dim=1)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class MultiBranchVAE03(nn.Module):
    """仅编码器按通道独立。"""

    def __init__(self, latent_Num: int = 6, latent_value: int = 16):
        super().__init__()

        def make_encoder_branch() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(1, 8, 4, 2, 1),
                nn.BatchNorm2d(8),
                nn.LeakyReLU(0.2),
                nn.Conv2d(8, 16, 4, 2, 1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.2),
                nn.Conv2d(16, 32, 4, 2, 1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.ReLU(),
            )

        self.encoder_branches = nn.ModuleList([make_encoder_branch() for _ in range(4)])
        self.fc_mu = nn.Linear(64 * 4 * latent_Num * latent_Num, latent_value)
        self.fc_logvar = nn.Linear(64 * 4 * latent_Num * latent_Num, latent_value)

        # 共享解码器
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
        branches_out = [branch(x[:, i:i + 1]) for i, branch in enumerate(self.encoder_branches)]
        x_cat = torch.cat(branches_out, dim=1)
        x_flat = x_cat.view(x_cat.size(0), -1)
        return self.fc_mu(x_flat), self.fc_logvar(x_flat)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

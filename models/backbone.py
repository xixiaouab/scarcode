import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union


class SEBlock3d(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super(SEBlock3d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class ConvBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(ConvBlock3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualSEBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super(ResidualSEBlock3d, self).__init__()
        self.conv1 = ConvBlock3d(in_channels, out_channels)
        self.dropout = nn.Dropout3d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        self.se = SEBlock3d(out_channels)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        return out + residual


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super(DownSampleBlock, self).__init__()
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.res_block = ResidualSEBlock3d(in_channels, out_channels, dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pool = self.max_pool(x)
        return self.res_block(x_pool)


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super(UpSampleBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.res_block = ResidualSEBlock3d(in_channels, out_channels, dropout_rate)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        diff_d = skip.size(2) - x.size(2)
        diff_h = skip.size(3) - x.size(3)
        diff_w = skip.size(4) - x.size(4)

        x = F.pad(x, [
            diff_w // 2, diff_w - diff_w // 2,
            diff_h // 2, diff_h - diff_h // 2,
            diff_d // 2, diff_d - diff_d // 2
        ])

        x = torch.cat([skip, x], dim=1)
        return self.res_block(x)


class UNet3DEncoder(nn.Module):
    def __init__(
            self,
            in_channels: int = 4,
            base_filters: int = 16,
            depth: int = 4,
            dropout_rate: float = 0.0
    ):
        super(UNet3DEncoder, self).__init__()
        self.depth = depth
        self.inc = ResidualSEBlock3d(in_channels, base_filters, dropout_rate)

        self.downs = nn.ModuleList()
        current_filters = base_filters

        for _ in range(depth):
            self.downs.append(DownSampleBlock(current_filters, current_filters * 2, dropout_rate))
            current_filters *= 2

        self.out_channels = current_filters

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips = []
        x = self.inc(x)
        skips.append(x)

        for down in self.downs:
            x = down(x)
            skips.append(x)

        return x, skips


class UNet3DDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels: int,
            base_filters: int = 16,
            depth: int = 4,
            dropout_rate: float = 0.0
    ):
        super(UNet3DDecoder, self).__init__()
        self.ups = nn.ModuleList()
        current_filters = encoder_channels

        for _ in range(depth):
            self.ups.append(UpSampleBlock(current_filters, current_filters // 2, dropout_rate))
            current_filters //= 2

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        for i, up in enumerate(self.ups):
            skip = skips[-(i + 2)]
            x = up(x, skip)
        return x


class UNetBackbone(nn.Module):
    def __init__(
            self,
            in_channels: int = 4,
            base_filters: int = 16,
            depth: int = 4,
            dropout_rate: float = 0.0
    ):
        super(UNetBackbone, self).__init__()
        self.encoder = UNet3DEncoder(in_channels, base_filters, depth, dropout_rate)

        encoder_out_channels = base_filters * (2 ** depth)
        self.decoder = UNet3DDecoder(encoder_out_channels, base_filters, depth, dropout_rate)

        self.final_filters = base_filters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, skips = self.encoder(x)
        x = self.decoder(x, skips)
        return x
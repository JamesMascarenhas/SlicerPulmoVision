# PulmoBackend/unet3d.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3d(nn.Module):
    """
    Two (Conv3d + BN + ReLU) blocks.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down3d(nn.Module):
    """
    Downscaling with maxpool followed by DoubleConv3d.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = DoubleConv3d(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up3d(nn.Module):
    """
    Upscaling then concatenation with skip features, followed by DoubleConv3d.
    Uses trilinear upsampling + 1x1 conv instead of ConvTranspose3d to stay light.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.double_conv = DoubleConv3d(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.conv1x1(x)

        # Pad if needed (for odd sizes)
        diff_d = skip.size(2) - x.size(2)
        diff_h = skip.size(3) - x.size(3)
        diff_w = skip.size(4) - x.size(4)
        x = F.pad(
            x,
            [
                diff_w // 2,
                diff_w - diff_w // 2,
                diff_h // 2,
                diff_h - diff_h // 2,
                diff_d // 2,
                diff_d - diff_d // 2,
            ],
        )

        x = torch.cat([skip, x], dim=1)
        return self.double_conv(x)


class OutConv3d(nn.Module):
    """
    Final 1x1 conv to map to 1-channel logits.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    """
    Lightweight 3D U-Net for lung tumor segmentation.

    Input:  N x 1 x D x H x W
    Output: N x 1 x D x H x W (logits)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 16):
        super().__init__()

        # Encoder
        self.inc = DoubleConv3d(in_channels, base_channels)
        self.down1 = Down3d(base_channels, base_channels * 2)
        self.down2 = Down3d(base_channels * 2, base_channels * 4)
        self.down3 = Down3d(base_channels * 4, base_channels * 8)

        # Bottleneck
        self.bot = DoubleConv3d(base_channels * 8, base_channels * 16)

        # Decoder
        self.up3 = Up3d(base_channels * 16, base_channels * 8)
        self.up2 = Up3d(base_channels * 8, base_channels * 4)
        self.up1 = Up3d(base_channels * 4, base_channels * 2)
        self.up0 = Up3d(base_channels * 2, base_channels)

        self.outc = OutConv3d(base_channels, out_channels)

    def forward(self, x):
        # Encoder
        x0 = self.inc(x)       # N, C
        x1 = self.down1(x0)    # N, 2C
        x2 = self.down2(x1)    # N, 4C
        x3 = self.down3(x2)    # N, 8C

        # Bottleneck
        xb = self.bot(x3)      # N, 16C

        # Decoder with skip connections
        x = self.up3(xb, x3)   # N, 8C
        x = self.up2(x, x2)    # N, 4C
        x = self.up1(x, x1)    # N, 2C
        x = self.up0(x, x0)    # N, C

        logits = self.outc(x)
        return logits


"""
BASNet_3050_lite.py
Light BASNet-style model for Salient Object Detection on RTX 3050 / 4GB VRAM.

How to use:
  Option 1: copy this file into your model/ folder and import:
      from model.BASNet_3050_lite import BASNet
  Option 2: rename it to BASNet.py to replace your current heavy model.

Main changes compared with the uploaded model:
  - Keep ResNet34 encoder, but use a much lighter BASNet decoder.
  - Remove the very heavy RefUnet with ASPP + AttentionGate at full resolution.
  - Use a small residual refine head for cleaner boundaries.
  - Use GroupNorm for small batch training.
  - Return 8 sigmoid outputs to remain compatible with your current training loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ------------------------------------------------------------
# Utility layers
# ------------------------------------------------------------
def _make_gn(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """Choose a safe GroupNorm group count for any channel number."""
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def replace_bn_with_gn(module: nn.Module, max_groups: int = 8) -> nn.Module:
    """Replace BatchNorm2d with GroupNorm while preserving BN affine weights."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            gn = _make_gn(child.num_features, max_groups=max_groups)
            if child.affine:
                with torch.no_grad():
                    gn.weight.copy_(child.weight.data)
                    gn.bias.copy_(child.bias.data)
            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child, max_groups=max_groups)
    return module


class ConvGNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, dilation: int = 1):
        super().__init__()
        pad = dilation if k == 3 else 0
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=pad, dilation=dilation, bias=False),
            _make_gn(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class ResidualBlockGN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__()
        self.conv1 = ConvGNReLU(in_ch, out_ch, 3, dilation=dilation)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            _make_gn(out_ch),
        )
        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            _make_gn(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv2(self.conv1(x)) + self.shortcut(x))

class LiteRefineHead(nn.Module):
    """Small residual refinement head. Much cheaper than full RefUnet."""
    def __init__(self, channels: int = 32):
        super().__init__()
        self.refine = nn.Sequential(
            ConvGNReLU(1, channels, 3),
            ConvGNReLU(channels, channels, 3),
            nn.Conv2d(channels, 1, 3, padding=1),
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits + self.refine(logits)

def _resnet34_encoder(pretrained: bool = True) -> nn.Module:
    """Load ResNet34 with a safe fallback for different torchvision versions."""
    if pretrained:
        try:
            weights = models.ResNet34_Weights.DEFAULT
            return models.resnet34(weights=weights)
        except Exception:
            try:
                return models.resnet34(pretrained=True)
            except Exception:
                return models.resnet34(weights=None)
    return models.resnet34(weights=None)

# ------------------------------------------------------------
# BASNet 3050 Lite
# ------------------------------------------------------------
class BASNet(nn.Module):
    def __init__(self, n_channels: int = 3, n_classes: int = 1, pretrained: bool = True):
        super().__init__()

        resnet = _resnet34_encoder(pretrained=pretrained)
        replace_bn_with_gn(resnet, max_groups=8)

        # Keep your original BASNet-style stem: no 7x7 stride-2 conv, no maxpool.
        self.inconv = ConvGNReLU(n_channels, 64, 3)
        self.encoder1 = resnet.layer1   # H
        self.encoder2 = resnet.layer2   # H/2
        self.encoder3 = resnet.layer3   # H/4
        self.encoder4 = resnet.layer4   # H/8

        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        # Light context stages. Original file used many 512-channel blocks here.
        self.stage5 = ResidualBlockGN(512, 512, dilation=1)   # H/16
        self.stage6 = ResidualBlockGN(512, 512, dilation=2)   # H/32

        # Lightweight bridge/context.
        self.bridge = nn.Sequential(
            ConvGNReLU(512, 512, 3, dilation=2),
            ConvGNReLU(512, 512, 3, dilation=4),
        )

        # Light decoder: one residual block per level instead of three convs per level.
        self.dec6 = ResidualBlockGN(1024, 512, dilation=2)
        self.dec5 = ResidualBlockGN(1024, 512, dilation=1)
        self.dec4 = ResidualBlockGN(1024, 256, dilation=1)
        self.dec3 = ResidualBlockGN(512, 128, dilation=1)
        self.dec2 = ResidualBlockGN(256, 64, dilation=1)
        self.dec1 = ResidualBlockGN(128, 64, dilation=1)

        # Side outputs. They stay compatible with your 8-output loss.
        self.outconvb = nn.Conv2d(512, n_classes, 3, padding=1)
        self.outconv6 = nn.Conv2d(512, n_classes, 3, padding=1)
        self.outconv5 = nn.Conv2d(512, n_classes, 3, padding=1)
        self.outconv4 = nn.Conv2d(256, n_classes, 3, padding=1)
        self.outconv3 = nn.Conv2d(128, n_classes, 3, padding=1)
        self.outconv2 = nn.Conv2d(64, n_classes, 3, padding=1)
        self.outconv1 = nn.Conv2d(64, n_classes, 3, padding=1)

        self.refine = LiteRefineHead(channels=32)

    @staticmethod
    def _up_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)

    @staticmethod
    def _up_to_size(x: torch.Tensor, size) -> torch.Tensor:
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor):
        out_size = x.shape[2:]

        hx = self.inconv(x)
        h1 = self.encoder1(hx)           # H
        h2 = self.encoder2(h1)           # H/2
        h3 = self.encoder3(h2)           # H/4
        h4 = self.encoder4(h3)           # H/8

        h5 = self.stage5(self.pool(h4))  # H/16
        h6 = self.stage6(self.pool(h5))  # H/32
        hbg = self.bridge(h6)            # H/32

        hd6 = self.dec6(torch.cat([hbg, h6], dim=1))

        hx = self._up_to(hd6, h5)
        hd5 = self.dec5(torch.cat([hx, h5], dim=1))

        hx = self._up_to(hd5, h4)
        hd4 = self.dec4(torch.cat([hx, h4], dim=1))

        hx = self._up_to(hd4, h3)
        hd3 = self.dec3(torch.cat([hx, h3], dim=1))

        hx = self._up_to(hd3, h2)
        hd2 = self.dec2(torch.cat([hx, h2], dim=1))

        hx = self._up_to(hd2, h1)
        hd1 = self.dec1(torch.cat([hx, h1], dim=1))

        d1_logits = self.outconv1(hd1)
        d0_logits = self.refine(d1_logits)

        db = self._up_to_size(self.outconvb(hbg), out_size)
        d6 = self._up_to_size(self.outconv6(hd6), out_size)
        d5 = self._up_to_size(self.outconv5(hd5), out_size)
        d4 = self._up_to_size(self.outconv4(hd4), out_size)
        d3 = self._up_to_size(self.outconv3(hd3), out_size)
        d2 = self._up_to_size(self.outconv2(hd2), out_size)
        d1 = self._up_to_size(d1_logits, out_size)
        d0 = self._up_to_size(d0_logits, out_size)

        return (
            torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3),
            torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6), torch.sigmoid(db)
        )

if __name__ == "__main__":
    # quick shape check
    model = BASNet(3, 1, pretrained=False)
    model.eval()
    with torch.no_grad():
        y = model(torch.randn(1, 3, 192, 192))
    print([tuple(t.shape) for t in y])

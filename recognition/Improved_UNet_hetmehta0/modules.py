"""PyTorch-only 2D U-Nets; original implementation assisted by OpenAI Codex.

Architecture references:
Ronneberger et al. (2015), https://arxiv.org/abs/1505.04597
Isensee et al. (2018), https://arxiv.org/abs/1802.10508
The improved model is an explicitly documented 2D adaptation, not a reproduction.
"""
import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, incoming, outgoing):
        super().__init__(
            nn.Conv2d(incoming, outgoing, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(outgoing, outgoing, 3, padding=1), nn.ReLU(inplace=True))


class UNet2D(nn.Module):
    """Standard four-level U-Net; same padding preserves native input size."""
    def __init__(self, num_classes, base_channels=16):
        super().__init__()
        widths = [base_channels * 2**i for i in range(5)]
        self.encoder = nn.ModuleList(
            DoubleConv(1 if i == 0 else widths[i - 1], width)
            for i, width in enumerate(widths))
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.ModuleList(
            nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2)
            for i in reversed(range(4)))
        self.decoder = nn.ModuleList(DoubleConv(2 * widths[i], widths[i])
                                     for i in reversed(range(4)))
        self.head = nn.Conv2d(widths[0], num_classes, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.encoder):
            x = block(x if i == 0 else self.pool(x))
            skips.append(x)
        for up, block, skip in zip(self.upsample, self.decoder, reversed(skips[:-1])):
            x = up(x)
            # Odd input dimensions may lose one pixel at each pooling level.
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat((skip, x), dim=1))
        return self.head(x)


def norm_activation(channels):
    return nn.Sequential(nn.InstanceNorm2d(channels, affine=True),
                         nn.LeakyReLU(0.01, inplace=False))


class ResidualContext(nn.Module):
    """Pre-activation context block with spatial dropout between convolutions."""
    def __init__(self, channels, dropout=0.3):
        super().__init__()
        self.branch = nn.Sequential(
            norm_activation(channels), nn.Conv2d(channels, channels, 3, padding=1),
            nn.Dropout2d(dropout), norm_activation(channels),
            nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return x + self.branch(x)


class Localization(nn.Sequential):
    def __init__(self, incoming, outgoing):
        super().__init__(
            nn.Conv2d(incoming, outgoing, 3, padding=1), norm_activation(outgoing),
            nn.Conv2d(outgoing, outgoing, 1), norm_activation(outgoing))


class ImprovedUNet2D(nn.Module):
    """Residual encoder and summed multi-scale logits provide deep supervision.

Deliberate deviations from the cited 3D model: 2D kernels, one MRI channel,
configurable width, direct interpolation to skip sizes, no test-time ensemble.
"""
    def __init__(self, num_classes, base_channels=16, dropout=0.3):
        super().__init__()
        widths = [base_channels * 2**i for i in range(5)]
        self.entry = nn.ModuleList(
            nn.Conv2d(1 if i == 0 else widths[i - 1], width, 3,
                      stride=1 if i == 0 else 2, padding=1)
            for i, width in enumerate(widths))
        self.context = nn.ModuleList(ResidualContext(w, dropout) for w in widths)
        self.up_conv = nn.ModuleList(
            nn.Sequential(nn.Conv2d(widths[i + 1], widths[i], 3, padding=1),
                          norm_activation(widths[i]))
            for i in reversed(range(4)))
        self.localize = nn.ModuleList(Localization(2 * widths[i], widths[i])
                                      for i in reversed(range(4)))
        self.segmentation_heads = nn.ModuleList(
            nn.Conv2d(widths[i], num_classes, 1) for i in (2, 1, 0))

    def forward(self, x):
        full_size, skips = x.shape[-2:], []
        for entry, block in zip(self.entry, self.context):
            x = block(entry(x))
            skips.append(x)
        maps = []
        for index, (up, local, skip) in enumerate(
                zip(self.up_conv, self.localize, reversed(skips[:-1]))):
            x = up(F.interpolate(x, size=skip.shape[-2:], mode="nearest"))
            x = local(torch.cat((skip, x), dim=1))
            if index >= 1:
                maps.append(self.segmentation_heads[index - 1](x))
        # Every head receives gradients through the final supervised segmentation.
        return sum(F.interpolate(logits, size=full_size, mode="bilinear",
                                 align_corners=False) for logits in maps)


def build_model(name, num_classes, base_channels=16):
    if num_classes < 2 or base_channels < 1:
        raise ValueError("Need >= 2 classes and positive base channels")
    constructors = {"baseline": UNet2D, "improved": ImprovedUNet2D}
    if name not in constructors:
        raise ValueError(f"Unknown architecture: {name}")
    return constructors[name](num_classes, base_channels)

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class PSPHead(nn.Module):
    """PSPNet head module for segmentation."""

    def __init__(
        self, in_channels: int, channels: int, pool_sizes: tuple = (1, 2, 3, 6)
    ):
        super(PSPHead, self).__init__()
        self.ppm = PyramidPoolingModule(in_channels, channels, pool_sizes)
        self.bottleneck = nn.Conv2d(
            in_channels + len(pool_sizes) * channels, channels, kernel_size=1
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.ppm(x)
        x = self.bottleneck(x)
        return x


class PyramidPoolingModule(nn.Module):
    """Pyramid Pooling Module used in PSPNet."""

    def __init__(
        self, in_channels: int, channels: int, pool_sizes: tuple = (1, 2, 3, 6)
    ):
        super(PyramidPoolingModule, self).__init__()
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, channels, size) for size in pool_sizes]
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.size()[2:]
        pyramids = [x]
        for stage in self.stages:
            pyramids.append(
                F.interpolate(
                    stage(x), size=(h, w), mode="bilinear", align_corners=True
                )
            )
        return torch.cat(pyramids, dim=1)

    def _make_stage(self, in_channels: int, channels: int, bin_size: int):
        """AdaptiveAvgPool2d + 1x1 Conv2d."""
        prior = nn.AdaptiveAvgPool2d((bin_size, bin_size))
        conv = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling Module used in DeeplabV3."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        dilations: tuple[int, ...] = (1, 6, 12, 18),
    ):
        super(ASPPModule, self).__init__()
        self.stages = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=channels,
                    kernel_size=1 if dilation == 1 else 3,
                    padding=0 if dilation == 1 else dilation,
                    dilation=dilation,
                    bias=False,
                )
                for dilation in dilations
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        pyramids = []
        for stage in self.stages:
            pyramids.append(stage(x))
        return torch.cat(pyramids, dim=1)


class ASPPHead(nn.Module):
    def __init__(
        self, in_channels: int, channels: int, dilations: tuple = (1, 6, 12, 18)
    ):
        super(ASPPHead, self).__init__()
        self.img_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
        )
        self.asppm = ASPPModule(in_channels, channels, dilations)
        self.bottleneck = nn.Conv2d(
            (len(dilations) + 1) * channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.size()[2:]
        out = torch.cat(
            [
                F.interpolate(
                    self.img_pool(x), size=(h, w), mode="bilinear", align_corners=True
                ),
                self.asppm(x),
            ],
            dim=1,
        )
        out = self.bottleneck(out)
        return out

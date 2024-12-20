"""ResNet backbone for segmentation."""

from torch import nn, Tensor
from torchvision import models


class BasicBlock(nn.Module):
    """Residual block in ResNet."""

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample=None,
    ) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = _conv3x3(in_planes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.conv2(h)
        h = self.bn2(h)
        if self.downsample:
            identity = self.downsample(x)
        h += identity
        h = self.relu(h)
        return h


class Bottleneck(nn.Module):
    """Bottleneck in ResNet."""

    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample=None,
    ) -> None:
        super(Bottleneck, self).__init__()
        self.conv1 = _conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = _conv3x3(planes, planes, stride=stride, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = _conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu(h)
        h = self.conv3(h)
        h = self.bn3(h)
        if self.downsample:
            identity = self.downsample(x)
        h += identity
        h = self.relu(h)
        return h


class ResNet(nn.Module):
    """ResNet."""

    cfg = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }

    def __init__(
        self,
        depth: int,
        replace_stride_with_dilation: list[bool] = [False, False, False],
    ) -> None:
        super(ResNet, self).__init__()
        if depth not in self.cfg:
            raise ValueError(f"ResNet-{depth} is not supported.")
        layers = self.cfg[depth]
        block = BasicBlock if depth in (18, 34) else Bottleneck
        self.in_planes = 64
        self.dilation = 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)
        c1 = self.maxpool(h)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c1, c2, c3, c4, c5

    def _make_layer(
        self,
        block: type["BasicBlock | Bottleneck"],
        planes: int,
        num_blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        prev_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.in_planes, planes * block.expansion, stride=stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [
            block(
                self.in_planes,
                planes,
                stride,
                dilation=prev_dilation,
                downsample=downsample,
            )
        ]
        self.in_planes = planes * block.expansion
        for _ in range(num_blocks - 1):
            layers.append(block(self.in_planes, planes, dilation=self.dilation))
        return nn.Sequential(*layers)


def resnet18(pretrained: bool = True) -> ResNet:
    """ResNet-18 backbone."""
    model = ResNet(depth=18)
    if pretrained:
        pretrained_model = models.resnet18(weights="IMAGENET1K_V1")
        model.load_state_dict(pretrained_model.state_dict(), strict=False)
    return model


def resnet34(pretrained: bool = True) -> ResNet:
    """ResNet-34 backbone."""
    model = ResNet(depth=34)
    if pretrained:
        pretrained_model = models.resnet34(weights="IMAGENET1K_V1")
        model.load_state_dict(pretrained_model.state_dict(), strict=False)
    return model


def resnet50(
    replace_stride_with_dilation: list[bool] = [False, False, False],
    pretrained: bool = True,
) -> ResNet:
    """ResNet-50 backbone."""
    model = ResNet(depth=50, replace_stride_with_dilation=replace_stride_with_dilation)
    if pretrained:
        pretrained_model = models.resnet50(weights="IMAGENET1K_V1")
        model.load_state_dict(pretrained_model.state_dict(), strict=False)
    return model


def resnet101(
    replace_stride_with_dilation: list[bool] = [False, False, False],
    pretrained: bool = True,
) -> ResNet:
    """ResNet-101 backbone."""
    model = ResNet(depth=101, replace_stride_with_dilation=replace_stride_with_dilation)
    if pretrained:
        pretrained_model = models.resnet101(weights="IMAGENET1K_V1")
        model.load_state_dict(pretrained_model.state_dict(), strict=False)
    return model


def resnet152(
    replace_stride_with_dilation: list[bool] = [False, False, False],
    pretrained: bool = True,
) -> ResNet:
    """ResNet-152 backbone."""
    model = ResNet(depth=152, replace_stride_with_dilation=replace_stride_with_dilation)
    if pretrained:
        pretrained_model = models.resnet152(weights="IMAGENET1K_V1")
        model.load_state_dict(pretrained_model.state_dict(), strict=False)
    return model


def _conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution."""
    return nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False,
    )


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )

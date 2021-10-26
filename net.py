import torch.nn
import torch.nn as nn
from dorefanet import *
from typing import Type, Any, Callable, List, Optional, Tuple, Union


class AlexNet(torch.nn.Module):
    """

    """
    def __init__(self, w_bits, a_bits, num_classes=1000):
        super(AlexNet, self).__init__()
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.num_classes = num_classes

        self.first_layer = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))

        self.quantized_layer_1 = nn.Sequential(
            QuantizationConv2d(96, 256, kernel_size=5, padding=2, w_bits=self.w_bits),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            QuantizationActivation(self.a_bits),
            nn.MaxPool2d(3, 2),

            QuantizationConv2d(256, 384, kernel_size=3, w_bits=self.w_bits),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            QuantizationActivation(self.a_bits),

            QuantizationConv2d(384, 384, kernel_size=3, w_bits=self.w_bits),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            QuantizationActivation(self.a_bits),

            QuantizationConv2d(384, 256, kernel_size=3, w_bits=self.w_bits),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            QuantizationActivation(self.a_bits),
            nn.MaxPool2d(3, 2),

        )

        self.quantized_layer_2 = nn.Sequential(
            QuantizationFullyConnected(256*3*3, 4096, w_bits=self.w_bits),
            nn.ReLU(inplace=True),
            QuantizationActivation(self.a_bits),

            QuantizationFullyConnected(4096, 4096, w_bits=self.w_bits),
            nn.ReLU(inplace=True),
            QuantizationActivation(self.a_bits)
        )

        self.last_layer = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.first_layer(x)
        x = self.quantized_layer_1(x)
        x = torch.flatten(x, 1)
        x = self.quantized_layer_2(x)
        x = self.last_layer(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, w_bits=32, a_bits=32):
        super(BasicBlock, self).__init__()

        # self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.quant_conv1 = QuantizationConv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False, w_bits=w_bits)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.quant_activation = QuantizationActivation(a_bits)

        # self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.quant_conv2 = QuantizationConv2d(planes, planes, 3, padding=1, bias=False, w_bits=w_bits)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample= downsample

    def forward(self, x):
        residual = x

        out = self.quant_conv1(x)
        # out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.quant_activation(out)

        out = self.quant_conv2(out)
        # out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.quant_activation(out)

        return out


class ResNet20(torch.nn.Module):
    """
    Cifar10 데이터셋을 활용하기 위한 ResNet20 모델.

    Fitst Layer, Last Layer는 양자화를 진행하지 않는다(FP32).

    Layer 순서는 Quant_Conv -> BN -> ReLU -> Quant_Activation

    Reference:
            Paper 참조(3~4page): "https://arxiv.org/pdf/1512.03385.pdf"
    """
    def __init__(
        self,
        block: BasicBlock,
        layers: List[int],
        w_bits: int,
        a_bits: int,
        num_classes: int=10
    ) -> None:
        super(ResNet20, self).__init__()

        self.w_bits = w_bits
        self.a_bits = a_bits
        self.in_planes = 16

        self.conv = nn.Conv2d(3, self.in_planes, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block: Type[BasicBlock], planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        layers = []

        if (stride != 1) or (self.in_planes != planes * block.expansion): # block.expansion은 항상 1
            downsample = nn.Sequential(
                QuantizationConv2d(self.in_planes, planes, 1, stride=stride, w_bits=self.w_bits),
                nn.BatchNorm2d(planes)
            )

        layers.append(block(self.in_planes, planes, stride, downsample, w_bits=self.w_bits, a_bits=self.a_bits))

        self.in_planes = planes * block.expansion

        for num_block in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, w_bits=self.w_bits, a_bits=self.a_bits))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        out = self.fc(out)

        return out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def alexnet(**kwargs):
    model = AlexNet(**kwargs)
    return model


def resnet20(w_bits, a_bits, **kwargs):
    r"""
    ResNet-20 모델 함수.

    Parameters
    ----------
    w_bits: int
        가중치 양자화 bits

    a_bits: int
        활성함수 양자화 bits
    """
    model = ResNet20(BasicBlock, [3, 3, 3], w_bits, a_bits, **kwargs)

    return model


if __name__ == '__main__':
    w_bits, a_bits = 32, 32
    model = resnet20(w_bits, a_bits)
    print(model)

    for i, (name, param) in enumerate(model.named_parameters()):
        print(param)
        print(name)

        if i == 3:
            break
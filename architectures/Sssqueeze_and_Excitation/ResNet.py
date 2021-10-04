"""
Sssqueeze_and_Excitation/ResNet.py

Variation of Squeeze_and_Excitation(SE), so we call this algorithm as SssE.
As described in README.md, squeeze operation in SE - global average pooling - is replaced to global average pooling * standard-deviation.

sample model configuration:
    "model": {
        "backbone": "resnet18",
        "variation": {
            "type": "SssE",
            "config": {
                "reduction": 16
            }
        }
    }
"""
import torch
import torch.nn as nn

from ..Baseline import ResNet_Baseline



__all__ = [
    'resnet18_SssE',
    'resnet34_SssE',
    'resnet50_SssE'
]


class SssELayer(nn.Module):
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.excitation = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        
    
    def forward(self, x: torch.Tensor):
        b, c, _, _ = x.size()
        mean = torch.nn.functional.adaptive_avg_pool2d(x, 1).view(b, c)
        std = torch.std(x, dim=[2, 3])
        squeeze = mean * std
        excitation = self.excitation(squeeze).view(b, c, 1, 1)
        return x * excitation.expand_as(x)
    
    
def conv3x3_SssE(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1_SssE(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class BasicBlock_SssE(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, *, config=None):
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3_SssE(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3_SssE(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()
        self.ssse = SssELayer(planes, config['reduction'])
        self.downsample = downsample
        self.stride = stride

        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ssse(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out
    
    
    
class Bottleneck_SssE(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, config=None):
        super().__init__()
        width = int(planes * (base_width / 64.)) * groups
        
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1_SssE(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3_SssE(width, width, stride, groups, dilation)
        self.bn2 = nn.BatchNorm2d(width)
        self.relu2 = nn.ReLU()
        self.conv3 = conv1x1_SssE(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU()
        self.ssse = SssELayer(planes * 4, config['reduction'])
        self.downsample = downsample
        self.stride = stride

        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.ssse(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out
    
    
def resnet18_SssE(target_dataset, config=None):
    return ResNet_Baseline(BasicBlock_SssE, [2, 2, 2, 2], target_dataset, config=config)


def resnet34_SssE(target_dataset, config=None):
    return ResNet_Baseline(BasicBlock_SssE, [3, 4, 6, 3], target_dataset, config=config)


def resnet50_SssE(target_dataset, config=None):
    return ResNet_Baseline(BasicBlock_SssE, [3, 4, 6, 3], target_dataset, config=config)
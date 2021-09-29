"""
Quantization_Aware_Training/ResNet.py

Quantization Aware Training scheme applied ResNet implementation.
PACT is used as activation function instead of ReLU.

sample model configuration:
    "model": {
        "backbone": "resnet18",
        "variation": {
            "type": "QAT",
            "config": {
                "weight": 8,
                "activation": 8,

                "weight_scaling_per_output_channel": true
            }
        }
    }
"""
import torch
import torch.nn as nn
import brevitas.nn as qnn

from ..Baseline import ResNet_Baseline
from ..common_components import *



__all__ = [
    'resnet18_QAT',
    'resnet34_QAT',
    'resnet50_QAT'
]


def conv3x3_QAT(config, in_planes, out_planes, stride=1, groups=1, dilation=1):
    return qnn.QuantConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                           padding=dilation, groups=groups, bias=False, dilation=dilation,
                           weight_bit_width=config['weight'], 
                           weight_scaling_per_output_channel=config['weight_scaling_per_output_channel'])


def conv1x1_QAT(config, in_planes, out_planes, stride=1):
    return qnn.QuantConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,
                           weight_bit_width=config['weight'], 
                           weight_scaling_per_output_channel=config['weight_scaling_per_output_channel'])


class BasicBlock_QAT(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, config=None):
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
            
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3_QAT(config, inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = QuantPACTReLU(bit_width=config['activation'])
        self.conv2 = conv3x3_QAT(config, planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = QuantPACTReLU(bit_width=config['activation'])
        self.downsample = downsample
        self.stride = stride
        
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out
    
    
        
class Bottleneck_QAT(nn.Module):
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
        self.conv1 = conv1x1_QAT(config, inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu1 = QuantPACTReLU(bit_width=config['activation'])
        self.conv2 = conv3x3_QAT(config, width, width, stride, groups, dilation)
        self.bn2 = nn.BatchNorm2d(width)
        self.relu2 = QuantPACTReLU(bit_width=config['activation'])
        self.conv3 = conv1x1_QAT(config, width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = QuantPACTReLU(bit_width=config['activation'])
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

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out
    
    

def resnet18_QAT(target_dataset, config=None):
    return ResNet_Baseline(BasicBlock_QAT, [2, 2, 2, 2], target_dataset, config=config)


def resnet34_QAT(target_dataset, config=None):
    return ResNet_Baseline(BasicBlock_QAT, [3, 4, 6, 3], target_dataset, config=config)


def resnet50_QAT(target_dataset, config=None):
    return ResNet_Baseline(BasicBlock_QAT, [3, 4, 6, 3], target_dataset, config=config)
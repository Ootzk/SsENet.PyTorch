"""
ResNet.py

ResNet architecture implementation with or without SE variant algorithm.
source code refer: https://github.com/moskomule/senet.pytorch

sample model configuration:
    "model": {
        "backbone": "resnet18",
        # no config if you want Baseline
        "config": {
            "algorithm": ("SE" / "gap" / "gmp" / "std" / "gapXstd" / "random"),
            # "SE" == "gap"
            "reduction": 16
        }
    }
"""
from typing import Type, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'resnet18',
    'resnet34',
    'resnet50'
]



def conv3x3(in_planes: int,
            out_planes: int,
            stride: int=1,
            groups: int=1,
            dilation: int=1) -> nn.Conv2d:
    
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)



def conv1x1(in_planes: int,
            out_planes: int,
            stride: int=1) -> nn.Conv2d:
    
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class SE_layer(nn.Module):
    def __init__(self,
                 channel: int,
                 algorithm: str="SE",
                 reduction: int=16) -> None:
        super().__init__()
        
        assert algorithm in ["SE", "gap", "gmp", "std", "gapXstd", "random"]
        self.algorithm = algorithm
        
        self.excitation = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        
    
    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        
        b, c, _, _ = x.size()
        
        if self.algorithm == "SE" or self.algorithm == "gap":
            squeeze = F.adaptive_avg_pool2d(x, 1).view(b, c)
        elif self.algorithm == "gmp":
            squeeze = F.adaptive_max_pool2d(x, 1).view(b, c)
        elif self.algorithm == "std":
            squeeze = torch.std(x, dim=[2, 3])
        elif self.algorithm == "gapXstd":
            squeeze = F.adaptive_avg_pool2d(x, 1).view(b, c) * torch.std(x, dim=[2, 3])
        elif self.algorithm == "random":
            squeeze = torch.rand((b, c), device=x.get_device())
            
        y = self.excitation(squeeze).view(b, c, 1, 1)
        
        return x * y.expand_as(x)
    
    
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 *,
                 config: Optional[dict] = None) -> None:
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()
        
        if config is not None:
            self.se = SE_layer(planes, config["algorithm"], config['reduction'])
        self.downsample = downsample
        self.stride = stride

        
    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if hasattr(self, "se"):
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out
    
    
    
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, 
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 *, 
                 config: Optional[dict] = None) -> None:
        super().__init__()
        width = int(planes * (base_width / 64.)) * groups
        
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = nn.BatchNorm2d(width)
        self.relu2 = nn.ReLU()
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU()
        if config is not None:
            self.se = SE_layer(planes * 4, config['algorithm'], config['reduction'])
        self.downsample = downsample
        self.stride = stride

        
    def forward(self, 
                x: torch.Tensor):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if hasattr(self, "se"):
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out
    
    
    
class ResNet(nn.Module):
    def __init__(self, 
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 target_dataset: str,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 *,
                 config: Optional[dict])-> None:
        super().__init__()
        
        assert target_dataset in ['CIFAR10', 'CIFAR100', 'ImageNet', 'DEBUG']
        self.target_dataset = target_dataset
        if target_dataset == 'CIFAR10' or target_dataset == 'DEBUG':
            self.num_classes = 10
        elif target_dataset == 'CIFAR100':
            self.num_classes = 100
        elif target_dataset == 'ImageNet':
            self.num_classes = 1000
            
        if config is not None:
            assert all(attr in config for attr in ["algorithm", "reduction"])
        
        self.config = config
        
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        if target_dataset == 'ImageNet':
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.ReLU()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)
                

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, config=self.config))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation, config=self.config))

        return nn.Sequential(*layers)
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        if hasattr(self, 'maxpool'):
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    
    
def resnet18(target_dataset, config=None):
    return ResNet(BasicBlock, [2, 2, 2, 2], target_dataset, config=config)


def resnet34(target_dataset, config=None):
    return ResNet(BasicBlock, [3, 4, 6, 3], target_dataset, config=config)


def resnet50(target_dataset, config=None):
    return ResNet(BasicBlock, [3, 4, 6, 3], target_dataset, config=config)
# coding=utf-8
import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """DepthwiseSeparableConv
    Implement of depthwise separable convolution
    
    Args: 
        in_channels: Number of channels in the input data
        out_channels: Number of channels produced by the convolution
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution.
        width_multiplier: The hyper-parameter introduced by Google's paper(https://arxiv.org/abs/1704.04861) uses to 
                          control the model's width(the channel)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, width_multiplier=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        in_channels = round(in_channels*width_multiplier)
        out_channels = round(out_channels*width_multiplier)
        
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=1,
                                   groups=in_channels, bias=False)
        self.batchnorm0 = nn.BatchNorm2d(in_channels)
        
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.batchnorm0(x)
        x = self.relu(x)
        
        x = self.pointwise(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        return x
        
        
class MobileNet(nn.Module):
    """MobileNet
    A pytorch implementation of MobileNet.
    The paper about MobileNet is Google's paper(https://arxiv.org/abs/1704.04861)
    
    Args:
        in_channels: Number of channels in the input data
        num_classes: Number of classes want to classify
        width_multiplier: The hyper-parameter introduced by Google's paper(https://arxiv.org/abs/1704.04861) uses to 
                          control the model's width(the channel)
        
    Return:
        logits: the result after softmax activation
    """
    def __init__(self, in_channels, num_classes=1000, width_multiplier=1):
        super(MobileNet, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, round(32*width_multiplier), kernel_size=3, stride=2, padding=1, bias=False)
        self.batchnorm = nn.BatchNorm2d(round(32*width_multiplier))
        self.relu = nn.ReLU()
        
        self.layer1 = self._make_layer(32, 64, width_multiplier=width_multiplier)
        self.layer2 = self._make_layer(64, 128, stride=2, width_multiplier=width_multiplier)
        self.layer3 = self._make_layer(128, 128, width_multiplier=width_multiplier)
        self.layer4 = self._make_layer(128, 256, stride=2, width_multiplier=width_multiplier)
        self.layer5 = self._make_layer(256, 256, width_multiplier=width_multiplier)
        self.layer6 = self._make_layer(256, 512, stride=2, width_multiplier=width_multiplier)
        
        layer7by5 = []
        for i in range(5):
            layer7by5.append(self._make_layer(512, 512, width_multiplier=width_multiplier))
        self.layer7 = nn.Sequential(*layer7by5)
        
        self.layer8 = self._make_layer(512, 1024, stride=2, width_multiplier=width_multiplier)
        self.layer9 = self._make_layer(1024, 1024, width_multiplier=width_multiplier)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(round(1024*width_multiplier), num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        logits = self.softmax(x)
        
        return logits
    
    def _make_layer(self, in_channels, out_channels, kernel_size=3, stride=1,  width_multiplier=1):
        return DepthwiseSeparableConv(in_channels, out_channels, kernel_size, stride,  width_multiplier)



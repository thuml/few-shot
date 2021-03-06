# This code is modified from https://github.com/kjunelee/MetaOptNet

import torch.nn as nn
import torch.nn.functional as F

from few_shot.config import cfg
from few_shot.dropblock import DropBlock


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.C1 = conv3x3(inplanes, planes)
        self.BN1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.C2 = conv3x3(planes, planes)
        self.BN2 = nn.BatchNorm2d(planes)
        self.C3 = conv3x3(planes, planes)
        self.BN3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample

        # dropblock
        self.drop_rate = drop_rate
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        # get the num of batches
        num_batches_tracked = int(self.BN1.num_batches_tracked.cpu().data)
        residual = x

        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu(out)

        out = self.C2(out)
        out = self.BN2(out)
        out = self.relu(out)

        out = self.C3(out)
        out = self.BN3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * num_batches_tracked, 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training)

        return out


class ResNet12(nn.Module):

    def __init__(self, block, flatten=True, drop_rate=0.0, dropblock_size=5):
        self.inplanes = 3
        super(ResNet12, self).__init__()

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        if flatten:
            self.avgpool = nn.AvgPool2d(5, stride=1)
            self.final_feat_dim = 640
        else:
            self.final_feat_dim = [640, 5, 5]
        self.flatten = flatten
        self.num_batches_tracked = 0

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size)]
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.flatten:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


def resnet12(flatten=True):
    """Constructs a ResNet-12 model.
    """
    model = ResNet12(BasicBlock, flatten=flatten, drop_rate=cfg.train.drop_rate,
                     dropblock_size=cfg.train.dropblock_size)
    return model

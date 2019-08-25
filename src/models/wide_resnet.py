# codes were based on https://github.com/louis2889184/pytorch-adversarial-training/blob/master/cifar-10/src/model/model.py
# original author: louis2889184

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import ModelTemplate, per_image_standardization_tf


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual
        self.in_p = in_planes
        self.out_p = out_planes

        self.avg_pool = torch.nn.AvgPool2d(stride, stride, padding=0)  # no padding is in effect tf.avgpool with 'VALID'

    def forward(self, x):
        if self.activate_before_residual:
            out = self.relu1(self.bn1(x))
            orig_x = out
        else:
            orig_x = x
            out = self.relu1(self.bn1(x))
        if self.in_p != self.out_p:
            orig_x = self.avg_pool(orig_x)
            orig_x = F.pad(orig_x, (0,0,0,0, (self.out_p - self.in_p)//2, (self.out_p - self.in_p)//2), mode='constant', value=0)

        out = self.relu2(self.bn2(self.conv1(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)

        return out + orig_x


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(nb_layers, in_planes, out_planes, block, stride, dropRate, activate_before_residual)

    def _make_layer(self, nb_layers, in_planes, out_planes, block, stride, dropRate, activate_before_residual):
        layers = []
        for i in range(nb_layers):
            if i==0:
                layers.append(block(in_planes, out_planes, stride, dropRate, activate_before_residual))
            else:
                layers.append(block(out_planes, out_planes, 1, dropRate, False))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(ModelTemplate):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, True)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, False)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, False)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, *args):
        x = per_image_standardization_tf(x)
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)  # The size of each channel in the end (after passing through two strides of two 32/4=8)
        out = out.view(-1, self.nChannels)

        return self.fc(out)


if __name__ == '__main__':
    i = torch.FloatTensor(4, 3, 32, 32)

    n = WideResNet(depth=34, num_classes=10, widen_factor=10, dropRate=0.0)

# print(n(i).size())
  
""" DenseNet in PyTorch.
Official paper at https://arxiv.org/pdf/1608.06993.pdf
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import models


class ColorNet_40_x(nn.Module):
    def __init__(self, growth_rate=12, num_classes=8):
        super(ColorNet_40_x, self).__init__()
        num_nets = 7

        self.rgb_densenet = models.densenet_40_x(growth_rate=growth_rate, num_classes=num_classes)
        self.lab_densenet = models.densenet_40_x(growth_rate=growth_rate, num_classes=num_classes)
        self.hsv_densenet = models.densenet_40_x(growth_rate=growth_rate, num_classes=num_classes)
        self.yuv_densenet = models.densenet_40_x(growth_rate=growth_rate, num_classes=num_classes)
        self.ycbcr_densenet = models.densenet_40_x(growth_rate=growth_rate, num_classes=num_classes)
        self.hed_densenet = models.densenet_40_x(growth_rate=growth_rate, num_classes=num_classes)
        self.yiq_densenet = models.densenet_40_x(growth_rate=growth_rate, num_classes=num_classes)

        self.bn = nn.BatchNorm1d(num_nets*num_classes)
        self.linear = nn.Linear(num_nets*num_classes, num_classes)



    def forward(self, rgb, lab, hsv, yuv, ycbcr, hed, yiq):
        out_rgb = self.rgb_densenet(rgb)
        out_lab = self.lab_densenet(lab)
        out_hsv = self.lab_densenet(hsv)
        out_yuv = self.lab_densenet(yuv)
        out_ycbcr = self.lab_densenet(ycbcr)
        out_hed = self.lab_densenet(hed)
        out_yiq = self.lab_densenet(yiq)

        scores = torch.cat([out_rgb, out_lab, out_hsv, out_yuv, out_ycbcr, out_hed, out_yiq], 1)

        out = self.linear(self.bn(scores))

        return out


def test():
    net = ColorNet_40_x(growth_rate=12, num_classes=8).cuda()
    rgb_img = torch.randn(4, 3, 32, 32)
    lab_img = torch.randn(4, 3, 32, 32)
    y = net(rgb_img.cuda(), lab_img.cuda(), lab_img.cuda(), lab_img.cuda(), lab_img.cuda(), lab_img.cuda(), lab_img.cuda())
    print(y)
test()
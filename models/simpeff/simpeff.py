import torch
from torch import nn
from torchvision.ops.misc import ConvNormActivation
from torch.nn import Linear, AvgPool2d, Sigmoid

from .bridges import *


class Simpeff(torch.nn.Module):  # tm fixed
    def __init__(self):
        super().__init__()
        self.conv0 = ConvNormActivation( 1, 16, 3, 1, padding=1)
        self.conv1 = ConvNormActivation(16, 32, 3, 2, padding=1)
        self.conv2 = ConvNormActivation(32, 16, 1, 1)
        self.conv3 = ConvNormActivation(16, 32, 3, 1, padding=1)
        self.conv4 = ConvNormActivation(32, 64, 3, 2, padding=1)
        self.conv5 = ConvNormActivation(64, 32, 1, 1)
        self.conv6 = ConvNormActivation(32, 64, 3, 1, padding=1)
        self.conv7 = ConvNormActivation(64, 32, 1, 1)
        self.conv8 = ConvNormActivation(32, 64, 3, 1, padding=1)
        self.conv9 = ConvNormActivation(64, 32, 1, 1)

        self.dila10_13 = DilatedModule(2, 2)
        self.dila14_17 = DilatedModule(4, 4)

        self.conv19 = ConvNormActivation(64, 32, 1, 1)
        self.conv21 = ConvNormActivation(64, 128, 3, 2, padding=1)
        self.conv22 = ConvNormActivation(128, 64, 1, 1)
        self.conv23 = ConvNormActivation(64, 128, 3, 1, padding=1)
        self.conv24 = ConvNormActivation(128, 64, 1, 1)
        self.conv25 = ConvNormActivation(64, 128, 3, 1, padding=1)

        self.pass27_30 = PassthroughModule()

        self.conv31 = ConvNormActivation(128, 256, 3, 1, padding=1)
        self.conv32 = ConvNormActivation(256, 6, 1, 1)

        self.avgpool33 = AvgPool2d((128, 160))
        self.fc35 = Linear(6, 4)
        self.sigmoid36 = Sigmoid()
        
    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        y = self.conv9(x)
        z = self.dila10_13(y)
        m = self.dila14_17(z)

        n = torch.cat((y, z), 1)
        p = self.conv19(n)

        q = torch.cat((m, p), 1)
        q = self.conv21(q)
        q = self.conv22(q)
        q = self.conv23(q)
        q = self.conv24(q)
        q = self.conv25(q)
        q = self.pass27_30(x, q)
        q = self.conv31(q)
        q = self.conv32(q)
        q = self.avgpool33(q)

        q = torch.flatten(q, 1)
        q = self.fc35(q)
        q = self.sigmoid36(q)
        
        return q

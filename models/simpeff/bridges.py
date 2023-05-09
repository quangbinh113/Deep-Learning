import torch
from torch import nn
from torchvision.ops.misc import ConvNormActivation

class DilatedModule(nn.Module):
    def __init__(self, dilation, padding):
        super().__init__()
        self.dilation = dilation
        self.dilated_conv = ConvNormActivation(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=padding, padding_mode='replicate', dilation=dilation)
        self.conv1 = ConvNormActivation(in_channels=64, out_channels=32, kernel_size=1, stride=1)
        self.conv2 = ConvNormActivation(in_channels=64, out_channels=32, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        y = self.dilated_conv(x)
        y = self.conv1(y)
        z = torch.cat((x, y), 1)
        z = self.conv2(z)
        return z

class PassthroughModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvNormActivation(in_channels=64, out_channels=16, kernel_size=1, stride=1)
        self.conv2 = ConvNormActivation(in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        
    def forward(self, x, y):
        x = self.conv1(x)
        x = torch.cat(
            [x[:,:,::2,::2], x[:,:,::2,1::2], x[:,:,1::2,::2], x[:,:,1::2,1::2]], 1)
        
        x = torch.cat((x, y), 1)
        x = self.conv2(x)
        return x
    
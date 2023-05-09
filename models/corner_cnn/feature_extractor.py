import torchvision
from torch import nn, Tensor
import torchvision.ops as O

from .mlp import TwoMLPHead

class CustomeResnet34(nn.Module):
    def __init__(self, in_channels=1, out_channels=512):
        resnet = torchvision.models.resnet34(pretrained=True)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        resnet.fc = TwoMLPHead(out_channels)

        # inital frozen (change during training)
        for param in resnet.parameters():
            param.requires_grad = False
        resnet.conv1.weight.requires_grad = True
        for _, weight in resnet.fc.named_parameters():
            weight.requires_grad = True
    
    def forward(self, x):
        x = self.resnet(x)
        return x

# ---DEPRECATED
class LightWeightCNN(nn.Module): 
    def __init__(self, in_channels, out_channels=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        conv_block = []
        channels = [in_channels, 8, 16, 32]
        for i in range(1, len(channels)):
            conv = nn.Sequential(*[
                nn.Conv2d(in_channels=channels[i-1], out_channels=channels[i], kernel_size=3, stride=1, padding="same", bias=False),
                nn.BatchNorm2d(num_features=channels[i]),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=channels[i], out_channels=channels[i], kernel_size=3, stride=1, padding="same", bias=False),
                nn.BatchNorm2d(num_features=channels[i]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            ])
            conv_block += list(conv)
        self.conv_block = nn.Sequential(*conv_block)
        self.bottleneck = O.misc.ConvNormActivation(in_channels=channels[-1], out_channels=out_channels, kernel_size=1, stride=1, padding="same")
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.mlp_head = TwoMLPHead(out_channels)
        
    def forward(self, x):
        x = self.conv_block(x)
        x = self.bottleneck(x)
        x = self.global_avg_pool(x)
        x = self.mlp_head(x)
        return x

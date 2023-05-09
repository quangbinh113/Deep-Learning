from torch import nn, Tensor

class TwoMLPHead(nn.Module):
    def __init__(self, in_features=512):
        super().__init__()
        self.flatten = nn.Flatten()
        self.loc_fc = nn.Linear(in_features, 4)
        self.cls_fc = nn.Linear(in_features, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.flatten(x)
        box = self.sigmoid(self.loc_fc(x))
        score = self.sigmoid(self.cls_fc(x))
        return box, score
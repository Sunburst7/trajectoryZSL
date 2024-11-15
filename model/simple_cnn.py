import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding, dilation) -> None:
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.PReLU()
        self.max_pool = nn.MaxPool2d((2, 2), stride=2)

    def forward(self, x):
        return self.max_pool(self.relu(self.bn(self.conv(x))))


class SimpleCNN(nn.Module):
    def __init__(self, num_class, features_dim) -> None:
        super(SimpleCNN, self).__init__()
        self.DCL_1 = nn.Sequential(
            ConvLayer(8, 16, (3, 3), (1, 1), 1),
            ConvLayer(16, 32, (3, 3), (1, 1), 1),
            ConvLayer(32, 64, (3, 3), (1, 1), 1),
            nn.Flatten()
        )
        self.DCL_2 = nn.Sequential(
            ConvLayer(8, 16, (3, 3), (1, 1), 2),
            ConvLayer(16, 32, (3, 3), (1, 1), 2),
            ConvLayer(32, 64, (3, 3), (1, 1), 2),
            nn.Flatten()
        )
        self.DCL_3 = nn.Sequential(
            ConvLayer(8, 16, (3, 3), (1, 1), 3),
            ConvLayer(16, 32, (3, 3), (1, 1), 3),
            ConvLayer(32, 64, (3, 3), (1, 1), 3),
            nn.Flatten()
        )
        self.mlp = nn.Sequential(
            nn.Linear(3584, 1024),
            nn.PReLU(),
            nn.Linear(1024, 512),
            nn.PReLU(),
            nn.Linear(512, features_dim),
            nn.PReLU()
        )
        self.cls_head = nn.Linear(features_dim, num_class)

    def forward(self, x):
        x1 = self.DCL_1(x)
        x2 = self.DCL_2(x)
        x3 = self.DCL_3(x)
        x = torch.hstack((x1, x2, x3))
        # x = x1
        features = self.mlp(x)
        cls_logits = self.cls_head(features)
        return features, cls_logits

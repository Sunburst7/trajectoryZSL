import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.DCL_1 = nn.Sequential(
            nn.Conv2d(8, 16, (3, 3), padding=(1, 1), dilation=1),    # [16, 8, 32, 64]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),                         # [16, 8, 16, 32]
            nn.Conv2d(16, 32, (3, 3), padding=(1, 1), dilation=1),   # [16, 16, 16, 32]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),                         # [16, 16, 8, 16]
            nn.Conv2d(32, 64, (3, 3), padding=(1, 1), dilation=1),  # [16, 32, 8, 16]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),                          # [16, 32, 4, 8]
        )
        self.DCL_3 = nn.Sequential(
            nn.Conv2d(8, 16, (3, 3), padding=(1, 1), dilation=3),    # [16, 8, 32, 64]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),                         # [16, 8, 16, 32]
            nn.Conv2d(16, 32, (3, 3), padding=(1, 1), dilation=3),   # [16, 16, 16, 32]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),                         # [16, 16, 8, 16]
            nn.Conv2d(32, 64, (3, 3), padding=(1, 1), dilation=3),  # [16, 32, 8, 16]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),                          # [16, 32, 4, 8]
        )
        self.DCL_5 = nn.Sequential(
            nn.Conv2d(8, 16, (3, 3), padding=(1, 1), dilation=5),    # [16, 8, 32, 64]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),                         # [16, 8, 16, 32]
            nn.Conv2d(16, 32, (3, 3), padding=(1, 1), dilation=5),   # [16, 16, 16, 32]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),                         # [16, 16, 8, 16]
            nn.Conv2d(32, 64, (3, 3), padding=(1, 1), dilation=5),  # [16, 32, 8, 16]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),                          # [16, 32, 4, 8]
        )

    def forward(self, x):
        print(x.shape, x.dtype)
        x1 = self.DCL_1(x)
        x2 = self.DCL_3(x)
        x3 = self.DCL_5(x)
        print(x1.shape)
        print(x2.shape)
        print(x3.shape)
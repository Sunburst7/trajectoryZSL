import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_class, feature_dim) -> None:
        super(MyModel, self).__init__()

        self.conv1 = nn.Conv2d()

    def forward(self, x):
        pass
import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple

class AisDataset(Dataset):
    """航迹数据集
    """
    def __init__(self, X, Y) -> None:
        super().__init__()
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        return torch.tensor(self.X[index]), torch.tensor(self.Y[index])
    
    def __len__(self):
        return len(self.X)
    
    def __str__(self) -> str:
        return f"samples length={len(self.X)}, a sample which is class_{self.Y[0]}: {self.X[0]}"
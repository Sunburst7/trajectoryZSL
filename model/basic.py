import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from util.mahalanobis import MahalanobisLayer
from tqdm import tqdm

class BasicModel(nn.Module):
    def __init__(self, seen_class, features_dim) -> None:
        super(BasicModel, self).__init__()
        self.known_thresholds:Dict[int, torch.Tensor] = {known_class : 0 for known_class in seen_class} # 可见类的距离阈值
        self.distances:Dict[int, list] = {known_class : [] for known_class in seen_class} # 每个可见类对应样本到聚类中心的距离
        self.maha = MahalanobisLayer(features_dim, decay=0.99)

    def update_thresholds(self):
        distances = {k : torch.stack(v) for (k, v) in self.distances.items()} # self.distances[i] = (num_samples, num_class, 1)
        distances_std = {k : torch.std(v, dim=0) for (k, v) in distances.items()} # distances[i] = (num_samples, num_class)
        for (cls_id, distance) in distances.items():
            distances[cls_id] = torch.sort(distance, dim=0)[0]
            # 3-sigma
            outlier_indics = torch.where(distances[cls_id] >= 3 * distances_std[cls_id])[0]
            outlier_index = outlier_indics[0].item()
            tqdm.write(f"{cls_id}: {outlier_index}th value={distances[cls_id][outlier_index]}")
            self.known_thresholds[cls_id] = distances[cls_id][outlier_index]
            # 分位数
            # self.known_thresholds[cls_id] =  distances[cls_id][int(0.95 * distances[cls_id].shape[0])]
            # print(f"{cls_id}: threshold value={self.known_thresholds[cls_id]}")

    def calculate_distance(self, centers: nn.Parameter, feature: torch.Tensor, mode="mahalanobis") -> torch.Tensor:
        """计算特征向量到所有聚类中心的平方马氏距离

        Args:
            centers (nn.Parameter): 聚类中心向量 [num_class, features_dim]
            feature (torch.Tensor): 特征向量 [1, features_dim]

        Returns:
            torch.Tensor: 特征向量到每个聚类中心的马氏距离 [num_class(seen), 1]
        """
        if mode == "mahalanobis":
            return torch.stack([self.maha(feature, center).squeeze(dim=0) for center in centers])
        elif mode == "euclidan":
            return torch.stack([torch.norm(feature - center, p=2) for center in centers])
        elif mode == "manhattan":
            return torch.stack([torch.norm(feature - center, p=1) for center in centers])

    def dist_clearing(self):
        for saving_list in self.distances.values():
            saving_list.clear()

    def batch_dist_saving(self, centers: nn.Parameter, features: torch.Tensor, labels: torch.Tensor):
        for bid in range(features.shape[0]):
            true_label = labels[bid].item()
            self.distances[true_label].append(self.calculate_distance(centers, features[bid], mode="mahalanobis")[true_label])


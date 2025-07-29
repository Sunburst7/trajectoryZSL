import torch
import torch.nn as nn
import torch.nn.init as init
from model.GaussinHead import GaussinHead
from model.ProbHead import ProbHead

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes, proj_layer, hidden_dim, feat_dim,  margin, devices):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.feat_dim = feat_dim
        self.devices = devices
        self.margin = margin
        self.proj = proj_layer
        # self.p = nn.Parameter(torch.randn(hidden_dim, feat_dim).to(devices), requires_grad=True)
        # 投影矩阵，用于将 feat_dim 投影到 hidden_dim
        # 使用 ModuleList 管理 GaussinHead 模块
        self.gaussins = nn.ModuleList([ProbHead(hidden_dim) for _ in range(num_classes)])

    def forward(self, x, labels, is_train=True):
        """
        Args:
            x: 特征矩阵，形状 (batch_size, feat_dim).
            labels: ground truth 标签，形状 (batch_size).
        """
        batch_size = x.size(0)
        dim = x.size(1)

        if is_train:
            for i, x_ in enumerate(x):
                self.gaussins[labels[i]].update_params(x_)

        # 将整个 batch 一次性投影到 hidden_dim 空间，形状变为 (batch_size, hidden_dim)
        x_proj = self.proj(x)
        
        # 堆叠所有类别的均值和逆协方差矩阵
        means = torch.stack([gauss.get_mean() for gauss in self.gaussins], dim=0)  # (num_classes, hidden_dim)
        inv_covs = torch.stack([gauss.inv_obj_cov for gauss in self.gaussins], dim=0)  # (num_classes, hidden_dim, hidden_dim)
        
        # 计算 Mahalanobis 距离：
        # 将 x_proj 扩展为 (batch_size, 1, hidden_dim)，means 扩展为 (1, num_classes, hidden_dim)
        delta = x_proj.unsqueeze(1) - means.unsqueeze(0)  # (batch_size, num_classes, hidden_dim)
        # 利用 einsum 批量计算每个样本到每个类别中心的平方 Mahalanobis 距离
        dist_sq = torch.einsum('bni,nij,bnj->bn', delta, inv_covs, delta)
        distmat = torch.sqrt(dist_sq + 1e-12)  # (batch_size, num_classes)
        
        # 根据 margin 计算损失
        zeros = torch.zeros_like(distmat)
        # margin_distmat = torch.maximum(zeros, self.margin - distmat)
        
        # 构造 mask，确定样本真实类别和其它类别
        classes = torch.arange(self.num_classes, device=self.devices).unsqueeze(0).expand(batch_size, -1)
        labels_expanded = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        center_mask = (labels_expanded == classes)
        # margin_mask = (labels_expanded != classes)
        
        loss = (distmat * center_mask.float()).clamp(min=1e-12, max=1e+12).sum() / batch_size
        # margin_loss = (margin_distmat * margin_mask.float()).clamp(min=1e-12, max=1e+12).sum() / batch_size 

        return loss, distmat
    
    def get_centers(self) -> torch.Tensor:
        return torch.stack([gauss.get_mean() for gauss in self.gaussins])
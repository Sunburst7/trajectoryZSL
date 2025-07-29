import torch
import torch.nn as nn
import torch.nn.init as init
from ..model.GaussinHead import GaussinHead

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes, feat_dim, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t() # x^2 + centers^2
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2) # x^2 + centers^2 - 2x * centers
        zeros = torch.zeros(distmat.shape).to(distmat.device)
        margin_distmat = torch.maximum(zeros, self.margin - distmat)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes) # the distance of each sample vector to the semantics center [bs, num_class]
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        margin_dist = margin_distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        margin_loss = margin_dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss, margin_loss
    
    def get_centers(self) -> nn.Parameter:
        return self.centers
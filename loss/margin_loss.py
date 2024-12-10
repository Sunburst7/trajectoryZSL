import torch
import torch.nn as nn

class MarginLoss(nn.Module):
    """Margin loss.
    
    Args:
        margin: the maximum margin to other class in the semantic space
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
        centers(nn.Parameter) : the semantic center of each class, model owned.
    """
    def __init__(self, margin, num_classes, feat_dim, centers: nn.Parameter, use_gpu=True):
        super(MarginLoss, self).__init__()
        self.margin = margin
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = centers
        self.use_gpu = use_gpu


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
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.ne(classes.expand(batch_size, self.num_classes))

        margin_distmat = margin_distmat * mask.float()
        loss = margin_distmat.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
import torch
import torch.nn as nn
import math
import torch.nn.init as init
import torch.nn.functional as F
from model.ProbHead import ProbHead

class Criterion(nn.Module):
    def __init__(self, cfg, weight_dict, losses):
        super().__init__()
        self.cfg = cfg
        self.losses = losses
        self.weight_dict = weight_dict

    def loss_label(self, outputs, targets): 
        logits = outputs['pred_logits']
        return {"loss_label": F.cross_entropy(logits, targets)}
        # src_prob = torch.softmax(outputs['pred_logits'], dim=-1) * outputs['pred_objectness']
        # src_logits = torch.log(src_prob / (1 - src_prob))
        # batch_size = len(targets)

    def loss_dummy(self, outputs, targets):
        dummy_logits = outputs['pred_logits'].clone()
        for i in range(dummy_logits.shape[0]):
            dummy_logits[i][targets[i]] = -float('inf')
        dummy_y = torch.ones_like(targets) * self.cfg.dataset.unseen_class[0]
        loss_dummy = F.cross_entropy(dummy_logits, dummy_y)
        return {"loss_dummy": loss_dummy}

    def get_loss(self, loss, outputs, targets):
        loss_map = {
            'loss_label': self.loss_label,
            'loss_dummy': self.loss_dummy
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets)

    def forward(self, outputs, targets):
        # calculate true pred_logits

        # logits = outputs['pred_logits']
        # logit_max = logits[:, :-1].max(dim=-1, keepdim=True).values
        # # prob_unknown = 1 - (logit_max.exp() / logits.exp().sum(dim=-1, keepdim=True)) 
        
        # logits = logits - logit_max
        # tmp = logits[:, :-1].exp().sum(dim=-1, keepdim=True)
        # # assign unknown logit s.t. p(unknown) = p_original(known - max_known)
        # unknown_logit = (tmp - 1).log() + tmp.log() - (1 + logits[:, -1:].exp()).log()
        # logits = torch.cat([logits[:, :-1], unknown_logit], dim=-1)

        # outputs['pred_logits'] = logits
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))

        outputs['losses'] = losses
        return outputs
    
    def pred(self, outputs, targets):
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))

        outputs['losses'] = losses
        return outputs

        
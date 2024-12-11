import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from collections import OrderedDict

from layers.Embed import DataEmbedding
import numpy as np
from typing import List
from torchvision.models.vision_transformer import vit_b_16

class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def forward(self, x: torch.Tensor, query: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        x = x + self.attn(self.ln_1(query), self.ln_1(x), self.ln_1(x), need_weights=False, attn_mask=self.attn_mask)[0]
        x = x + self.mlp(self.ln_2(x))
        return x


class TFModel(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, cfg):
        super(TFModel, self).__init__()
        self.seq_len = cfg.seq_len
        self.pred_len = cfg.pred_len
        self.output_attention = cfg.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding(cfg.seq_len, cfg.d_model, dropout=cfg.dropout)
        # Encoder
        self.time_encoders = nn.Sequential(OrderedDict([
            (f'cross_attn_{l}', CrossAttention(d_model=cfg.d_model, n_head=cfg.n_heads, dropout=cfg.dropout))
            for l in range(cfg.e_layers)
        ]))
           
        self.freq_encoders = nn.Sequential(OrderedDict([
             (f'cross_attn_{l}', CrossAttention(d_model=cfg.d_model, n_head=cfg.n_heads, dropout=cfg.dropout))
            for l in range(cfg.e_layers)
        ]))

        # Decoder
        # self.act = F.gelu
        # self.dropout = nn.Dropout(cfg.dropout)
        # self.projection = nn.Linear(cfg.d_model * cfg.enc_in, cfg.num_class)

    def forward(self, x):
        # Embedding
        time_stack = []
        freq_stack = []
        x = torch.permute(x, (0, 2, 1))
        x_f = torch.abs(torch.fft.fft(x, dim=-1)) / x.shape[-1]
        for t_multihead_attn, f_multihead_attn in zip(self.time_encoders, self.freq_encoders):
            x_ = t_multihead_attn(x, query=x_f)
            x_f_ = f_multihead_attn(x_f, query=x)
            x = x_
            x_f = x_f_

            time_stack.append(x)
            freq_stack.append(x_f)
        return time_stack, freq_stack     
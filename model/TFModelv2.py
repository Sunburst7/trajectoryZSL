import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from collections import OrderedDict

# from layers.Embed import DataEmbedding
import numpy as np
from typing import List
from torchvision.models.vision_transformer import vit_b_16
from util.utils import initialize_weights
from config.deafault import get_cfg_defaults

class SelfEncoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_ff=None, dropout=0.1, activation="gelu"):
        super(SelfEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn

class CrossEncoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_ff=None, dropout=0.1, activation="gelu"):
        super(CrossEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.cross_attention_time = nn.MultiheadAttention(d_model, n_head, dropout)
        self.conv1_dis = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2_dis = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1_dis = nn.LayerNorm(d_model)
        self.norm2_dis = nn.LayerNorm(d_model)
        self.dropout_dis = nn.Dropout(dropout)
        self.activation_time = F.relu if activation == "relu" else F.gelu

        self.cross_attention_freq = nn.MultiheadAttention(d_model, n_head, dropout)
        self.conv1_con = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2_con = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1_con = nn.LayerNorm(d_model)
        self.norm2_con = nn.LayerNorm(d_model)
        self.dropout_con = nn.Dropout(dropout)
        self.activation_freq = F.relu if activation == "relu" else F.gelu

    def forward(self, x_time, x_freq, attn_mask=None, tau=None, delta=None):
        x_time_raw = x_time
        x_freq_raw = x_freq
        x_time_add, x_time_attn = self.cross_attention_time(x_time_raw, x_freq_raw, x_freq_raw)
        x_time_out = x_time_raw + self.dropout_dis(x_time_add)
        x_time_out = self.norm1_dis(x_time_out)

        y_time = x_time_out
        y_time = self.dropout_dis(self.activation_time(self.conv1_dis(y_time.transpose(-1, -2))))
        y_time = self.dropout_dis(self.conv2_dis(y_time).transpose(-1, -2))
        x_time_out = x_time_out + y_time

        x_time_out = self.norm2_dis(x_time_out + x_time_raw)



        x_freq_add, x_freq_attn = self.cross_attention_freq(x_freq_raw, x_time_raw, x_time_raw)
        x_freq_out = x_freq_raw + self.dropout_con(x_freq_add)

        x_freq_out = self.norm1_con(x_freq_out)

        y_freq = x_freq_out
        y_freq = self.dropout_con(self.activation_freq(self.conv1_con(y_freq.transpose(-1, -2))))
        y_freq = self.dropout_con(self.conv2_con(y_freq).transpose(-1, -2))
        x_freq_out = x_freq_out + y_freq

        #  residual
        x_freq_out = self.norm2_con(x_freq_out+x_freq_raw)

        return x_time_out, x_freq_out, x_time_attn, x_freq_attn
    
class Attention_EncoderBlock(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.1, activation="gelu"):
        super(Attention_EncoderBlock, self).__init__()
        self.intra_time_attention = SelfEncoderLayer(n_head, d_model,dropout=dropout,activation=activation)
        self.intra_freq_attention = SelfEncoderLayer(n_head, d_model,dropout=dropout,activation=activation)
        self.cross_attention = CrossEncoderLayer(n_head, d_model,dropout=dropout,activation=activation)

    def forward(self, x_time, x_freq, attn_mask=None, tau=None, delta=None):
        x_time_raw,dis_att = self.intra_time_attention(x_time)
        x_freq_raw,con_att = self.intra_freq_attention(x_freq)

        x_time_out, x_freq_out, dis_con_att, con_dis_att = self.cross_attention(x_time_raw,x_freq_raw)


        return x_time_out, x_freq_out, dis_att,con_att,dis_con_att,con_dis_att
    
class PatchEmbed(nn.Module):
    def __init__(self, seq_len, patch_size=8, in_chans=3, embed_dim=384):
        super().__init__()
        stride = patch_size // 2
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        x_out = self.proj(x).flatten(2).transpose(1, 2)
        return x_out

class TFModel(nn.Module):

    def __init__(self, cfg):
        super(TFModel, self).__init__()
        self.seq_len = cfg.seq_len
        self.pred_len = cfg.pred_len
        self.output_attention = cfg.output_attention
        # Embedding
        # self.enc_embedding = DataEmbedding(cfg.seq_len, cfg.d_model, dropout=cfg.dropout)
        self.flat = nn.Flatten()
        # Encoder
        self.encoders = nn.Sequential(OrderedDict([
            (f'attn_{l}', Attention_EncoderBlock(d_model=cfg.d_model, n_head=cfg.n_heads, dropout=cfg.dropout))
            for l in range(cfg.e_layers)
        ]))
        
        self.act = F.gelu
        self.dropout = nn.Dropout(cfg.dropout)
        self.projection = nn.Linear(cfg.d_model * cfg.enc_in, cfg.num_class)

    def forward(self, x):
        # Embedding
        # x_f = torch.abs(torch.fft.fft(x, dim=-1)) / x.shape[-1]
        t = torch.permute(x, (0, 2, 1)) # B C N
        f = torch.abs(torch.fft.fft(t)) / x.shape[-1] # B C N
        for encoder in self.encoders:
            t, f, _, _, _, _ = encoder(t, f)
        return self.projection(self.flat(t))


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    model = TFModel(cfg)
    x = torch.randn(16, 8, 100)
    y = model(x)
    print(y.shape)
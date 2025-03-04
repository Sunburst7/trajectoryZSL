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

class ComplexDropout(nn.Module):
    def __init__(self, p=0.5):
        super(ComplexDropout, self).__init__()
        self.p = p
        self.scale = 1.0 / (1.0 - p)  # 为了保持输出期望值不变

    def forward(self, x):
        # B C C -> B C C 2
        x = torch.view_as_real(x)
        if self.training:  # 只在训练模式下使用 dropout
            mask = (torch.rand_like(x[..., 0]) > self.p).float().unsqueeze(3)  # 生成0和1的掩码
            mask = mask * self.scale  # 对保留的单元进行缩放
            # 应用相同的掩码到两个输入上
            x = x * mask
        return torch.view_as_complex(x)
    
class ComplexLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(ComplexLayerNorm, self).__init__()
        # 分别为实部和虚部创建独立的 LayerNorm 层
        self.real_norm = nn.LayerNorm(normalized_shape=normalized_shape)
        self.imag_norm = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x):
        x = torch.view_as_real(x)
        # 提取实部和虚部
        real_part = x[..., 0] 
        imag_part = x[..., 1]

        # 对实部和虚部分别进行 LayerNorm
        real_normed = self.real_norm(real_part)
        imag_normed = self.imag_norm(imag_part)

        # 重新组合实部和虚部
        return torch.view_as_complex(torch.stack([real_normed, imag_normed], dim=-1))
    
class ComplexReLU(nn.Module):
    def __init__(self):
        super(ComplexReLU, self).__init__()

    def forward(self, z):
        # z is a complex tensor: z = x + iy
        x = z.real
        y = z.imag
        magnitude = torch.sqrt(x**2 + y**2)
        
        # Apply the smoothed ReLU-like transformation
        real_part = 0.5 * (x + magnitude)
        imag_part = 0.5 * y
        
        return torch.complex(real_part, imag_part)

class ComplexLinear(nn.Module):
    def __init__(self, d_in, d_out, bias:bool = True):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.weight = nn.Parameter(torch.randn((d_out, d_in, 2), dtype=torch.float32) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.randn((d_out, 2), dtype=torch.float32) * 0.02)
        else:
            self.register_parameter("bias", None)
    
    def forward(self, x):
        return F.linear(x, torch.view_as_complex(self.weight), torch.view_as_complex(self.bias))


class ComplexMultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.linear_Q = ComplexLinear(d_model, d_model)
        self.linear_K = ComplexLinear(d_model, d_model)
        self.linear_V = ComplexLinear(d_model, d_model)
        self.d_model = d_model
        self.n_heads = n_heads
        self.norm = ComplexLayerNorm(d_model)
        self.dropout = ComplexDropout(p=dropout)


    def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor):
        # 输入要求是复数
        Q = self.linear_Q(q) # B C N type=torch.complex64
        K = self.linear_K(k) # B C N type=torch.complex64
        V = self.linear_V(v) # B C N type=torch.complex64
        Q_ = torch.concat(torch.split(Q, self.d_model // self.n_heads, dim=2), dim=0) # B*h C N/h type=torch.complex64
        K_ = torch.concat(torch.split(K, self.d_model // self.n_heads, dim=2), dim=0) # B*h C N/h type=torch.complex64
        V_ = torch.concat(torch.split(V, self.d_model // self.n_heads, dim=2), dim=0) # B*h C N/h type=torch.complex64

        QK = torch.bmm(Q_, K_.transpose(2, 1)) / K_.shape[-1] ** 0.5
        attn = torch.view_as_complex(F.softmax(torch.view_as_real(QK), dim=-2))

        # Dropouts
        outputs = self.dropout(attn)
        # Weighted sum
        outputs = torch.bmm(outputs, V_)
        outputs = torch.concat(torch.split(outputs, outputs.shape[0] // self.n_heads, dim=0), dim=2)
        # Residual connection
        outputs = outputs + q
        # Normalize
        outputs = self.norm(outputs)
        return outputs, attn

class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout)
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.attn_mask = attn_mask

    def forward(self, x: torch.Tensor, query: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        x = x + self.attn(query, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        x = self.ln_1(x)
        x = x + self.mlp(x)
        return self.ln_2(x)
    
class ComplexCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = ComplexMultiheadAttention(d_model, n_head, dropout)
        self.ln_1 = ComplexLayerNorm(d_model)
        self.ln_2 = ComplexLayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", ComplexLinear(d_model, d_model * 4)),
            ("gelu", ComplexReLU()),
            ("c_proj", ComplexLinear(d_model * 4, d_model))
        ]))
        self.attn_mask = attn_mask

    def forward(self, x: torch.Tensor, query: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        x = x + self.attn(query, x, x)[0]
        x = self.ln_1(x)
        x = x + self.mlp(x)
        return self.ln_2(x)
    
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
        self.time_norm = nn.LayerNorm(cfg.d_model) if cfg.normalize_before else None
        self.freq_norm = ComplexLayerNorm(cfg.d_model) if cfg.normalize_before else None
        self.flat = nn.Flatten()
        # Encoder
        self.time_encoders = nn.Sequential(OrderedDict([
            (f'cross_attn_{l}', CrossAttention(d_model=cfg.d_model, n_head=cfg.n_heads, dropout=cfg.dropout))
            for l in range(cfg.e_layers)
        ]))
           
        self.freq_encoders = nn.Sequential(OrderedDict([
             (f'cross_attn_{l}', ComplexCrossAttention(d_model=cfg.d_model, n_head=cfg.n_heads, dropout=cfg.dropout))
            for l in range(cfg.e_layers)
        ]))


        # Classifier head
        self.time_weight = nn.Parameter(torch.randn((cfg.d_model * cfg.num_feature), dtype=torch.float32) * 0.02)
        self.freq_weight = nn.Parameter(torch.randn((cfg.d_model * cfg.num_feature, 2), dtype=torch.float32) * 0.02)
        self.head = nn.Linear(cfg.d_model * cfg.num_feature, cfg.num_class)

        # Decoder
        self.act = F.gelu
        self.dropout = nn.Dropout(cfg.dropout)
        # self.projection = nn.Linear(cfg.d_model * cfg.enc_in, cfg.num_class)

    def forward(self, x):
        # Embedding
        time_stack = []
        freq_stack = []
        # x_f = torch.abs(torch.fft.fft(x, dim=-1)) / x.shape[-1]
        x = torch.permute(x, (0, 2, 1)) # B C N
        x_f = torch.fft.fft(x) # B C N
        if self.time_norm:
            x = self.time_norm(x)
        if self.freq_norm:
            x_f = self.freq_norm(x_f)
        for t_multihead_attn, f_multihead_attn in zip(self.time_encoders, self.freq_encoders):
            x = t_multihead_attn(x, query=torch.fft.ifft(x_f).to(x.dtype))
            x_f = f_multihead_attn(x_f, query=torch.fft.fft(x))
            time_stack.append(self.flat(x))
            freq_stack.append(self.flat(x_f))
        return torch.stack(time_stack, dim=0), torch.stack(freq_stack, dim=0)     
    
    def classification(self, x):
        x = torch.permute(x, (0, 2, 1)) # B C N
        x_f = torch.fft.fft(x) # B C N
        if self.time_norm:
            x = self.time_norm(x)
        if self.freq_norm:
            x_f = self.freq_norm(x_f)
        for t_multihead_attn, f_multihead_attn in zip(self.time_encoders, self.freq_encoders):
            x = t_multihead_attn(x, query=torch.fft.ifft(x_f).to(x.dtype))
            x_f = f_multihead_attn(x_f, query=torch.fft.fft(x))
        x = self.flat(x) * self.time_weight
        x_f = self.flat(x) * torch.view_as_complex(self.freq_weight)
        return self.head(self.dropout(self.act(x + torch.abs(x_f) / x_f.shape[-1])))

if __name__ == "__main__":
    multihead_attn = CrossAttention(1024, 8, 0.1)
    complex_multihead_attn = ComplexCrossAttention(1024, 8, 0.1)
    x = torch.randn((16, 4, 1024))
    x_f = torch.fft.fft(x)
    print(multihead_attn(x, torch.fft.ifft(x_f).to(x.dtype)))
    print(complex_multihead_attn(x_f, x_f))
import numpy as np
import torch
import matplotlib.pyplot as plt
import os 
import torch.nn as nn
import torch.nn.init as init

def standardized(matrix: np.ndarray, axis=0):
    """Z-Score 标准化

    Args:
        matrix (np.ndarray): _description_
        axis (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    return (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)

def normalized(matrix: np.ndarray, axis=0):
    """最小-最大归一化

    Args:
        matrix (np.ndarray): _description_
        axis (int, optional): _description_. Defaults to 0.
    """
    return (matrix - matrix.min(axis=0)) / (matrix.max(axis=0) - matrix.min(axis=0))      

def normalize_to_range(matrix: np.ndarray, a, b, axis=0):
    """
    将数据归一化到指定的范围 [a, b]
    """
    # 计算数据的最小值和最大值
    min_val = matrix.min(axis=0)
    max_val = matrix.max(axis=0)
    
    # 归一化公式
    return a + (matrix - min_val) * (b - a) / (max_val - min_val)  


def bacth_STFT(x: torch.Tensor, n_fft, hop_len, win_len, window:torch.Tensor, verbose, path):
    """_summary_

    Args:
        x (torch.Tensor): _description_
        n_fft (_type_): _description_
        hop_len (_type_): _description_
        win_len (_type_): _description_
        window (torch.Tensor): _description_
        verbose (_type_): 是否保存图片
        path (_type_): 保存根目录

    Returns:
        _type_: STFT后的tensor

    x = bacth_STFT(x, cfg.stft.n_fft, cfg.stft.hop_length, cfg.stft.window_length, torch.hamming_window(cfg.stft.window_length).to(devices[0]), verbose=False)
    """
    # x [bs, seq_len, feature_dim]
    x = x.permute(0, 2, 1) # x [bs, feature_dim, seq_len]
    x_ = []

    def draw_and_save(img_vector: np.ndarray, path):
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img_vector, aspect='auto', origin='lower')
        plt.colorbar(label='Magnitude')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Time (s)')
        plt.savefig(path)
        plt.close()

    for i, single_sample in enumerate(x):
        stft_sample = torch.stft(single_sample, n_fft, hop_len, win_len, window, normalized=True, return_complex=True)[:, :-1, :-1]
        stft_sample = torch.cat((torch.view_as_real(stft_sample)[:, :, :, 0], torch.view_as_real(stft_sample)[:, :, :, 1]))
        if verbose == True:
            for j, time_freq_matrix in enumerate(stft_sample):
                draw_and_save(time_freq_matrix.cpu().numpy(), os.path.join(path, "temp", f"sample_{i}_features_{j}.png"))
        x_.append(stft_sample)
    return torch.stack(x_)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            init.ones_(m.weight)
            init.zeros_(m.bias)
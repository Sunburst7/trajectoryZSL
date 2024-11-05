import torch
import torch.nn as nn

class STFTLayer(nn.Module):
    """短时傅里叶变换层

        Args:
            n_fft (_type_): _description_
            hop_len (_type_): the distance between neighboring sliding window frames
            win_len (_type_): the size of window frame and STFT filter
            window (torch.Tensor): window function
            return_complex (bool, optional): 是否返回复数类型. Defaults to True.
    """

    def __init__(self, n_fft, hop_len, win_len, window:torch.Tensor, return_complex:bool=True) -> None:
        super(STFTLayer, self).__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.window = window
        self.return_complex = return_complex

    def forward(self, x):
        return torch.stft(x, self.n_fft, self.hop_len, self.win_len, self.window, return_complex=self.complex)
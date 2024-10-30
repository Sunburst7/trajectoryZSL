import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 创建一个 n 行 4 列的 ndarray（例如，n = 50）
n = 60
original_data = np.random.rand(n, 4)

# 创建原始数据的 x 轴坐标（假设均匀分布）
x_original = np.linspace(0, 1, n)

# 创建线性插值函数
interp_funcs = [interp1d(x_original, original_data[:, i], kind='linear', fill_value='extrapolate') for i in range(original_data.shape[1])]

# 生成新的 x 轴坐标（100 行）
x_new = np.linspace(0, 1, 100)

# 使用插值函数生成新的数据
new_data = np.array([interp_func(x_new) for interp_func in interp_funcs]).T

print("原始数据: " + str(original_data.shape))
print(original_data)
print("\n重构后的数据: " + str(new_data.shape))
print(new_data)



import torch
import numpy as np
import matplotlib.pyplot as plt

# 生成一个示例信号：1秒的正弦波
fs = 100  # 采样频率
t = np.linspace(0, 1, fs)
signal = .5 * np.sin(2 * np.pi * 100 * t)  # 100Hz 正弦波

# 转换为 PyTorch 张量
signal_tensor = torch.tensor(signal, dtype=torch.float32)

# STFT 参数
n_fft = 256
hop_length = 128
win_length = 256
window = torch.hann_window(win_length)

# 计算 STFT
stft_result = torch.stft(signal_tensor, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=window, return_complex=True)

# 获取幅度谱
magnitude = torch.abs(stft_result)

# 绘制 STFT 幅度谱
plt.figure(figsize=(10, 6))
plt.imshow(magnitude.T.numpy(), aspect='auto', origin='lower', extent=[0, 1, 0, fs / 2])
plt.colorbar(label='Magnitude')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title('STFT Magnitude Spectrum')
plt.show()


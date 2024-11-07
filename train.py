import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision
import random 
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from ais_dataset import AisDataset
from model.simple_transformer import SimpleTransformer
from model.simple_cnn import SimpleCNN

random.seed(42)
ROOT_DATA_PATH = os.path.join('/data2', 'hh', 'workspace', 'data', 'ais')
NUM_CLASS = 14
LNG_AND_LAT_THRESHOLD = 1
NUM_SAMPLE_ROW = 1024
# stft参数
N_FFT = 128
WINDOM_LENGTH = 128
HOP_LENGTH = 16
WINDOW_FUNCTION = "Hamming"
NUM_SAMPLE_FEATURES = 4
RATIO = 0.7
IS_GZSL = False
SEEN_CLASS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13]
UNSEEN_CLASS = [10]
RANDOM_SEED = 42

num_epoch = 10
num_class = 14
batch_size = 16
learning_rate = 1e-2
wd = 0
encoder_layer_num = 6
features_dim = 128
device = [i for i in range(torch.cuda.device_count())]

# model = SimpleTransformer(input_dim=NUM_SAMPLE_FEATURES, feature_dim=features_dim, num_heads=4, num_layers=encoder_layer_num, num_classes=NUM_CLASS)
model = SimpleCNN(num_class=NUM_CLASS).to(device[0])
model = torch.nn.DataParallel(model, device_ids=device)
criterion = nn.CrossEntropyLoss()  # cross entropy loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd) # model optimizer

def create_dataloader(path, batch_size):
    dataset = AisDataset.load(path)
    return Data.DataLoader(dataset, batch_size=batch_size)

def draw_and_save(img_vector: np.ndarray, path):
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_vector, aspect='auto', origin='lower')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Time (s)')

    plt.savefig(path)
    plt.close()

def bacth_STFT(x: torch.Tensor, n_fft, hop_len, win_len, window:torch.Tensor, verbose:bool=False):
    # x [bs, seq_len, feature_dim]
    x = x.permute(0, 2, 1) # x [bs, feature_dim, seq_len]
    x_ = []
    for i, single_sample in enumerate(x):
        stft_sample = torch.stft(single_sample, n_fft, hop_len, win_len, window, normalized=True, return_complex=True)[:, :-1, :-1]
        stft_sample = torch.cat((torch.view_as_real(stft_sample)[:, :, :, 0], torch.view_as_real(stft_sample)[:, :, :, 1]))
        if verbose == True:
            for j, time_freq_matrix in enumerate(stft_sample):
                draw_and_save(time_freq_matrix.cpu().numpy(), f"./temp/sample_{i}_features_{j}_.png")
        x_.append(stft_sample)
    return torch.stack(x_)

train_filepath = os.path.join(ROOT_DATA_PATH, f'train_seqLen_{NUM_SAMPLE_ROW}_ratio_{RATIO}_isGZSL_{IS_GZSL}.pkl')
valid_filepath = os.path.join(ROOT_DATA_PATH, f'valid_seqLen_{NUM_SAMPLE_ROW}_ratio_{RATIO}_isGZSL_{IS_GZSL}.pkl')
test_filepath = os.path.join(ROOT_DATA_PATH, f'test_seqLen_{NUM_SAMPLE_ROW}_ratio_{RATIO}_isGZSL_{IS_GZSL}.pkl')
train_loader = create_dataloader(train_filepath, batch_size)
valid_loader = create_dataloader(valid_filepath, batch_size)
test_loader = create_dataloader(test_filepath, batch_size)

def run_epoch(model, loader, is_train:bool):
    model.train()
    for epoch in tqdm(range(num_epoch), desc="训练"):
        pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
        for i, (x, y) in pbar:
            x = x.to(device[0])
            x = bacth_STFT(x, N_FFT, HOP_LENGTH, WINDOM_LENGTH, torch.hamming_window(WINDOM_LENGTH).to(device[0]), verbose=True)
            y = y.type(torch.LongTensor).to(device[0])
            logits = model(x)
            loss_1 = criterion(logits, y)
            print(loss_1)
            exit()
            

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("GPU is running...")

    # print(f"Number of train samples: {len(train_loader)}, Number of test samples: {len(test_loader)}")
    # print(f"Model size: {get_model_size(model.module):.4f}MB")
    run_epoch(model, train_loader, is_train=True)


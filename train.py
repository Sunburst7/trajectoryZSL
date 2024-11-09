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
ROOT_PROJECT_PATH = os.path.join('/data2', 'hh', 'workspace', 'trajectoryZSL')
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
DEVICES = [i for i in range(torch.cuda.device_count())]

# model = SimpleTransformer(input_dim=NUM_SAMPLE_FEATURES, feature_dim=features_dim, num_heads=4, num_layers=encoder_layer_num, num_classes=NUM_CLASS)
model = SimpleCNN(num_class=NUM_CLASS).to(DEVICES[0])
model = torch.nn.DataParallel(model, device_ids=DEVICES)
criterion = nn.CrossEntropyLoss()  # cross entropy loss
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wd) # model optimizer

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
                draw_and_save(time_freq_matrix.cpu().numpy(), os.path.join(ROOT_PROJECT_PATH, "temp", f"sample_{i}_features_{j}.png"))
        x_.append(stft_sample)
    return torch.stack(x_)

train_filepath = os.path.join(ROOT_DATA_PATH, f'train_seqLen_{NUM_SAMPLE_ROW}_ratio_{RATIO}_isGZSL_{IS_GZSL}.pkl')
valid_filepath = os.path.join(ROOT_DATA_PATH, f'valid_seqLen_{NUM_SAMPLE_ROW}_ratio_{RATIO}_isGZSL_{IS_GZSL}.pkl')
test_filepath = os.path.join(ROOT_DATA_PATH, f'test_seqLen_{NUM_SAMPLE_ROW}_ratio_{RATIO}_isGZSL_{IS_GZSL}.pkl')
train_loader = create_dataloader(train_filepath, batch_size)
valid_loader = create_dataloader(valid_filepath, batch_size)
test_loader = create_dataloader(test_filepath, batch_size)

class Trainer:
    def __init__(self, model, train_dataset, valid_dataset, test_dataset, config, savedir=None, devices=torch.device("cpu")):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.savedir = savedir
        self.devices = devices
        self.model = model.to(devices[0])

    
    def run_epoch(self, model, loader, epoch:int, is_train:bool):
        devices = self.devices
        model.train(is_train)
        pbar = tqdm(enumerate(loader), total=len(loader))
        # losses = []
        num_total, num_correct = 0, 0
        for i, (x, y) in pbar:
            x = x.to(devices[0])
            x = bacth_STFT(x, N_FFT, HOP_LENGTH, WINDOM_LENGTH, torch.hamming_window(WINDOM_LENGTH).to(devices[0]), verbose=False)
            y = y.type(torch.LongTensor).to(devices[0])
            logits = model(x)
            loss_classify = criterion(logits, y)
            num_total += x.shape[0]
            num_correct += torch.sum(torch.argmax(logits, dim=-1) == y)
            if is_train:
                model.zero_grad()
                loss_classify.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                pbar.set_description(f"Training... epoch {epoch + 1} iter {i}: loss {loss_classify.item():.5f}. lr {learning_rate:e} acc {num_correct / num_total: .4f}")
            else:
                pbar.set_description(f"Testing.... iter {i}: loss {loss_classify.item():.5f}. lr {learning_rate:e} acc {num_correct / num_total: .4f}")


    def train(self):
        epoch_pbar = tqdm(range(num_epoch), desc=f"开始训练")
        for epoch in epoch_pbar:
            epoch_pbar.set_description(f"训练的第{epoch + 1}个epoch")
            self.run_epoch(self.model, train_loader, epoch=epoch, is_train=True)
            self.run_epoch(self.model, valid_loader, 0 , is_train=False)

    def test(self):
        self.run_epoch(self.model, test_loader, 0 , is_train=False)
        

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("GPU is running...")

    # print(f"Number of train samples: {len(train_loader)}, Number of test samples: {len(test_loader)}")
    # print(f"Model size: {get_model_size(model.module):.4f}MB")
    t = Trainer(model, train_loader, valid_loader, test_loader, None, savedir=None, devices=DEVICES)
    t.train()
    t.test()


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import random 
import os
from tqdm import tqdm

from ais_dataset import AisDataset
from model.simple_transformer import SimpleTransformer

random.seed(42)
ROOT_DATA_PATH = os.path.join('/data2', 'hh', 'workspace', 'data', 'ais')
NUM_CLASS = 14
LNG_AND_LAT_THRESHOLD = 1
NUM_SAMPLE_ROW = 1024
# stft参数
N_FFT = 64
WINDOM_LENGTH = 64
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

model = torch.nn.DataParallel(
    SimpleTransformer(input_dim=NUM_SAMPLE_FEATURES, feature_dim=features_dim, num_heads=4, num_layers=encoder_layer_num, num_classes=NUM_CLASS), device_ids=device)
criterion = nn.CrossEntropyLoss()  # cross entropy loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd) # model optimizer

def create_dataloader(path, batch_size):
    dataset = AisDataset.load(path)
    return Data.DataLoader(dataset, batch_size=batch_size)

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
            x = torch.stft(x, N_FFT, HOP_LENGTH, WINDOM_LENGTH, torch.hamming_window, return_complex=False)
            y = y.to(device[0])
            print(x.shape)
            print(x)
            exit()
            

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("GPU is running...")

    # print(f"Number of train samples: {len(train_loader)}, Number of test samples: {len(test_loader)}")
    # print(f"Model size: {get_model_size(model.module):.4f}MB")
    run_epoch(model, train_loader, is_train=True)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data as Data
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Literal
import optuna
from optuna import Trial

from ais_dataset import AisDataReader
from loss.info_nce import InfoNCE
from model.TFModel import TFModel
from util.lookhead import Lookahead
from util.utils import initialize_weights
from config.deafault import get_cfg_defaults 

cfg = get_cfg_defaults()
# cfg.merge_from_file("./config/TFModel.yaml")
cfg.freeze()
print(cfg)

model = TFModel(cfg.model)
model = torch.nn.DataParallel(model, device_ids=cfg.model.devices)
criterion_cls = nn.CrossEntropyLoss()  # cross entropy loss
optimizer = Lookahead(optim.RAdam(model.parameters(), lr=cfg.model.learning_rate))
scheduler = StepLR(optimizer, step_size=5, gamma=0.9)

def create_dataloader(path, batch_size):
    X, Y = AisDataReader.load_binary(path)
    dataset = Data.TensorDataset(torch.Tensor(X).float(), torch.Tensor(Y).long())
    return Data.DataLoader(dataset, batch_size)

train_filepath = os.path.join(cfg.dataset.root_data_path, f'train_seqLen_{cfg.dataset.seq_len}_rate_{cfg.dataset.ratio}_isGZSL_{cfg.dataset.is_gzsl}.pkl')
valid_filepath = os.path.join(cfg.dataset.root_data_path, f'valid_seqLen_{cfg.dataset.seq_len}_rate_{cfg.dataset.ratio}_isGZSL_{cfg.dataset.is_gzsl}.pkl')
test_filepath = os.path.join(cfg.dataset.root_data_path, f'test_seqLen_{cfg.dataset.seq_len}_rate_{cfg.dataset.ratio}_isGZSL_{cfg.dataset.is_gzsl}.pkl')
train_loader = create_dataloader(train_filepath, cfg.model.batch_size)
valid_loader = create_dataloader(valid_filepath, cfg.model.batch_size)
test_loader = create_dataloader(test_filepath, cfg.model.batch_size)
print(f"train length: {len(train_loader) * cfg.model.batch_size}")
print(f"valid length: {len(valid_loader) * cfg.model.batch_size}")
print(f"test length: {len(test_loader) * cfg.model.batch_size}")

def model_pretrain(model, model_optimizer, criterion: InfoNCE, train_loader, config, devices):
    model = model.to(devices[0])
    total_loss = []
    model.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    num_total, num_correct = 0, 0
    for i, (x, y) in pbar:
        x = x.to(devices[0])
        y = y.to(devices[0])
        time_features, freq_features = model(x)
        if freq_features[0].dtype not in [torch.complex32, torch.complex64]:
            freq_features = torch.view_as_complex(freq_features)
        loss = 0
        for time_feature, freq_feature in zip(time_features, freq_features):
            # freq_feature B C N. 每个样本正例应该是对应的
            pos = torch.fft.ifft(freq_feature).to(time_feature.dtype)
            neg = torch.stack([pos[torch.arange(pos.shape[0], device=pos.device) != i] for i in range(pos.shape[0])])
            loss += criterion(time_feature, pos, neg)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
        optimizer.step()
        total_loss.append(loss.item())
        # print(f"epoch {epoch} {stage + 'ing'}... iter {i}: loss {loss.item():.5f}. lr {scheduler.get_last_lr()[0]:e} and {scheduler_cent.get_last_lr()[0]:e} acc {num_correct / num_total: .4f}")
        pbar.set_description(f"training... iter {i}: loss {loss.item():.5f}. lr {scheduler.get_last_lr()[0]:e}")


        

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("GPU is running...")

    
    model_pretrain(model, optimizer, InfoNCE(negative_mode='paired'), train_loader, cfg.model, cfg.model.devices)


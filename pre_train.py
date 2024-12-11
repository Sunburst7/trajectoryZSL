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

class PreTrainer:
    def __init__(self, model, train_dataset, config, savedir=None, devices=torch.device("cpu")):
        self.train_dataset = train_dataset
        self.config = config
        self.savedir = savedir
        self.devices = devices
        self.model = model.to(devices[0])
        # self.tsne = Tsne(cfg.model.d_center, num_class=cfg.dataset.num_class,seen_class=cfg.dataset.seen_class, unseen_class=cfg.dataset.unseen_class)
        initialize_weights(model)
    
    def run_epoch(self, model, loader, epoch:int):
        devices = self.devices
        model.train(True)
        pbar = tqdm(enumerate(loader), total=len(loader))
        # losses = []
        num_total, num_correct = 0, 0
        for i, (x, y) in pbar:
            x = x.to(devices[0])
            y = y.to(devices[0])
            features, logits = model(x)
            # self.tsne.append(features.detach().cpu().numpy(), y.detach().cpu().numpy()) # 保存训练/测试样本准备t-sne
            loss_cls = criterion_cls(logits, y)
            loss = loss_cls
            optimizer.zero_grad()
            num_total += x.shape[0]
            num_correct += torch.sum(torch.argmax(logits, dim=-1) == y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            optimizer.step()
            # print(f"epoch {epoch} {stage + 'ing'}... iter {i}: loss {loss.item():.5f}. lr {scheduler.get_last_lr()[0]:e} and {scheduler_cent.get_last_lr()[0]:e} acc {num_correct / num_total: .4f}")
            pbar.set_description(f"epoch {epoch} training... iter {i}: loss {loss.item():.5f}. lr {scheduler.get_last_lr()[0]:e} and {cfg.model.learning_rate_cent:e} acc {num_correct / num_total: .4f}")
            scheduler.step()

        return num_correct / num_total

    def train(self):
        epoch_pbar = tqdm(range(cfg.model.num_epoch), desc=f"开始训练")
        for epoch in epoch_pbar:
            epoch_pbar.set_description(f"训练的第{epoch}个epoch")
            self.run_epoch(self.model, train_loader, epoch=epoch)
        

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("GPU is running...")

    t = PreTrainer(model, train_loader, None, savedir=None, devices=cfg.model.devices)
    t.train()


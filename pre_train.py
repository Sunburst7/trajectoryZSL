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
from util.early_stop import EarlyStopping
from config.deafault_1 import get_cfg_defaults 

def model_pretrain(model, epoch, model_optimizer, criterion: InfoNCE, train_loader, config, devices):
    tqdm.write(f"epoch: {epoch}")
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
        contrastive_time_loss = 0
        contrastive_freq_loss = 0
        for time_feature, freq_feature in zip(time_features, freq_features):
            # freq_feature B C N. 每个样本正例应该是对应的
            pos = torch.fft.ifft(freq_feature).to(time_feature.dtype)
            neg = torch.stack([pos[torch.arange(pos.shape[0], device=pos.device) != i] for i in range(pos.shape[0])])
            contrastive_time_loss += criterion(time_feature, pos, neg)

            pos_freq = torch.fft.fft(time_feature)
            neg_freq = torch.stack([pos_freq[torch.arange(pos_freq.shape[0], device=pos_freq.device) != i] for i in range(pos_freq.shape[0])])
            contrastive_freq_loss += criterion(freq_feature, pos_freq, neg_freq)
        loss = contrastive_time_loss + contrastive_freq_loss
        model_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
        model_optimizer.step()
        total_loss.append(loss.item())
        tqdm.write(f"pre_training... iter {i}: tloss {contrastive_time_loss.item():.5f} floss {contrastive_freq_loss.item():.5f}. lr {config.lr:e}")
        pbar.set_description(f"pre_training... iter {i}: tloss {contrastive_time_loss.item():.5f} floss {contrastive_freq_loss.item():.5f}. lr {config.lr:e}")

def model_finetuning(model, epoch, model_optimizer, criterion, train_loader, valid_loader, config, devices):
    tqdm.write(f"epoch: {epoch}")
    model = model.to(devices[0])
    total_loss = []
    model.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    num_total, num_correct = 0, 0
    for i, (x, y) in pbar:
        x = x.to(devices[0])
        y = y.to(devices[0])
        logits = model.module.classification(x)
        loss = criterion(logits, y)
        model_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
        model_optimizer.step()
        total_loss.append(loss.item())
        num_total += x.shape[0]
        num_correct += torch.sum(torch.argmax(logits, dim=-1) == y)
        tqdm.write(f"fine_tune training... iter {i}: loss {np.average(total_loss):.5f}. lr {config.ft_lr:e}. acc {num_correct / num_total: .4f}")
        pbar.set_description(f"fine_tune training... iter {i}: loss {np.average(total_loss):.5f}. lr {config.ft_lr:e}. acc {num_correct / num_total: .4f}")
    
    
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    num_total, num_correct = 0, 0
    for i, (x, y) in pbar:
        x = x.to(devices[0])
        y = y.to(devices[0])
        logits = model.module.classification(x)
        loss = criterion(logits, y)
        total_loss.append(loss.item())
        num_total += x.shape[0]
        num_correct += torch.sum(torch.argmax(logits, dim=-1) == y)
        tqdm.write(f"fine_tune validing... iter {i}: loss {np.average(total_loss):.5f}. lr {config.ft_lr:e}. acc {num_correct / num_total: .4f}")
        pbar.set_description(f"fine_tune validing... iter {i}: loss {np.average(total_loss):.5f}. lr {config.ft_lr:e}. acc {num_correct / num_total: .4f}")
    return num_correct / num_total

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("GPU is running...")

    cfg = get_cfg_defaults()
    cfg.merge_from_file("./config/TFModel.yaml")

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


    def get_str_setting(cfg):
        setting = 'TF_bs_{}_lr_{}_ftlr_{}_nh_{}_el_{}_dm_{}_dff_{}_wd_{}_drop_{}'.format(
            cfg.model.batch_size,
            cfg.model.lr,
            cfg.model.ft_lr,
            cfg.model.n_heads,
            cfg.model.e_layers,
            cfg.model.d_model,
            cfg.model.d_ff,
            cfg.model.wd,
            cfg.model.dropout,
            )
        return setting
  
    if cfg.search_param:
        def objective(trial: Trial):
            cfg.model.batch_size = trial.suggest_categorical("batch_size", [8, 16, 32]) 
            cfg.model.lr = trial.suggest_float('lr', 1e-4, 1e-2) 
            cfg.model.ft_lr = trial.suggest_float('ft_lr', 1e-4, 1e-2) 
            cfg.model.n_heads = trial.suggest_categorical('n_heads', [2, 4, 8])
            cfg.model.e_layers = trial.suggest_int('e_layers', 2, 6)
            # cfg.model.d_model = trial.suggest_int("d_model", 32, 256, step=16) 
            cfg.model.d_ff = trial.suggest_categorical("d_ff", [128, 256, 512, 1024]) 
            # cfg.model.wd = trial.suggest_float('wd', 0.01, 0.2) 
            # cfg.model.dropout = trial.suggest_float('dropout', 0.01, 0.2) 

            model = TFModel(cfg.model)
            model = torch.nn.DataParallel(model, device_ids=cfg.model.devices)
            # initialize_weights(model)
            criterion_cls = nn.CrossEntropyLoss()  # cross entropy loss
            optimizer = Lookahead(optim.RAdam(model.parameters(), lr=cfg.model.lr))
            optimizer_ft = Lookahead(optim.RAdam(model.parameters(), lr=cfg.model.ft_lr))
            early_stopping = EarlyStopping(5)
            setting = get_str_setting(cfg)
            tqdm.write('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            for epoch in range(cfg.model.pre_train_epoch):
                model_pretrain(model, epoch, optimizer, InfoNCE(negative_mode='paired'), train_loader, cfg.model, cfg.model.devices)
            tqdm.write('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            # freeze the model
            for name, param in model.named_parameters():
                if name not in ["module.time_weight.weight", "module.time_weight.bias", "module.freq_weight.weight", "module.freq_weight.bias", "module.head.weight", "module.head.bias",]:
                    param.requires_grad = False
            # for name, param in model.named_parameters():
            #     print(f"{name}: requires_grad={param.requires_grad}")

            for epoch in range(cfg.model.ft_epoch):
                acc = model_finetuning(model, epoch, optimizer_ft, criterion_cls, train_loader, valid_loader, cfg.model, cfg.model.devices)
                early_stopping(-acc, None, None)
                if early_stopping.early_stop:
                    tqdm.write("Early stopping")
                    break
            torch.cuda.empty_cache()
            return acc

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)
        best_param = study.best_params
        print(f"Best value: {study.best_value} (params: {study.best_params})")
    else:
        # best_params = {'batch_size': 24, 'n_heads': 2, 'e_layers': 3, 'd_model': 112, 'd_ff': 704, 'learning_rate': 0.0009576973048766614, 'learning_rate_cent': 0.7930508351347259, 'eta_cent': 0.003324700654716918, 'eta_margin': 0.38542286870029036}
        # for key, value in best_params.items(): # 使用字典更新Namespace对象
        #     setattr(cfg.model, key, value)
        cfg.freeze()
        print(cfg)

        model = TFModel(cfg.model)
        model = torch.nn.DataParallel(model, device_ids=cfg.model.devices)
        criterion_cls = nn.CrossEntropyLoss()  # cross entropy loss
        optimizer = Lookahead(optim.RAdam(model.parameters(), lr=cfg.model.lr))
        optimizer_ft = Lookahead(optim.RAdam(model.parameters(), lr=cfg.model.ft_lr))
        early_stopping = EarlyStopping(5)
        setting = get_str_setting(cfg)
        tqdm.write('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        for epoch in range(cfg.model.pre_train_epoch):
            model_pretrain(model, epoch, optimizer, InfoNCE(negative_mode='paired'), train_loader, cfg.model, cfg.model.devices)
        tqdm.write('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        # freeze the model
        for name, param in model.named_parameters():
            if name not in ["module.time_weight.weight", "module.time_weight.bias", "module.freq_weight.weight", "module.freq_weight.bias", "module.head.weight", "module.head.bias",]:
                param.requires_grad = False
        # for name, param in model.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")

        for epoch in range(cfg.model.ft_epoch):
            acc = model_finetuning(model, epoch, optimizer_ft, criterion_cls, train_loader, valid_loader, cfg.model, cfg.model.devices)
            early_stopping(-acc, None, None)
            if early_stopping.early_stop:
                tqdm.write("Early stopping")
                break

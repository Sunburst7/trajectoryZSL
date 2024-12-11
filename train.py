import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR
import torch.utils.data as Data
import torch.nn.init as init
import torchvision
import random 
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Literal
import optuna
from optuna import Trial

from ais_dataset import AisDataReader
from model.iTransformer import iTransformer
from model.simple_cnn import SimpleCNN
from loss.center_loss import CenterLoss
from loss.margin_loss import MarginLoss
from util.lookhead import Lookahead
from util.mahalanobis import MahalanobisLayer
from util.tsne import Tsne
from util.early_stop import EarlyStopping
from config.deafault import get_cfg_defaults 


cfg = get_cfg_defaults()
# cfg.merge_from_file("./config/test.yaml")
cfg.freeze()
print(cfg)

model = iTransformer(cfg.model, cfg.dataset.seen_class)
model = torch.nn.DataParallel(model, device_ids=cfg.model.devices)
criterion_cls = nn.CrossEntropyLoss()  # cross entropy loss
criterion_cent = CenterLoss(num_classes=cfg.dataset.num_class, feat_dim=cfg.model.d_center)
criterion_margin = MarginLoss(cfg.model.margin, num_classes=cfg.dataset.num_class, feat_dim=cfg.model.d_center, centers=criterion_cent.get_centers())
optimizer = Lookahead(optim.RAdam(model.parameters(), lr=cfg.model.learning_rate))
scheduler = StepLR(optimizer, step_size=5, gamma=0.9)
optimizer_cent = Lookahead(optim.SGD(criterion_cent.parameters(), lr=cfg.model.learning_rate_cent), k=3)

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

class Trainer:
    def __init__(self, model, train_dataset, valid_dataset, test_dataset, config, savedir=None, devices=torch.device("cpu")):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.savedir = savedir
        self.devices = devices
        self.model = model.to(devices[0])
        self.tsne = Tsne(cfg.model.d_center, num_class=cfg.dataset.num_class,seen_class=cfg.dataset.seen_class, unseen_class=cfg.dataset.unseen_class)
        self.early_stopping = EarlyStopping(patience=cfg.model.patience, verbose=True)
        initialize_weights(model)
    
    def semantic_classify(self, centers: nn.Parameter, features: torch.Tensor):
        """计算批样本对每个已知类的聚类中心的距离

        Args:
            centers (nn.Parameter): _description_
            features (torch.Tensor): _description_

        Returns:
            torch.Tensor: 预测的标签
        """
        results = []
        for bid in range(features.shape[0]):
            predict_label = -1
            min_dist = float('inf')
            if_know = False
            dists = self.model.module.calculate_distance(centers, features[bid], mode="euclidan")
            for certain_label in cfg.dataset.seen_class:
                # if dists[certain_label] < self.model.module.known_thresholds[certain_label]:
                #     if_know = True
                if dists[certain_label] < self.model.module.known_thresholds[certain_label] and dists[certain_label] < min_dist:
                    if_know = True
                    min_dist = dists[certain_label]
                    predict_label = certain_label

            results.append(predict_label)
        return torch.Tensor(results).cuda()

    
    def run_epoch(self, model, loader, epoch:int, stage:Literal['train', 'valid', 'test']):
        eta_cent = cfg.model.eta_cent
        eta_cls = cfg.model.eta_cls
        eta_margin = cfg.model.eta_margin
        devices = self.devices
        is_train = stage == 'train'
        model.train(is_train)
        pbar = tqdm(enumerate(loader), total=len(loader))
        concat_feature_stack = [] # 中间特征与输出特征组合后的一个存储栈
        label_stack = []
        # losses = []
        num_total, num_correct = 0, 0
        for i, (x, y) in pbar:
            x = x.to(devices[0])
            y = y.to(devices[0])
            features, logits = model.module.classification(x, None)
            self.tsne.append(features.detach().cpu().numpy(), y.detach().cpu().numpy()) # 保存训练/测试样本准备t-sne
            loss_cls = criterion_cls(logits, y)
            loss_cent = criterion_cent(features, y)
            loss_margin = criterion_margin(features, y)
            loss = loss_cls  + loss_cent * eta_cent + loss_margin * eta_cent
            optimizer.zero_grad()
            optimizer_cent.zero_grad()
            num_total += x.shape[0]
            if is_train:
                num_correct += torch.sum(torch.argmax(logits, dim=-1) == y)
                self.model.module.batch_dist_saving(criterion_cent.get_centers(), features, y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                optimizer.step()
                for param in criterion_cent.parameters():
                    param.grad.data *= (1. / eta_cent) # multiple (1./alpha) in order to remove the effect of alpha on updating centers
                optimizer_cent.step()
                # print(f"epoch {epoch} {stage + 'ing'}... iter {i}: loss {loss.item():.5f}. lr {scheduler.get_last_lr()[0]:e} and {scheduler_cent.get_last_lr()[0]:e} acc {num_correct / num_total: .4f}")
                pbar.set_description(f"epoch {epoch} {stage + 'ing'}... iter {i}: loss {loss.item():.5f}. lr {scheduler.get_last_lr()[0]:e} and {cfg.model.learning_rate_cent:e} acc {num_correct / num_total: .4f}")
            else:
                predict_labels = self.semantic_classify(criterion_cent.get_centers(), features)
                # TODO: 目前是单未知类的替换
                predict_labels = torch.where(predict_labels == -1, cfg.dataset.unseen_class[0] if len(cfg.dataset.unseen_class) > 0 else -1, predict_labels)
                num_correct += torch.sum(predict_labels == y)
                # num_correct += torch.sum(torch.argmax(logits, dim=-1) == y)
                # print(f"epoch {epoch} {stage + 'ing'}.... iter {i}: loss {loss.item():.5f}. acc {num_correct / num_total: .4f}")
                pbar.set_description(f"epoch {epoch} {stage + 'ing'}.... iter {i}: loss {loss.item():.5f}. acc {num_correct / num_total: .4f}")
                

        self.tsne.append(criterion_cent.get_centers().detach().cpu().numpy(), np.arange(cfg.dataset.num_class))
        self.tsne.cal_and_save(os.path.join(cfg.root_project_path, "temp", f"epoch_{epoch}_{stage}_tsne.png"), stage)
        self.tsne.clear()

        if is_train:
            scheduler.step()
            # scheduler_cent.step()
            # 每个epoch更新已知类的阈值，清空保存的距离（聚类中心更新需要重新计算）
            self.model.module.update_thresholds()
            self.model.module.dist_clearing()
        
        if stage == 'valid':
            self.early_stopping(-num_correct / num_total, None, None)
            if self.early_stopping.early_stop:
                tqdm.write("Early stopping")
                return num_correct / num_total

        return num_correct / num_total

    def train(self):
        epoch_pbar = tqdm(range(cfg.model.num_epoch), desc=f"开始训练")
        for epoch in epoch_pbar:
            epoch_pbar.set_description(f"训练的第{epoch}个epoch")
            self.run_epoch(self.model, train_loader, epoch=epoch, stage='train')
            self.run_epoch(self.model, valid_loader, epoch=epoch, stage='valid')
            torch.save({"model_state_dict" : self.model.state_dict(), "known_thresholds": self.model.module.known_thresholds}, 
                os.path.join(cfg.root_project_path, "model", "model.pth"))
            torch.save(criterion_cent.state_dict(), os.path.join(cfg.root_project_path, "model", "center.pth"))
            checkpoint = torch.load(os.path.join(cfg.root_project_path, "model", "model.pth"), weights_only=True)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.module.known_thresholds = checkpoint["known_thresholds"]
            criterion_cent.load_state_dict(torch.load(os.path.join(cfg.root_project_path, "model", "center.pth"), weights_only=True))
            if epoch % 10 == 0:
                self.run_epoch(self.model, test_loader, epoch, stage='test')

    def test(self):
        checkpoint = torch.load(os.path.join(cfg.root_project_path, "model", "model.pth"), weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.module.known_thresholds = checkpoint["known_thresholds"]
        criterion_cent.load_state_dict(torch.load(os.path.join(cfg.root_project_path, "model", "center.pth"), weights_only=True))
        valid_acc = self.run_epoch(self.model, valid_loader, epoch=0, stage='valid')
        test_acc = self.run_epoch(self.model, test_loader, 0 , stage='test')
        

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("GPU is running...")

    # print(f"Number of train samples: {len(train_loader)}, Number of test samples: {len(test_loader)}")
    # print(f"Model size: {get_model_size(model.module):.4f}MB")

    # def objective(trial: Trial):
    #     cfg.model.margin = trial.suggest_int('cfg.model.margin', 1, 32)
    #     eta_cent = trial.suggest_float('eta_cent', 5e-4, 5e-1, step=1e-4)
    #     eta_margin = trial.suggest_float('eta_cent', 1e-4, 5e-1, step=1e-4)

    #     # best_params = {'e_layers': 8, 'batch_size': 16, 'd_model': 128, 'd_ff': 120, 'top_k': 6, 'd_quantile': 100, 'learning_rate': 0.0009037400676189364, 'enc_in': 16}
    #     # for key, value in best_params.items(): # 使用字典更新Namespace对象
    #     #     setattr(args, key, value)

    #     avg_acc = 0
    #     for i in range(args.itr):
    #         print(str(i) + ' fold: ')
    #         exp = Exp(args, i)  # set experiments
    #         setting = get_str_setting(args, i)
    #         print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    #         exp.train(setting)
    #         print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    #         acc, _ = exp.test(setting)
    #         torch.cuda.empty_cache()
    #         avg_acc += acc
    #     return avg_acc / args.itr

    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=5)
    # best_param = study.best_params
    # print(f"Best value: {study.best_value} (params: {study.best_params})")

    t = Trainer(model, train_loader, valid_loader, test_loader, None, savedir=None, devices=cfg.model.devices)
    t.train()
    t.test()


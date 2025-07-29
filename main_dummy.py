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
from loss.center_lossv2 import CenterLoss
from util.lookhead import Lookahead
from util.tsne import Tsne
from util.early_stop import EarlyStopping
from util.utils import draw_acc
from config.deafault import get_cfg_defaults 


cfg = get_cfg_defaults()
cfg.merge_from_file("./config/aircraft.yaml")

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

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        if epoch < 10:
            lr_adjust = {epoch: args.learning_rate / 2 * (1 + math.cos(epoch / args.num_epoch * math.pi))}
        else:
            lr_adjust = {
                10: 9e-4, 
                12: 7e-4, 
                14: 5e-4, 
                16: 3e-4, 
                18: 1e-4,
                20: 7e-5,
                25: 5e-5,
                30: 3e-5
            }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.num_epoch * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        tqdm.write('Updating learning rate to {}'.format(lr))

def create_dataloader(path, batch_size): 
    X, Y = AisDataReader.load_binary(path)
    dataset = Data.TensorDataset(torch.Tensor(X).float(), torch.Tensor(Y).long())
    print(f"loading data(length: {len(dataset)}) from {path}")
    return Data.DataLoader(dataset, batch_size)

def get_str_setting(cfg):
    setting = 'bs_{}_nh_{}_el_{}_dm_{}_dff_{}_beta_{}_lr_{}'.format(
        cfg.model.batch_size,
        cfg.model.n_heads,
        cfg.model.e_layers,
        cfg.model.d_model,
        cfg.model.d_ff,
        cfg.model.beta,
        cfg.model.learning_rate,
        )
    return setting

train_filepath = os.path.join(cfg.dataset.root_data_path, f'train_seqLen_{cfg.dataset.seq_len}_rate_{cfg.dataset.ratio}_isGZSL_{cfg.dataset.is_gzsl}.pkl')
valid_filepath = os.path.join(cfg.dataset.root_data_path, f'valid_seqLen_{cfg.dataset.seq_len}_rate_{cfg.dataset.ratio}_isGZSL_{cfg.dataset.is_gzsl}.pkl')
test_filepath = os.path.join(cfg.dataset.root_data_path, f'test_seqLen_{cfg.dataset.seq_len}_rate_{cfg.dataset.ratio}_isGZSL_{cfg.dataset.is_gzsl}.pkl')
train_loader = create_dataloader(train_filepath, cfg.model.batch_size)
valid_loader = create_dataloader(valid_filepath, cfg.model.batch_size)
test_loader = create_dataloader(test_filepath, cfg.model.batch_size)

class Trainer:
    def __init__(self, model, train_dataset, valid_dataset, test_dataset, config, savedir=None, devices=torch.device("cpu")):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.savedir = savedir
        self.devices = devices
        self.model = model.to(devices[0])
        self.tsne = Tsne(cfg.model.d_center, num_class=cfg.model.num_class,seen_class=cfg.dataset.seen_class, unseen_class=cfg.dataset.unseen_class)
        
        self.criterion_cls = nn.CrossEntropyLoss()  # cross entropy loss
        self.proj_layer = nn.Linear(cfg.model.d_center, cfg.model.d_model, device=devices[0])
        self.criterion_cent = CenterLoss(num_classes=cfg.dataset.num_class, 
                                         feat_dim=cfg.model.d_center,
                                         hidden_dim=cfg.model.d_model,
                                         proj_layer=self.proj_layer, 
                                         margin=cfg.model.margin, devices=devices[0])
        self.optimizer = Lookahead(optim.RAdam(model.parameters(), lr=cfg.model.learning_rate))
        self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.9)
        self.optimizer_cent = optim.SGD(self.proj_layer.parameters(), lr=cfg.model.learning_rate_cent)
        self.early_stopping = EarlyStopping(patience=cfg.model.patience, verbose=True)
        # initialize_weights(model)

    def run_epoch(self, model, loader, epoch:int, stage:Literal['train', 'valid', 'test']):
        criterion_cls = self.criterion_cls
        optimizer = self.optimizer
        scheduler = self.scheduler

        devices = self.devices
        is_train = stage == 'train'
        model.train(is_train)
        pbar = tqdm(enumerate(loader), total=len(loader))
        losses = []
        num_total, num_correct = 0, 0
        
        for i, (x, y) in pbar:
            x = x.to(devices[0])
            y = y.to(devices[0])
            features, logits = model.module.classification(x, None)
            # self.tsne.append(features.cpu().detach().numpy(), y.cpu().detach().numpy())
            loss = criterion_cls(logits, y)
            if stage == 'train':
                dummy_logits = logits.clone()
                for i in range(dummy_logits.shape[0]):
                    dummy_logits[i][y[i]] = -float('inf')
                dummy_y = torch.ones_like(y) * self.config.dataset.unseen_class[0]
                loss_dummy = criterion_cls(dummy_logits, dummy_y)
                # loss_margin = criterion_margin(features, y)
                #   + loss_cent * eta_cent + loss_margin * eta_margin
                loss += loss_dummy * cfg.model.beta
            losses.append(loss.item())
            # center_losses.append(loss_cent.item())
            optimizer.zero_grad()
            num_total += x.shape[0]
            if is_train:
                num_correct += torch.sum(torch.argmax(logits, dim=-1) == y)
                # self.model.module.batch_dist_saving(criterion_cent.get_centers(), features, y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                optimizer.step()
                pbar.set_description(f"epoch {epoch} {stage + 'ing'}... iter {i}: loss {np.average(losses):.5f}.  lr {scheduler.get_last_lr()[0]:e} acc {num_correct / num_total: .4f}")
            else:
                num_correct += torch.sum(torch.argmax(logits, dim=-1) == y)
                pbar.set_description(f"epoch {epoch} {stage + 'ing'}... iter {i}: loss {np.average(losses):.5f}.  lr {scheduler.get_last_lr()[0]:e} acc {num_correct / num_total: .4f}")
                

        if is_train:
            # adjust_learning_rate(self.optimizer, epoch + 1, self.config.model)
            scheduler.step()
            # scheduler_cent.step()
            # if epoch % 5 == 0:
            #     self.tsne.cal_and_save(os.path.join(cfg.root_project_path, "temp", f"tsne_{epoch}.png"), stage)
            # self.tsne.clear()
        
        tqdm.write(f"epoch {epoch} {stage + 'ing'}....: loss {np.average(losses):.5f}. lr {scheduler.get_last_lr()[0]:e}, acc {num_correct / num_total: .4f}")
        return num_correct / num_total

    def train(self):
        if not os.path.exists(os.path.join(self.savedir, cfg.dataset.name)):
            os.mkdir(os.path.join(self.savedir, cfg.dataset.name))

        epoch_pbar = tqdm(range(cfg.model.num_epoch), desc=f"开始训练")
        best_valid_acc = 0
        train_acc_list = []
        valid_acc_list = []
        test_acc_list = []
        test_epochs = []
        for epoch in epoch_pbar:
            epoch_pbar.set_description(f"训练的第{epoch}个epoch")
            train_acc_list.append(self.run_epoch(self.model, train_loader, epoch=epoch, stage='train').cpu().numpy())
            valid_acc = self.run_epoch(self.model, valid_loader, epoch=epoch, stage='valid')
            valid_acc_list.append(valid_acc.cpu().numpy())
            if valid_acc >= best_valid_acc:
                best_valid_acc = valid_acc
                # save model
                folder_name = get_str_setting(cfg)
                save_childdir = os.path.join(self.savedir, cfg.dataset.name, folder_name)
                if not os.path.exists(save_childdir):
                    os.mkdir(save_childdir)
                torch.save({"model_state_dict" : self.model.state_dict(), "known_thresholds": self.model.module.known_thresholds}, 
                    os.path.join(save_childdir, "model.pth"))

            if epoch % 5 == 0:
                test_epochs.append(epoch)
                test_acc_list.append(self.run_epoch(self.model, test_loader, epoch, stage='test').cpu().numpy())
            self.early_stopping(-valid_acc, None, None)
            if self.early_stopping.early_stop:
                tqdm.write("Early stopping")
                test_epochs.append(epoch + 1)
                test_acc_list.append(self.run_epoch(self.model, test_loader, epoch, stage='test').cpu().numpy())
                draw_acc(train_acc_list, valid_acc_list, test_acc_list, test_epochs, os.path.join(cfg.root_project_path, "temp", f"{get_str_setting(cfg)}.png"))
                return valid_acc
        test_epochs.append(epoch + 1)
        test_acc_list.append(self.run_epoch(self.model, test_loader, epoch, stage='test').cpu().numpy())
        draw_acc(train_acc_list, valid_acc_list, test_acc_list, test_epochs, os.path.join(cfg.root_project_path, "temp", f"{get_str_setting(cfg)}.png"))
        return valid_acc

    def test(self):
        # load model
        folder_name = get_str_setting(cfg)
        save_childdir = os.path.join(self.savedir, cfg.dataset.name, folder_name)
        checkpoint = torch.load(os.path.join(save_childdir, "model.pth"), weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.module.known_thresholds = checkpoint["known_thresholds"]
        valid_acc = self.run_epoch(self.model, valid_loader, 0 , stage='valid')
        test_acc = self.run_epoch(self.model, test_loader, 0 , stage='test')
        return valid_acc, test_acc
        

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("GPU is running...")

    # print(f"Number of train samples: {len(train_loader)}, Number of test samples: {len(test_loader)}")
    # print(f"Model size: {get_model_size(model.module):.4f}MB")

    if cfg.search_param:
        def objective(trial: Trial):
            cfg.model.batch_size = trial.suggest_categorical("batch_size", [8, 16, 32]) 
            cfg.model.n_heads = trial.suggest_categorical('n_heads', [2, 4, 8])
            cfg.model.e_layers = trial.suggest_int('e_layers', 2, 8)
            cfg.model.d_model = trial.suggest_categorical('d_model', [128, 256, 512])
            cfg.model.d_ff = 4 * cfg.model.d_model
            cfg.model.learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3) 
            cfg.model.beta = trial.suggest_float('beta', 1e-1, 0.8)
            cfg.model.d_center = cfg.model.d_model * cfg.model.enc_in

            model = iTransformer(cfg.model, cfg.dataset.seen_class)
            model = torch.nn.DataParallel(model, device_ids=cfg.model.devices)
            t = Trainer(model, train_loader, valid_loader, test_loader, cfg, savedir=os.path.join(cfg.root_project_path, 'checkpoints'), devices=cfg.model.devices)
            setting = get_str_setting(cfg)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            t.train()
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            valid_acc, test_acc = t.test()
            torch.cuda.empty_cache()
            return valid_acc, test_acc
        
        study = optuna.create_study(directions=["maximize", "maximize"])
        study.optimize(objective, n_trials=100)
        print(f"Number of trials on the Pareto front: {len(study.best_trials)}")
        optuna.visualization.plot_pareto_front(study, target_names=["valid_acc", "test_acc"])
        trial_with_highest_valid_accuracy = max(study.best_trials, key=lambda t: t.values[0])
        print("Trial with highest accuracy: ")
        print(f"\tnumber: {trial_with_highest_valid_accuracy.number}")
        print(f"\tparams: {trial_with_highest_valid_accuracy.params}")
        print(f"\tvalues: {trial_with_highest_valid_accuracy.values}")
    else:
        #best_params = {'batch_size': 8, 'n_heads': 8, 'e_layers': 5, 'd_model': 512, 'learning_rate': 0.0008287875374506874, 'beta': 0.9484880167535646}
        # best_params = {'batch_size': 16, 'n_heads': 8, 'e_layers': 7, 'd_model': 256, 'learning_rate': 0.0009922578876425468, 'beta': 0.76}
        best_params = {'batch_size': 32, 'n_heads': 4, 'e_layers': 7, 'd_model': 256, 'learning_rate': 1e-4, 'beta': 0.5}
        for key, value in best_params.items(): # 使用字典更新Namespace对象
            setattr(cfg.model, key, value)
        cfg.model.d_ff = 4 * cfg.model.d_model
        cfg.model.d_center = cfg.model.d_model * cfg.model.enc_in
        cfg.freeze()
        print(cfg)

        model = iTransformer(cfg.model, cfg.dataset.seen_class)
        model = torch.nn.DataParallel(model, device_ids=cfg.model.devices)
        t = Trainer(model, train_loader, valid_loader, test_loader, cfg, savedir=os.path.join(cfg.root_project_path, 'checkpoints'), devices=cfg.model.devices)
        setting = get_str_setting(cfg)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        acc = t.train()
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        t.test() 

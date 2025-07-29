import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR
import torch.utils.data as Data
import torch.nn.init as init
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
from model.UnknownDetector import UnknownDetector
from loss.criterion import Criterion
from util.metric import Metric
from util.lookhead import Lookahead
from util.tsne import Tsne
from util.early_stop import EarlyStopping
from util.utils import draw_acc
from config.deafault import get_cfg_defaults 

from momentfm import MOMENTPipeline
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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
    setting = 'bs_{}_beta_{}_lr_{}'.format(
        cfg.model.batch_size,
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
        losses_train = ["loss_label", "loss_dummy"]
        losses_test = ["loss_label"]
        weight_dict = {"loss_label": 1, "loss_dummy": cfg.model.beta}
        self.criterion_train = Criterion(config, weight_dict, losses_train)
        self.criterion_test = Criterion(config, weight_dict, losses_test)
        self.optimizer = Lookahead(optim.RAdam(model.parameters(), lr=cfg.model.learning_rate))
        self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.9)
        self.metric = Metric(cfg.dataset.unseen_class)
        self.early_stopping = EarlyStopping(patience=cfg.model.patience, verbose=True)
        self.object_logit = nn.BatchNorm1d(1).to(devices[0])
        if config.model.osr == 'bc':
            self.ud = UnknownDetector(cfg, cfg.model.STD_COEF_1, cfg.model.STD_COEF_2)
        # initialize_weights(model)

    
    def run_train_epoch(self, model, loader, epoch:int):
        optimizer = self.optimizer
        scheduler = self.scheduler
        devices = self.devices
        model.train(True)
        pbar = tqdm(enumerate(loader), total=len(loader))
        loss_list = []
        self.metric.reset()
        for i, (x, y) in pbar:
            x = x.to(devices[0])
            x = torch.permute(x, (0, 2, 1))
            y = y.to(devices[0])
            outputs = model(x_enc=x)
            output_dict = {'pred_logits': outputs.logits}
            # self.tsne.append(features.cpu().detach().numpy(), y.cpu().detach().numpy())
           
            # embeddings = outputs.embeddings.mean(dim=1) # [batch_size, n+patch, num_channel * d_model]
            # obj_feature = embeddings.norm(dim=-1, keepdim=True)
            # objectness = torch.sigmoid(self.object_logit(obj_feature))

            output_dict = self.criterion_train(output_dict, y) 
            loss_dict, logits = output_dict['losses'], output_dict['pred_logits']
            weight_dict = self.criterion_train.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            optimizer.zero_grad()
            losses.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            optimizer.step()
            
            loss_list.append(losses.item())
            y_pred = torch.argmax(logits, dim=-1)
            if cfg.model.osr == 'bc':
                y_pred = self.ud(x, y_pred)
            self.metric(y_pred, y)
            pbar.set_description(f"epoch {epoch} training... iter {i}: loss {np.average(loss_list):.5f}. lr {scheduler.get_last_lr()[0]:e}")
        
        tqdm.write(f"epoch {epoch} finish training(loss {np.average(loss_list):.5f}. lr {scheduler.get_last_lr()[0]:e}): \n{self.metric}")
        return self.metric.get_metrics()
    
    def run_test_epoch(self, model, loader, epoch:int, stage:Literal['valid', 'test']):
        optimizer = self.optimizer
        scheduler = self.scheduler
        devices = self.devices
        model.train(False)
        pbar = tqdm(enumerate(loader), total=len(loader))
        loss_list = []
        self.metric.reset()
        for i, (x, y) in pbar:
            x = x.to(devices[0])
            x = torch.permute(x, (0, 2, 1))
            y = y.to(devices[0])
            outputs = model(x_enc=x)
            output_dict = {'pred_logits': outputs.logits}
            # self.tsne.append(features.cpu().detach().numpy(), y.cpu().detach().numpy())
           
            # embeddings = outputs.embeddings.mean(dim=1) # [batch_size, n+patch, num_channel * d_model]
            # obj_feature = embeddings.norm(dim=-1, keepdim=True)
            # objectness = torch.sigmoid(self.object_logit(obj_feature))

            output_dict = self.criterion_test.pred(output_dict, y) 
            loss_dict, logits = output_dict['losses'], output_dict['pred_logits']
            weight_dict = self.criterion_test.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
            loss_list.append(losses.item())
            y_pred = torch.argmax(logits, dim=-1)
            if cfg.model.osr == 'bc':
                y_pred = self.ud(x, y_pred)
            self.metric(y_pred, y)
            pbar.set_description(f"epoch {epoch} {stage}ing... iter {i}: loss {np.average(loss_list):.5f}. lr {scheduler.get_last_lr()[0]:e}")
        
        tqdm.write(f"epoch {epoch} finish {stage}ing(loss {np.average(loss_list):.5f}. lr {scheduler.get_last_lr()[0]:e}): \n{self.metric}")
        return self.metric.get_metrics()

    def train(self):
        if not os.path.exists(os.path.join(self.savedir, cfg.dataset.name)):
            os.mkdir(os.path.join(self.savedir, cfg.dataset.name))

        epoch_pbar = tqdm(range(cfg.model.num_epoch), desc=f"开始训练")
        best_valid_acc = 0
        train_acc_list = []
        valid_acc_list = []
        test_ur_list = []
        test_epochs = []
        for epoch in epoch_pbar:
            epoch_pbar.set_description(f"训练的第{epoch}个epoch")
            train_result = self.run_train_epoch(self.model, train_loader, epoch=epoch)
            train_acc_list.append(train_result['k_acc'])
            valid_result = self.run_test_epoch(self.model, valid_loader, epoch=epoch, stage='valid')
            valid_acc_list.append(valid_result['k_acc'])
            if valid_result['k_acc'] >= best_valid_acc:
                best_valid_acc = valid_result['k_acc']
                # save model
                folder_name = get_str_setting(cfg)
                save_childdir = os.path.join(self.savedir, cfg.dataset.name, folder_name)
                if not os.path.exists(save_childdir):
                    os.mkdir(save_childdir)
                torch.save({"model_state_dict" : self.model.state_dict()}, 
                    os.path.join(save_childdir, "model.pth"))

            if epoch % 2 == 0:
                test_epochs.append(epoch)
                test_result = self.run_test_epoch(self.model, test_loader, epoch, stage='test')
                test_ur_list.append(test_result['u_recall'])
            
            self.early_stopping(-valid_result['k_acc'], None, None)
            if self.early_stopping.early_stop:
                tqdm.write("Early stopping")
                test_epochs.append(epoch + 1)
                test_result = self.run_test_epoch(self.model, test_loader, epoch, stage='test')
                test_ur_list.append(test_result['u_recall'])
                draw_acc(train_acc_list, valid_acc_list, test_ur_list, test_epochs, os.path.join(cfg.root_project_path, "temp", f"{get_str_setting(cfg)}.png"))
                return valid_result['k_acc']
            
        test_epochs.append(epoch + 1)
        test_result = self.run_test_epoch(self.model, test_loader, epoch, stage='test')
        test_ur_list.append(test_result['u_recall'])
        draw_acc(train_acc_list, valid_acc_list, test_ur_list, test_epochs, os.path.join(cfg.root_project_path, "temp", f"{get_str_setting(cfg)}.png"))
        return valid_result['k_acc']

    def test(self):
        # load model
        folder_name = get_str_setting(cfg)
        save_childdir = os.path.join(self.savedir, cfg.dataset.name, folder_name)
        checkpoint = torch.load(os.path.join(save_childdir, "model.pth"), weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        valid_result = self.run_test_epoch(self.model, valid_loader, 0 , stage='valid')
        test_result = self.run_test_epoch(self.model, test_loader, 0 , stage='test')
        return valid_result["k_acc"], test_result['u_recall']
        

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("GPU is running...")

    if cfg.search_param:
        def objective(trial: Trial):
            # cfg.model.batch_size = trial.suggest_categorical("batch_size", [8, 16, 24, 32]) 
            # cfg.model.learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3) 
            # cfg.model.beta = trial.suggest_float('beta', 0.2, 0.5)
            cfg.model.STD_COEF_1 = trial.suggest_categorical("STD_COEF_1", [0, 0.5, 0.75, 1, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 4]) 
            cfg.model.STD_COEF_2 = trial.suggest_categorical("STD_COEF_2", [0, 0.5, 0.75, 1, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 4]) 
            model = MOMENTPipeline.from_pretrained(
                "AutonLab/MOMENT-1-large", 
                model_kwargs={
                    "task_name": "classification",
                    "n_channels": cfg.dataset.num_feature,
                    "num_class": cfg.dataset.num_class,
                },
            )
            model.init()
            model = torch.nn.DataParallel(model, device_ids=cfg.model.devices)
            t = Trainer(model, train_loader, valid_loader, test_loader, cfg, savedir=os.path.join(cfg.root_project_path, 'checkpoints'), devices=cfg.model.devices)
            setting = get_str_setting(cfg)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            t.train()
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            k_acc, ur = t.test()
            torch.cuda.empty_cache()
            return k_acc, ur
        
        study = optuna.create_study(directions=["maximize", "maximize"])
        study.optimize(objective, n_trials=50)
        print(f"Number of trials on the Pareto front: {len(study.best_trials)}")
        optuna.visualization.plot_pareto_front(study, target_names=["UR"])
        trial_with_highest_valid_accuracy = max(study.best_trials, key=lambda t: t.values[0])
        print("Trial with highest accuracy: ")
        print(f"\tnumber: {trial_with_highest_valid_accuracy.number}")
        print(f"\tparams: {trial_with_highest_valid_accuracy.params}")
        print(f"\tvalues: {trial_with_highest_valid_accuracy.values}")
    else:
        #best_params = {'batch_size': 8, 'n_heads': 8, 'e_layers': 5, 'd_model': 512, 'learning_rate': 0.0008287875374506874, 'beta': 0.9484880167535646}
        # best_params = {'batch_size': 16, 'n_heads': 8, 'e_layers': 7, 'd_model': 256, 'learning_rate': 0.0009922578876425468, 'beta': 0.76}
        # best_params = {'batch_size': 32, 'n_heads': 4, 'e_layers': 7, 'd_model': 256, 'learning_rate': 0.00024233339175292422, 'beta': 0.42724296145552965}
        best_params = {'batch_size': 32, 'learning_rate': 0.00024233339175292422, 'beta': 0.2}
        for key, value in best_params.items(): # 使用字典更新Namespace对象
            setattr(cfg.model, key, value)
        # cfg.model.d_ff = 4 * cfg.model.d_model
        # cfg.model.d_center = cfg.model.d_model * cfg.model.enc_in

        # res = []
        # for c1 in [0, 0.5, 0.75, 1, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 4]:
        #     for c2 in [0, 0.5, 0.75, 1, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 4]:
        #         print("start traing... c1: {}, c2: {}".format(c1, c2))
        # setattr(cfg.model, 'STD_COEF_1', c1)
        # setattr(cfg.model, 'STD_COEF_2', c2)
        # # cfg.freeze()
        # print(cfg.model.STD_COEF_1, cfg.model.STD_COEF_2)

        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large", 
            model_kwargs={
                "task_name": "classification",
                "n_channels": cfg.dataset.num_feature,
                "num_class": cfg.dataset.num_class,
            },
        )
        model.init()
        model = torch.nn.DataParallel(model, device_ids=cfg.model.devices)
        t = Trainer(model, train_loader, valid_loader, test_loader, cfg, savedir=os.path.join(cfg.root_project_path, 'checkpoints'), devices=cfg.model.devices)
        setting = get_str_setting(cfg)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        t.train()
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        k_acc, ur = t.test() 
        print(k_acc, ur)
        #         res.append([c1, c2, k_acc, ur])
        # print(res)
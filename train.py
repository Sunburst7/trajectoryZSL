import torch
import torch.nn as nn
import torch.optim as optim
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

from ais_dataset import AisDataReader
from model.simple_transformer import SimpleTransformer
from model.simple_cnn import SimpleCNN
from loss.center_loss import CenterLoss
from loss.margin_loss import MarginLoss
from mahalanobis import MahalanobisLayer
from tsne import Tsne

# dataset 参数
ROOT_DATA_PATH = os.path.join('/data2', 'hh', 'workspace', 'data', 'ais')
ROOT_PROJECT_PATH = os.path.join('/data2', 'hh', 'workspace', 'trajectoryZSL')
NUM_CLASS = 14
LNG_AND_LAT_THRESHOLD = 1
NUM_SEQ_LEN = 1024
RATIO = 0.7
IS_GZSL = False
SEEN_CLASS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13]
UNSEEN_CLASS = [10]
# stft参数
N_FFT = WINDOM_LENGTH = 128
HOP_LENGTH = 16
WINDOW_FUNCTION = "Hamming"
NUM_SAMPLE_FEATURES = 4

# 模型超参数
RANDOM_SEED = 2024
MARGIN = 24
eta_cent = 5e-2
eta_cls = 1
eta_margin = 0.1

num_epoch = 50
batch_size = 8
learning_rate = 5e-4
wd = 0.05
encoder_layer_num = 6
features_dim = 128
DEVICES = [i for i in range(torch.cuda.device_count())]
DEVICES = [0, 1, 2, 3]

# model = SimpleTransformer(input_dim=NUM_SAMPLE_FEATURES, feature_dim=features_dim, num_heads=4, num_layers=encoder_layer_num, num_classes=NUM_CLASS)
model = SimpleCNN(num_class=NUM_CLASS, features_dim=features_dim).to(DEVICES[0])
model = torch.nn.DataParallel(model, device_ids=DEVICES)
criterion_cls = nn.CrossEntropyLoss()  # cross entropy loss
criterion_cent = CenterLoss(num_classes=NUM_CLASS, feat_dim=features_dim)
criterion_margin = MarginLoss(MARGIN, num_classes=NUM_CLASS, feat_dim=features_dim, centers=criterion_cent.get_centers())
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wd, momentum=0.9) # model optimizer
optimizer_cent = optim.SGD(criterion_cent.parameters(), lr=0.5)

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


train_filepath = os.path.join(ROOT_DATA_PATH, f'train_seqLen_{NUM_SEQ_LEN}_rate_{RATIO}_isGZSL_{IS_GZSL}.pkl')
valid_filepath = os.path.join(ROOT_DATA_PATH, f'valid_seqLen_{NUM_SEQ_LEN}_rate_{RATIO}_isGZSL_{IS_GZSL}.pkl')
test_filepath = os.path.join(ROOT_DATA_PATH, f'test_seqLen_{NUM_SEQ_LEN}_rate_{RATIO}_isGZSL_{IS_GZSL}.pkl')
train_loader = create_dataloader(train_filepath, batch_size)
valid_loader = create_dataloader(valid_filepath, batch_size)
test_loader = create_dataloader(test_filepath, batch_size)
print(f"train length: {len(train_loader) * batch_size}")
print(f"valid length: {len(valid_loader) * batch_size}")
print(f"test length: {len(test_loader) * batch_size}")

class Trainer:
    def __init__(self, model, train_dataset, valid_dataset, test_dataset, config, savedir=None, devices=torch.device("cpu")):
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.savedir = savedir
        self.devices = devices
        self.model = model.to(devices[0])
        self.known_thresholds:Dict[int, torch.Tensor] = {known_class : 0 for known_class in SEEN_CLASS} # 可见类的距离阈值
        self.distances:Dict[int, list] = {known_class : [] for known_class in SEEN_CLASS} # 每个可见类对应样本到聚类中心的距离
        self.maha = MahalanobisLayer(features_dim, decay=0.99)
        self.tsne = Tsne(feature_dim=features_dim, num_class=NUM_CLASS, seen_class=SEEN_CLASS)
        initialize_weights(model)
    
    def update_thresholds(self):
        distances = {k : torch.stack(v).squeeze(dim=1) for (k, v) in self.distances.items()} # self.distances[i] = (num_samples, num_class, 1)
        distances_std = {k : torch.std(v, dim=0) for (k, v) in distances.items()} # distances[i] = (num_samples, num_class)
        for (cls_id, distance) in distances.items():
            distances[cls_id] = torch.sort(distance, dim=0)[0]
            # 3-sigma
            outlier_indics = torch.where(distances[cls_id] >= 3 * distances_std[cls_id])[0]
            outlier_index = outlier_indics[0].item()
            print(f"{cls_id}: {outlier_index}th value={distances[cls_id][outlier_index]}")
            self.known_thresholds[cls_id] = distances[cls_id][outlier_index]
            # 分位数
            # self.known_thresholds[cls_id] =  distances[cls_id][int(0.95 * distances[cls_id].shape[0])]
            # print(f"{cls_id}: threshold value={self.known_thresholds[cls_id]}")

    def calculate_distance(self, centers: nn.Parameter, feature: torch.Tensor) -> torch.Tensor:
        """计算特征向量到所有聚类中心的平方马氏距离

        Args:
            centers (nn.Parameter): 聚类中心向量 [num_class, features_dim]
            feature (torch.Tensor): 特征向量 [1, features_dim]

        Returns:
            torch.Tensor: 特征向量到每个聚类中心的马氏距离 [num_class(seen), 1]
        """
        return torch.stack([self.maha(feature, center) for center in centers])
    
    def dist_clearing(self):
        for saving_list in self.distances.values():
            saving_list.clear()

    def batch_dist_saving(self, centers: nn.Parameter, features: torch.Tensor, labels: torch.Tensor):
        for bid in range(features.shape[0]):
            true_label = labels[bid].item()
            self.distances[true_label].append(self.calculate_distance(centers, features[bid])[true_label])
    
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
            dists = self.calculate_distance(centers, features[bid])
            for certain_label in SEEN_CLASS:
                # if dists[certain_label] < self.known_thresholds[certain_label]:
                #     if_know = True
                if dists[certain_label] < self.known_thresholds[certain_label] and dists[certain_label] < min_dist:
                    if_know = True
                    min_dist = dists[certain_label]
                    predict_label = certain_label

            results.append(predict_label)
        return torch.Tensor(results).cuda()

    
    def run_epoch(self, model, loader, epoch:int, stage:Literal['train', 'valid', 'test']):
        devices = self.devices
        is_train = stage == 'train'
        model.train(is_train)
        pbar = tqdm(enumerate(loader), total=len(loader))
        # losses = []
        num_total, num_correct = 0, 0
        for i, (x, y) in pbar:
            x = x.to(devices[0])
            x = bacth_STFT(x, N_FFT, HOP_LENGTH, WINDOM_LENGTH, torch.hamming_window(WINDOM_LENGTH).to(devices[0]), verbose=False)
            y = y.to(devices[0])
            features, logits, crude_features = model(x)
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
                self.batch_dist_saving(criterion_cent.get_centers(), features, y)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                for param in criterion_cent.parameters():
                    param.grad.data *= (1. / eta_cent) # multiple (1./alpha) in order to remove the effect of alpha on updating centers
                optimizer_cent.step()
                pbar.set_description(f"epoch {epoch} {stage + 'ing'}... iter {i}: loss {loss.item():.5f}. lr {learning_rate:e} acc {num_correct / num_total: .4f}")
            else:
                # features = torch.hstack((features, crude_features))
                # U, S, V = torch.pca_lowrank(features, q=features_dim)
                # features = torch.matmul(features, V[:, :features_dim])
                predict_labels = self.semantic_classify(criterion_cent.get_centers(), features)
                # TODO: 目前是单未知类的替换
                predict_labels = torch.where(predict_labels == -1, UNSEEN_CLASS[0] if len(UNSEEN_CLASS) > 0 else -1, predict_labels)
                num_correct += torch.sum(predict_labels == y)
                pbar.set_description(f"epoch {epoch} {stage + 'ing'}.... iter {i}: loss {loss.item():.5f}. lr {learning_rate:e} acc {num_correct / num_total: .4f}")

        if is_train:
            self.tsne.append(criterion_cent.get_centers().detach().cpu().numpy(), np.arange(NUM_CLASS))
            self.tsne.cal_and_save(os.path.join(ROOT_PROJECT_PATH, "temp", f"epoch_{epoch}_tsne.png"))
            self.tsne.clear()
            # 每个epoch更新已知类的阈值，清空保存的距离（聚类中心更新需要重新计算）
            self.update_thresholds()
            self.dist_clearing()

    def train(self):
        epoch_pbar = tqdm(range(num_epoch), desc=f"开始训练")
        for epoch in epoch_pbar:
            epoch_pbar.set_description(f"训练的第{epoch}个epoch")
            self.run_epoch(self.model, train_loader, epoch=epoch, stage='train')
            self.run_epoch(self.model, valid_loader, 0 , stage='valid')
            if epoch % 10 == 0:
                self.run_epoch(self.model, test_loader, epoch / 10 , stage='test')

    def test(self):
        self.run_epoch(self.model, test_loader, 0 , stage='test')
        

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("GPU is running...")

    # print(f"Number of train samples: {len(train_loader)}, Number of test samples: {len(test_loader)}")
    # print(f"Model size: {get_model_size(model.module):.4f}MB")
    t = Trainer(model, train_loader, valid_loader, test_loader, None, savedir=None, devices=DEVICES)
    t.train()
    t.test()


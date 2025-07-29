import pickle
import numpy as np
from typing import Tuple,List
import os
import random
import pandas as pd
import numpy as np
import math
from tqdm import tqdm
from util.utils import standardized, normalized, normalize_to_range
from geopy.distance import geodesic
from tslearn.barycenters import softdtw_barycenter
from tslearn.metrics import dtw_path_from_metric
from scipy.signal import correlate

def get_augmented_data(x, mean=0, std=0.25):
    """Returns a distorted version of the train data to be used in hyperparameter optimization"""

    aug_data = x.copy()
    if np.unique(np.isnan(aug_data)):
        aug_data[np.isnan(aug_data)] = mean
    for i in range(len(x)):
        for j in range(x[0].shape[1]):
            noise = np.random.normal(mean, std, len(x[0]))
            aug_data[i, :, j] += noise

    cut_idx = int(aug_data.shape[1] / 2)
    temp1 = aug_data[:, cut_idx:]
    temp2 = aug_data[:, :cut_idx]

    aug_data = np.hstack((temp1, temp2))
    
    return np.flip(aug_data)

class AisDataReader():
    """航迹数据集
    """
    gps_type = {
        '歼-10A' :0,
        '歼-10B' :1,
        '歼-10C' :2,
        '歼-11B' :3,
        '歼-11BS':4,
        '歼-16'  :5,
        '歼-20'  :6,
        '苏-30'  :7
    }

    gps_type_1 = {
        '歼-10A' :0,
        '歼-10B' :0,
        '歼-10C' :0,
        '歼-11B' :1,
        '歼-11BS':1,
        '歼-16'  :2,
        '歼-20'  :3,
        '苏-30'  :4
    }


    def __init__(self, dname, dpath, seen_class:List[int], unseen_class:List[int], seq_len=1024, num_feature=4, rate=0.7, is_gzsl=False) -> None:
        super().__init__()
        self.dname = dname
        self.seen_class = seen_class
        self.unseen_class = unseen_class
        self.num_class = len(seen_class) + len(unseen_class)
        self.dpath = dpath
        self.raw_data_map = {}
        self.seq_len = seq_len
        self.num_feature = num_feature
        self.rate = rate
        self.is_gzsl = is_gzsl
        # self.X = np.ndarray((0, self.seq_len, self.num_feature), dtype=np.float32)
        # self.Y = np.ndarray((0), dtype=np.int32)
        self.X = []
        self.Y = []
        self.cls_count = [0]
        self.interval = 100

        self.load_data_with_fft()
        self.split_train_test(self.rate)
        self.split_unknown(rate=0.5)
        pass

    def get_data(self):
        self.X_train_split, self.Y_train_split, self.X_test_split, self.Y_test_split, self.X_unk_split, self.Y_unk_split
    
    def save(self, fpath) -> None:
        with open(os.path.join(fpath, f"train_seqLen_{self.seq_len}_rate_{self.rate}_isGZSL_{self.is_gzsl}.pkl"), 'wb') as f: #保存为二进制文件
            pickle.dump((self.X_train_split, self.Y_train_split), f)
        with open(os.path.join(fpath, f"valid_seqLen_{self.seq_len}_rate_{self.rate}_isGZSL_{self.is_gzsl}.pkl"), 'wb') as f: #保存为二进制文件
            pickle.dump((self.X_test_split, self.Y_test_split), f)
        with open(os.path.join(fpath, f"test_seqLen_{self.seq_len}_rate_{self.rate}_isGZSL_{self.is_gzsl}.pkl"), 'wb') as f: #保存为二进制文件
            pickle.dump((self.X_unk_split, self.Y_unk_split), f)

    def cal_speed(self, lng, lat, alt):
        horizontal_dis = np.array([geodesic((lat[i], lng[i]), (lat[i + 1], lng[i + 1])).meters for i in range(len(lat) - 1)])
        vertiacal_dis = alt[1:] - alt[:-1]
        distances_3d = np.sqrt(horizontal_dis**2 + vertiacal_dis**2)
        speeds = distances_3d / 1
        return speeds


    def load_data(self):
        cnt = 0
        pbar = tqdm(range(self.num_class), leave=True, position=0, desc="读取原始数据文件...")
        if self.dname == 'ais':
            for i in pbar:
                arr = []
                dir_name = os.path.join(self.dpath, str(i))
                for file_name in os.listdir(dir_name):
                    single_sample = pd.read_csv(os.path.join(dir_name, file_name), sep=' ', names=["time", "lng", "lat", "sog", "cog"]).drop(columns=['time']).to_numpy(dtype=np.float32)
                    # [seq_len, 4]
                    # 找到速度中第一个不为0的位置
                    non_zero_index = np.argmax(single_sample[:, 2] != 0)
                    end_index = min(non_zero_index + self.seq_len, len(single_sample))
                    single_sample = single_sample[non_zero_index:end_index]
                    cur_seq_len = single_sample.shape[0]
                    if cur_seq_len < self.seq_len:
                        new_data = np.zeros((self.seq_len, single_sample.shape[1]))
                        new_data[:cur_seq_len, :] = single_sample
                        new_data[cur_seq_len:, 0:2] = single_sample[-1, 0:2]
                        single_sample = new_data
                    # norm_single_sample = normalize_to_range(single_sample, -1, 1)
                    # if np.all(single_sample[:, 2] == 0) or np.any(np.isnan(norm_single_sample)):
                    #     continue
                    arr.append(single_sample)
                self.raw_data_map[i] = np.array(arr)
                cnt += len(arr)
                self.cls_count.append(cnt)
                self.X = np.vstack((self.X, self.raw_data_map[i]))
                self.Y = np.hstack((self.Y, [i] * len(arr)))
                pbar.set_description(desc=f"正在处理第{i}类, 总计共{cnt}个样本, shape={self.raw_data_map[i].shape}", refresh=True)
        elif self.dname == 'aircraft':
            for date_dir in os.listdir(self.dpath):
                for file in os.path.join(self.dpath, date_dir):
                    print(os.path.join(self.dpath, date_dir, file))
        else:
            raise ValueError(f"Unknown dataset name: {self.dname}")

        for k, samples in self.raw_data_map.items():
            print(f"the {k}-th class has {len(samples)} samples")


    def load_data_with_fft(self):
        cnt = 0
        if self.dname == 'ais':
            pbar = tqdm(range(self.num_class), leave=True, position=0, desc="读取原始数据文件...")
            for i in pbar:
                arr = []
                dir_name = os.path.join(self.dpath, str(i))
                for file_name in os.listdir(dir_name):
                    single_sample = pd.read_csv(os.path.join(dir_name, file_name), sep=' ', names=["time", "lng", "lat", "sog", "cog"]).drop(columns=['time']).to_numpy(dtype=np.float32)
                    # [seq_len, 4]
                    narr = np.quantile(single_sample, np.linspace(0, 1, self.seq_len), axis=0)
                    new_narr = np.abs(np.fft.fft(narr[:, (0, 1, 2, 3)], axis=0)) / self.seq_len
                    arr.append(np.hstack((narr, new_narr)))
                self.raw_data_map[i] = np.array(arr)
                cnt += len(arr)
                self.cls_count.append(cnt)
                self.X = np.vstack((self.X, self.raw_data_map[i]))
                self.Y = np.hstack((self.Y, [i] * len(arr)))
                pbar.set_description(desc=f"正在处理第{i}类, 总计共{cnt}个样本, shape={self.raw_data_map[i].shape}", refresh=True)
        elif self.dname == 'aircraft':
            for k, v in self.gps_type.items():
                self.raw_data_map[v] = []
            for date_dir in os.listdir(self.dpath):
                if date_dir == 'barycenters':
                    continue
                if not os.path.isdir(os.path.join(self.dpath, date_dir)):
                    continue
                for file in os.listdir(os.path.join(self.dpath, date_dir)):
                    df = pd.read_excel(os.path.join(self.dpath, date_dir, file), engine='openpyxl', 
                                                names=["id", "type", "time", "lng", "lat", "heading", "height", "nor_speed", "sky_speed", "east_speed"])
                    label = self.gps_type[df['type'][0]]
                    df.drop(columns=['id', 'type', 'time'], inplace=True)
                    df = df[(df["lat"] <= 90) & (df["lat"] >= -90)]
                    df["speed"] = pd.Series(
                        self.cal_speed(df["lng"].to_numpy(), df["lat"].to_numpy(), df["height"].to_numpy()))
                    narr = df.to_numpy(dtype=np.float32)
                    # num_splits = (len(narr) + self.interval - 1) // self.interval
                    # num_splits -= 1
                    # narr = narr[:num_splits * self.interval]
                    # data_list = np.stack([narr[i::num_splits, :] for i in range(num_splits)], axis=0)
                    # # data_return = np.ndarray((num_splits, self.seq_len, self.num_feature), dtype=np.float32)
                    # for i, data in enumerate(data_list):
                    #     data = np.quantile(data, np.linspace(0, 1, self.seq_len), axis=0)
                    #     # new_data = np.abs(np.fft.fft(data, axis=0)) / self.seq_len
                    #     # data = np.hstack((data, new_data))
                    #     # data = new_data
                    #     self.raw_data_map[label].append(data)
                    print(f"正在处理{file}")
                    self.raw_data_map[label].append(narr)
            
            for k in self.raw_data_map.keys():
                cnt += len(self.raw_data_map[k])
                self.cls_count.append(cnt)
                self.X.extend(self.raw_data_map[k])
                self.Y.extend([k] * len(self.raw_data_map[k]))
                # self.X = np.vstack((self.X, np.stack(self.raw_data_map[k])))
                # self.Y = np.hstack((self.Y, [k] * len(self.raw_data_map[k])))
            
            self.raw_data_map.clear()
        else:
            raise ValueError(f"Unknown dataset name: {self.dname}")
    
    def split_train_test(self, rate=0.7, seed=2024):
        np.random.seed(seed)

        self.train_indices = []
        self.test_indices = []
        # self.train_map={}
        # self.test_map={}

        for idx, certain_class in enumerate(self.seen_class):
            class_idx = list(range(self.cls_count[certain_class],self.cls_count[certain_class+1]))
            class_train_idx = list(np.random.choice(class_idx, size=int(len(class_idx)*rate), replace=False))
            class_test_idx = list(set(class_idx)-set(class_train_idx))
            # self.train_map[idx]=self.X[class_train_idx]
            # self.test_map[idx]=self.X[class_test_idx]
            self.train_indices += class_train_idx
            self.test_indices += class_test_idx

        random.shuffle(self.train_indices)
        random.shuffle(self.test_indices)
        self.X_train = [self.X[i] for i in self.train_indices]
        self.X_test = [self.X[i] for i in self.test_indices]
        self.Y_train = [self.Y[i] for i in self.train_indices]
        self.Y_test = [self.Y[i] for i in self.test_indices]

        self.X_train_split = np.ndarray((0, self.seq_len, self.num_feature), dtype=np.float32)
        self.X_test_split = np.ndarray((0, self.seq_len, self.num_feature), dtype=np.float32)
        self.Y_train_split = np.ndarray((0), dtype=np.int32)
        self.Y_test_split = np.ndarray((0), dtype=np.int32)

        for i, narr in enumerate(self.X_train):
            num_splits = (len(narr) + self.interval - 1) // self.interval
            num_splits -= 1
            narr = narr[:num_splits * self.interval]
            data_list = np.stack([np.quantile(
                narr[i::num_splits, :], 
                np.linspace(0, 1, self.seq_len), 
                axis=0) for i in range(num_splits)], axis=0)
            self.X_train_split = np.vstack((self.X_train_split, data_list))
            self.Y_train_split = np.hstack((self.Y_train_split, np.ones(num_splits) * self.Y_train[i]))

        for i, narr in enumerate(self.X_test):
            num_splits = (len(narr) + self.interval - 1) // self.interval
            num_splits -= 1
            narr = narr[:num_splits * self.interval]
            data_list = np.stack([np.quantile(
                narr[i::num_splits, :], 
                np.linspace(0, 1, self.seq_len), 
                axis=0) for i in range(num_splits)], axis=0)
            self.X_test_split = np.vstack((self.X_test_split, data_list))
            self.Y_test_split = np.hstack((self.Y_test_split, np.ones(num_splits) * self.Y_test[i]))

        permuted_indices = np.random.permutation(len(self.X_train_split))
        self.X_train_split = self.X_train_split[permuted_indices]
        self.Y_train_split = self.Y_train_split[permuted_indices]

        permuted_indices_test = np.random.permutation(len(self.X_test_split))
        self.X_test_split = self.X_test_split[permuted_indices_test]
        self.Y_test_split = self.Y_test_split[permuted_indices_test]

    def split_unknown(self, rate=0.7, seed=2024):
        np.random.seed(seed)

        self.unknown_test_indices = []
        self.X_unk_split = np.ndarray((0, self.seq_len, self.num_feature), dtype=np.float32)
        self.Y_unk_split = np.ndarray((0), dtype=np.int32)
        # self.unknown_test_map={}

        for idx, certain_class in enumerate(self.unseen_class):
            class_indices=list(range(self.cls_count[certain_class],self.cls_count[certain_class+1]))
            unknown_test_class_idx=list(np.random.choice(class_indices, size=int(len(class_indices)*(1-rate)+1), replace=False))
            # self.unknown_test_map[idx + len(self.train_class)]=self.X[unknown_test_class_idx]
            self.unknown_test_indices+=unknown_test_class_idx
        self.X_unknown_test = [self.X[i] for i in self.unknown_test_indices]
        self.Y_unknown_test = [self.Y[i] for i in self.unknown_test_indices]
        for i, narr in enumerate(self.X_unknown_test):
            num_splits = (len(narr) + self.interval - 1) // self.interval
            num_splits -= 1
            narr = narr[:num_splits * self.interval]
            data_list = np.stack([np.quantile(
                narr[i::num_splits, :], 
                np.linspace(0, 1, self.seq_len), 
                axis=0) for i in range(num_splits)], axis=0)
            self.X_unk_split = np.vstack((self.X_unk_split, data_list))
            self.Y_unk_split = np.hstack((self.Y_unk_split, np.ones(num_splits) * self.Y_unknown_test[i]))

    def calculate_barycenter(self, fpath):
        # Calculate Barycenters
        # Sort by the target values so that it can be split for each class 
        sorted_idxs = np.argsort(self.Y_train_split, axis=0)
        x_train = self.X_train_split[sorted_idxs]
        y_train = self.Y_train_split[sorted_idxs]

        # Group by target, i.e. one split for each class
        splits = np.split(x_train, np.unique(y_train, return_index = True)[1][1:])

        bc_dir = os.path.join(fpath, 'barycenters')
        os.makedirs(bc_dir, exist_ok=True)
        bc_list = [] 

        # Calculate barycenters for each class (split)
        c_ = 0
        for split in tqdm(splits, desc=f"Calculating barycenters for..."):
            bc = softdtw_barycenter(split) # barycenters for the given class 
            bc_list.append(bc)
            np.save(os.path.join(bc_dir, f'bc_{c_}'), bc) # save files
            c_ += 1

        print(f'Calculating distances...')
        distances_all = {}
        for i in tqdm(range(len(self.seen_class))):
            distances_for_class = []
            for x in splits[i]:
                distance = dtw_path_from_metric(x, bc_list[i], metric='sqeuclidean')[1]
                distances_for_class.append(math.log(distance))
            distances_all[i] = (distances_for_class)
        
        with open(os.path.join(fpath, f"all_dist.pkl"), 'wb') as f: #保存为二进制文件
            pickle.dump(distances_all, f)

        print(f'Calculating correlate...')
        cc_all = {}
        for i in tqdm(range(len(self.seen_class))):
            cc_for_each_class = []
            for x in splits[i]:
                cc_for_each_dim = []
                for c in range(x_train.shape[2]):
                    cc = np.max(correlate(x[:, c], bc_list[i][:, c]))
                    cc_for_each_dim.append(cc)
                cc_for_each_class.append(cc_for_each_dim)
            cc_all[i] = cc_for_each_class

        with open(os.path.join(fpath, f"all_cc.pkl"), 'wb') as f: #保存为二进制文件
            pickle.dump(cc_all, f)

        # print(f'Creating augmented samples...')
        # aug_train = get_augmented_data(x_train)
        # aug_train[np.isnan(aug_train)] = mask_value

        # # Save files
        # aug_dir = f'augmented/{dataset}'
        # os.makedirs(aug_dir, exist_ok=True)
        # np.save(os.path.join(aug_dir, f'{dataset}_train_augmented'), aug_train)
    
    
    @staticmethod
    def load_binary(fpath):
        with open(fpath, 'rb') as f:
            return pickle.load(f)
        
if __name__ == '__main__':
    from config.deafault import get_cfg_defaults 

    cfg = get_cfg_defaults()
    cfg.merge_from_file("./config/ais.yaml")
    cfg.freeze()
    print(cfg)

    data_reader = AisDataReader(cfg.dataset.name, cfg.dataset.root_data_path, cfg.dataset.seen_class, cfg.dataset.unseen_class, cfg.dataset.seq_len, cfg.dataset.num_feature, cfg.dataset.ratio)
    data_reader.save(os.path.join(cfg.dataset.root_data_path))
    data_reader.calculate_barycenter(cfg.dataset.root_data_path)
    X_train, Y_train = data_reader.load_binary(os.path.join(cfg.dataset.root_data_path, 
                                                            f'train_seqLen_{cfg.dataset.seq_len}_rate_{cfg.dataset.ratio}_isGZSL_{cfg.dataset.is_gzsl}.pkl'))
    X_test, Y_test = data_reader.load_binary(os.path.join(cfg.dataset.root_data_path, 
                                                          f'valid_seqLen_{cfg.dataset.seq_len}_rate_{cfg.dataset.ratio}_isGZSL_{cfg.dataset.is_gzsl}.pkl'))
    X_unknown, Y_unknown = data_reader.load_binary(os.path.join(cfg.dataset.root_data_path, 
                                                                f'test_seqLen_{cfg.dataset.seq_len}_rate_{cfg.dataset.ratio}_isGZSL_{cfg.dataset.is_gzsl}.pkl'))
    print(f"train: {len(X_train)}")
    print(f"valid: {len(X_test)}")
    print(f"test : {len(X_unknown)}")
    
import pickle
import numpy as np
from typing import Tuple,List
import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from util.utils import standardized, normalized, normalize_to_range


class AisDataReader():
    """航迹数据集
    """
    gps_type = {
        '歼-10A':0,
        '歼-10B':1,
        '歼-10C':2,
        '歼-11B':3,
        '歼-11BS':4,
        '歼-16':5,
        '歼-20':6,
        '苏-30':7
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
        self.X = np.ndarray((0, self.seq_len, self.num_feature), dtype=np.float32)
        self.Y = np.ndarray((0), dtype=np.int32)
        self.cls_count = [0]
        self.interval = 100

        self.load_data_with_fft()
        self.split_train_test(self.rate)
        self.split_unknown(rate=0.5)
        pass

    def get_data(self):
        self.X_train, self.Y_train, self.X_test, self.Y_test, self.X_unknown_test, self.Y_unknown_test
    
    def save(self, fpath) -> None:
        with open(os.path.join(fpath, f"train_seqLen_{self.seq_len}_rate_{self.rate}_isGZSL_{self.is_gzsl}.pkl"), 'wb') as f: #保存为二进制文件
            pickle.dump((self.X_train, self.Y_train), f)
        with open(os.path.join(fpath, f"valid_seqLen_{self.seq_len}_rate_{self.rate}_isGZSL_{self.is_gzsl}.pkl"), 'wb') as f: #保存为二进制文件
            pickle.dump((self.X_test, self.Y_test), f)
        with open(os.path.join(fpath, f"test_seqLen_{self.seq_len}_rate_{self.rate}_isGZSL_{self.is_gzsl}.pkl"), 'wb') as f: #保存为二进制文件
            pickle.dump((self.X_unknown_test, self.Y_unknown_test), f)

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
                if not os.path.isdir(os.path.join(self.dpath, date_dir)):
                    continue
                for file in os.listdir(os.path.join(self.dpath, date_dir)):
                    single_sample = pd.read_excel(os.path.join(self.dpath, date_dir, file), 
                                                names=["id", "type", "time", "lng", "lat", "heading", "height", "nor_speed", "sky_speed", "east_speed"])
                    label = self.gps_type[single_sample['type'][0]]
                    single_sample.drop(columns=['id', 'type', 'time'], inplace=True)
                    single_sample = single_sample.to_numpy(dtype=np.float32)
                    num_splits = (len(single_sample) + self.interval - 1) // self.interval
                    data_list = np.array_split(single_sample, num_splits)
                    # data_return = np.ndarray((num_splits, self.seq_len, self.num_feature), dtype=np.float32)
                    for i, data in enumerate(data_list):
                        data = np.quantile(data, np.linspace(0, 1, self.seq_len), axis=0)
                        new_data = np.abs(np.fft.fft(data, axis=0)) / self.seq_len
                        # data = np.hstack((data, new_data))
                        data = new_data
                        self.raw_data_map[label].append(data)
                    print(f"正在处理{file}, 总计共{num_splits}个样本")
            
            for k in self.raw_data_map.keys():
                cnt += len(self.raw_data_map[k])
                self.cls_count.append(cnt)
                self.X = np.vstack((self.X, np.stack(self.raw_data_map[k])))
                self.Y = np.hstack((self.Y, [k] * cnt))
                    
        else:
            raise ValueError(f"Unknown dataset name: {self.dname}")

        for k, samples in self.raw_data_map.items():
            print(f"the {k}-th class has {len(samples)} samples")
    
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
        self.X_train = self.X[self.train_indices]
        self.X_test = self.X[self.test_indices]
        self.Y_train = self.Y[self.train_indices]
        self.Y_test = self.Y[self.test_indices]

    def split_unknown(self, rate=0.7, seed=2024):
        np.random.seed(seed)

        self.unknown_test_indices = []
        # self.unknown_test_map={}

        for idx, certain_class in enumerate(self.unseen_class):
            class_indices=list(range(self.cls_count[certain_class],self.cls_count[certain_class+1]))
            unknown_test_class_idx=list(np.random.choice(class_indices, size=int(len(class_indices)*(1-rate)+1), replace=False))
            # self.unknown_test_map[idx + len(self.train_class)]=self.X[unknown_test_class_idx]
            self.unknown_test_indices+=unknown_test_class_idx
        self.X_unknown_test = self.X[self.unknown_test_indices]
        self.Y_unknown_test = self.Y[self.unknown_test_indices]

    
    @staticmethod
    def load_binary(fpath):
        with open(fpath, 'rb') as f:
            return pickle.load(f)
        
if __name__ == '__main__':
    from config.deafault import get_cfg_defaults 

    cfg = get_cfg_defaults()
    cfg.merge_from_file("./config/aircraft.yaml")
    cfg.freeze()
    print(cfg)

    data_reader = AisDataReader(cfg.dataset.name, cfg.dataset.root_data_path, cfg.dataset.seen_class, cfg.dataset.unseen_class, cfg.dataset.seq_len, cfg.dataset.num_feature, cfg.dataset.ratio)
    data_reader.save(os.path.join(cfg.dataset.root_data_path))
    X_train, Y_train = data_reader.load_binary(os.path.join(cfg.dataset.root_data_path, 
                                                            f'train_seqLen_{cfg.dataset.seq_len}_rate_{cfg.dataset.ratio}_isGZSL_{cfg.dataset.is_gzsl}.pkl'))
    X_test, Y_test = data_reader.load_binary(os.path.join(cfg.dataset.root_data_path, 
                                                          f'valid_seqLen_{cfg.dataset.seq_len}_rate_{cfg.dataset.ratio}_isGZSL_{cfg.dataset.is_gzsl}.pkl'))
    X_unknown, Y_unknown = data_reader.load_binary(os.path.join(cfg.dataset.root_data_path, 
                                                                f'test_seqLen_{cfg.dataset.seq_len}_rate_{cfg.dataset.ratio}_isGZSL_{cfg.dataset.is_gzsl}.pkl'))
    print(f"train: {len(X_train)}")
    print(f"valid: {len(X_test)}")
    print(f"test : {len(X_unknown)}")
    
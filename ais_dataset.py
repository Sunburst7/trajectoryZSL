import pickle
import numpy as np
from typing import Tuple,List
import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import standardized, normalized, normalize_to_range


class AisDataReader():
    """航迹数据集
    """
    def __init__(self, dpath, seen_class:List[int], unseen_class:List[int], seq_len=1024, features_dim=4, rate=0.7, is_gzsl=False) -> None:
        super().__init__()
        self.seen_class = seen_class
        self.unseen_class = unseen_class
        self.num_class = len(seen_class) + len(unseen_class)
        self.dpath = dpath
        self.raw_data_map = {}
        self.seq_len = seq_len
        self.features_dim = 4
        self.rate = 0.7
        self.is_gzsl = is_gzsl
        self.X = np.ndarray((0, self.seq_len, self.features_dim), dtype=np.float32)
        self.Y = np.ndarray((0), dtype=np.int32)
        self.cls_count = [0]

        self.load_data()
        self.split_train_test()
        self.split_unknown()
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
                norm_single_sample = normalized(single_sample)
                if np.all(single_sample[:, 2] == 0) or np.any(np.isnan(norm_single_sample)):
                    continue
                arr.append(single_sample)
            self.raw_data_map[i] = np.array(arr)
            cnt += len(arr)
            self.cls_count.append(cnt)
            self.X = np.vstack((self.X, self.raw_data_map[i]))
            self.Y = np.hstack((self.Y, [i] * len(arr)))
            pbar.set_description(desc=f"正在处理第{i}类, 总计共{cnt}个样本, shape={self.raw_data_map[i].shape}", refresh=True)

        for k, samples in self.raw_data_map.items():
            print(f"the {k}-th class has {len(samples)} samples")
    
    def split_train_test(self, rate=0.7, seed=2024):
        np.random.seed(seed)

        self.train_indices = []
        self.test_indices = []
        # self.train_map={}
        # self.test_map={}

        for idx, certain_class in enumerate(SEEN_CLASS):
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

        for idx, certain_class in enumerate(UNSEEN_CLASS):
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
    ROOT_DATA_PATH = os.path.join('/data2', 'hh', 'workspace', 'data', 'ais')
    SEEN_CLASS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13]
    UNSEEN_CLASS = [10]

    data_reader = AisDataReader(ROOT_DATA_PATH, SEEN_CLASS, UNSEEN_CLASS)
    data_reader.save(os.path.join(ROOT_DATA_PATH))
    X_train, Y_train = data_reader.load_binary(os.path.join(ROOT_DATA_PATH, f'train_seqLen_{1024}_rate_{0.7}_isGZSL_{False}.pkl'))
    X_test, Y_test = data_reader.load_binary(os.path.join(ROOT_DATA_PATH, f'valid_seqLen_{1024}_rate_{0.7}_isGZSL_{False}.pkl'))
    X_unknown, Y_unknown = data_reader.load_binary(os.path.join(ROOT_DATA_PATH, f'test_seqLen_{1024}_rate_{0.7}_isGZSL_{False}.pkl'))
    print(f"train: {len(X_train)}")
    print(f"valid: {len(X_test)}")
    print(f"test : {len(X_unknown)}")
    
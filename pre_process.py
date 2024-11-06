import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys
import random
import pickle
from typing import List, Dict

from ais_dataset import AisDataset 

ROOT_DATA_PATH = os.path.join('/data2', 'hh', 'workspace', 'data', 'ais')
NUM_CLASS = 14
LNG_AND_LAT_THRESHOLD = 1
NUM_SAMPLE_ROW = 1024
NUM_SAMPLE_FEATURES = 4
RATIO = 0.7
USE_COMPLEX = False
IS_GZSL = False
SEEN_CLASS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13]
UNSEEN_CLASS = [10]
RANDOM_SEED = 42
raw_data = {} # raw_data每个索引对应一个DataFrame数组
processed_data = {} #每一个索引对应一个ndarray 第一维是样本数

def split_samples(df: pd.DataFrame, limited_row_num: int, thresholds_dict: dict) -> pd.DataFrame:
    """按照列元素变化幅度与行数分割样本

    Args:
        df (pd.DataFrame): dataframe
        limited_row_num (int): 最大行数
        thresholds_dict (dict): 每列的元素diff阈值 是一个字典索引 与df的列名对应

    Examples:
        data = {
            'A': np.random.rand(13),  # 随机生成200个元素
            'B': np.random.rand(13)
        }
        df = pd.DataFrame(data)
        threshold = .5 # 定义差值阈值
        row_limit = 5 # 定义行数阈值
        print(split_samples(df, row_limit, {'A': threshold, 'B': threshold}))
    """
    segment = 0
    seg_len = 0 # 当前段的长度
    segments = []
    diff_column_names = []
    for column_name in df.columns:
        if column_name in thresholds_dict:
            df[f'diff_{column_name}'] = df[column_name].diff().abs()
            diff_column_names.append(column_name)

    for i in range(len(df)):
        if i == 0:
            segments.append(segment)
            seg_len += 1
            continue
    
        if seg_len % limited_row_num == 0 and seg_len > 0:
            segment += 1
            segments.append(segment)
            seg_len = 1 
            continue
        # 如果差值大于阈值，增加段数
        for c_name in diff_column_names:
            if df[f'diff_{c_name}'].iloc[i] > thresholds_dict[c_name]:
                segment += 1
                seg_len = 0
                break

        segments.append(segment)
        seg_len += 1

    df['segment'] = segments 
    df.drop(columns=[f'diff_{name}' for name in diff_column_names], inplace=True)
    return df

def ais_pad_with_tail(data: np.ndarray, max_num: int):
    """根据末尾值的经纬度填充AIS数据

    Args:
        data (np.ndarray): 待填充数据
        max_num (int): 填充的个数
    
    Examples:
        data = np.array([
            [120., 30., 10., 90.],
            [121., 31., 15., 85.]
        ])
        print(ais_pad_with_tail(data, 5))
    """
    assert max_num >= data.shape[0]
    rows = data.shape[0]
    new_data = np.zeros((max_num, data.shape[1]))
    new_data[:rows, :] = data
    new_data[rows:, 0:2] = data[-1, 0:2]
    return new_data

def split_zsl_dataset(data: Dict[int, np.ndarray], seen_classes: List[int], unseen_classes: List[int], ratio: float, is_generalized: bool):
    """划分ZSL数据集为训练集、验证集、测试集

    Args:
        data (Dict[int, np.ndarray]): AIS数据集，每个类的字典项为一个ndarray数据，第一维度是样本数
        seen_classes (List[int]): 一个包含可见类的列表
        unseen_classes (List[int]): 一个不包含可见类的列表
        ratio (float): 划分比例
        is_generalized (bool): 是泛化零样本学习吗

    Examples:
        test_dic = {}
        for i in range(NUM_CLASS):
            arr = []
            for j in range(10):
                arr.append(np.full((4,), i))
            test_dic[i] = np.array(arr)

        train_set, valid_set, test_set = split_zsl_dataset(test_dic, SEEN_CLASS, UNSEEN_CLASS, 0.7, False)
        print(train_set)
        print(valid_set)
        print(test_set)
    """
    list_shape = list(data[0].shape)
    list_shape[0] = 0
    tuple_shape = tuple(list_shape)
    random.seed(42)
    train_x = valid_x = test_x = np.ndarray(tuple_shape)
    train_y = valid_y = test_y = np.ndarray((0), dtype=int)
    all_seen_x = all_unseen_x = np.ndarray(tuple_shape)
    all_seen_y = all_unseen_y = np.ndarray((0), dtype=int)
    for seen_cls_id in seen_classes:
        all_seen_x = np.vstack((all_seen_x, data[seen_cls_id]))
        all_seen_y = np.hstack((all_seen_y, np.array(len(data[seen_cls_id]) * [seen_cls_id])))
    for unseen_cls_id in unseen_classes:
        all_unseen_x = np.vstack((all_unseen_x, data[unseen_cls_id]))
        all_unseen_y = np.hstack((all_unseen_y, np.array(len(data[unseen_cls_id]) * [unseen_cls_id])))
    seen_indics = list(range(0, all_seen_x.shape[0]))
    unseen_indics = list(range(0, all_unseen_x.shape[0]))
    random.shuffle(seen_indics)
    random.shuffle(unseen_indics)
    trainset_len = int(ratio * len(seen_indics))
    if is_generalized == False:
        train_x = all_seen_x[seen_indics[:trainset_len]]
        train_y = all_seen_y[seen_indics[:trainset_len]]
        valid_x = all_seen_x[seen_indics[trainset_len:]]
        valid_y = all_seen_y[seen_indics[trainset_len:]]
        test_x = all_unseen_x[unseen_indics[:int((1 - ratio) * len(unseen_indics))]]
        test_y = all_unseen_y[unseen_indics[:int((1 - ratio) * len(unseen_indics))]]
    else:
        train_x = all_seen_x[seen_indics[:trainset_len]]
        train_y = all_seen_y[seen_indics[:trainset_len]]
        valid_x = all_seen_x[seen_indics[trainset_len: (len(seen_indics) + trainset_len) // 2]]
        valid_y = all_seen_y[seen_indics[trainset_len: (len(seen_indics) + trainset_len) // 2]]
        test_x = all_unseen_x[unseen_indics[:int((1 - ratio) * len(unseen_indics))]]
        test_y = all_unseen_y[unseen_indics[:int((1 - ratio) * len(unseen_indics))]]
        test_x = np.vstack((test_x, all_seen_x[seen_indics[(len(seen_indics) + trainset_len) // 2 : ]]))
        test_y = np.hstack((test_y, all_seen_y[seen_indics[(len(seen_indics) + trainset_len) // 2 : ]]))
    return AisDataset(train_x, train_y), AisDataset(valid_x, valid_y), AisDataset(test_x, test_y)

for i in range(NUM_CLASS):
    arr = []
    dir_name = os.path.join(ROOT_DATA_PATH, str(i))
    for file_name in os.listdir(dir_name):
        arr.append(pd.read_csv(os.path.join(dir_name, file_name), sep=' ', names=["time", "lng", "lat", "sog", "cog"]))
    raw_data[i] = arr

pbar = tqdm(range(NUM_CLASS), leave=True, position=0)
for i in pbar:
    arr = []
    for j in range(len(raw_data[i])):
        # 根据 segment 列分组
        grouped = split_samples(raw_data[i][j], NUM_SAMPLE_ROW, {"lng" : 1, "lat" : 1}).groupby('segment')
        # 获取分组后的 DataFrame
        for sid, group in grouped:
            np_data = group.drop(columns=['segment', 'time']).to_numpy(dtype=np.float32)
            # 对于不足NUM_SAMPLE_ROW的样本使用最后一个经纬度值，速度航向为0填充到NUM_SAMPLE_ROW
            pad_np_data = ais_pad_with_tail(np_data, NUM_SAMPLE_ROW) if np_data.shape[0] < NUM_SAMPLE_ROW else np_data
            if USE_COMPLEX == True:
                complex_pad_np_data = np.ndarray((pad_np_data.shape[0], 2), dtype=complex)
                complex_pad_np_data[:, 0] = pad_np_data[:, 0] + 1j * pad_np_data[:, 1]
                complex_pad_np_data[:, 1] = pad_np_data[:, 2] + 1j * pad_np_data[:, 3]
                arr.append(complex_pad_np_data)
            else:
                arr.append(pad_np_data)

    arr = np.array(arr)
    processed_data[i] = arr
    pbar.set_description_str(desc=f"正在处理第{i}类，共{len(arr)}个样本, shape={arr.shape}", refresh=True)


train_set, valid_set, test_set = split_zsl_dataset(processed_data, SEEN_CLASS, UNSEEN_CLASS, RATIO, IS_GZSL)
train_filepath = os.path.join(ROOT_DATA_PATH, f'train_seqLen_{NUM_SAMPLE_ROW}_ratio_{RATIO}_isGZSL_{IS_GZSL}.pkl')
valid_filepath = os.path.join(ROOT_DATA_PATH, f'valid_seqLen_{NUM_SAMPLE_ROW}_ratio_{RATIO}_isGZSL_{IS_GZSL}.pkl')
test_filepath = os.path.join(ROOT_DATA_PATH, f'test_seqLen_{NUM_SAMPLE_ROW}_ratio_{RATIO}_isGZSL_{IS_GZSL}.pkl')
# 保存为二进制文件
train_set.save(train_filepath) 
train_set = AisDataset.load(train_filepath)
print(f"train: {train_set}")
valid_set.save(valid_filepath)
valid_set = AisDataset.load(valid_filepath)
print(f"valid: {valid_set}")
test_set.save(test_filepath)
test_set = AisDataset.load(test_filepath)
print(f"test : {test_set}")





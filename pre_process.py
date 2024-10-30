import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys
import pickle

from torch_kits.split_samples import split_samples

ROOT_DATA_PATH = os.path.join('/data2', 'hh', 'workspace', 'data', 'ais')
NUM_CLASS = 14
LNG_AND_LAT_THRESHOLD = 1
NUM_SAMPLE_ROW = 100
raw_data = {} # raw_data每个索引对应一个DataFrame数组
processed_data = {}

for i in range(NUM_CLASS):
    arr = []
    dir_name = os.path.join(ROOT_DATA_PATH, str(i))
    for file_name in os.listdir(dir_name):
        arr.append(pd.read_csv(os.path.join(dir_name, file_name), sep=' ', names=["time", "lng", "lat", "sog", "cog"]))
    raw_data[i] = arr

for i in tqdm(range(NUM_CLASS), desc=f"处理第{i}类样本"):
    arr = []
    for j in range(len(raw_data[i])):
        # 根据 segment 列分组
        grouped = split_samples(raw_data[i][j], NUM_SAMPLE_ROW, {"lng" : 1, "lat" : 1}).groupby('segment')
        # 获取分组后的 DataFrame
        for sid, group in grouped:
            arr.append(group.drop(columns=['segment', 'time']).to_numpy())
        # TODO: 对于不足NUM_SAMPLE_ROW的样本使用最后一个经纬度值，速度航向为0填充到NUM_SAMPLE_ROW

    # 保存为二进制文件
    with open(os.path.join(ROOT_DATA_PATH, f'np_class_{i}.pkl'), 'wb') as f: # List[ndarray]保存为二进制文件
        pickle.dump(arr, f)

# with open(os.path.join(ROOT_DATA_PATH, f'np_class_0.pkl'), 'rb') as f:
#     loaded_array_list = pickle.load(f)
#     print(len(loaded_array_list))




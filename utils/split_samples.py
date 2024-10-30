import pandas as pd

def split_samples(df: pd.DataFrame, limited_row_num: int, thresholds_dict: dict) -> pd.DataFrame:
    """按照列元素变化幅度与行数分割样本

    Args:
        df (pd.DataFrame): dataframe
        limited_row_num (int): 最大行数
        thresholds_dict (dict): 每列的元素diff阈值 是一个字典索引 与df的列名对应
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

# # 创建示例 DataFrame
# data = {
#     'A': np.random.rand(13),  # 随机生成200个元素
#     'B': np.random.rand(13)
# }
# df = pd.DataFrame(data)

# # 定义差值阈值
# threshold = .5
# # 定义行数阈值
# row_limit = 5
# print(split_samples(df, row_limit, {'A': threshold, 'B': threshold}))
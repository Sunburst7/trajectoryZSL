import numpy as np

def standardized(matrix: np.ndarray, axis=0):
    """Z-Score 标准化

    Args:
        matrix (np.ndarray): _description_
        axis (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    return (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)

def normalized(matrix: np.ndarray, axis=0):
    """最小-最大归一化

    Args:
        matrix (np.ndarray): _description_
        axis (int, optional): _description_. Defaults to 0.
    """
    return (matrix - matrix.min(axis=0)) / (matrix.max(axis=0) - matrix.min(axis=0))      

def normalize_to_range(matrix: np.ndarray, a, b, axis=0):
    """
    将数据归一化到指定的范围 [a, b]
    """
    # 计算数据的最小值和最大值
    min_val = matrix.min(axis=0)
    max_val = matrix.max(axis=0)
    
    # 归一化公式
    return a + (matrix - min_val) * (b - a) / (max_val - min_val)  



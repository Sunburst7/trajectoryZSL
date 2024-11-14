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
    arr_normalized = (matrix - matrix.min(axis=0)) / (matrix.max(axis=0) - matrix.min(axis=0))        



import numpy as np
from sklearn import preprocessing
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class Tsne():
    def __init__(self, feature_dim, num_class) -> None:
        # t-SNE降维处理
        self.feature_dim = feature_dim
        self.num_class = num_class
        self.vec_array = np.ndarray((0, feature_dim))
        self.label_array = np.ndarray((0))
        self.tsne = TSNE(n_components=2, verbose=1, random_state=42)
    
    def append(self, samples: np.ndarray, labels: np.ndarray):
        self.vec_array = np.vstack((self.vec_array, samples))
        self.label_array = np.hstack((self.label_array, labels))

    def clear(self):
        self.vec_array = np.ndarray((0, self.feature_dim))
        self.label_array = np.ndarray((0))

    def cal_and_save(self, path):
        result = self.tsne.fit_transform(self.vec_array)

        # 归一化处理
        scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        result = scaler.fit_transform(result)

        fig, ax = plt.subplots()
        ax.set_title('t-SNE process')
        ax.scatter(result[:-self.num_class,0], result[:-self.num_class,1], c=self.label_array[:-self.num_class], s=5, alpha=0.4, cmap='jet')
        ax.scatter(result[-self.num_class:,0], result[-self.num_class:,1], c=self.label_array[-self.num_class:], s=50, marker='*', cmap='jet')
        plt.savefig(path)
        plt.close()
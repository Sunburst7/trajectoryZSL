import numpy as np
from sklearn import preprocessing
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import Dict, Literal

class Tsne():
    def __init__(self, feature_dim, num_class, seen_class, unseen_class) -> None:
        # t-SNE降维处理
        self.feature_dim = feature_dim
        self.num_class = num_class
        self.seen_class = seen_class
        self.unseen_class = unseen_class
        self.vec_array = np.ndarray((0, feature_dim))
        self.label_array = np.ndarray((0))
        self.tsne = TSNE(n_components=3, perplexity=40, verbose=0, random_state=42)
    
    def append(self, samples: np.ndarray, labels: np.ndarray):
        self.vec_array = np.vstack((self.vec_array, samples))
        self.label_array = np.hstack((self.label_array, labels))

    def clear(self):
        self.vec_array = np.ndarray((0, self.feature_dim))
        self.label_array = np.ndarray((0))

    def cal_and_save(self, path, stage:Literal['train', 'valid', 'test']):
        result = self.tsne.fit_transform(self.vec_array)
        colors = plt.cm.tab20.colors
        # 归一化处理
        scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        result = scaler.fit_transform(result)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_title('t-SNE process')
        center_vec = self.vec_array[-self.num_class:]
        center_label = self.label_array[-self.num_class:]
        self.vec_array = self.vec_array[:-self.num_class]
        self.label_array = self.label_array[:-self.num_class]
        if stage == 'test':
            for label_idx in self.unseen_class:
                ax.scatter(
                    self.vec_array[self.label_array==label_idx, 0],
                    self.vec_array[self.label_array==label_idx, 1],
                    self.vec_array[self.label_array==label_idx, 2],
                    color=colors[label_idx],
                    s=5,
                    alpha=0.4,
                )
        else:
            for label_idx in self.seen_class:
                ax.scatter(
                    self.vec_array[self.label_array==label_idx, 0],
                    self.vec_array[self.label_array==label_idx, 1],
                    self.vec_array[self.label_array==label_idx, 2],
                    color=colors[label_idx],
                    s=5,
                    alpha=0.4,
                )
        for label_idx in self.seen_class:
            ax.scatter(
                center_vec[center_label==label_idx, 0],
                center_vec[center_label==label_idx, 1],
                center_vec[center_label==label_idx, 2],
                color=colors[label_idx],
                s=50, marker='*'
            )
        plt.legend([str(i) for i in self.seen_class], loc='upper right')
        # ax.scatter(result[:-self.num_class,0], result[:-self.num_class,1], c=self.label_array[:-self.num_class], s=5, alpha=0.4, cmap='jet')
        # ax.scatter(result[-self.num_class:,0], result[-self.num_class:,1], c=self.label_array[-self.num_class:], s=50, marker='*', cmap='jet')
        plt.savefig(path)
        plt.close()
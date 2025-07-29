import torch
import torch.nn as nn

class Metric(nn.Module):
    def __init__(self, unk_class):
        super(Metric, self).__init__()
        self.metric = ['k_acc', 'u_acc', 'u_recall', 'u_f1']  # 需要计算的指标
        self.unk_class = set(unk_class)  # 使用集合加速查找
        self.reset()

    def reset(self):
        """重置所有统计信息"""
        self.K_TP = 0  # 已知类别的 True Positive
        self.K_FP = 0  # 已知类别的 False Positive
        self.K_TN = 0  # 已知类别的 True Negative
        self.K_FN = 0  # 已知类别的 False Negative

        self.U_TP = 0  # 未知类别的 True Positive
        self.U_FP = 0  # 未知类别的 False Positive
        self.U_TN = 0  # 未知类别的 True Negative
        self.U_FN = 0  # 未知类别的 False Negative

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        更新统计情况（并行化实现）
        :param y_true: 真实标签 (torch.Tensor)
        :param y_pred: 预测标签 (torch.Tensor)
        """
        # 将标签转换为 CPU（如果是在 GPU 上）
        y_true = y_true.cpu()
        y_pred = y_pred.cpu()

        # 判断是否为未知类别
        is_true_unk = torch.tensor([true.item() in self.unk_class for true in y_true], dtype=torch.bool)
        is_pred_unk = torch.tensor([pred.item() in self.unk_class for pred in y_pred], dtype=torch.bool)

        # 计算已知类别的统计
        known_true = y_true[~is_true_unk]
        known_pred = y_pred[~is_true_unk]
        self.K_TP += torch.sum(known_true == known_pred).item()
        self.K_FP += torch.sum(known_true != known_pred).item()

        # 计算未知类别的统计
        unknown_true = y_true[is_true_unk]
        unknown_pred = y_pred[is_true_unk]
        self.U_TP += torch.sum(unknown_true == unknown_pred).item()
        self.U_FP += torch.sum(is_pred_unk & ~is_true_unk).item()
        self.U_FN += torch.sum(~is_pred_unk & is_true_unk).item()

    def compute_k_acc(self):
        """
        计算已知类别的准确率 (k_acc)
        """
        total_known = self.K_TP + self.K_FP
        if total_known == 0:
            return 0.0
        return self.K_TP / total_known

    def compute_u_acc(self):
        """
        计算未知类别的准确率 (u_acc)
        """
        total_unknown = self.U_TP + self.U_FP
        if total_unknown == 0:
            return 0.0
        return self.U_TP / total_unknown

    def compute_u_recall(self):
        """
        计算未知类别的召回率 (u_recall)
        """
        total_unknown = self.U_TP + self.U_FN
        if total_unknown == 0:
            return 0.0
        return self.U_TP / total_unknown

    def compute_u_f1(self):
        """
        计算未知类别的 F1 分数 (u_f1)
        """
        precision = self.U_TP / (self.U_TP + self.U_FP) if (self.U_TP + self.U_FP) > 0 else 0.0
        recall = self.compute_u_recall()
        if (precision + recall) == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def get_metrics(self):
        """
        返回所有指标的计算结果
        """
        metrics = {}
        if 'k_acc' in self.metric:
            metrics['k_acc'] = self.compute_k_acc()
        if 'u_acc' in self.metric:
            metrics['u_acc'] = self.compute_u_acc()
        if 'u_recall' in self.metric:
            metrics['u_recall'] = self.compute_u_recall()
        if 'u_f1' in self.metric:
            metrics['u_f1'] = self.compute_u_f1()
        return metrics

    def __str__(self):
        """
        打印所有指标的计算结果
        """
        s = ""
        metrics = self.get_metrics()
        for metric_name, value in metrics.items():
            s += f"{metric_name}: {value:.4f}\n"
        return s


if __name__ == '__main__':
    # 假设未知类别为 0
    unk_class = [0]
    metric = Metric(unk_class)

    # 假设有一批真实标签和预测标签
    y_true = torch.tensor([1, 2, 0, 1, 0])
    y_pred = torch.tensor([1, 2, 1, 0, 0])

    # 更新统计
    metric(y_true, y_pred)

    # 打印指标
    metric.print_metrics()
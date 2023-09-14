from scipy.spatial.distance import directed_hausdorff
from collections import *
import pandas as pd
from statistics import mean
from torch.utils.tensorboard import SummaryWriter

class MetricsStatistics:
    def __init__(self, save_dir="./results/"):
        self.epsilon = 1e-6
        self.func_dct = {
            "Precision": self.cal_precision,
            "Recall": self.cal_recall,
            "Specificity": self.cal_specificity,
            "Jaccard": self.cal_jaccard_index,
            "Dice": self.cal_dice,
            "Hausdorff": self.cal_hausdorff
        }
        self.save_dir = save_dir
        self.metric_values = defaultdict(list) # check epoch 临时用
        self.metric_epochs = defaultdict(list) # 保存了指定epoch的各样本平均值
        self.summary_writer = SummaryWriter(log_dir=save_dir)

    def cal_epoch_metric(self, metrics, label_type, label, pred): # 计算并保存样本指标
        for x in metrics:self.metric_values["{}-{}".format(x, label_type)].append(self.func_dct[x](label, pred))

    def record_result(self, epoch):
        self.metric_epochs["epoch"].append(epoch)
        for k, v in self.metric_values.items():
            self.summary_writer.add_scalar(k, mean(v), epoch)
            self.metric_epochs[k].append(mean(v))
        pd.DataFrame(self.metric_epochs).to_excel("{}/metrics_statistics.xlsx".format(self.save_dir), index=False)
        self.metric_values.clear()
    
    def close(self):
        self.summary_writer.close()

    def cal_confusion_matrix(self, pred, label):
        TP = ((pred == 1) & (label == 1)).sum().item()
        FP = ((pred == 0) & (label == 1)).sum().item()
        FN = ((pred == 1) & (label == 0)).sum().item()
        TN = ((pred == 0) & (label == 0)).sum().item()
        return TP, FP, FN, TN

    def cal_precision(self, pred, label):
        TP, FP, FN, TN = self.cal_confusion_matrix(pred, label)
        return TP / (TP + FP + self.epsilon)

    def cal_recall(self, pred, label):
        TP, FP, FN, TN = self.cal_confusion_matrix(pred, label)
        return TP / (TP + FN + self.epsilon)

    def cal_specificity(self, pred, label):
        TP, FP, FN, TN = self.cal_confusion_matrix(pred, label)
        return TN / (TN + FP + self.epsilon)

    def cal_jaccard_index(self, pred, label):
        intersection = (pred & label).sum().item()
        union = (pred | label).sum().item()
        jaccard_index = intersection / (union + self.epsilon)
        return jaccard_index

    def cal_dice(self, pred, label):
        intersection = (pred & label).sum().item()
        union = pred.sum().item() + label.sum().item()
        dice = 2 * intersection / (union + self.epsilon)
        return dice

    def cal_hausdorff(self, pred, label):
        array1 = pred.cpu().numpy()
        array2 = label.cpu().numpy()
        dist1 = directed_hausdorff(array1, array2)[0]
        dist2 = directed_hausdorff(array2, array1)[0]
        hausdorff_dist = max(dist1, dist2)
        return hausdorff_dist
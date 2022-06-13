# 本文件生成若干数据集，并且包装成类
import numpy as np
from sklearn import datasets


class MyDataset(object):
    sklearn_datasets = ["iris", "wine"]
    available_datasets = sklearn_datasets
    
    def __init__(self, name):
        super(MyDataset, self).__init__()
        assert name in MyDataset.available_datasets
        global datasets
        if name in MyDataset.sklearn_datasets:
            dataset = getattr(datasets, f"load_{name}")()
            self.X = dataset.data
            self.y = dataset.target
            self.feature_names = dataset.feature_names
            self.label_names = dataset.target_names
    
    def get_data(self):
        return self.X, self.y, self.feature_names, self.label_names


if __name__ == '__main__':
    iris = MyDataset(name="iris")
    X, y, feature_names, label_names = iris.get_data()

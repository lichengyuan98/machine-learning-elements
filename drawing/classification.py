import numpy as np
import matplotlib.pyplot as plt

def draw_dicision_rigion(X, clf, step=0.1, ax=None):
    """
    利用分类器clf绘制样本X范围内的决策边界
    :param X: 样本数据[N, dim=2]，N为样本数量，dim为特征维度
    :type X: ndarray
    :param clf: 分类器
    :type clf: Any
    :param step: 区域划分精细程度，越小分界面越光滑
    :type step: float
    :param ax: 画布
    :type ax: matplotlib.Axes
    :return: 决策边界图
    :rtype: None
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.contourf(xx, yy, Z, alpha=0.4)
        plt.show()
    else:
        ax.contourf(xx, yy, Z, alpha=0.4)
    return None


def draw_samples(X, labels, ax=None):
    """
    绘制样本点
    :param X: 样本数据[N, dim=2]，N为样本数量，dim为特征维度
    :type X: np.ndarray
    :param labels: 样本数据标签[N]， N为样本数量
    :type labels: np.ndarray
    :param ax: 画布
    :type ax: matplotlib.Axes
    :return: 样本散点图
    :rtype: None
    """
    label_types = list(set(labels))
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for type in label_types:
            ax.scatter(X[labels == type, 0], X[labels == type, 1], label=f"class {type}", alpha=0.4, edgecolors="black")
        plt.legend()
        plt.show()
    else:
        for type in label_types:
            ax.scatter(X[labels == type, 0], X[labels == type, 1], label=f"class {type}", alpha=0.4, edgecolors="black")
    return None
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from einops import rearrange

matplotlib.use("TKAgg")

# %% 建立数据集
x_ticks = 51
x = np.linspace(-2, 2, x_ticks)
y = x
X, Y = np.meshgrid(x, y)
Z = X * np.exp(-X ** 2 - Y ** 2)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2)
plt.show(block=True)

# %% 数据集维度变化
dataset = np.stack([X, Y, Z], axis=0)
dataset = rearrange(dataset, "dim X Y -> (X Y) dim")
data = dataset[:, :2]
labels = dataset[:, -1]
# %% SVC
from sklearn.svm import SVR

svr = SVR(C=1, epsilon=0., # C为模型复杂度的正则系数，epsilon为支撑向量的最大偏离值
          kernel="rbf")
svr.fit(data, labels)
# 预测
label_pred = svr.predict(data).reshape(-1, 1)
dataset_pred = np.concatenate([data, label_pred], axis=1)
dataset_pred = rearrange(dataset_pred, "(X Y) dim -> dim X Y", X=x_ticks)
# 绘图，对比原始数据和预测数据
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(dataset_pred[0, :, :], dataset_pred[1, :, :], dataset_pred[2, :, :],
           color="k", s=0.1)
ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, lw=0.5, color="r")
plt.show(block=True)
# %% NuSVR
from sklearn.svm import NuSVR

nusvr = NuSVR(C=1, nu=1, # C为正则系数，Nu为用于支持向量的样本比例
              kernel="rbf")
nusvr.fit(data, labels)
# 预测
label_pred = nusvr.predict(data).reshape(-1, 1)
dataset_pred = np.concatenate([data, label_pred], axis=1)
dataset_pred = rearrange(dataset_pred, "(X Y) dim -> dim X Y", X=x_ticks)
# 绘图，对比原始数据和预测数据
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(dataset_pred[0, :, :], dataset_pred[1, :, :], dataset_pred[2, :, :],
           color="k", s=0.1)
ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, lw=0.5, color="r")
plt.show(block=True)
#%% LinearSVR （利用多项式特征进行拟合）
from sklearn.svm import LinearSVR
from sklearn.preprocessing import PolynomialFeatures # 使用多项式特征
polyfeatures = PolynomialFeatures(degree = 4, interaction_only = False, include_bias = True)
data_poly = polyfeatures.fit_transform(data)


linearsvr = LinearSVR(C=1, epsilon=0)
linearsvr.fit(data_poly, labels)
# 预测
label_pred = linearsvr.predict(data_poly).reshape(-1, 1)
dataset_pred = np.concatenate([data_poly[:, 1:3], label_pred], axis=1)
dataset_pred = rearrange(dataset_pred, "(X Y) dim -> dim X Y", X=x_ticks)
# 绘图，对比原始数据和预测数据
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(dataset_pred[0, :, :], dataset_pred[1, :, :], dataset_pred[2, :, :],
           color="k", s=0.1)
ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, lw=0.5, color="r")
plt.show(block=True)
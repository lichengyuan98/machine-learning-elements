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
# %%
from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=100, max_depth=10,
                                     bootstrap=True, max_samples=0.5, max_features=0.5,
                                     random_state=0)
rf_regressor.fit(data, labels)
# 预测
label_pred = rf_regressor.predict(data).reshape(-1, 1)
dataset_pred = np.concatenate([data, label_pred], axis=1)
dataset_pred = rearrange(dataset_pred, "(X Y) dim -> dim X Y", X=x_ticks)
# 绘图，对比原始数据和预测数据
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(dataset_pred[0, :, :], dataset_pred[1, :, :], dataset_pred[2, :, :],
           color="k", s=0.1)
ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, lw=0.5, color="r")
plt.show(block=True)
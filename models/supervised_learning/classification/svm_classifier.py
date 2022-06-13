# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC, LinearSVC, NuSVC
from drawing.classification import draw_samples, draw_dicision_rigion

# %% 准备测试数据
from sklearn.datasets import make_classification
X, labels = make_classification(n_samples=100,
                                n_features=2, n_informative=2, n_repeated=0, n_redundant=0,
                                n_classes=3, n_clusters_per_class=1,
                                random_state=0)
draw_samples(X, labels=labels)
# %% SVC
svc = SVC(C=0.9, kernel="rbf")
svc.fit(X, labels)

fig, ax = plt.subplots(1,1)
draw_dicision_rigion(X, svc,step=0.05, ax=ax)
draw_samples(X, labels, ax)
plt.show()
#%% LinearSVC
linearsvc = LinearSVC(C=0.9)
linearsvc.fit(X, labels)

fig, ax = plt.subplots(1,1)
draw_dicision_rigion(X, linearsvc,step=0.05, ax=ax)
draw_samples(X, labels, ax)
plt.show()
#%% NuSVC
nusvc = NuSVC(nu=0.01, kernel="rbf")
nusvc.fit(X, labels)

fig, ax = plt.subplots(1,1)
draw_dicision_rigion(X, nusvc,step=0.05, ax=ax)
draw_samples(X, labels, ax)
plt.show()


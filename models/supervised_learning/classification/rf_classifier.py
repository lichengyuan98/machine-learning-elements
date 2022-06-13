# %%
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from drawing.classification import draw_samples, draw_dicision_rigion

# %% 准备测试数据

X, labels = make_classification(n_samples=400,
                                n_features=2, n_informative=2, n_redundant=0, n_repeated=0,
                                n_classes=4, n_clusters_per_class=1,
                                random_state=0)
draw_samples(X, labels)

#%%
rf_classifier = RandomForestClassifier(n_estimators=100, criterion="entropy", max_depth=10, # 树的参数和分裂方法
                                       # bootstrap=True, max_features="sqrt", max_samples=0.8,# 数据采样方法
                                       bootstrap=False,# 数据采样方法
                                       random_state=0)
rf_classifier.fit(X, labels)

fig, ax = plt.subplots(1,1)
draw_dicision_rigion(X, rf_classifier, ax=ax)
draw_samples(X, labels, ax=ax)
plt.show()
print(rf_classifier.score(X, labels))
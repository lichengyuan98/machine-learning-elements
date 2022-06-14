# %% 导入包
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from drawing.classification import draw_dicision_rigion, draw_samples

# %% 创建数据
X, labels = make_classification(n_samples=400, #
                                n_features=2, n_informative=2, n_redundant=0, n_repeated=0,
                                n_classes=4, n_clusters_per_class=1,
                                random_state=0)
draw_samples(X, labels)
# %% XGBoost
xgb_classifier = XGBClassifier(n_estimators=30, max_depth=20, min_child_weight=1,  # 树特征
                               learning_rate=0.2, gamma=0., reg_lambda=1,  # 学习方法
                               subsample=1, colsample_bytree=1,  # 采样方法
                               seed=0)
xgb_classifier.fit(X, labels)

fig, ax = plt.subplots(1, 1)
draw_dicision_rigion(X, xgb_classifier, step=0.01, ax=ax)
draw_samples(X, labels, ax=ax)
plt.show()
print(xgb_classifier.score(X, labels))
# %% Adaboost
adb_classifier = AdaBoostClassifier(n_estimators=100,
                                    base_estimator=DecisionTreeClassifier(max_depth=3),
                                    random_state=0)
adb_classifier.fit(X, labels)

fig, ax = plt.subplots(1, 1)
draw_dicision_rigion(X, adb_classifier, step=0.1, ax=ax)
draw_samples(X, labels, ax=ax)
plt.show()
print(adb_classifier.score(X, labels))
# %%  GDB
gdb_classifier = GradientBoostingClassifier(n_estimators=100, max_depth=10,
                                            subsample=0.8, max_features=0.9,
                                            random_state=0)
gdb_classifier.fit(X, labels)

fig, ax = plt.subplots(1, 1)
draw_dicision_rigion(X, gdb_classifier, step=0.1, ax=ax)
draw_samples(X, labels, ax=ax)
plt.show()
print(gdb_classifier.score(X, labels))

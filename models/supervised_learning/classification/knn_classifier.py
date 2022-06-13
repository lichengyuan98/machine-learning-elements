#%%
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from drawing.classification import draw_samples, draw_dicision_rigion

# %% 准备测试数据

X, labels = make_classification(n_samples=400,
                                n_features=2, n_informative=2, n_redundant=0, n_repeated=0,
                                n_classes=4, n_clusters_per_class=1,
                                random_state=0)
draw_samples(X, labels)

#%%

knn_classifier = KNeighborsClassifier(n_neighbors=1)
knn_classifier.fit(X, labels)

fig, ax = plt.subplots(1,1)
draw_dicision_rigion(X, knn_classifier, ax=ax)
draw_samples(X, labels, ax=ax)
plt.show()
print(knn_classifier.score(X, labels))

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import mglearn

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

 

iris_dataset = load_iris() #鸢尾花数据集

 

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

 #数据拆分，最佳比例是数据集：测试集 = 3：1

 

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', 

	hist_kwds={"bins": 20}, s=60, alpha=.8, cmap=mglearn.cm3)  #展示散点图矩阵

 

knn = KNeighborsClassifier(n_neighbors=1) #knn对算法进行了封装，包含了模型构建算法与预测算法

 

knn.fit(X_train, y_train) #构建模型

 

X_new = np.array([[5, 2.9, 1, 0.2]])

prediction = knn.predict(X_new)

print("Prediction: {}".format(prediction))

print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))

 

y_pred = knn.predict(X_test)

print("Test set predictions:\n {}".format(y_pred))

print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

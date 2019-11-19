from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
dataset = load_boston()
x_data = dataset.data # 导入所有特征变量
y_data = dataset.target # 导入目标值（房价）
name_data = dataset.feature_names #导入特征名

#选取所有的样本作为测试样本和训练样本
x_train=x_data
y_train=y_data
x_test=x_data
y_test=y_data

#采用最大最小标准化进行数据归一化处理
from sklearn import preprocessing
#分别初始化对特征和目标值的标准化器
min_max_scaler = preprocessing.MinMaxScaler()
#分别对训练和测试数据的特征以及目标值进行标准化处理
x_train_train=min_max_scaler.fit_transform(x_train)
x_train_test=min_max_scaler.fit_transform(x_test)
y_train=min_max_scaler.fit_transform(y_train.reshape(-1,1))#reshape(-1,1)指将它转化为1列，行自动确定
y_test=min_max_scaler.fit_transform(y_test.reshape(-1,1))#reshape(-1,1)指将它转化为1列，行自动确定

#使用线性回归模型LinearRegression对波士顿房价数据进行训练及预测
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
#使用训练数据进行参数估计
lr.fit(x_train,y_train)
#回归预测
lr_y_predict=lr.predict(x_test)

from sklearn.metrics import r2_score
score = r2_score(y_test, lr_y_predict)
print(score)


#输出得分
0.7406426641094094

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
from dateutil import relativedelta
import scipy.stats as sci
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import NuSVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


df = pd.read_excel("load.xlsx")
df['DATA_TIME'] = pd.to_datetime(df['DATA_TIME'])

X = df.iloc[:, 1:8]
Y = df.iloc[:, 9:10]

#随机划分训练集和测试集
train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.3)

standard_scaler_x = preprocessing.MinMaxScaler()
standard_scaler_y = preprocessing.MinMaxScaler()

#将属性缩放到0-1之间
train_x = standard_scaler_x.fit_transform(train_x)
train_y = standard_scaler_y.fit_transform(train_y).ravel()

test_x = standard_scaler_x.transform(test_x)
test_y = standard_scaler_y.transform(test_y).ravel()

def svm_cross_validation(x, y):
    model = NuSVR(kernel='rbf')
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs = 8, verbose=1)
    grid_search.fit(x, y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = NuSVR(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'])
    model.fit(x, y)
    return model

model_svm = svm_cross_validation(train_x, train_y)

predict_svm = model_svm.predict(test_x)

print("SVM Mean Absolute Error: " + str(mean_absolute_error(predict_svm, test_y)))

def plot_result(label, predict_y):
    plt.plot(range(label.size), label, color='navy')
    plt.plot(range(label.size), predict_y, color='red')
    plt.show()

print("反归一化")
test_yy = np.array(test_y).reshape(1, -1)
origin_test_y = standard_scaler_y.inverse_transform(test_yy).ravel()
predict_yy = np.array(predict_svm).reshape(1, -1)
origin_predict_y = standard_scaler_y.inverse_transform(predict_yy).ravel()

print("计算准确率误差")
#print("Mean Absolute Error: " + str(mean_absolute_error(predict_y, test_y)))
error = 1 - mean_absolute_error(predict_svm, test_y)
print(error)
plot_result(origin_predict_y[0:50], origin_test_y[0:50])

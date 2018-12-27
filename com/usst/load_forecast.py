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

#font = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc')
#matplotlib.rcParams['font.family'] = ['WenQuanYi Micro Hei']
#matplotlib.rcParams['axes.unicode_minus'] = False

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

model_xgb = XGBRegressor()
#model_svm = svm_cross_validation(train_x, train_y)

model_xgb.fit(train_x, train_y, verbose=False)

predict_xgb = model_xgb.predict(test_x)
#predict_svm = model_svm.predict(test_x)

print("Xgboost Mean Absolute Error: " + str(mean_absolute_error(predict_xgb, test_y)))
#print("SVM Mean Absolute Error: " + str(mean_absolute_error(predict_svm, test_y)))

def plot_result(label, predict_y):
    plt.plot(range(label.size), label, color='navy')
    plt.plot(range(label.size), predict_y, color='red')
    plt.show()

plot_result(predict_xgb, test_y)
#plot_result(predict_svm, test_y)
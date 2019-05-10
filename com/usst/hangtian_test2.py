# -*- coding: utf-8 -*-
import cx_Oracle
import datetime
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sqlalchemy import create_engine
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
# conn = cx_Oracle.connect('system', 'tiger', '192.168.1.114:1521/orcl1')
# cursor = conn.cursor()
#相对误差
def precession(label, predict_y):
    e = np.sum(np.abs(label - predict_y)) / np.sum(np.abs(label))
    print(1 - e)
def plot_result(label, predict_y):
    plt.plot(range(label.size), label, color='navy')
    plt.plot(range(label.size), predict_y, color='red')
    plt.show()

df = pd.read_excel("sheet_historyPower2.xlsx")


#随机划分训练集和测试集
X = pd.concat([df.iloc[:,1:3],df.iloc[:,6:10],df.iloc[:,11:12]],axis = 1)
train_x = X.iloc[1:590, 0:]
train_y = df.iloc[1:590, 10:11]
test_x = X.iloc[590:, 0:]
test_y = df.iloc[590:, 10:11]
print(test_x)

standard_scaler_x = preprocessing.MinMaxScaler()
standard_scaler_y = preprocessing.MinMaxScaler()

#将属性缩放到0-1之间
train_x = standard_scaler_x.fit_transform(train_x)
train_y = standard_scaler_y.fit_transform(train_y).ravel()

test_x = standard_scaler_x.transform(test_x)
test_y = standard_scaler_y.transform(test_y).ravel()


model_xgb = XGBRegressor()
#model_svm = svm_cross_validation(train_x, train_y)

model_xgb.fit(train_x, train_y, verbose=False)

predict_xgb = model_xgb.predict(test_x)
#predict_svm = model_svm.predict(test_x)

print("反归一化")
test_yy = np.array(test_y).reshape(1, -1)
origin_test_y = standard_scaler_y.inverse_transform(test_yy).ravel()
predict_yy = np.array(predict_xgb).reshape(1, -1)
origin_predict_y = standard_scaler_y.inverse_transform(predict_yy).ravel()

print("准确率")
precession(origin_test_y, origin_predict_y)

plot_result(origin_predict_y[:], origin_test_y[:])
#
# conn.commit()
# cursor.close()
# conn.close()
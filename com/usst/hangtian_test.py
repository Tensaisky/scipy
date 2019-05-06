# -*- coding: utf-8 -*-
import cx_Oracle
import datetime
import os

# os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
# conn = cx_Oracle.connect('system', 'tiger', '192.168.1.108:1521/orcl1')
# cursor = conn.cursor()

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
from sqlalchemy import create_engine
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

def precession(label, predict_y):
    e = np.sum(np.abs(label - predict_y)) / np.sum(np.abs(label))
    print(1 - e)
    
def plot_result(label, predict_y):
    plt.plot(range(label.size), label, color='navy')
    plt.plot(range(label.size), predict_y, color='red')
    plt.show()

engine = create_engine('oracle+cx_oracle://system:tiger@192.168.1.114:1521/orcl1')

df = pd.read_sql('SELECT * FROM SYSTEM."sheet_historyPower"',engine)
# df = pd.read_excel("sheet_historyPower.xlsx")

X = pd.concat([df.iloc[:,1:7],df.iloc[:,11:12]],axis = 1)
Y = df.iloc[:, 10:11]

train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.08)

train_x = X.iloc[1:420, :]
train_y = df.iloc[1:420, 10:11]
test_x = X.iloc[420:, :]
test_y = df.iloc[420:, 10:11]
print(test_x)

# print(test_x)
# print(test_y)

standard_scaler_x = preprocessing.MinMaxScaler()
standard_scaler_y = preprocessing.MinMaxScaler()

#将属性缩放到0-1之间
train_x = standard_scaler_x.fit_transform(train_x)
train_y = standard_scaler_y.fit_transform(train_y).ravel()

test_x = standard_scaler_x.transform(test_x)
test_y = standard_scaler_y.transform(test_y).ravel()


model_xgb = XGBRegressor()

model_xgb.fit(train_x, train_y, verbose=False)

predict_xgb = model_xgb.predict(test_x)

print("误差: " + str(1-mean_absolute_error(predict_xgb, test_y)))

# print("反归一化")
test_yy = np.array(test_y).reshape(1, -1)
origin_test_y = standard_scaler_y.inverse_transform(test_yy).ravel()
predict_yy = np.array(predict_xgb).reshape(1, -1)
origin_predict_y = standard_scaler_y.inverse_transform(predict_yy).ravel()

print(origin_predict_y)
print(type(origin_predict_y))

print("准确率")
precession(origin_test_y, origin_predict_y)

plot_result(origin_predict_y[0:], origin_test_y[0:])

# conn.commit()
# cursor.close()
# conn.close()
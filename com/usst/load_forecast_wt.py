import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
from dateutil import relativedelta
import scipy.stats as sci
import pywt
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import NuSVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

#font = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc')
#matplotlib.rcParams['font.family'] = ['WenQuanYi Micro Hei']
#matplotlib.rcParams['axes.unicode_minus'] = False
def printc(str, color="35"):
    print("\n\033[1;"+ color +"m" + str + "\033[0m")

def plot_result(label, predict_y):
    plt.plot(range(label.size), label, color='navy')
    plt.plot(range(label.size), predict_y, color='red')
    plt.show()

def precession(label, predict_y):
    e = np.sum(np.abs(label - predict_y)) / np.sum(np.abs(label))
    print(1 - e)
    
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

df = pd.read_excel("load.xlsx")
df['DATA_TIME'] = pd.to_datetime(df['DATA_TIME'])
df = df.iloc[:,0:]
df_DATA_TIME = df.iloc[:, 0:1]
df_day_of_week = df.iloc[:,7:8]
df_holiday = df.iloc[:,8:9]
df_load = df.iloc[:,9:10]

printc("用db5小波进行yes_ave 3层分解")
yes_ave = np.array(df.iloc[:,1:2]).reshape(1,-1)
# print(ML.shape)
yes_ave_wt = pywt.wavedec(yes_ave, 'db5', level=3)
yes_ave_cA3, yes_ave_cD3, yes_ave_cD2, yes_ave_cD1 = yes_ave_wt

yes_ave_cA3 = np.array(yes_ave_cA3).flatten()
yes_ave_cD3 = np.array(yes_ave_cD3).flatten()
yes_ave_cD2 = np.array(yes_ave_cD2).flatten()
yes_ave_cD1 = np.array(yes_ave_cD1).flatten()

yes_ave_cA3_ = pywt.upcoef('a', yes_ave_cA3, 'db5', level=3, take=311)
yes_ave_cD1_ = pywt.upcoef('d', yes_ave_cD1, 'db5', level=1, take=311)
yes_ave_cD2_ = pywt.upcoef('d', yes_ave_cD2, 'db5', level=2, take=311)
yes_ave_cD3_ = pywt.upcoef('d', yes_ave_cD3, 'db5', level=3, take=311)

yes_ave_cA3_ = yes_ave_cA3_.tolist()
yes_ave_cD1_ = yes_ave_cD1_.tolist()
yes_ave_cD2_ = yes_ave_cD2_.tolist()
yes_ave_cD3_ = yes_ave_cD3_.tolist()

printc("用db5小波进行yes_load 3层分解")
yes_load = np.array(df.iloc[:,2:3]).reshape(1,-1)
# print(ML.shape)
yes_load_wt = pywt.wavedec(yes_load, 'db5', level=3)
yes_load_cA3, yes_load_cD3, yes_load_cD2, yes_load_cD1 = yes_load_wt

yes_load_cA3 = np.array(yes_load_cA3).flatten()
yes_load_cD3 = np.array(yes_load_cD3).flatten()
yes_load_cD2 = np.array(yes_load_cD2).flatten()
yes_load_cD1 = np.array(yes_load_cD1).flatten()

yes_load_cA3_ = pywt.upcoef('a', yes_load_cA3, 'db5', level=3, take=311)
yes_load_cD1_ = pywt.upcoef('d', yes_load_cD1, 'db5', level=1, take=311)
yes_load_cD2_ = pywt.upcoef('d', yes_load_cD2, 'db5', level=2, take=311)
yes_load_cD3_ = pywt.upcoef('d', yes_load_cD3, 'db5', level=3, take=311)

yes_load_cA3_ = yes_load_cA3_.tolist()
yes_load_cD1_ = yes_load_cD1_.tolist()
yes_load_cD2_ = yes_load_cD2_.tolist()
yes_load_cD3_ = yes_load_cD3_.tolist()

printc("用db5小波进行week_before 3层分解")
week_before = np.array(df.iloc[:,3:4]).reshape(1,-1)
# print(ML.shape)
week_before_wt = pywt.wavedec(week_before, 'db5', level=3)
week_before_cA3, week_before_cD3, week_before_cD2, week_before_cD1 = week_before_wt

week_before_cA3 = np.array(week_before_cA3).flatten()
week_before_cD3 = np.array(week_before_cD3).flatten()
week_before_cD2 = np.array(week_before_cD2).flatten()
week_before_cD1 = np.array(week_before_cD1).flatten()

week_before_cA3_ = pywt.upcoef('a', week_before_cA3, 'db5', level=3, take=311)
week_before_cD1_ = pywt.upcoef('d', week_before_cD1, 'db5', level=1, take=311)
week_before_cD2_ = pywt.upcoef('d', week_before_cD2, 'db5', level=2, take=311)
week_before_cD3_ = pywt.upcoef('d', week_before_cD3, 'db5', level=3, take=311)

week_before_cA3_ = week_before_cA3_.tolist()
week_before_cD1_ = week_before_cD1_.tolist()
week_before_cD2_ = week_before_cD2_.tolist()
week_before_cD3_ = week_before_cD3_.tolist()

printc("用db5小波进行yes_max 3层分解")
yes_max = np.array(df.iloc[:,4:5]).reshape(1,-1)
# print(ML.shape)
yes_max_wt = pywt.wavedec(yes_max, 'db5', level=3)
yes_max_cA3, yes_max_cD3, yes_max_cD2, yes_max_cD1 = yes_max_wt

yes_max_cA3 = np.array(yes_max_cA3).flatten()
yes_max_cD3 = np.array(yes_max_cD3).flatten()
yes_max_cD2 = np.array(yes_max_cD2).flatten()
yes_max_cD1 = np.array(yes_max_cD1).flatten()

yes_max_cA3_ = pywt.upcoef('a', yes_max_cA3, 'db5', level=3, take=311)
yes_max_cD1_ = pywt.upcoef('d', yes_max_cD1, 'db5', level=1, take=311)
yes_max_cD2_ = pywt.upcoef('d', yes_max_cD2, 'db5', level=2, take=311)
yes_max_cD3_ = pywt.upcoef('d', yes_max_cD3, 'db5', level=3, take=311)

yes_max_cA3_ = yes_max_cA3_.tolist()
yes_max_cD1_ = yes_max_cD1_.tolist()
yes_max_cD2_ = yes_max_cD2_.tolist()
yes_max_cD3_ = yes_max_cD3_.tolist()

printc("用db5小波进行temperature 3层分解")
temperature = np.array(df.iloc[:,5:6]).reshape(1,-1)
# print(ML.shape)
temperature_wt = pywt.wavedec(temperature, 'db5', level=3)
temperature_cA3, temperature_cD3, temperature_cD2, temperature_cD1 = temperature_wt

temperature_cA3 = np.array(temperature_cA3).flatten()
temperature_cD3 = np.array(temperature_cD3).flatten()
temperature_cD2 = np.array(temperature_cD2).flatten()
temperature_cD1 = np.array(temperature_cD1).flatten()

temperature_cA3_ = pywt.upcoef('a', temperature_cA3, 'db5', level=3, take=311)
temperature_cD1_ = pywt.upcoef('d', temperature_cD1, 'db5', level=1, take=311)
temperature_cD2_ = pywt.upcoef('d', temperature_cD2, 'db5', level=2, take=311)
temperature_cD3_ = pywt.upcoef('d', temperature_cD3, 'db5', level=3, take=311)

temperature_cA3_ = temperature_cA3_.tolist()
temperature_cD1_ = temperature_cD1_.tolist()
temperature_cD2_ = temperature_cD2_.tolist()
temperature_cD3_ = temperature_cD3_.tolist()

printc("用db5小波进行dew 3层分解")
dew = np.array(df.iloc[:,6:7]).reshape(1,-1)
# print(ML.shape)
dew_wt = pywt.wavedec(dew, 'db5', level=3)
dew_cA3, dew_cD3, dew_cD2, dew_cD1 = dew_wt

dew_cA3 = np.array(dew_cA3).flatten()
dew_cD3 = np.array(dew_cD3).flatten()
dew_cD2 = np.array(dew_cD2).flatten()
dew_cD1 = np.array(dew_cD1).flatten()

dew_cA3_ = pywt.upcoef('a', dew_cA3, 'db5', level=3, take=311)
dew_cD1_ = pywt.upcoef('d', dew_cD1, 'db5', level=1, take=311)
dew_cD2_ = pywt.upcoef('d', dew_cD2, 'db5', level=2, take=311)
dew_cD3_ = pywt.upcoef('d', dew_cD3, 'db5', level=3, take=311)

dew_cA3_ = dew_cA3_.tolist()
dew_cD1_ = dew_cD1_.tolist()
dew_cD2_ = dew_cD2_.tolist()
dew_cD3_ = dew_cD3_.tolist()

printc("用db5小波进行load 3层分解")
load = np.array(df.iloc[:,9:10]).reshape(1,-1)
# print(ML.shape)
load_wt = pywt.wavedec(load, 'db5', level=3)
load_cA3, load_cD3, load_cD2, load_cD1 = load_wt

load_cA3 = np.array(load_cA3).flatten()
load_cD3 = np.array(load_cD3).flatten()
load_cD2 = np.array(load_cD2).flatten()
load_cD1 = np.array(load_cD1).flatten()

load_cA3_ = pywt.upcoef('a', load_cA3, 'db5', level=3, take=311)
load_cD1_ = pywt.upcoef('d', load_cD1, 'db5', level=1, take=311)
load_cD2_ = pywt.upcoef('d', load_cD2, 'db5', level=2, take=311)
load_cD3_ = pywt.upcoef('d', load_cD3, 'db5', level=3, take=311)

load_cA3_ = load_cA3_.tolist()
load_cD1_ = load_cD1_.tolist()
load_cD2_ = load_cD2_.tolist()
load_cD3_ = load_cD3_.tolist()

# 对每个频段进行预测
printc("cA3数据")
df_cA3 = pd.DataFrame({
        "昨日平均负载":list(yes_ave_cA3_),
        "昨日此刻负载":list(yes_load_cA3_),
        "周前负载":list(week_before_cA3_),
        "昨日最大负载":list(yes_max_cA3_),
        "温度":list(temperature_cA3_),
        "dew":list(dew_cA3_),
        # "最高温度cA3":list(HT_cA3_)
})
df_cA3.insert(0,'DATA_TIME',df_DATA_TIME)
df_cA3.insert(7,'day_of_week',df_day_of_week)
df_cA3.insert(8,'holiday',df_holiday)
df_cA3.insert(9,'load',load_cA3_);

printc("准备训练和预测数据：")
X_cA3 = df_cA3.iloc[:, 1:9]
Y_cA3 = df_cA3.iloc[:, 9:10]

#随机划分训练集和测试集
train_x_cA3,test_x_cA3,train_y_cA3,test_y_cA3=train_test_split(X_cA3,Y_cA3,test_size=0.3)
print(np.ndim(train_x_cA3))

printc("归一化")
standard_scaler_x_cA3 = preprocessing.MinMaxScaler()
standard_scaler_y_cA3 = preprocessing.MinMaxScaler()

#将属性缩放到0-1之间
train_x_cA3 = standard_scaler_x_cA3.fit_transform(train_x_cA3)
train_y_cA3 = standard_scaler_y_cA3.fit_transform(train_y_cA3).ravel()
test_x_cA3 = standard_scaler_x_cA3.transform(test_x_cA3)
test_y_cA3= standard_scaler_y_cA3.transform(test_y_cA3).ravel()

# model_xgb_cA3 = XGBRegressor()
# model_xgb_cA3.fit(train_x_cA3, train_y_cA3, verbose=False)
# predict_y_cA3 = model_xgb_cA3.predict(test_x_cA3)

model_svm = svm_cross_validation(train_x_cA3, train_y_cA3)
predict_y_cA3 = model_svm.predict(test_x_cA3)

printc("反归一化")
test_yy_cA3 = np.array(test_y_cA3).reshape(1, -1)
origin_test_y_cA3 = standard_scaler_y_cA3.inverse_transform(test_yy_cA3).ravel()
predict_yy_cA3 = np.array(predict_y_cA3).reshape(1, -1)
origin_predict_y_cA3=standard_scaler_y_cA3.inverse_transform(predict_yy_cA3).ravel()

printc("cD3数据")
df_cD3 = pd.DataFrame({
        "昨日平均负载":list(yes_ave_cD3_),
        "昨日此刻负载":list(yes_load_cD3_),
        "周前负载":list(week_before_cD3_),
        "昨日最大负载":list(yes_max_cD3_),
        "温度":list(temperature_cD3_),
        "dew":list(dew_cD3_),
        # "最高温度cD3":list(HT_cD3_)
})
df_cD3.insert(0,'DATA_TIME',df_DATA_TIME)
df_cD3.insert(7,'day_of_week',df_day_of_week)
df_cD3.insert(8,'holiday',df_holiday)
df_cD3.insert(9,'load',load_cD3_);

printc("准备训练和预测数据：")
X_cD3 = df_cD3.iloc[:, 1:9]
Y_cD3 = df_cD3.iloc[:, 9:10]

#随机划分训练集和测试集
train_x_cD3,test_x_cD3,train_y_cD3,test_y_cD3=train_test_split(X_cD3,Y_cD3,test_size=0.3)
print(np.ndim(train_x_cD3))

printc("归一化")
standard_scaler_x_cD3 = preprocessing.MinMaxScaler()
standard_scaler_y_cD3 = preprocessing.MinMaxScaler()

#将属性缩放到0-1之间
train_x_cD3 = standard_scaler_x_cD3.fit_transform(train_x_cD3)
train_y_cD3 = standard_scaler_y_cD3.fit_transform(train_y_cD3).ravel()
test_x_cD3 = standard_scaler_x_cD3.transform(test_x_cD3)
test_y_cD3= standard_scaler_y_cD3.transform(test_y_cD3).ravel()

# model_xgb_cD3 = XGBRegressor()
# model_xgb_cD3.fit(train_x_cD3, train_y_cD3, verbose=False)
# predict_y_cD3 = model_xgb_cD3.predict(test_x_cD3)

model_svm = svm_cross_validation(train_x_cD3, train_y_cD3)
predict_y_cD3 = model_svm.predict(test_x_cD3)

printc("反归一化")
test_yy_cD3 = np.array(test_y_cD3).reshape(1, -1)
origin_test_y_cD3 = standard_scaler_y_cD3.inverse_transform(test_yy_cD3).ravel()
predict_yy_cD3 = np.array(predict_y_cD3).reshape(1, -1)
origin_predict_y_cD3=standard_scaler_y_cD3.inverse_transform(predict_yy_cD3).ravel()

printc("cD2数据")
df_cD2 = pd.DataFrame({
        "昨日平均负载":list(yes_ave_cD2_),
        "昨日此刻负载":list(yes_load_cD2_),
        "周前负载":list(week_before_cD2_),
        "昨日最大负载":list(yes_max_cD2_),
        "温度":list(temperature_cD2_),
        "dew":list(dew_cD2_),
        # "最高温度cD2":list(HT_cD2_)
})
df_cD2.insert(0,'DATA_TIME',df_DATA_TIME)
df_cD2.insert(7,'day_of_week',df_day_of_week)
df_cD2.insert(8,'holiday',df_holiday)
df_cD2.insert(9,'load',load_cD2_);

printc("准备训练和预测数据：")
X_cD2 = df_cD2.iloc[:, 1:9]
Y_cD2 = df_cD2.iloc[:, 9:10]

#随机划分训练集和测试集
train_x_cD2,test_x_cD2,train_y_cD2,test_y_cD2=train_test_split(X_cD2,Y_cD2,test_size=0.3)
print(np.ndim(train_x_cD2))

printc("归一化")
standard_scaler_x_cD2 = preprocessing.MinMaxScaler()
standard_scaler_y_cD2 = preprocessing.MinMaxScaler()

#将属性缩放到0-1之间
train_x_cD2 = standard_scaler_x_cD2.fit_transform(train_x_cD2)
train_y_cD2 = standard_scaler_y_cD2.fit_transform(train_y_cD2).ravel()
test_x_cD2 = standard_scaler_x_cD2.transform(test_x_cD2)
test_y_cD2= standard_scaler_y_cD2.transform(test_y_cD2).ravel()

# model_xgb_cD2 = XGBRegressor()
# model_xgb_cD2.fit(train_x_cD2, train_y_cD2, verbose=False)
# predict_y_cD2 = model_xgb_cD2.predict(test_x_cD2)

model_svm = svm_cross_validation(train_x_cD2, train_y_cD2)
predict_y_cD2 = model_svm.predict(test_x_cD2)

printc("反归一化")
test_yy_cD2 = np.array(test_y_cD2).reshape(1, -1)
origin_test_y_cD2 = standard_scaler_y_cD2.inverse_transform(test_yy_cD2).ravel()
predict_yy_cD2 = np.array(predict_y_cD2).reshape(1, -1)
origin_predict_y_cD2=standard_scaler_y_cD2.inverse_transform(predict_yy_cD2).ravel()

printc("cD1数据")
df_cD1 = pd.DataFrame({
        "昨日平均负载":list(yes_ave_cD1_),
        "昨日此刻负载":list(yes_load_cD1_),
        "周前负载":list(week_before_cD1_),
        "昨日最大负载":list(yes_max_cD1_),
        "温度":list(temperature_cD1_),
        "dew":list(dew_cD1_),
        # "最高温度cD1":list(HT_cD1_)
})
df_cD1.insert(0,'DATA_TIME',df_DATA_TIME)
df_cD1.insert(7,'day_of_week',df_day_of_week)
df_cD1.insert(8,'holiday',df_holiday)
df_cD1.insert(9,'load',load_cD1_);

printc("准备训练和预测数据：")
X_cD1 = df_cD1.iloc[:, 1:9]
Y_cD1 = df_cD1.iloc[:, 9:10]

#随机划分训练集和测试集
train_x_cD1,test_x_cD1,train_y_cD1,test_y_cD1=train_test_split(X_cD1,Y_cD1,test_size=0.3)
print(np.ndim(train_x_cD1))

printc("归一化")
standard_scaler_x_cD1 = preprocessing.MinMaxScaler()
standard_scaler_y_cD1 = preprocessing.MinMaxScaler()

#将属性缩放到0-1之间
train_x_cD1 = standard_scaler_x_cD1.fit_transform(train_x_cD1)
train_y_cD1 = standard_scaler_y_cD1.fit_transform(train_y_cD1).ravel()
test_x_cD1 = standard_scaler_x_cD1.transform(test_x_cD1)
test_y_cD1= standard_scaler_y_cD1.transform(test_y_cD1).ravel()

# model_xgb_cD1 = XGBRegressor()
# model_xgb_cD1.fit(train_x_cD1, train_y_cD1, verbose=False)
# predict_y_cD1 = model_xgb_cD1.predict(test_x_cD1)

model_svm = svm_cross_validation(train_x_cD1, train_y_cD1)
predict_y_cD1 = model_svm.predict(test_x_cD1)

printc("反归一化")
test_yy_cD1 = np.array(test_y_cD1).reshape(1, -1)
origin_test_y_cD1 = standard_scaler_y_cD1.inverse_transform(test_yy_cD1).ravel()
predict_yy_cD1 = np.array(predict_y_cD1).reshape(1, -1)
origin_predict_y_cD1=standard_scaler_y_cD1.inverse_transform(predict_yy_cD1).ravel()

printc("叠加")
test_y = test_y_cA3 + test_y_cD1 + test_y_cD2 + test_y_cD3
origin_test_y = origin_test_y_cA3 + origin_test_y_cD1 + origin_predict_y_cD2 + origin_predict_y_cD3
origin_predict_y = origin_predict_y_cA3 + origin_predict_y_cD1 + origin_predict_y_cD2 +origin_predict_y_cD3
predict_y = predict_y_cA3 + predict_y_cD1 + predict_y_cD2 + predict_y_cD3

# printc("计算准确率误差")
# precession(test_y, predict_y)
printc("计算准确率误差")
#print("Mean Absolute Error: " + str(mean_absolute_error(predict_y, test_y)))
error = 1 - mean_absolute_error(predict_y, test_y)
print(error)

printc("结果可视化")
plot_result(test_y, predict_y)
plot_result(origin_test_y,origin_predict_y)
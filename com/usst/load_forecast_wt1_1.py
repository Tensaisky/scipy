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

# 输入输出都小波变换，高频预测高频，低频预测低频，叠加还原原始信号对比误差

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
    print("误差：")
    print(1 - e)

def plot_result(label, predict_y):
    plt.plot(range(label.size), label, color='navy')
    plt.plot(range(label.size), predict_y, color='red')
    plt.show()
    
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

# 小波分解（自己定义一个名称，x表格起始行，y表格终点行）
def py_wt(str,x,y):
    signal = np.array(df.iloc[:,x:y]).reshape(1, -1)
    signal_wt = pywt.wavedec(signal, 'db5', level=3)
    signal_cA3, signal_cD3, signal_cD2, signal_cD1 = signal_wt

    signal_cA3 = np.array(signal_cA3).flatten()
    signal_cD3 = np.array(signal_cD3).flatten()
    signal_cD2 = np.array(signal_cD2).flatten()
    signal_cD1 = np.array(signal_cD1).flatten()
    
    signal_cA3_ = pywt.upcoef('a', signal_cA3, 'db5', level=3, take=311)
    signal_cD1_ = pywt.upcoef('d', signal_cD1, 'db5', level=1, take=311)
    signal_cD2_ = pywt.upcoef('d', signal_cD2, 'db5', level=2, take=311)
    signal_cD3_ = pywt.upcoef('d', signal_cD3, 'db5', level=3, take=311)
    
    signal_cA3_ = signal_cA3_.tolist()
    signal_cD1_ = signal_cD1_.tolist()
    signal_cD2_ = signal_cD2_.tolist()
    signal_cD3_ = signal_cD3_.tolist()
    print("signal_cA3:")
    print(signal_cA3)
    
    df_signal = pd.DataFrame({
        str+"_cA3":list(signal_cA3_),
        str+"_cD3":list(signal_cD3_),
        str+"_cD2":list(signal_cD2_),
        str+"_cD1":list(signal_cD1_),
    })
    print("df_signal")
    print(df_signal)
    return df_signal
    

df = pd.read_excel("load.xlsx")
df['DATA_TIME'] = pd.to_datetime(df['DATA_TIME'])
df = df.iloc[:,0:]
df_DATA_TIME = df.iloc[:, 0:1]
df_day_of_week = df.iloc[:,7:8]
df_holiday = df.iloc[:,8:9]
df_load = df.iloc[:,9:10]

printc("用db5小波进行层分解")

yes_ave_wt = py_wt("yes_ave",1,2)
yes_load_wt = py_wt("yes_load",2,3)
week_before_wt = py_wt("week_before",3,4)
yes_max_wt = py_wt("yes_max",4,5)
temperature_wt = py_wt("temperature",5,6)
dew_wt = py_wt("dew",6,7)
#load_wt = py_wt("load",9,10)

# 行对对齐
#df_wt = pd.concat([df_DATA_TIME,yes_ave_wt,yes_load_wt,week_before_wt,yes_max_wt,temperature_wt,dew_wt,df_day_of_week,df_holiday,load_wt],axis=1)
df_wt = pd.concat([df_DATA_TIME,yes_ave_wt,yes_load_wt,week_before_wt,yes_max_wt,temperature_wt,dew_wt,df_day_of_week,df_holiday,df_load],axis=1)
print(df_wt.head())
#随机划分训练集和测试集
X=df_wt.iloc[:,1:27]
# print("训练数据")
# print(X)
Y=df_wt.iloc[:,27:28]
train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.3)
print(np.ndim(train_x))

#随机划分训练集和测试集
train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.3)

standard_scaler_x = preprocessing.MinMaxScaler()
standard_scaler_y = preprocessing.MinMaxScaler()

#将属性缩放到0-1之间
train_x = standard_scaler_x.fit_transform(train_x)
train_y = standard_scaler_y.fit_transform(train_y).ravel()

test_x = standard_scaler_x.transform(test_x)
test_y = standard_scaler_y.transform(test_y).ravel()


# model_xgb = XGBRegressor()
# model_xgb.fit(train_x, train_y, verbose=False)
# predict_xgb = model_xgb.predict(test_x)
#print("Xgboost Mean Absolute Error: " + str(1-mean_absolute_error(predict_xgb, test_y)))

model_svm = svm_cross_validation(train_x, train_y)
predict_svm = model_svm.predict(test_x)
print("SVM Mean Absolute Error: " + str(mean_absolute_error(predict_svm, test_y)))


print("反归一化")
test_yy = np.array(test_y).reshape(1, -1)
origin_test_y = standard_scaler_y.inverse_transform(test_yy).ravel()
predict_yy = np.array(predict_svm).reshape(1, -1)
origin_predict_y = standard_scaler_y.inverse_transform(predict_yy).ravel()

print("计算准确率误差")
precession(origin_test_y, origin_predict_y)

#plot_result(origin_predict_y, origin_test_y)
plot_result(origin_predict_y[0:50], origin_test_y[0:50])
import numpy as np
import pandas as pd
import matplotlib.pyplot as pWC
import pywt
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import NuSVR
from sklearn.metrics import mean_absolute_error

def printc(str, color="35"):
    print("\n\033[1;"+ color +"m" + str + "\033[0m")

def plot_resuWC(label, predict_y):
    pWC.plot(range(label.size), label, color='navy')
    pWC.plot(range(label.size), predict_y, color='red')
    pWC.show()

def precession(label, predict_y):
    e = np.sum(np.abs(label - predict_y)) / np.sum(np.abs(label))
    print(1 - e)

def svm_cross_validation(x, y):
    model = NuSVR(kernel='rbf')
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs =8, verbose=1)
    grid_search.fit(x, y)#训练模型
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = NuSVR(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'])
    model.fit(x, y)
    return model

df = pd.read_excel("caijiapo.xlsx")
df['发电量/kWh'] = df['发电量/kWh'].map(lambda x: x*10000)
printc("原始数据")
df = df.iloc[:,1:]
df_date = df.iloc[:, 0:1]

weather = df['天气状况'].str.extract('(.*)\((.*)℃~(.*)℃\)', expand=False)
weather.columns = ['日类型', '周前发电量', '最高温度']
weather['周前发电量'] = weather['周前发电量'].astype('float')
weather['最高温度'] = weather['最高温度'].astype('float')
printc("正则提取天气状况中的温度和日类型")
print(weather.head())

df = pd.concat([df.iloc[:,0:4], weather], axis=1)
printc("拼接负荷和天气数据")
print(df.head())

class_mapping = {label:idx for idx,label in enumerate(np.unique(df['日类型']))}
printc("对日类型分类并采用one_hot编码：")
print(class_mapping)

df['日类型'] = df['日类型'].map(class_mapping)
printc("处理完日类型之后")
df['日类型'] = df['日类型'].astype('float')
print(df.head())

printc("提取 日前发电量、日前最大负荷")
df['日前发电量/kWh'] = df['发电量/kWh'].shift(1)
df['日前最大负荷/MW'] = df['最大负荷/MW'].shift(1)
print(df.head())

printc("提取 周前发电量、周前最大负荷")
df['周前发电量/kWh'] = df['发电量/kWh'].shift(7)
df['周前最大负荷/MW'] = df['最大负荷/MW'].shift(7)
print(df.iloc[7:11, :])

printc("删除最大负荷和天气状况")
df = df.drop("最大负荷/MW", axis=1)
df = df.drop("天气状况", axis=1)
print(df.iloc[7:11, :])

printc("删除开始一周（由于开始一周没有前一周数据），重置索引")
df = df.iloc[7:, :]
df = df.reset_index(drop=True)
print(df.head())

# printc("生成Excel表")
# df.to_excel("cjp2.xlsx",sheet_name="xcjp",index=False,header=True)

##########################################################################
##########################################################################
printc("用db5小波进行日前发电量3层分解")
DAG = np.array(df.iloc[:,5:6]).reshape(1,-1)
# print(ML.shape)
DAG_wt = pywt.wavedec(DAG, 'db5', level=3)
DAG_cA3, DAG_cD3, DAG_cD2, DAG_cD1 = DAG_wt

DAG_cA3 = np.array(DAG_cA3).flatten()
DAG_cD3 = np.array(DAG_cD3).flatten()
DAG_cD2 = np.array(DAG_cD2).flatten()
DAG_cD1 = np.array(DAG_cD1).flatten()

DAG_cA3_ = pywt.upcoef('a', DAG_cA3, 'db5', level=3, take=311)
DAG_cD1_ = pywt.upcoef('d', DAG_cD1, 'db5', level=1, take=311)
DAG_cD2_ = pywt.upcoef('d', DAG_cD2, 'db5', level=2, take=311)
DAG_cD3_ = pywt.upcoef('d', DAG_cD3, 'db5', level=3, take=311)

DAG_cA3_ = DAG_cA3_.tolist()
DAG_cD1_ = DAG_cD1_.tolist()
DAG_cD2_ = DAG_cD2_.tolist()
DAG_cD3_ = DAG_cD3_.tolist()

printc("用db5小波进行周前发电量3层分解")
WC = np.array(df.iloc[:,4:5]).reshape(1,-1)
WC_wt = pywt.wavedec(WC, 'db5', level=3)
WC_cA3, WC_cD3, WC_cD2, WC_cD1 = WC_wt

WC_cA3 = np.array(WC_cA3).flatten()
WC_cD3 = np.array(WC_cD3).flatten()
WC_cD2 = np.array(WC_cD2).flatten()
WC_cD1 = np.array(WC_cD1).flatten()

WC_cA3_ = pywt.upcoef('a', WC_cA3, 'db5', level=3, take=311)
WC_cD1_ = pywt.upcoef('d', WC_cD1, 'db5', level=1, take=311)
WC_cD2_ = pywt.upcoef('d', WC_cD2, 'db5', level=2, take=311)
WC_cD3_ = pywt.upcoef('d', WC_cD3, 'db5', level=3, take=311)

WC_cA3_ = WC_cA3_.tolist()
WC_cD1_ = WC_cD1_.tolist()
WC_cD2_ = WC_cD2_.tolist()
WC_cD3_ = WC_cD3_.tolist()

# printc("用db5小波进行最高温度3层分解")
# HT = np.array(df.iloc[:,5:6]).reshape(1,-1)
# HT_wt = pywt.wavedec(HT, 'db5', level=3)
# HT_cA3, HT_cD3, HT_cD2, HT_cD1 = HT_wt
#
# HT_cA3 = np.array(HT_cA3).flatten()
# HT_cD3 = np.array(HT_cD3).flatten()
# HT_cD2 = np.array(HT_cD2).flatten()
# HT_cD1 = np.array(HT_cD1).flatten()
#
# HT_cA3_ = pywt.upcoef('a', HT_cA3, 'db5', level=3, take=311)
# HT_cD1_ = pywt.upcoef('d', HT_cD1, 'db5', level=1, take=311)
# HT_cD2_ = pywt.upcoef('d', HT_cD2, 'db5', level=2, take=311)
# HT_cD3_ = pywt.upcoef('d', HT_cD3, 'db5', level=3, take=311)
#
# HT_cA3_ = HT_cA3_.tolist()
# HT_cD1_ = HT_cD1_.tolist()
# HT_cD2_ = HT_cD2_.tolist()
# HT_cD3_ = HT_cD3_.tolist()

printc("用db5小波进行发电量3层分解")
Q = np.array(df.iloc[:,1:2]).reshape(1,-1)
Q_wt = pywt.wavedec(Q, 'db5', level=3)
Q_cA3, Q_cD3, Q_cD2, Q_cD1 = Q_wt

Q_cA3 = np.array(Q_cA3).flatten()
Q_cD3 = np.array(Q_cD3).flatten()
Q_cD2 = np.array(Q_cD2).flatten()
Q_cD1 = np.array(Q_cD1).flatten()

Q_cA3_ = pywt.upcoef('a', Q_cA3, 'db5', level=3, take=311)
Q_cD1_ = pywt.upcoef('d', Q_cD1, 'db5', level=1, take=311)
Q_cD2_ = pywt.upcoef('d', Q_cD2, 'db5', level=2, take=311)
Q_cD3_ = pywt.upcoef('d', Q_cD3, 'db5', level=3, take=311)

Q_cA3_ = Q_cA3_.tolist()
Q_cD1_ = Q_cD1_.tolist()
Q_cD2_ = Q_cD2_.tolist()
Q_cD3_ = Q_cD3_.tolist()

##########################################################################
##########################################################################
printc("cA3数据")
df_cA3 = pd.DataFrame({
        "发电量cA3":list(Q_cA3_),
        "日前发电量cA3":list(DAG_cA3_),
        "周前发电量cA3":list(WC_cA3_)
        # "最高温度cA3":list(HT_cA3_)
})
df_cA3.insert(0,'date',df_date)
print(df_cA3.head())

printc("准备训练和预测数据：")
X_cA3 = df_cA3.iloc[:, 2:]
Y_cA3 = df_cA3.iloc[:, 1:2]

train_x_cA3,test_x_cA3,train_y_cA3,test_y_cA3=train_test_split(X_cA3,Y_cA3,test_size=0.3)
print(np.ndim(train_x_cA3))

printc("归一化")
standard_scaler_x_cA3 = preprocessing.MinMaxScaler()
standard_scaler_y_cA3 = preprocessing.MinMaxScaler()

train_x_cA3 = standard_scaler_x_cA3.fit_transform(train_x_cA3)
train_y_cA3 = standard_scaler_y_cA3.fit_transform(train_y_cA3).ravel()

test_x_cA3 = standard_scaler_x_cA3.transform(test_x_cA3)
test_y_cA3= standard_scaler_y_cA3.transform(test_y_cA3).ravel()

printc("参数寻优与建模")
model = svm_cross_validation(train_x_cA3, train_y_cA3)

printc("预测")
predict_y_cA3 = model.predict(test_x_cA3)

printc("反归一化")
test_yy_cA3 = np.array(test_y_cA3).reshape(1, -1)
origin_test_y_cA3 = standard_scaler_y_cA3.inverse_transform(test_yy_cA3).ravel()
predict_yy_cA3 = np.array(predict_y_cA3).reshape(1, -1)
origin_predict_y_cA3=standard_scaler_y_cA3.inverse_transform(predict_yy_cA3).ravel()

##########################################################################
##########################################################################
printc("cD1数据")
df_cD1 = pd.DataFrame({
        "发电量cD1":list(Q_cD1_),
        "日前发电量cA3":list(DAG_cA3_),
        "周前发电量cD1":list(WC_cD1_)
        # "最高温度cD1":list(HT_cD1_)
})
df_cD1.insert(0,'date',df_date)
print(df_cD1.head())

printc("准备训练和预测数据：")
X_cD1 = df_cD1.iloc[:, 2:]
Y_cD1 = df_cD1.iloc[:, 1:2]

train_x_cD1,test_x_cD1,train_y_cD1,test_y_cD1=train_test_split(X_cD1,Y_cD1,test_size=0.3)
# print(np.ndim(train_x_cD1))

printc("归一化")
standard_scaler_x_cD1 = preprocessing.MinMaxScaler()
standard_scaler_y_cD1 = preprocessing.MinMaxScaler()

train_x_cD1 = standard_scaler_x_cD1.fit_transform(train_x_cD1)
train_y_cD1 = standard_scaler_y_cD1.fit_transform(train_y_cD1).ravel()

test_x_cD1 = standard_scaler_x_cD1.transform(test_x_cD1)
test_y_cD1= standard_scaler_y_cD1.transform(test_y_cD1).ravel()

printc("参数寻优与建模")
model = svm_cross_validation(train_x_cD1, train_y_cD1)

printc("预测")
predict_y_cD1 = model.predict(test_x_cD1)

printc("反归一化")
test_yy_cD1= np.array(test_y_cD1).reshape(1, -1)
origin_test_y_cD1 = standard_scaler_y_cD1.inverse_transform(test_yy_cD1).ravel()
predict_yy_cD1 = np.array(predict_y_cD1).reshape(1, -1)
origin_predict_y_cD1 = standard_scaler_y_cD1.inverse_transform(predict_yy_cD1).ravel()

##########################################################################
##########################################################################
printc("cD2数据")
df_cD2 = pd.DataFrame({
        "发电量cD2":list(Q_cD2_),
        "日前发电量cA3":list(DAG_cA3_),
        "周前发电量cD2":list(WC_cD2_)
        # "最高温度cD2":list(HT_cD2_)
})
df_cD2.insert(0,'date',df_date)
print(df_cD2.head())

printc("准备训练和预测数据：")
X_cD2 = df_cD2.iloc[:, 2:]
Y_cD2 = df_cD2.iloc[:, 1:2]

train_x_cD2,test_x_cD2,train_y_cD2,test_y_cD2=train_test_split(X_cD2,Y_cD2,test_size=0.3)
print(np.ndim(train_x_cD2))

printc("归一化")
standard_scaler_x_cD2 = preprocessing.MinMaxScaler()
standard_scaler_y_cD2 = preprocessing.MinMaxScaler()

train_x_cD2 = standard_scaler_x_cD2.fit_transform(train_x_cD2)
train_y_cD2= standard_scaler_y_cD2.fit_transform(train_y_cD2).ravel()

test_x_cD2 = standard_scaler_x_cD2.transform(test_x_cD2)
test_y_cD2= standard_scaler_y_cD2.transform(test_y_cD2).ravel()

printc("参数寻优与建模")
model = svm_cross_validation(train_x_cD2, train_y_cD2)

printc("预测")
predict_y_cD2 = model.predict(test_x_cD2)

printc("反归一化")
test_yy_cD2 = np.array(test_y_cD2).reshape(1, -1)
origin_test_y_cD2 = standard_scaler_y_cD2.inverse_transform(test_yy_cD2).ravel()
predict_yy_cD2 = np.array(predict_y_cD2).reshape(1, -1)
origin_predict_y_cD2 = standard_scaler_y_cD2.inverse_transform(predict_yy_cD2).ravel()

##########################################################################
##########################################################################
printc("cD3数据")
df_cD3 = pd.DataFrame({
        "发电量cD3":list(Q_cD3_),
        "日前发电量cA3":list(DAG_cA3_),
        "周前发电量cD3":list(WC_cD3_)
        # "最高温度cD3":list(HT_cD3_)
})
df_cD3.insert(0,'date',df_date)
print(df_cD3.head())

printc("准备训练和预测数据：")
X_cD3 = df_cD3.iloc[:, 2:]
Y_cD3 = df_cD3.iloc[:, 1:2]

train_x_cD3,test_x_cD3,train_y_cD3,test_y_cD3=train_test_split(X_cD3,Y_cD3,test_size=0.3)
print(np.ndim(train_x_cD3))

printc("归一化")
standard_scaler_x_cD3 = preprocessing.MinMaxScaler()
standard_scaler_y_cD3 = preprocessing.MinMaxScaler()

train_x_cD3 = standard_scaler_x_cD3.fit_transform(train_x_cD3)
train_y_cD3 = standard_scaler_y_cD3.fit_transform(train_y_cD3).ravel()

test_x_cD3 = standard_scaler_x_cD3.transform(test_x_cD3)
test_y_cD3= standard_scaler_y_cD3.transform(test_y_cD3).ravel()

printc("参数寻优与建模")
model = svm_cross_validation(train_x_cD3, train_y_cD3)

printc("预测")
predict_y_cD3 = model.predict(test_x_cD3)

printc("反归一化")
test_yy_cD3 = np.array(test_y_cD3).reshape(1, -1)
origin_test_y_cD3 = standard_scaler_y_cD3.inverse_transform(test_yy_cD3).ravel()
predict_yy_cD3 = np.array(predict_y_cD3).reshape(1, -1)
origin_predict_y_cD3 = standard_scaler_y_cD3.inverse_transform(predict_yy_cD3).ravel()
print(origin_predict_y_cD3)
##########################################################################
##########################################################################

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
plot_resuWC(test_y, predict_y)
plot_resuWC(origin_test_y,origin_predict_y)
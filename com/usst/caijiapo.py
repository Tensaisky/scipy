import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import NuSVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

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


df = pd.read_excel("caijiapo.xlsx")
df['发电量/kWh'] = df['发电量/kWh'].map(lambda x: x*10000)
printc("原始数据")
df = df.iloc[:,1:]
print(df.head())

weather = df['天气状况'].str.extract('(.*)\((.*)℃~(.*)℃\)', expand=False)
weather.columns = ['日类型', '最低温度', '最高温度']
weather['最低温度'] = weather['最低温度'].astype('float')
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
print(df.iloc[7:11,:])

printc("删除最大负荷和天气状况")
df = df.drop("最大负荷/MW", axis=1)
df = df.drop("天气状况", axis=1)
print(df.iloc[7:11,:])

printc("删除开始一周（由于开始一周没有前一周数据），重置索引")
df = df.iloc[7:,:]
df = df.reset_index(drop=True)
print(df.head())

printc("准备训练和预测数据：")
X = df.iloc[:, 2:]
Y = df.iloc[:, 1:2]

train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.3)

printc("归一化")
standard_scaler_x = preprocessing.MinMaxScaler()
standard_scaler_y = preprocessing.MinMaxScaler()

train_x = standard_scaler_x.fit_transform(train_x)
train_y = standard_scaler_y.fit_transform(train_y).ravel()

test_x = standard_scaler_x.transform(test_x)
test_y = standard_scaler_y.transform(test_y).ravel()


printc("参数寻优与建模")
model = svm_cross_validation(train_x, train_y)

#model = XGBRegressor()
#model.fit(train_x, train_y, verbose=False)

printc("预测")
predict_y = model.predict(test_x)

printc("反归一化")
test_yy = np.array(test_y).reshape(1, -1)
origin_test_y = standard_scaler_y.inverse_transform(test_yy).ravel()
predict_yy = np.array(predict_y).reshape(1, -1)
origin_predict_y = standard_scaler_y.inverse_transform(predict_yy).ravel()
print(origin_predict_y)

print("Mean Absolute Error: " + str(mean_absolute_error(predict_y, test_y)))
plot_result(test_y, predict_y)
plot_result(origin_test_y,origin_predict_y)
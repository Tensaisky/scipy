import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import NuSVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error




def plot_result(label, predict_y):
    plt.plot(range(label.size), label, color='navy')
    plt.plot(range(label.size), predict_y, color='red')
    plt.show()


def precession(label, predict_y):
    e = np.sum(np.abs(label - predict_y)) / np.sum(np.abs(label))
    print(1 - e)


df = pd.read_excel("caijiapo.xlsx")
df['发电量/kWh'] = df['发电量/kWh'].map(lambda x: x*10000)

df = df.iloc[:,1:]


weather = df['天气状况'].str.extract('(.*)\((.*)℃~(.*)℃\)', expand=False)
weather.columns = ['日类型', '最低温度', '最高温度']
weather['最低温度'] = weather['最低温度'].astype('float')
weather['最高温度'] = weather['最高温度'].astype('float')


df = pd.concat([df.iloc[:,0:4], weather], axis=1)



class_mapping = {label:idx for idx,label in enumerate(np.unique(df['日类型']))}



df['日类型'] = df['日类型'].map(class_mapping)

df['日类型'] = df['日类型'].astype('float')



df['日前发电量/kWh'] = df['发电量/kWh'].shift(1)
df['日前最大负荷/MW'] = df['最大负荷/MW'].shift(1)



df['周前发电量/kWh'] = df['发电量/kWh'].shift(7)
df['周前最大负荷/MW'] = df['最大负荷/MW'].shift(7)


df = df.drop("最大负荷/MW", axis=1)
df = df.drop("天气状况", axis=1)



df = df.iloc[7:,:]
df = df.reset_index(drop=True)


X = df.iloc[:, 2:]
Y = df.iloc[:, 1:2]

train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.3)


standard_scaler_x = preprocessing.MinMaxScaler()
standard_scaler_y = preprocessing.MinMaxScaler()

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


# model = svm_cross_validation(train_x, train_y)

model = XGBRegressor()
model.fit(train_x, train_y, verbose=False)


predict_y = model.predict(test_x)

print("Mean Absolute Error: " + str(mean_absolute_error(predict_y, test_y)))
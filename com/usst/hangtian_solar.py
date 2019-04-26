import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import NuSVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


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
#相对误差
def precession(label, predict_y):
    e = np.sum(np.abs(label - predict_y)) / np.sum(np.abs(label))
    print(1 - e)


df = pd.read_excel("hangtian_solar.xlsx")
df['时间'] = pd.to_datetime(df['时间'])
print(df.head())

# X = pd.concat([df.iloc[:, 1:2],df.iloc[:, 4:5],df.iloc[:, 4:5]],axis=1)
X = df.iloc[:, 1:18]
Y = df.iloc[:, 18:19]

train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.3)

standard_scaler_x = preprocessing.MinMaxScaler()
standard_scaler_y = preprocessing.MinMaxScaler()

train_x = standard_scaler_x.fit_transform(train_x)
train_y = standard_scaler_y.fit_transform(train_y).ravel()

test_x = standard_scaler_x.transform(test_x)
test_y = standard_scaler_y.transform(test_y).ravel()

model_xgb = XGBRegressor()

model_xgb.fit(train_x, train_y, verbose=False)

predict_xgb = model_xgb.predict(test_x)

print("Xgboost Mean Absolute Error: " + str(1-mean_absolute_error(predict_xgb, test_y)))

def plot_result(label, predict_y):
    plt.plot(range(label.size), label, color='navy')
    plt.plot(range(label.size), predict_y, color='red')
    plt.show()

print("反归一化")
test_yy = np.array(test_y).reshape(1, -1)
origin_test_y = standard_scaler_y.inverse_transform(test_yy).ravel()
predict_yy = np.array(predict_xgb).reshape(1, -1)
origin_predict_y = standard_scaler_y.inverse_transform(predict_yy).ravel()

print("计算准确率误差")
precession(origin_test_y, origin_predict_y)

plot_result(origin_predict_y[0:50], origin_test_y[0:50])
#plot_result(predict_xgb[0:50], test_y[0:50])
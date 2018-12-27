import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import NuSVR
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.externals import joblib

df = pd.read_csv("pv_power.csv")
df = df.iloc[:, 2:8]

X = df.iloc[:, 0:5]
y = df.iloc[:, 5:6]

test_X = df.iloc[20000:, 0:5]
test_y = df.iloc[20000:, 5:6]

min_max_scaler_X = preprocessing.StandardScaler()
X = min_max_scaler_X.fit_transform(X)

min_max_scaler_y = preprocessing.StandardScaler()
y = min_max_scaler_y.fit_transform(y).ravel()

test_X = min_max_scaler_X.transform(test_X)
test_y = min_max_scaler_y.transform(test_y).ravel()

svr = NuSVR(kernel='rbf', C=1000)
y_lin = svr.fit(X, y).predict(test_X)


def save_result(label, predict_y, i):
    plt.plot(range(label.size), label, color='navy')
    plt.plot(range(label.size), predict_y, color='red')
    path = str(i) + ".jpg"
    plt.savefig(path, dpi=240)
    plt.show()


def precesion(label, predict_y):
    e = np.sum(np.abs(label - predict_y)) / np.sum(np.abs(label))
    print(1 - e)


precesion(test_y, y_lin)
save_result(test_y, y_lin, 3)

joblib.dump(svr, 'svr_pv.pkl')
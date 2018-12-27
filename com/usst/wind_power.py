import sys
import pandas as pd
import datetime
import sqlalchemy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import NuSVR
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.externals import joblib

try:
    engine = sqlalchemy.create_engine('mysql+pymysql://root:Aa123@localhost:3306/zyzn_nxgl')
except sqlalchemy.exc.OperationalError as e:
    print('Error is '+str(e))
    sys.exit()
except sqlalchemy.exc.InternalError as e:
    print('Error is '+str(e))
    sys.exit()

df = pd.read_sql("wind_power", engine)
# df = pd.read_csv("D:/resources/wind_power.csv")
df = df.iloc[:, 2:8]
df_X = df.iloc[0:50000, 1:6]
df_y = df.iloc[0:50000, 0:1]

df_test_X = df.iloc[50000:60000, 1:6]
df_test_y = df.iloc[50000:60000, 0:1]

min_max_scaler_X = preprocessing.StandardScaler()
X = min_max_scaler_X.fit_transform(df_X)

min_max_scaler_y = preprocessing.StandardScaler()
y = min_max_scaler_y.fit_transform(df_y).ravel()

test_X = min_max_scaler_X.transform(df_test_X)
test_y = min_max_scaler_y.transform(df_test_y).ravel()

svr = NuSVR(kernel='rbf', C=1000)
y_lin = svr.fit(X, y).predict(test_X)

def save_result(label, predict_y, i):
    plt.plot(range(label.size), label, color='navy')
    plt.plot(range(label.size), predict_y, color='red')
    path = "C:/Users/WORK01/Desktop/" + str(i) + ".jpg"
    plt.savefig(path, dpi=240)
    plt.show()


def precession(label, predict_y):
    e = np.sum(np.abs(label - predict_y)) / np.sum(np.abs(label))
    print(1 - e)


precession(test_y, y_lin)
save_result(test_y, y_lin, 3)

joblib.dump(svr, 'svr1.pkl')
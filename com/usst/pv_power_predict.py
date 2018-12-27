import sys
import json
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

date_start = sys.argv[1]
# date_start = "2015-04-12"
date_end = datetime.datetime.strptime(date_start, "%Y-%m-%d")
date_end = date_end + datetime.timedelta(days=1)
date_end = date_end.strftime("%Y-%m-%d")

df = pd.read_sql("pv_power", engine)
df_test = df[(df.DATA_TIME > date_start) & (df.DATA_TIME < date_end)].iloc[:, 2:8]

now = datetime.datetime.now()
datet = datetime.datetime(2015, now.month, now.day, now.hour, now.minute, now.second)
daten = datet.strftime("%Y-%m-%d")
datet = datet.strftime("%Y-%m-%d %H:%M:%S")

df_train = df.iloc[:, 2:8]
df_X = df_train.iloc[:, 0:5]
df_y = df_train.iloc[:, 5:6]

df_test_X = df_test.iloc[:, 0:5]
df_test_y = df_test.iloc[:, 5:6]

min_max_scaler_X = preprocessing.StandardScaler()
X = min_max_scaler_X.fit_transform(df_X)

min_max_scaler_y = preprocessing.StandardScaler()
y = min_max_scaler_y.fit_transform(df_y).ravel()

test_X = min_max_scaler_X.transform(df_test_X)
test_y = min_max_scaler_y.transform(df_test_y).ravel()

svr = joblib.load('C:/Users/WORK01/Desktop/scipy/com/usst/svr_pv.pkl')
y_predict = svr.predict(test_X)
y_predict = y_predict.reshape(1, -1)
y_predict = min_max_scaler_y.inverse_transform(y_predict)
test_y = min_max_scaler_y.inverse_transform(test_y.reshape(1, -1))


def save_result(label, predict_y, i):
    plt.plot(range(label.size), label, color='navy')
    plt.plot(range(label.size), predict_y, color='red')
    path = "C:/Users/WORK01/Desktop/" + str(i) + ".jpg"
    plt.savefig(path, dpi=240)
    plt.show()


def precess(label, predict_y):
    e = np.sum(np.abs(label - predict_y)) / np.sum(np.abs(label))
    return np.abs(1-e)


# save_result(test_y.ravel(), y_predict.ravel(), 4)
precession = precess(test_y.ravel(), y_predict.ravel())
precession = precession * 100
precession = round(precession, 2)
# print(precession)
# print(test_y.ravel())
# print(y_predict.ravel())

if (date_start == daten):
    label = df[(df.DATA_TIME > date_start) & (df.DATA_TIME < datet)].iloc[:, 7:8]
    precession = 0
else:
    label = df[(df.DATA_TIME > date_start) & (df.DATA_TIME < date_end)].iloc[:, 7:8]

label = label.as_matrix().reshape(-1).tolist()
for i in range(y_predict.shape[1]):
    y_predict[0, i] = round(y_predict[0, i], 3)

y_predict = y_predict.reshape(-1).tolist()

result = []

result.append(label)
result.append(y_predict)
result.append([precession])
print(json.dumps(result))
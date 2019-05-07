# -*- coding: utf-8 -*-
import cx_Oracle
import datetime
import os

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
conn = cx_Oracle.connect('system', 'tiger', '192.168.1.114:1521/orcl1')
cursor = conn.cursor()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sqlalchemy import create_engine
from xgboost import XGBRegressor

def getCountOfPreSheet():
    sql = """
        select count(*) from SYSTEM."sheet_prePower"
    """
    result = cursor.execute(sql)
    count = 0
    for i in result:
        count = i[0]
    return count
def getRowNumTimeOfPrePowerSheet(rownum_in):
    # 输入int类型rownum（方便循环），转成str用于查询某一行的时间
    rownum = rownum_in
    rownum = str(rownum)
    
    sql = """
        select * from (SELECT rownum no , SYSTEM."sheet_prePower"."时间" FROM SYSTEM."sheet_prePower" )
         where no =
      """ + rownum + """
    """
    result = cursor.execute(sql)
    date_time = '2019/3/26 11:00:00'
    date_time = datetime.datetime.strptime(date_time, '%Y/%m/%d %H:%M:%S')
    for i in result:
        date_time = i[1]
        # print(date_time)
    return date_time
def storePrePower(date_time_begin,prePower):
    # 输入一个datetime时间和最大负荷
    date_time_begin2 = date_time_begin + datetime.timedelta(minutes=(2))
    date_time_begin_for_search = date_time_begin.strftime("%Y/%m/%d %H:%M:%S")
    date_time_begin_for_search2 = date_time_begin2.strftime("%Y/%m/%d %H:%M:%S")
    
    prePower = str(prePower)
    sqlExit = """
    UPDATE SYSTEM."sheet_prePower"
    set SYSTEM."sheet_prePower"."预测负荷" =('
    """ + prePower + """
    ')
    WHERE
    SYSTEM."sheet_prePower"."时间" BETWEEN to_date('
    """ + date_time_begin_for_search + """
    ','yyyy-mm-dd hh24:mi:ss') AND to_date('
    """ + date_time_begin_for_search2 + """
    ','yyyy-mm-dd hh24:mi:ss')
    """
    cursor.execute(sqlExit)
    conn.commit()
    
engine = create_engine('oracle+cx_oracle://system:tiger@192.168.1.114:1521/orcl1')

df = pd.read_sql('SELECT * FROM SYSTEM."sheet_historyPower"',engine)
df2 = pd.read_sql('SELECT * FROM SYSTEM."sheet_prePower"',engine)
X = pd.concat([df.iloc[:,1:7],df.iloc[:,11:13]],axis = 1)
Y = df.iloc[:, 10:11]
X2 = df2.iloc[:,1:8]

train_x = X.iloc[1:, 0:7]
train_y = df.iloc[1:, 10:11]
test_x = X2

standard_scaler_x = preprocessing.MinMaxScaler()
standard_scaler_y = preprocessing.MinMaxScaler()
train_x = standard_scaler_x.fit_transform(train_x)
train_y = standard_scaler_y.fit_transform(train_y).ravel()
test_x = standard_scaler_x.transform(test_x)

model_xgb = XGBRegressor()
model_xgb.fit(train_x, train_y, verbose=False)
predict_xgb = model_xgb.predict(test_x)
predict_yy = np.array(predict_xgb).reshape(1, -1)
origin_predict_y = standard_scaler_y.inverse_transform(predict_yy).ravel()

origin_predict_y = origin_predict_y.tolist()
print(origin_predict_y)
print(type(origin_predict_y))

number = getCountOfPreSheet()
preDate = []
for num in range(number):
    preDate.append(getRowNumTimeOfPrePowerSheet(num+1))
for i in range(number):
    storePrePower(preDate[i],origin_predict_y[i])

conn.commit()
cursor.close()
conn.close()
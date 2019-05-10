import cx_Oracle
import os
import  pandas as pd
import datetime
import numpy as np
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
class Oracle_Connect:
    date = ""
    def __init__(self,date):
        self.date = date

    # 提供数据库连接驱动
    def connect_driver(self):
        connnect = cx_Oracle.connect('system/tiger@192.168.1.114:1521/orcl1')
        cursor = connnect.cursor()
        return  cursor,connnect
    def getPvData(self):
        cursor, conn = self.connect_driver()
        sql = 'SELECT * FROM SYSTEM ."solar_2" '
        # sql = 'SELECT SYSTEM ."solar_2"."总功率" FROM SYSTEM ."solar_2" WHERE SYSTEM ."solar_2"."时间" BETWEEN "TO_DATE"' + "('2019-03-29 00:00:00', 'yyyy-mm-dd hh24:mi:ss')" + "AND " + "TO_DATE" + "('2019-03-29 23:59:59', 'yyyy-mm-dd hh24:mi:ss')"
        cursor.execute(sql)
        rows = cursor.fetchall()  # 得到所有数据集
        pv_Alldata = pd.DataFrame(rows)
        pv_Alldata.columns= ['时间','温度','辐照度','逆变1直流功率','逆变1交流功率','逆变1累计电度','逆变2直流功率','逆变2交流功率','逆变2累计电度','逆变3直流功率','逆变3交流功率','逆变3累计电度','逆变4直流功率','逆变4交流功率','逆变4累计电度','逆变5直流功率','逆变5交流功率','逆变5累计电度','总功率','总电度']
        print(pv_Alldata)
        print(len(pv_Alldata['总功率']))
        return pv_Alldata
conn_model = Oracle_Connect("a")
conn_model.getPvData()
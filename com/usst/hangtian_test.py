# -*- coding: utf-8 -*-
import cx_Oracle
import datetime
import os

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
conn = cx_Oracle.connect('system', 'tiger', '192.168.1.108:1521/orcl1')
cursor = conn.cursor()

date_time = '2019/3/25 11:00:00'
date_time2 = '2019/3/25 11:02:00'

sqlExit = """
SELECT
SYSTEM."sheet_historyPower"."时间"
FROM
SYSTEM."sheet_historyPower"
WHERE
SYSTEM."sheet_historyPower"."时间" BETWEEN to_date('
""" + date_time + """
','yyyy-mm-dd hh24:mi:ss') AND to_date('
""" + date_time2 + """
','yyyy-mm-dd hh24:mi:ss')
"""
result = cursor.execute(sqlExit)
hasRecord = 0
for j in result:
    if j:
        hasRecord = 1
print(hasRecord)


conn.commit()
cursor.close()
conn.close()
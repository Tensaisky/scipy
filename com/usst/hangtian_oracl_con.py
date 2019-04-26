# -*- coding: utf-8 -*-
import cx_Oracle
import datetime
import os
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
conn = cx_Oracle.connect('system','tiger','192.168.1.106:1521/orcl1')

cursor = conn.cursor()
date_time = '2019/3/28 11:00:00'
date_time2 = '2019/3/28 11:00:00'
# str to datetime
date_time = datetime.datetime.strptime(date_time,'%Y/%m/%d %H:%M:%S')
print(type(date_time))
print(date_time)
# datetime to str
date_time2 = (date_time+ datetime.timedelta(minutes=(2))).strftime("%Y/%m/%d %H:%M:%S")
print(date_time2)
print(type(date_time2))
print()
sql = """
SELECT
SYSTEM."wind_2"."时间"
FROM
SYSTEM."wind_2"
WHERE
SYSTEM."wind_2"."时间" BETWEEN to_date('
""" + date_time + """
','yyyy-mm-dd hh24:mi:ss') AND to_date('
""" + date_time2 + """
','yyyy-mm-dd hh24:mi:ss')
"""

# sql = """
# SELECT
# SYSTEM."wind_2"."时间"
# FROM
# SYSTEM."wind_2"
# WHERE
# SYSTEM."wind_2"."时间" LIKE to_date('
# """ + date_time + """
# ','yyyy-mm-dd hh24:mi:ss')
# """
result = cursor.execute(sql)
for i in result:
    print(i)
conn.commit()
cursor.close()
conn.close()


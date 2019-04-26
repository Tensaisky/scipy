# -*- coding: utf-8 -*-
import cx_Oracle
import datetime
import os
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
conn = cx_Oracle.connect('system','tiger','192.168.1.106:1521/orcl1')

cursor = conn.cursor()
date_time = '2019/3/28 11:00:00'
date_time2 = '2019/3/28 11:02:00'

# 取第一个数据
sql = """
select SYSTEM."Sheet_mainsPower"."时间" from SYSTEM."Sheet_mainsPower" where rownum<2
"""

result = cursor.execute(sql)

for i in result:
    date_time_begin = i[0]
date_time_begin2 = date_time_begin+ datetime.timedelta(minutes=(2))

print(date_time_begin)
print(date_time_begin2)
print(type(date_time_begin))
print(type(date_time_begin2))

date_time_begin_for_search = date_time_begin.strftime("%Y/%m/%d %H:%M:%S")
date_time_begin_for_search2 = date_time_begin2.strftime("%Y/%m/%d %H:%M:%S")
print(date_time_begin)
print(date_time_begin2)
print(type(date_time_begin))
print(type(date_time_begin2))

sql1 = """
SELECT
SYSTEM."wind_2"."交流输出功率W"
FROM
SYSTEM."wind_2"
WHERE
SYSTEM."wind_2"."时间" BETWEEN to_date('
""" + date_time_begin_for_search + """
','yyyy-mm-dd hh24:mi:ss') AND to_date('
""" + date_time_begin_for_search2 + """
','yyyy-mm-dd hh24:mi:ss')
"""
result1 = cursor.execute(sql1)
for j in result1:
    print(j[0])

conn.commit()
cursor.close()
conn.close()


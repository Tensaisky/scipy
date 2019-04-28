# -*- coding: utf-8 -*-
import cx_Oracle
import datetime
import os
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
conn = cx_Oracle.connect('system','tiger','192.168.1.108:1521/orcl1')
cursor = conn.cursor()

rownum = 4
rownum = str(rownum)

sql = """
    select count(*) from SYSTEM."Sheet_mainsPower"
"""
result = cursor.execute(sql)
for i in result:
    print(i[0])
    print(type(i[0]))
    
conn.commit()
cursor.close()
conn.close()
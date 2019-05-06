# -*- coding: utf-8 -*-
import cx_Oracle
import datetime
import os
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
conn = cx_Oracle.connect('system','tiger','192.168.1.114:1521/orcl1')

cursor = conn.cursor()
date_time = '2019/3/26 11:00:00'
date_time2 = '2019/3/26 11:00:00'
# str 转 datetime
date_time = datetime.datetime.strptime(date_time,'%Y/%m/%d %H:%M:%S')


# 返回某一行得时间，从1开始
def getRowNumTimeOfSheet(rownum_in):
    # 输入int类型rownum（方便循环），转成str用于查询某一行的时间
    rownum = rownum_in
    rownum = str(rownum)
    
    sql = """
        select * from (SELECT rownum no , SYSTEM."sheet_historyPower"."时间" FROM SYSTEM."sheet_historyPower" )
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

def oneDayRecord(date_time_begin):
    # 输入一个datetime时间，以list格式返回这个日期的所有负荷数据
    date_time_begin = date_time_begin.strftime("%Y/%m/%d %H:%M:%S")
    date_time_begin = date_time_begin.split(' ')[0]
    date_time_begin = date_time_begin + ' 00:00:00'
    date_time_begin = datetime.datetime.strptime(date_time_begin, '%Y/%m/%d %H:%M:%S')
    date_time_begin2 = date_time_begin + datetime.timedelta(hours=(24))
    date_time_begin_for_search = date_time_begin.strftime("%Y/%m/%d %H:%M:%S")
    date_time_begin_for_search2 = date_time_begin2.strftime("%Y/%m/%d %H:%M:%S")
    print(date_time_begin_for_search)
    print(date_time_begin_for_search2)
    sqlExit = """
    SELECT
    SYSTEM."sheet_historyPower"."负荷"
    FROM
    SYSTEM."sheet_historyPower"
    WHERE
    SYSTEM."sheet_historyPower"."时间" BETWEEN to_date('
    """ + date_time_begin_for_search + """
    ','yyyy-mm-dd hh24:mi:ss') AND to_date('
    """ + date_time_begin_for_search2 + """
    ','yyyy-mm-dd hh24:mi:ss')
    """
    result = cursor.execute(sqlExit)
    dayRecord_list = []
    for j in result:
        # print(j)
        dayRecord_list.append(j[0])
    return dayRecord_list

def storeMaxPower(date_time_begin,maxPower):
    # 输入一个datetime时间和最大负荷
    date_time_begin = date_time_begin.strftime("%Y/%m/%d %H:%M:%S")
    date_time_begin = date_time_begin.split(' ')[0]
    date_time_begin = date_time_begin + ' 00:00:00'
    date_time_begin = datetime.datetime.strptime(date_time_begin, '%Y/%m/%d %H:%M:%S')
    date_time_begin2 = date_time_begin + datetime.timedelta(hours=(24))
    date_time_begin_for_search = date_time_begin.strftime("%Y/%m/%d %H:%M:%S")
    date_time_begin_for_search2 = date_time_begin2.strftime("%Y/%m/%d %H:%M:%S")
    maxPower = str(maxPower)
    sqlExit = """
    UPDATE SYSTEM."sheet_historyPower"
    set SYSTEM."sheet_historyPower"."日最大负荷" =('
    """ + maxPower + """
    ')
    WHERE
    SYSTEM."sheet_historyPower"."时间" BETWEEN to_date('
    """ + date_time_begin_for_search + """
    ','yyyy-mm-dd hh24:mi:ss') AND to_date('
    """ + date_time_begin_for_search2 + """
    ','yyyy-mm-dd hh24:mi:ss')
    """
    cursor.execute(sqlExit)
    conn.commit()
    
    
for i in range(444):
    time = getRowNumTimeOfSheet(i+1)
    dayRecord_list = oneDayRecord(time)
    # 日总负荷和平均负荷
    allPower = 0
    for j in range(len(dayRecord_list)):
        allPower = allPower + dayRecord_list[j]
    # print(allPower)
    averagePower = allPower/len(dayRecord_list)
    # print(averagePower)
    
    # 日最大负荷
    maxPower = 0
    for k in range(len(dayRecord_list)):
        if dayRecord_list[k] > maxPower:
            maxPower = dayRecord_list[k]
    
    storeMaxPower(time,maxPower)
    
conn.commit()
cursor.close()
conn.close()


# -*- coding: utf-8 -*-
import cx_Oracle
import datetime
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sqlalchemy import create_engine
from xgboost import XGBRegressor

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
conn = cx_Oracle.connect('system', 'tiger', '192.168.43.79:1521/orcl1')
cursor = conn.cursor()


# mainpower第一行数据得时间
def getFistTimeOfSheet():
    # 取第一个行的时间数据，以此时间为查询筛选标准
    # 返回数据格式为datetime
    sql = """
    select SYSTEM."Sheet_mainsPower"."时间" from SYSTEM."Sheet_mainsPower" where rownum<2
    """
    result = cursor.execute(sql)
    for i in result:
        date_time_begin = i[0]
    return date_time_begin


# 传入时间，返回昨天此刻时间
def getLastDayTime(daytimenow):
    # 传入当前日期时间datetime格式，返回昨天此刻时间数据
    daytimelast = daytimenow - datetime.timedelta(days=1)
    return daytimelast


# 返回表得行数
def getCountOfSheet():
    sql = """
        select count(*) from SYSTEM."Sheet_mainsPower"
    """
    result = cursor.execute(sql)
    count = 0
    for i in result:
        count = i[0]
    return count


def getCountOfHistorySheet():
    sql = """
        select count(*) from SYSTEM."sheet_historyPower"
    """
    result = cursor.execute(sql)
    count = 0
    for i in result:
        count = i[0]
    return count

def getHistorySheet():
    sql = """
        select * from SYSTEM."sheet_historyPower" order by SYSTEM."sheet_historyPower"."时间" desc
    """
    result = cursor.execute(sql)
    count = []
    num = 0
    for i in result:
        if num < 24 :
            num = num + 1
            count.append(i[0])
    return count

def getCountOfPreSheet():
    sql = """
        select count(*) from SYSTEM."sheet_prePower"
    """
    result = cursor.execute(sql)
    count = 0
    for i in result:
        count = i[0]
    return count


# 存储语句，因为最大负荷是是后期处理的，所以另外采用update语句更新
def storeData(list):
    # 传入数据list，保存
    # list = [date_time, 1, 2, 3, 4, 5, 6, 7, 8]
    sql = 'INSERT INTO SYSTEM."sheet_historyPower"(\"时间\",\"星期数\",\"当前整点数\",\"昨日此刻温度\",\"昨日此刻风速\",\"昨日此刻湿度\",\"昨日此刻负荷\",\"温度\",\"风速\",\"湿度\",\"负荷\") VALUES(:1,:2,:3,:4,:5,:6,:7,:8,:9,:10,:11)'
    cursor.execute(sql, list)


def storeMaxPower(date_time_begin, maxPower):
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


def storeAveragePower(date_time_begin, avePower):
    # 输入一个datetime时间和最大负荷
    date_time_begin = date_time_begin.strftime("%Y/%m/%d %H:%M:%S")
    date_time_begin = date_time_begin.split(' ')[0]
    date_time_begin = date_time_begin + ' 00:00:00'
    date_time_begin = datetime.datetime.strptime(date_time_begin, '%Y/%m/%d %H:%M:%S')
    date_time_begin2 = date_time_begin + datetime.timedelta(hours=(24))
    date_time_begin_for_search = date_time_begin.strftime("%Y/%m/%d %H:%M:%S")
    date_time_begin_for_search2 = date_time_begin2.strftime("%Y/%m/%d %H:%M:%S")
    avePower = str(avePower)
    sqlExit = """
    UPDATE SYSTEM."sheet_historyPower"
    set SYSTEM."sheet_historyPower"."日平均负荷" =('
    """ + avePower + """
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


def loopStoreMaxAndAverage(num):
    # 给出history表数据数量
    for i in range(num):
        time = getRowNumTimeOfSheet(i + 1)
        dayRecord_list = oneDayRecord(time)
        # 日总负荷和平均负荷
        allPower = 0
        for j in range(len(dayRecord_list)):
            allPower = allPower + dayRecord_list[j]
        averagePower = allPower / len(dayRecord_list)
        storeAveragePower(time, averagePower)
        # 日最大负荷
        maxPower = 0
        for k in range(len(dayRecord_list)):
            if dayRecord_list[k] > maxPower:
                maxPower = dayRecord_list[k]
        storeMaxPower(time, maxPower)


def storePreData(list):
    # 传入数据list，保存
    # list = [date_time, 1, 2, 3, 4, 5, 6, 7, 8]
    sql = 'INSERT INTO SYSTEM."sheet_prePower"(\"时间\",\"星期数\",\"当前整点数\",\"昨日此刻温度\",\"昨日此刻风速\",\"昨日此刻湿度\",\"昨日此刻负荷\",\"日最大负荷\") VALUES(:1,:2,:3,:4,:5,:6,:7,:8)'
    cursor.execute(sql, list)


def storePrePower(date_time_begin, prePower):
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


# 返回某一行得时间，从1开始
def getRowNumTimeOfSheet(rownum_in):
    # 输入int类型rownum（方便循环），转成str用于查询某一行的时间
    rownum = rownum_in
    rownum = str(rownum)
    
    sql = """
        select * from (SELECT rownum no , SYSTEM."Sheet_mainsPower"."时间" FROM SYSTEM."Sheet_mainsPower" )
         where no =
      """ + rownum + """
    """
    result = cursor.execute(sql)
    date_time = '2019/3/28 11:00:00'
    date_time = datetime.datetime.strptime(date_time, '%Y/%m/%d %H:%M:%S')
    for i in result:
        date_time = i[1]
        # print(date_time)
    return date_time


# 是否已经记录，有记录返回0，无记录返回1
def hasNoRecord(date_time_begin):
    # 输入datetime格式时间参数，转成str类型，只返回一个数据
    date_time_begin2 = date_time_begin + datetime.timedelta(minutes=(2))
    date_time_begin_for_search = date_time_begin.strftime("%Y/%m/%d %H:%M:%S")
    date_time_begin_for_search2 = date_time_begin2.strftime("%Y/%m/%d %H:%M:%S")
    sqlExit = """
    SELECT
    SYSTEM."sheet_historyPower"."时间"
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
    hasRecord = 1
    for j in result:
        if j:
            hasRecord = 0
    return hasRecord


def hasNoPreRecord(date_time_begin):
    # 输入datetime格式时间参数，转成str类型，只返回一个数据
    date_time_begin2 = date_time_begin + datetime.timedelta(minutes=(2))
    date_time_begin_for_search = date_time_begin.strftime("%Y/%m/%d %H:%M:%S")
    date_time_begin_for_search2 = date_time_begin2.strftime("%Y/%m/%d %H:%M:%S")
    sqlExit = """
    SELECT
    SYSTEM."sheet_prePower"."时间"
    FROM
    SYSTEM."sheet_prePower"
    WHERE
    SYSTEM."sheet_prePower"."时间" BETWEEN to_date('
    """ + date_time_begin_for_search + """
    ','yyyy-mm-dd hh24:mi:ss') AND to_date('
    """ + date_time_begin_for_search2 + """
    ','yyyy-mm-dd hh24:mi:ss')
    """
    result = cursor.execute(sqlExit)
    hasRecord = 1
    for j in result:
        if j:
            hasRecord = 0
    return hasRecord


# 某一日期的24个小时数据，返回列表
def oneDayRecord(date_time_begin):
    # 输入一个datetime时间，以list格式返回这个日期的所有负荷数据
    date_time_begin = date_time_begin.strftime("%Y/%m/%d %H:%M:%S")
    date_time_begin = date_time_begin.split(' ')[0]
    date_time_begin = date_time_begin + ' 00:00:00'
    date_time_begin = datetime.datetime.strptime(date_time_begin, '%Y/%m/%d %H:%M:%S')
    date_time_begin2 = date_time_begin + datetime.timedelta(hours=(24))
    date_time_begin_for_search = date_time_begin.strftime("%Y/%m/%d %H:%M:%S")
    date_time_begin_for_search2 = date_time_begin2.strftime("%Y/%m/%d %H:%M:%S")
    # print(date_time_begin_for_search)
    # print(date_time_begin_for_search2)
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


def getWindData(date_time_begin):
    # 输入datetime格式时间参数，转成str类型，只返回一个数据
    date_time_begin2 = date_time_begin + datetime.timedelta(minutes=(2))
    date_time_begin_for_search = date_time_begin.strftime("%Y/%m/%d %H:%M:%S")
    date_time_begin_for_search2 = date_time_begin2.strftime("%Y/%m/%d %H:%M:%S")
    
    sql = """
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
    windDate = 0
    result1 = cursor.execute(sql)
    for j in result1:
        windDate = j[0]
    return windDate


def getWindSpeed(date_time_begin):
    # 输入时间参数，只返回一个数据
    date_time_begin2 = date_time_begin + datetime.timedelta(minutes=(2))
    date_time_begin_for_search = date_time_begin.strftime("%Y/%m/%d %H:%M:%S")
    date_time_begin_for_search2 = date_time_begin2.strftime("%Y/%m/%d %H:%M:%S")
    
    sql = """
    SELECT
    SYSTEM."wind_2"."风速"
    FROM
    SYSTEM."wind_2"
    WHERE
    SYSTEM."wind_2"."时间" BETWEEN to_date('
    """ + date_time_begin_for_search + """
    ','yyyy-mm-dd hh24:mi:ss') AND to_date('
    """ + date_time_begin_for_search2 + """
    ','yyyy-mm-dd hh24:mi:ss')
    """
    windSpeed = 0
    result1 = cursor.execute(sql)
    for j in result1:
        windSpeed = j[0]
    return windSpeed


def getWindHumidity(date_time_begin):
    # 输入时间参数，只返回一个数据
    date_time_begin2 = date_time_begin + datetime.timedelta(minutes=(2))
    date_time_begin_for_search = date_time_begin.strftime("%Y/%m/%d %H:%M:%S")
    date_time_begin_for_search2 = date_time_begin2.strftime("%Y/%m/%d %H:%M:%S")
    
    sql = """
    SELECT
    SYSTEM."wind_2"."湿度"
    FROM
    SYSTEM."wind_2"
    WHERE
    SYSTEM."wind_2"."时间" BETWEEN to_date('
    """ + date_time_begin_for_search + """
    ','yyyy-mm-dd hh24:mi:ss') AND to_date('
    """ + date_time_begin_for_search2 + """
    ','yyyy-mm-dd hh24:mi:ss')
    """
    windHumidity = 0
    result1 = cursor.execute(sql)
    for j in result1:
        windHumidity = j[0]
    return windHumidity


def getSolarData(date_time_begin):
    # 输入时间参数，只返回一个数据
    date_time_begin2 = date_time_begin + datetime.timedelta(minutes=(2))
    date_time_begin_for_search = date_time_begin.strftime("%Y/%m/%d %H:%M:%S")
    date_time_begin_for_search2 = date_time_begin2.strftime("%Y/%m/%d %H:%M:%S")
    
    sql = """
    SELECT
    SYSTEM."solar_2"."总功率"
    FROM
    SYSTEM."solar_2"
    WHERE
    SYSTEM."solar_2"."时间" BETWEEN to_date('
    """ + date_time_begin_for_search + """
    ','yyyy-mm-dd hh24:mi:ss') AND to_date('
    """ + date_time_begin_for_search2 + """
    ','yyyy-mm-dd hh24:mi:ss')
    """
    solarDate = 0
    result = cursor.execute(sql)
    for j in result:
        solarDate = j[0]
    return solarDate


def getSolarTemp(date_time_begin):
    # 输入时间参数，只返回一个数据
    date_time_begin2 = date_time_begin + datetime.timedelta(minutes=(2))
    date_time_begin_for_search = date_time_begin.strftime("%Y/%m/%d %H:%M:%S")
    date_time_begin_for_search2 = date_time_begin2.strftime("%Y/%m/%d %H:%M:%S")
    
    sql = """
    SELECT
    SYSTEM."solar_2"."温度"
    FROM
    SYSTEM."solar_2"
    WHERE
    SYSTEM."solar_2"."时间" BETWEEN to_date('
    """ + date_time_begin_for_search + """
    ','yyyy-mm-dd hh24:mi:ss') AND to_date('
    """ + date_time_begin_for_search2 + """
    ','yyyy-mm-dd hh24:mi:ss')
    """
    solarTemp = 0
    result = cursor.execute(sql)
    for j in result:
        solarTemp = j[0]
    return solarTemp


def getmainsPowerData(date_time_begin):
    # 输入时间参数，只返回一个数据
    date_time_begin2 = date_time_begin + datetime.timedelta(minutes=(2))
    date_time_begin_for_search = date_time_begin.strftime("%Y/%m/%d %H:%M:%S")
    date_time_begin_for_search2 = date_time_begin2.strftime("%Y/%m/%d %H:%M:%S")
    
    sql = """
    SELECT
    SYSTEM."Sheet_mainsPower"."市电总功率"
    FROM
    SYSTEM."Sheet_mainsPower"
    WHERE
    SYSTEM."Sheet_mainsPower"."时间" BETWEEN to_date('
    """ + date_time_begin_for_search + """
    ','yyyy-mm-dd hh24:mi:ss') AND to_date('
    """ + date_time_begin_for_search2 + """
    ','yyyy-mm-dd hh24:mi:ss')
    """
    mainsPowerDate = 0
    result = cursor.execute(sql)
    for j in result:
        mainsPowerDate = j[0]
    return mainsPowerDate


def getRowNumTimeOfMainsPowerSheet(rownum_in):
    # 输入int类型rownum（方便循环），转成str用于查询某一行的时间
    rownum = rownum_in
    rownum = str(rownum)
    
    sql = """
        select * from (SELECT rownum no , SYSTEM."Sheet_mainsPower"."时间" FROM SYSTEM."Sheet_mainsPower" )
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


def getPreDayTime(daytimenow):
    # 传入当前日期时间datetime格式，返回昨天此刻时间数据
    daytimelast = daytimenow + datetime.timedelta(days=1)
    return daytimelast


def getWeek(date_time_begin):
    # 输入时间参数，只返回一个数据
    date_time_begin2 = date_time_begin + datetime.timedelta(minutes=(2))
    date_time_begin_for_search = date_time_begin.strftime("%Y/%m/%d %H:%M:%S")
    date_time_begin_for_search2 = date_time_begin2.strftime("%Y/%m/%d %H:%M:%S")
    
    sql = """
    SELECT
    SYSTEM."sheet_historyPower"."星期数"
    FROM
    SYSTEM."sheet_historyPower"
    WHERE
    SYSTEM."sheet_historyPower"."时间" BETWEEN to_date('
    """ + date_time_begin_for_search + """
    ','yyyy-mm-dd hh24:mi:ss') AND to_date('
    """ + date_time_begin_for_search2 + """
    ','yyyy-mm-dd hh24:mi:ss')
    """
    week = 0
    result = cursor.execute(sql)
    for j in result:
        week = j[0]
    return week


def getHour(date_time_begin):
    # 输入时间参数，只返回一个数据
    date_time_begin2 = date_time_begin + datetime.timedelta(minutes=(2))
    date_time_begin_for_search = date_time_begin.strftime("%Y/%m/%d %H:%M:%S")
    date_time_begin_for_search2 = date_time_begin2.strftime("%Y/%m/%d %H:%M:%S")
    
    sql = """
    SELECT
    SYSTEM."sheet_historyPower"."当前整点数"
    FROM
    SYSTEM."sheet_historyPower"
    WHERE
    SYSTEM."sheet_historyPower"."时间" BETWEEN to_date('
    """ + date_time_begin_for_search + """
    ','yyyy-mm-dd hh24:mi:ss') AND to_date('
    """ + date_time_begin_for_search2 + """
    ','yyyy-mm-dd hh24:mi:ss')
    """
    hour = 0
    result = cursor.execute(sql)
    for j in result:
        hour = j[0]
    return hour


def getTemp(date_time_begin):
    # 输入时间参数，只返回一个数据
    date_time_begin2 = date_time_begin + datetime.timedelta(minutes=(2))
    date_time_begin_for_search = date_time_begin.strftime("%Y/%m/%d %H:%M:%S")
    date_time_begin_for_search2 = date_time_begin2.strftime("%Y/%m/%d %H:%M:%S")
    
    sql = """
    SELECT
    SYSTEM."sheet_historyPower"."温度"
    FROM
    SYSTEM."sheet_historyPower"
    WHERE
    SYSTEM."sheet_historyPower"."时间" BETWEEN to_date('
    """ + date_time_begin_for_search + """
    ','yyyy-mm-dd hh24:mi:ss') AND to_date('
    """ + date_time_begin_for_search2 + """
    ','yyyy-mm-dd hh24:mi:ss')
    """
    temp = 0
    result = cursor.execute(sql)
    for j in result:
        temp = j[0]
    return temp


def getWind(date_time_begin):
    # 输入时间参数，只返回一个数据
    date_time_begin2 = date_time_begin + datetime.timedelta(minutes=(2))
    date_time_begin_for_search = date_time_begin.strftime("%Y/%m/%d %H:%M:%S")
    date_time_begin_for_search2 = date_time_begin2.strftime("%Y/%m/%d %H:%M:%S")
    
    sql = """
    SELECT
    SYSTEM."sheet_historyPower"."风速"
    FROM
    SYSTEM."sheet_historyPower"
    WHERE
    SYSTEM."sheet_historyPower"."时间" BETWEEN to_date('
    """ + date_time_begin_for_search + """
    ','yyyy-mm-dd hh24:mi:ss') AND to_date('
    """ + date_time_begin_for_search2 + """
    ','yyyy-mm-dd hh24:mi:ss')
    """
    wind = 0
    result = cursor.execute(sql)
    for j in result:
        wind = j[0]
    return wind


def getHumidity(date_time_begin):
    # 输入时间参数，只返回一个数据
    date_time_begin2 = date_time_begin + datetime.timedelta(minutes=(2))
    date_time_begin_for_search = date_time_begin.strftime("%Y/%m/%d %H:%M:%S")
    date_time_begin_for_search2 = date_time_begin2.strftime("%Y/%m/%d %H:%M:%S")
    
    sql = """
    SELECT
    SYSTEM."sheet_historyPower"."湿度"
    FROM
    SYSTEM."sheet_historyPower"
    WHERE
    SYSTEM."sheet_historyPower"."时间" BETWEEN to_date('
    """ + date_time_begin_for_search + """
    ','yyyy-mm-dd hh24:mi:ss') AND to_date('
    """ + date_time_begin_for_search2 + """
    ','yyyy-mm-dd hh24:mi:ss')
    """
    humidity = 0
    result = cursor.execute(sql)
    for j in result:
        humidity = j[0]
    return humidity


def getPower(date_time_begin):
    # 输入时间参数，只返回一个数据
    date_time_begin2 = date_time_begin + datetime.timedelta(minutes=(2))
    date_time_begin_for_search = date_time_begin.strftime("%Y/%m/%d %H:%M:%S")
    date_time_begin_for_search2 = date_time_begin2.strftime("%Y/%m/%d %H:%M:%S")
    
    sql = """
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
    power = 0
    result = cursor.execute(sql)
    for j in result:
        power = j[0]
    return power


def getMaxPower(date_time_begin):
    # 输入时间参数，只返回一个数据
    date_time_begin2 = date_time_begin + datetime.timedelta(minutes=(2))
    date_time_begin_for_search = date_time_begin.strftime("%Y/%m/%d %H:%M:%S")
    date_time_begin_for_search2 = date_time_begin2.strftime("%Y/%m/%d %H:%M:%S")
    
    sql = """
    SELECT
    SYSTEM."sheet_historyPower"."日最大负荷"
    FROM
    SYSTEM."sheet_historyPower"
    WHERE
    SYSTEM."sheet_historyPower"."时间" BETWEEN to_date('
    """ + date_time_begin_for_search + """
    ','yyyy-mm-dd hh24:mi:ss') AND to_date('
    """ + date_time_begin_for_search2 + """
    ','yyyy-mm-dd hh24:mi:ss')
    """
    power = 0
    result = cursor.execute(sql)
    for j in result:
        power = j[0]
    return power


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

# 生成pre表
preNum = getCountOfSheet() - 24
preTime = []
# 得到mainsPower最新的24小时时间
for i in range(24):
    lastTimeOfMainsPower = getRowNumTimeOfMainsPowerSheet((preNum + i + 1))
    preTime.append(lastTimeOfMainsPower)
print(preTime)
historyDate = getHistorySheet()
print("history")
print(historyDate)
print(type(historyDate))
# print(preTime)
# print(type(preTime))

# 利用时间查询history表里的数据，存到prePower表
# for i in range(24):
#     storePreList = []
#     preDate = getPreDayTime(preTime[i])
#     preWeek = getWeek(preTime[i]) + 1
#     if preWeek == 8:
#         preWeek = 1
#     preHour = getHour(preTime[i])
#     preTemp = getTemp(preTime[i])
#     preWind = getWind(preTime[i])
#     preHumidity = getHumidity(preTime[i])
#     prePower = getPower(preTime[i])
#     preMaxPower = getMaxPower(preTime[i])
#     storePreList.append(preDate)
#     storePreList.append(preWeek)
#     storePreList.append(preHour)
#     storePreList.append(preTemp)
#     storePreList.append(preWind)
#     storePreList.append(preHumidity)
#     storePreList.append(prePower)
#     storePreList.append(preMaxPower)
#     print(storePreList)


conn.commit()
cursor.close()
conn.close()
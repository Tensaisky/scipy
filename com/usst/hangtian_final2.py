# -*- coding: utf-8 -*-
import cx_Oracle
import datetime
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sqlalchemy import create_engine
from xgboost import XGBRegressor
from time import sleep

def getFistTimeOfSheet():
    sql = """
    select SYSTEM."Sheet_mainsPower"."时间" from SYSTEM."Sheet_mainsPower" where rownum<2
    """
    result = cursor.execute(sql)
    for i in result:
        date_time_begin = i[0]
    return date_time_begin
def getLastDayTime(daytimenow):
    daytimelast = daytimenow - datetime.timedelta(days=1)
    return daytimelast
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
def getCountOfPreSheet():
    sql = """
        select count(*) from SYSTEM."sheet_prePower"
    """
    result = cursor.execute(sql)
    count = 0
    for i in result:
        count = i[0]
    return count
def storeData(list):
    sql = 'INSERT INTO SYSTEM."sheet_historyPower"(\"时间\",\"星期数\",\"当前整点数\",\"昨日此刻温度\",\"昨日此刻风速\",\"昨日此刻湿度\",\"昨日此刻负荷\",\"温度\",\"风速\",\"湿度\",\"负荷\") VALUES(:1,:2,:3,:4,:5,:6,:7,:8,:9,:10,:11)'
    cursor.execute(sql, list)
def storeMaxPower(date_time_begin,maxPower):
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
    for i in range(num):
        time = getRowNumTimeOfSheet(i + 1)
        dayRecord_list = oneDayRecord(time)
        allPower = 0
        for j in range(len(dayRecord_list)):
            allPower = allPower + dayRecord_list[j]
        averagePower = allPower / len(dayRecord_list)
        storeAveragePower(time, averagePower)
        maxPower = 0
        for k in range(len(dayRecord_list)):
            if dayRecord_list[k] > maxPower:
                maxPower = dayRecord_list[k]
        storeMaxPower(time, maxPower)
def storePreData(list):
    sql = 'INSERT INTO SYSTEM."sheet_prePower"(\"时间\",\"星期数\",\"当前整点数\",\"昨日此刻温度\",\"昨日此刻风速\",\"昨日此刻湿度\",\"昨日此刻负荷\",\"日最大负荷\") VALUES(:1,:2,:3,:4,:5,:6,:7,:8)'
    cursor.execute(sql, list)
def storePrePower(date_time_begin, prePower):
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
def getRowNumTimeOfSheet(rownum_in):
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
    return date_time
def hasNoRecord(date_time_begin):
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
def oneDayRecord(date_time_begin):
    date_time_begin = date_time_begin.strftime("%Y/%m/%d %H:%M:%S")
    date_time_begin = date_time_begin.split(' ')[0]
    date_time_begin = date_time_begin + ' 00:00:00'
    date_time_begin = datetime.datetime.strptime(date_time_begin, '%Y/%m/%d %H:%M:%S')
    date_time_begin2 = date_time_begin + datetime.timedelta(hours=(24))
    date_time_begin_for_search = date_time_begin.strftime("%Y/%m/%d %H:%M:%S")
    date_time_begin_for_search2 = date_time_begin2.strftime("%Y/%m/%d %H:%M:%S")
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
        dayRecord_list.append(j[0])
    return dayRecord_list
def getWindData(date_time_begin):
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
    return date_time
def getPreDayTime(daytimenow):
    daytimelast = daytimenow + datetime.timedelta(days=1)
    return daytimelast
def getWeek(date_time_begin):
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
    return date_time

while(1):
    os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
    conn = cx_Oracle.connect('SYSTEM', '123456', 'localhost:1521/XE')
    cursor = conn.cursor()
    countOfData = getCountOfSheet()
    for rownum in range(countOfData):
        no = rownum + 1
        date_time = getRowNumTimeOfSheet(no)
        date_time = date_time - datetime.timedelta(minutes=1)
        date_time_last = getLastDayTime(date_time)
        lasttemp = getSolarTemp(date_time_last)
        lastspeed = getWindSpeed(date_time_last)
        lasthumidity = getWindHumidity(date_time_last)
        lastwindDate = getWindData(date_time_last)/1000
        lastsolarDate = getSolarData(date_time_last)
        lastmainsPower = getmainsPowerData(date_time_last)
        lastpower_all = lastwindDate + lastsolarDate + lastmainsPower
        temp = getSolarTemp(date_time)
        speed = getWindSpeed(date_time)
        humidity = getWindHumidity(date_time)
        windDate = getWindData(date_time)/1000
        solarDate = getSolarData(date_time)
        mainsPower = getmainsPowerData(date_time)
        power_all = windDate + solarDate + mainsPower
        date_time = date_time + datetime.timedelta(minutes=1)
        time_hour = date_time.strftime("%Y/%m/%d %H:%M:%S")
        time_hour = time_hour.split(' ')[1].split(':')[0]
        time_hour =  int(time_hour)
        date_time_str = date_time.strftime("%Y/%m/%d %H:%M:%S")
        week = datetime.datetime.strptime(date_time_str,'%Y/%m/%d %H:%M:%S').weekday()
        week = week + 1
        values = [date_time,week,time_hour,lasttemp,lastspeed,lasthumidity,lastpower_all,temp,speed,humidity,power_all]
        if hasNoRecord(date_time):
            storeData(values)
            conn.commit()
        else:
            do_nothing = 1
    countOfHistory = getCountOfHistorySheet()
    loopStoreMaxAndAverage(getCountOfHistorySheet())
    preNum = getCountOfSheet() - 48
    preTime = []
    for i in range(48):
        lastTimeOfMainsPower = getRowNumTimeOfMainsPowerSheet((preNum + i + 1))
        preTime.append(lastTimeOfMainsPower)
    for i in range(48):
        storePreList = []
        preDate = getPreDayTime(preTime[i])
        preWeek = getWeek(preTime[i]) + 1
        if preWeek == 8:
            preWeek = 1
        preHour = getHour(preTime[i])
        preTemp = getTemp(preTime[i])
        preWind = getWind(preTime[i])
        preHumidity = getHumidity(preTime[i])
        prePower = getPower(preTime[i])
        preMaxPower = getMaxPower(preTime[i])
        storePreList.append(preDate)
        storePreList.append(preWeek)
        storePreList.append(preHour)
        storePreList.append(preTemp)
        storePreList.append(preWind)
        storePreList.append(preHumidity)
        storePreList.append(prePower)
        storePreList.append(preMaxPower)
        if hasNoPreRecord(preDate):
            storePreData(storePreList)
            conn.commit()
        else:
            do_nothing = 1
    engine = create_engine('oracle+cx_oracle://SYSTEM:123456@localhost:1521/XE')
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
    number = getCountOfPreSheet()
    preDate = []
    for num in range(number):
        preDate.append(getRowNumTimeOfPrePowerSheet(num+1))
    for i in range(number):
        storePrePower(preDate[i],origin_predict_y[i])
    
    conn.commit()
    cursor.close()
    conn.close()
    print("predict done")
    sleep(300)
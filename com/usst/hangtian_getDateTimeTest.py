# -*- coding: utf-8 -*-
import cx_Oracle
import datetime
import os
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
conn = cx_Oracle.connect('system','tiger','192.168.1.106:1521/orcl1')
cursor = conn.cursor()

def getFistTimeOfSheet():
    # 取第一个行的时间数据，以此时间为查询筛选标准
    # 第一个为表中起始数据时间，第二个为第一个时间加两分钟时间，筛选数据用
    # 返回数据格式为datetime
    sql = """
    select SYSTEM."Sheet_mainsPower"."时间" from SYSTEM."Sheet_mainsPower" where rownum<2
    """
    result = cursor.execute(sql)
    for i in result:
        date_time_begin = i[0]
    date_time_begin2 = date_time_begin + datetime.timedelta(minutes=(2))
    return [date_time_begin,date_time_begin2]

def getWindDate(date_time_begin,date_time_begin2):
    # 输入时间参数，str类型，只返回一个数据
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

def getWindSpeed(date_time_begin, date_time_begin2):
    # 输入时间参数，str类型，只返回一个数据
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

def getWindHumidity(date_time_begin, date_time_begin2):
    # 输入时间参数，str类型，只返回一个数据
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

def getSolarDate(date_time_begin, date_time_begin2):
    # 输入时间参数，str类型，只返回一个数据
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

def getmainsPowerDate(date_time_begin, date_time_begin2):
    # 输入时间参数，str类型，只返回一个数据
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

date_time_begin = getFistTimeOfSheet()[0]
date_time_begin2 = getFistTimeOfSheet()[1]

windDate = getWindDate(date_time_begin,date_time_begin2)
solarDate = getSolarDate(date_time_begin,date_time_begin2)
mainsPower = getmainsPowerDate(date_time_begin,date_time_begin2)

power_all = windDate + solarDate + mainsPower
print(power_all)

windSpeed = getWindSpeed(date_time_begin,date_time_begin2)
print(windSpeed)

conn.commit()
cursor.close()
conn.close()


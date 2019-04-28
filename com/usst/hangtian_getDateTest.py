# -*- coding: utf-8 -*-
import cx_Oracle
import datetime
import os
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
conn = cx_Oracle.connect('system','tiger','192.168.1.108:1521/orcl1')
cursor = conn.cursor()

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

def getCountOfSheet():
    sql = """
        select count(*) from SYSTEM."Sheet_mainsPower"
    """
    result = cursor.execute(sql)
    count = 0
    for i in result:
        count = i[0]
    return count

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

def getWindDate(date_time_begin):
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
    # 输入时间参数，str类型，只返回一个数据
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
    # 输入时间参数，str类型，只返回一个数据
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

def getSolarDate(date_time_begin):
    # 输入时间参数，str类型，只返回一个数据
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
    # 输入时间参数，str类型，只返回一个数据
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

def getmainsPowerDate(date_time_begin):
    # 输入时间参数，str类型，只返回一个数据
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

date_time_begin = getFistTimeOfSheet()

windDate = getWindDate(date_time_begin)
solarDate = getSolarDate(date_time_begin)
mainsPower = getmainsPowerDate(date_time_begin)

power_all = windDate + solarDate + mainsPower
print(power_all)

print(getCountOfSheet())

conn.commit()
cursor.close()
conn.close()


import datetime

today = datetime.datetime.now().weekday()
print(today)

date_time = '2019/3/26 11:00:00'
date_time = datetime.datetime.strptime(date_time,'%Y/%m/%d %H:%M:%S')
date_time_week = date_time.weekday()
print(date_time_week)
print(type(date_time_week))
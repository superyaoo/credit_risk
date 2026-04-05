import numpy as np
import pandas as pd
import pytz
import time
import datetime
import re

# 引入
# import sys
# sys.path.append(r"/euler/")
# from euler_function.datetime_tools import timestamp_to_strftime,timestamp_to_datetime,diff_days_between_df_col

#时间戳转对应时区的str格式: df.created.apply(lambda x:timestamp_to_strftime(x,timezone_str="Asia/Kolkata",str_formate = "%Y-%m-%d")
def timestamp_to_strftime(timestamp,timezone_str,str_formate = "%Y-%m-%d %H:%M:%S"):
    """
    timestamp: 时间戳
    timezone_str: 例如'Asia/Kolkata' #印度 加尔各答； 'America/Bogota' #哥伦比亚 波哥大
    str_formate：输出字符串格式,默认是"%Y-%m-%d %H:%M:%S"
    return: 目标时区的时间str格式
    """
    if pd.isna(timestamp):
        return np.nan
    if len(str(timestamp)) >=13:
        timestamp = int(int(timestamp)/1000)
    # 将时间戳转换为 UTC 时间
    utc_dt = datetime.datetime.utcfromtimestamp(timestamp)
    # 创建时区对象
    local_tz = pytz.timezone(timezone_str)
    # 将 UTC 时间转换为指定时区的时间
    local_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(local_tz)
    return local_dt.strftime(str_formate)

# 方法一：计算时间戳2列的间隔天数,下面2个方法
# df_app_data['create_time'] = timestamp_to_datetime(df_app_data['create_stamp'],timezone_str="Asia/Kolkata")
# df_app_data['install_days']= diff_days_between_df_col(df_app_data.install_time,df_app_data.create_time) 

def timestamp_to_datetime(timestamp,timezone_str):
    # 将时间戳转为datatime
    """
    timestamp:df的一列pd.Series，时间戳格式，必须是13位；
    imezone_str: 例如'Asia/Kolkata' #印度 加尔各答； 'America/Bogota' #哥伦比亚 波哥大
    return: 返回对应时区的pd.Series，datetime格式；
    """
    if len(str(timestamp)) ==10:
        timestamp = int(timestamp*1000)
    ret = pd.to_datetime(timestamp, unit='ms')
    ret = ret.dt.tz_localize('UTC').dt.tz_convert(timezone_str)
    return ret

def diff_days_between_df_col(columns_old: pd.Series, columns_new: pd.Series) -> pd.Series:
    """
    columns_old:df的一列pd.Series，datetime格式；
    columns_new:df的一列pd.Series，datetime格式；
    return: (columns_new-columns_old)的间隔天数pd.Series，int类型
    """
    # 对两列都进行相同的处理，确保NA值被转换为pd.NaT
    columns_old = columns_old.where(pd.notna(columns_old), pd.NaT)
    columns_new = columns_new.where(pd.notna(columns_new), pd.NaT)

    # 获取差异的天数
    diff_days = (columns_new - columns_old).dt.days

    # 使用numpy的where函数处理NA情况
    diff_days = np.where(pd.notna(diff_days), diff_days, np.nan)

    return pd.Series(diff_days, index=columns_old.index)


#datetime时间格式转化为目标地时间格式
#current_timezone_str = 'UTC'
#target_timezone_str = 'America/Bogota'
def current_datetime_to_target_datetime(current_time,current_timezone_str,target_timezone_str):
    """
    current_time:服务器时间，datetime格式；
    current_timezone_str: 服务器时区str
    target_timezone_str: 目标时区str
    return：目的地时间,datetime格式；
    """
    current_tz = pytz.timezone(current_timezone_str)
    target_tz = pytz.timezone(target_timezone_str)

    current_time = current_tz.localize(current_time)
    target_time = current_time.astimezone(target_tz)
    return target_time


# +
def get_week_date(date):
    """
    date:为时间
    first_day:自然周的开始日期
    last_day:自然周的结束日期
    """
    iso_year, iso_week, _ = date.isocalendar()
    if date.month == 12 and iso_week == 1:
        iso_year += 1
    first_day = datetime.datetime.strptime(f'{iso_year}-W{iso_week}-1', "%Y-W%W-%w").date()
    last_day = first_day + datetime.timedelta(days=6)        
    return first_day, last_day




def get_week_range(year, week_num):
    """
    year:年份；
    week_num:当年的自然周
    first_day:自然周的开始日期
    last_day:自然周的结束日期
    """
    first_day = datetime.datetime.strptime(f'{year}-W{week_num}-1', "%Y-W%W-%w").date()
    last_day = first_day + datetime.timedelta(days=6)
    return first_day, last_day

def get_year_week_range(year):
    """
    year:年份;
    week_ranges:自然周及对应的时间区间
    """
    start_date = datetime.date(year, 1, 1)
    end_date = datetime.date(year, 12, 31)
    week_ranges = []
    while start_date <= end_date:
        week_num = start_date.isocalendar()[1]
        first_day, last_day = get_week_range(year, week_num)
        week_ranges.append((week_num, first_day, last_day))
        start_date += datetime.timedelta(days=7)
    return week_ranges

import os
import re
import traceback

import pandas as pd
import pymysql
import pytz
from sklearn.metrics import roc_auc_score
from sqlalchemy import create_engine
from datetime import timedelta, datetime
import functools

from tqdm import tqdm
import numpy as np

import logging

logger = logging.getLogger(__name__)

month_format = '%Y-%m'
day_formart = '%Y-%m-%d'
time_formart = '%Y-%m-%d %H:%M:%S'



def week_start_day(date_obj):
    """
    获取该日期对象，对应的每周第一天
    :param date_obj:
    :return:
    """
    weekday = date_obj.weekday()
    start_of_week = date_obj - timedelta(days=weekday)
    return start_of_week.strftime('%Y-%m-%d')


tidb_conf = {
    # 内网访问链接
    'inner': {'mx': ('172.20.1.186', 3306, 'AM', 'am_system'),
              'cl': ('172.20.1.186', 3306, 'AC', 'ac_system'),
              'co': ('172.20.1.186', 3306, 'AG', 'ag_system'),
              'ec': ('172.20.1.186', 3306, 'AEC', 'aec_system'),
              'pe': ('172.20.1.186', 3306, 'AL', 'al_system'),
              'th': ('172.16.0.12', 3306, 'ATH', 'ath_system'),
              'tz': ('110.238.79.30', 4000, 'AT', 'at_system'),
              'id': ('172.16.0.12', 3306, 'AF', 'af_system')
              },
    # 外网访问链接
    'outer': {
        'mx': ('47.253.106.38', 4000, 'AM', 'am_system'),
        'cl': ('47.253.106.38', 4000, 'AC', 'ac_system'),
        'co': ('47.253.106.38', 4000, 'AG', 'ag_system'),
        'ec': ('47.253.106.38', 4000, 'AEC', 'aec_system'),
        'pe': ('47.253.106.38', 4000, 'AL', 'al_system'),
        'th': ('8.215.28.149', 4000, 'ATH', 'ath_system'),
        'tz': ('110.238.79.30', 4000, 'AT', 'at_system'),
        'id': ('8.215.28.149', 4000, 'AF', 'af_system')
    }
}

doris_con = {
    # 内网访问链接
    'inner': {'mx': ('172.20.1.13', 9030, 'AM', 'am_system'),
              'cl': ('172.20.1.13', 9030, 'AC', 'ac_system'),
              'co': ('172.20.1.13', 9030, 'AG', 'ag_system'),
              'ec': ('172.20.1.13', 9030, 'AEC', 'aec_system'),
              'pe': ('172.20.1.13', 9030, 'AL', 'al_system'),
              'th': ('172.16.0.143', 9030, 'ATH', 'ath_system'),
              'id': ('172.16.0.143', 9030, 'AF', 'af_system'),
              'mw': ('172.20.1.13', 9030, 'AM',  'mw_system')
              },
    # 外网访问链接
    'outer': {
        'mx': ('47.253.56.86', 9030, 'AM', 'am_system'),
        'cl': ('47.253.56.86', 9030, 'AC', 'ac_system'),
        'co': ('47.253.56.86', 9030, 'AG', 'ag_system'),
        'ec': ('47.253.56.86', 9030, 'AEC', 'aec_system'),
        'pe': ('47.253.56.86', 9030, 'AL', 'al_system'),
        'th': ('8.215.28.149', 9030, 'ATH', 'ath_system'),
        'id': ('8.215.28.149', 9030, 'AF', 'af_system'),
        'mw': ('47.253.56.86', 9030, 'AM', 'mw_system')
    }
}

# 每个国家对应的country_id和与北京时间的时差
country_info = {"mx": ("52", -13),
                "cl": ("56", -13),
                "pe": ("51", -13),
                "co": ("57", -13),
                "ec": ("593", -13),
                "th": ("66", -1),
                "ph": ("63", 0),
                "in": ("91", -1),
                "id": ("62", -1),
                "ng": ("234", -1),
                "tz": ("255", -6),
                'mw': ("52", -13)}

service_prot_mapping = {
    "app_mx": 5100,
    "sms_mx": 5101,
    "call_mx": 5102,
    "loan_mx": 5103,
    "app_co": 5104,
    "sms_co": 5105,
    "call_co": 5106,
    "loan_co": 5107,
    "app_cl": 5108,
    "sms_cl": 5109,
    "call_cl": 5110,
    "loan_cl": 5111,
    "app_pe": 5112,
    "sms_pe": 5113,
    "call_pe": 5114,
    "loan_pe": 5115,
    "app_ec": 5116,
    "sms_ec": 5117,
    "call_ec": 5118,
    "loan_ec": 5119,
    "ui_cl": 5124,
    "ui_co": 5125,
    "ui_ec": 5126,
    "ui_mx": 5127,
    "ui_pe": 5128,
    "dev_cl": 5150,
    "dev_co": 5151,
    "dev_ec": 5152,
    "dev_mx": 5153,
    "dev_pe": 5154,
    "seq_mx": 5141,
    "seq_ec": 5142,
    "seq_cl": 5143,
    "seq_co": 5144,
    "seq_pe": 5145,
    "seq_th": 5146,
    "dev_th": 5155,
    "app_th": 5120,
    "sms_th": 5121,
    "call_th": 5122,
    "loan_th": 5123,
    "app_id": 5129,
    "sms_id": 5130,
    "call_id": 5131,
    "loan_id": 5132,
    "ui_th": 5137,
    "app_tz": 5138,
    "sms_tz": 5139,
    "call_tz": 5140,
    "dev_tz": 5156,
    "call_ng": 5157,
    "sms_ng": 5158,
    "app_ng": 5159,
    "cross_mx": 5164,
    "cross_id": 5136,
    "cross_ec": 5160,
    "cross_co": 5147,
    "cross_cl": 5148,
    "dev_id": 5135,
    "ui_id": 5134,
    "fdc_id": 5162
}

COUNTRY_TIME_ZONE = {"mx": -6,
                     "cl": -4,
                     "pe": -5,
                     "co": -5,
                     "ec": -5,
                     "th": 7,
                     "ph": 8,
                     "id": 7,
                     "in": 5.5,
                     "ng": 1,
                     "tz": 3}

COUNTRY_LANGUAGE={
    'mx': 'spanish',
    'cl': 'spanish',
    'pe': 'spanish',
    'co': 'spanish',
    'ec': 'spanish',
    'ng':'english',
    'th':'thai',
    'id':'english'
}


DORIES_INFO = {
    'inner':{
        'Asia':{'ip':'172.16.0.143','port':9030},
        'LatAm':{'ip':'172.20.1.13','port':9030}
    },
    'outer':{
        'Asia': {'ip': '8.215.28.149', 'port': 9030},
        'LatAm': {'ip': '47.253.56.86', 'port': 9030}
    }
}

SYS_INFO = {
    'am': {'continents':'LatAm','db': 'am_system', 'merchant_id': 'AM', 'country_abbr': 'mx',
            'country_id': "52", 'time_zone': -6,'china_time_delt': -14, 'language': 'spanish' },
    'm': {'continents':'LatAm','db': 'am_system', 'merchant_id': 'AM', 'country_abbr': 'mx',
            'country_id': "52", 'time_zone': -6,'china_time_delt': -14, 'language': 'spanish' },
    'mw': { 'continents':'LatAm', 'db': 'mw_system', 'merchant_id': 'MW', 'country_abbr': 'mx',
           'country_id': "52", 'time_zone': -6, 'china_time_delt': -14, 'language': 'spanish'},
    'ac': { 'continents':'LatAm', 'db': 'ac_system', 'merchant_id': 'AC', 'country_abbr': 'cl',
           'country_id': "56", 'time_zone': -4, 'china_time_delt': -12, 'language': 'spanish'},
    'aec': {'continents': 'LatAm', 'db': 'aec_system', 'merchant_id': 'AEC', 'country_abbr': 'ec',
            'country_id': "593", 'time_zone': -5, 'china_time_delt': -13, 'language': 'spanish'},
    'al': { 'continents':'LatAm', 'db': 'ag_system', 'merchant_id': 'AL', 'country_abbr': 'co',
           'country_id': "51", 'time_zone': -5, 'china_time_delt': -13, 'language': 'spanish'},
    'ag': { 'continents':'LatAm', 'db': 'ag_system', 'merchant_id': 'AG', 'country_abbr': 'co',
           'country_id': "57", 'time_zone': -5, 'china_time_delt': -13, 'language': 'spanish'},
    'ath': {'continents':'Asia','db': 'ath_system', 'merchant_id': 'ATH', 'country_abbr': 'th',
            'country_id': "66", 'time_zone': 7, 'china_time_delt': -1, 'language': 'tai'},
    'athl': {'continents': 'Asia', 'db': 'athl_system', 'merchant_id': 'ATHL', 'country_abbr': 'th',
            'country_id': "66", 'time_zone': 7, 'china_time_delt': -1, 'language': 'tai'},
    'athu': {'continents': 'Asia', 'db': 'athu_system', 'merchant_id': 'ATHU', 'country_abbr': 'th',
             'country_id': "66", 'time_zone': 7, 'china_time_delt': -1, 'language': 'tai'},
    'af': { 'continents':'Asia', 'db': 'af_system', 'merchant_id': 'AF', 'country_abbr': 'id',
           'country_id': "62", 'time_zone': 7, 'china_time_delt': -1, 'language': 'english'},
}

def get_sys_info(sys_name, dblink='inner')->dict:
    """
    基于系统的名称获取
    """
    sys_info = SYS_INFO[sys_name]
    dories_info = DORIES_INFO[dblink][sys_info['continents']]
    sys_info.update(dories_info)
    return sys_info

def local_time(sys_name):
    """
    返回当前国家的业务时间，
    :param sys_name: 对应国家的id
    :return: country_time,china_time 的tuple
    """
    time_delt = SYS_INFO[sys_name]['china_time_delt']
    china_time = datetime.now(pytz.timezone('Asia/Shanghai')).strftime(time_formart)
    country_time = pd.to_datetime(datetime.strptime(china_time, time_formart) + timedelta(hours=time_delt))
    return country_time,pd.to_datetime(china_time)

def get_dbcon(sys_name, user, passwd, dblink='inner'):
    """
    基于国家简称获取相应的tidb的数据库连接
    :param sys_name: 系统名称
    :param user: tidb账号
    :param passwd: tidb密码
    :param dblink: 内网网ip的， 默认inner 获取内网 ip ，outer 为外网
    :return:
    """
    sys_info = get_sys_info(sys_name,dblink)
    host = sys_info['ip']
    port = sys_info['port']
    db = sys_info['db']
    try:
        engine = create_engine(f'mysql+pymysql://{user}:{passwd}@{host}:{port}/{db}')
        return engine
    except Exception as e:
        logger.info(f"An error occurred: {e}")


def country_now(country_abbr):
    """
    返回当前国家的业务时间，
    :param country_abbr: 对应国家的id
    :return: country_time,china_time 的tuple
    """
    country_id, time_delt = country_info[country_abbr]
    china_time = datetime.now(pytz.timezone('Asia/Shanghai')).strftime(time_formart)
    country_time = pd.to_datetime(datetime.strptime(china_time, time_formart) + timedelta(hours=time_delt))
    return country_time,pd.to_datetime(china_time)

def time_trans(time, country_abbr):
    """
    根据时区信息，将多种格式时间格式转化为UTC-0的时间
    """
    from datetime import datetime, timedelta

    tz = COUNTRY_TIME_ZONE[country_abbr.lower()]
    time_len = len(str(time))
    time_format = "%a %b %d %H:%M:%S GMT%z %Y"
    if time_len == 10:
        format_time = datetime.utcfromtimestamp(int(time)) + timedelta(hours=tz)
    elif time_len == 13:
        format_time = datetime.utcfromtimestamp(int(time) // 1000) + timedelta(hours=tz)
    elif time_len == 34:
        format_time = datetime.strptime(str(time), time_format)
    else:
        format_time = datetime(2099, 12, 31, 23, 59, 59)

    return format_time


def parallel_process(task_function, task_list, process_num=10, tqdm_desc='task caculation'):
    """
    并发跑批函数,使用子进程的方式开启并发跑批任务，避免计算密集型的进程池中子进程内存不会定时GC的问题
    :param task_function: function
    :param task_list:
    :param process_num:
    :return:
    """

    def _task(task_function, queue, task_params):
        try:
            task_function(*task_params)
        finally:
            queue.put(True)

    from multiprocessing import Queue, Process
    tq = Queue()
    for i in range(process_num):
        tq.put(True)
    for task_params in tqdm(task_list, desc=tqdm_desc):
        tq.get()
        Process(target=_task, args=(task_function, tq, task_params)).start()
    for i in range(process_num):
        tq.get()
    tq.close()


def numerical_univerate(df: pd.DataFrame, feature: str, target: str, labels: list = None, mean_cols: list = [],
                        bins: int = 10, lamb: float = 0.001):
    feature_bin = f"{feature}_bin"
    if labels is None:
        df[feature_bin] = pd.qcut(df[feature], bins, duplicates='drop')
    else:
        df[feature_bin] = pd.cut(df[feature], labels)
    t_t = df[target].count()
    b_t = df[target].sum()
    g_t = t_t - b_t
    br = b_t / t_t
    dti = df.groupby(feature_bin).agg(
        bad=(target, 'sum'),
        total=(target, 'count'),
        brate=(target, 'mean')
    )
    dti['total_rate'] = dti['total'] / t_t
    dti['good'] = dti['total'] - dti['bad']
    dti['lift'] = dti['brate'] / br
    for mean_col in mean_cols:
        dti[mean_col] = df.groupby(feature_bin)[mean_col].mean()
    corr = df[feature].corr(df[target])
    # corr, _ = pointbiserialr(df[target], df[feature])  # 因为是二元分类问题，采用
    if corr < 0:
        dti = dti.sort_values(feature_bin, ascending=False)

    dti["woe"] = np.log(
        ((dti["good"] / g_t) + lamb) / ((dti["bad"] / b_t) + lamb)
    )
    dti['good_cum'] = dti["good"].cumsum()
    dti['bad_cum'] = dti["bad"].cumsum()
    dti['brate_cum'] = dti['bad_cum'] / (dti['good_cum'] + dti['bad_cum'])

    dti["ks"] = np.abs((dti["bad_cum"] / b_t) - (dti["good_cum"] / g_t))
    dti["iv"] = (dti["good"] / g_t - dti["bad"] / b_t) * dti["woe"]
    dti['iv'] = dti['iv'].sum()

    dti = dti.reset_index()
    dti.columns.name = None
    dti['bin_code'] = dti[feature_bin].cat.codes

    def _cum_calc_auc(n):
        if corr > 0:
            codes_list = range(0, n + 1)
        else:
            max_codes = dti['bin_code'].max()
            codes_list = range(n, max_codes + 1)

        df_new = df[df[feature_bin].cat.codes.isin(codes_list)]
        if (len(df_new) == 0) or ( df_new[target].nunique() == 1 ):
            auc = 0

        else:
            auc = roc_auc_score(df_new[target], df_new[feature])
            auc = auc if auc > 0.5 else 1 - auc
        return auc

    dti['auc_cum'] = dti['bin_code'].map(_cum_calc_auc)
    dti.rename({feature_bin: 'bin'}, axis=1, inplace=True)
    dti.insert(0, "target", [target] * dti.shape[0])
    dti.insert(0, "feature", [feature] * dti.shape[0])
    return dti[['feature', 'target', 'bin', 'bad', 'total', 'total_rate', 'brate', 'brate_cum', 'lift','woe', 'ks', 'iv',
                'auc_cum'] + mean_cols]


def list2sqlstr(arr: list):
    return "(" + ",".join(f"'{item}'" for item in arr) + ")"

def chunk_list(lst, chunk_size=1000):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def partition_path_parser(path):
    import re
    matchs = re.findall(r"([\w-]+)=([\w.-]+)", path)
    d = { k:v for k,v in matchs }
    return '/'.join([ f'{k}={v}' for k,v in matchs]),d

def data_of_dir(dir_path, contains_flags="", start_time=None, end_time=None):
    """
    基于关键字以及时间来扫描文件目录中的数据块
    :param dir_path: 待扫描的目录
    :param contains_flags: 关键字标签，同时支持 字符串以及 字符串数组两种格式
    :param start_time: 开始时间，如果 是'2023-01-01' 格式，则按照日期进行扫描，否则为月份扫描
    :param end_time: 结束时间，本函数采用左开右开的方式进行时间范围框定
    :return:
    """

    def check_date_format(date_str):
        patterns = {
            r"^\d{4}-\d{2}$": "Month",
            r"^\d{4}-\d{2}-\d{2}$": "Dayt"
        }
        for pattern, label in patterns.items():
            if re.match(pattern, date_str):
                return label
        return "Unknown"

    def _fetch_filenams(dir_path: str, contain_flag, start_time=None, end_time=None):
        file_paths = []
        contain_flag = contain_flag or ""
        if start_time is not None:
            time_type = check_date_format(start_time)
            if time_type == 'Dayt':
                pattern = r"\d{4}-\d{2}-\d{2}"
                end_time = '2999-12-01' if end_time is None else end_time
            elif time_type == 'Month':
                pattern = r"\d{4}-\d{2}"
                end_time = '2999-12' if end_time is None else end_time
            else:
                raise ValueError(f'调用函数的入参start_time={start_time}，格式错误')
        for file_name in os.listdir(dir_path):
            if (contain_flag in file_name) and (
            file_name.endswith(('.feather','.fth', '.pqt', '.parquet', '.csv', '.xlsx', '.pickle', '.pkl'))):
                if start_time is None:
                    file_paths.append(os.path.join(dir_path, file_name))
                else:
                    match = re.search(pattern, file_name)
                    time = match.group()
                    if (time >= start_time) and (time <= end_time):
                        file_paths.append(os.path.join(dir_path, file_name))
        file_paths.sort()
        return file_paths

    if isinstance(contains_flags, str) or contains_flags is None:
        return _fetch_filenams(dir_path, contains_flags, start_time, end_time)
    elif isinstance(contains_flags, list):
        file_names = None
        for contains_flag in contains_flags:  # type: ignore
            if file_names is None:
                file_names = _fetch_filenams(dir_path, contains_flag, start_time, end_time)
            else:
                file_names = file_names + _fetch_filenams(dir_path, contains_flag, start_time, end_time)
        return file_names

def batch_load_data(file_paths, load_function=pd.read_parquet):
    result = None
    for file_path in tqdm(file_paths):
        try:
            if result is None:
                result = load_function(file_path)
            else:
                result = pd.concat([result, load_function(file_path)])
        except Exception:
            print(f"文件加载异常:{file_path},跳过该文件\n{traceback.format_exc()}")
    return result

def load_datas(dir_path: str, contains_flags = "", start_time = None, end_time = None,load_function=pd.read_parquet):
    """批量加载文件夹中符合要求的数据
    注:为了限制数据格式的准确性，目前仅开放了 '.feather','.fth', '.pqt', '.parquet', '.csv', '.xlsx', '.pickle', '.pkl'结尾的数据。
    :param dir_path: 待扫描的目录
    :param contains_flags: 关键字标签，同时支持 字符串以及 字符串数组两种格式
    :param start_time: 开始时间，如果 是'2023-01-01' 格式，则按照日期进行扫描，否则为月份扫描
    :param end_time: 结束时间，本函数采用左开右开的方式进行时间范围框定
    :param load_function: 单独读取数据的函数
    """
    data_paths = data_of_dir(dir_path, contains_flags, start_time, end_time)
    df = batch_load_data(data_paths, load_function)
    return df


def random_list(l, n_split=20, random_state=42):
    """
    根据长度随机的生成切片序号
    :param l: 需要生成的数据的长度
    :param n_split: 切分多少片
    :param random_state: 随机种子
    :return:
    """
    arr = np.asarray(range(l))
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_split, shuffle=True, random_state=random_state)
    i = 0
    for i_train, i_test in kf.split(arr):
        arr[i_test] = i
        i = i + 1
    return arr


def gen_random_col(df, chunk_base_col='phone_number', n_split=20, random_state=42, random_col='random_no'):
    """
    根据指定得列生成随机切块编码
    :param df: 待生成随机编码得数据
    :param chunk_base_col: 基于指定得列随机生成编号
    :param random_state: 随机种子
    :param n_split: 切分多少片
    :param random_state: 随机种子
    :param random_col: 新生成得随机编码得字段名
    """
    unique_arr = df[chunk_base_col].unique()
    l = len(unique_arr)
    split_codes = random_list(l, n_split=n_split, random_state=random_state)
    map_s = pd.Series(split_codes, index=unique_arr)
    df[random_col] = df[chunk_base_col].map(map_s)
    return df


def ngrams(tokens, n):
    """生成给定长度的N-grams"""
    tokens = [str(x) for x in tokens]
    return [' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


class DBConnection:
    """数据库连接管理类"""
    
    def __init__(self, sys_name, user, passwd, dblink='inner'):
        """
        初始化数据库连接参数
        :param sys_name: 系统名称
        :param user: 数据库用户名
        :param passwd: 数据库密码
        :param dblink: 连接类型，'inner'为内网，'outer'为外网
        """
        self.sys_name = sys_name
        self.user = user
        self.passwd = passwd
        self.dblink = dblink
        self.engine = None
        self._init_connection_params()
    
    def _init_connection_params(self):
        """初始化连接参数"""
        sys_info = get_sys_info(self.sys_name, self.dblink)
        self.host = sys_info['ip']
        self.port = sys_info['port']
        self.db = sys_info['db']
    
    def connect(self):
        """创建数据库连接"""
        try:
            self.engine = create_engine(
                f'mysql+pymysql://{self.user}:{self.passwd}@{self.host}:{self.port}/{self.db}'
            )
            logger.info(f"成功连接到数据库 {self.db}")
            return self.engine
        except Exception as e:
            logger.error(f"数据库连接失败: {str(e)}")
            raise
    
    def close(self):
        """关闭数据库连接"""
        if self.engine is not None:
            try:
                self.engine.dispose()
                logger.info("数据库连接已关闭")
            except Exception as e:
                logger.error(f"关闭数据库连接时出错: {str(e)}")
            finally:
                self.engine = None
    
    def __enter__(self):
        """上下文管理器的进入方法"""
        return self.connect()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器的退出方法"""
        self.close()
        if exc_type is not None:
            logger.error(f"执行过程中出现错误: {str(exc_val)}")
            return False
        return True
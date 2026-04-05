import time
import traceback

import numpy as np
import requests
from tqdm import tqdm
import pandas as pd
import oss2
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
logger = logging.getLogger(__name__)
logging.getLogger("oss2").setLevel(logging.WARNING)

oss_conf = {
    'am': {"oss_id": ["<OSS_ID>", "<OSS_ID>", "<OSS_ID>"],
           "oss_secret": ["<OSS_SECRET>", "<OSS_SECRET>", "<OSS_SECRET>"],
           "oss_region": ["oss-us-west-1.aliyuncs.com", "oss-us-west-1.aliyuncs.com",
                          "oss-us-west-1.aliyuncs.com"],
           "oss_bucket": ["am-system-file", "ym-system-file", "mexico-system-file"]
    },
    'mw': {"oss_id": ["<OSS_ID>"],
           "oss_secret": ["<OSS_SECRET>"],
           "oss_region": ["oss-us-east-1.aliyuncs.com"],
           "oss_bucket": ["mw-system-file"]
    },
    'af': {"oss_id": ["<OSS_ID>", "<OSS_ID>"],
           "oss_secret": ["<OSS_SECRET>", "<OSS_SECRET>"],
           "oss_region": ["oss-ap-southeast-5.aliyuncs.com", "oss-ap-southeast-5.aliyuncs.com"],
           "oss_bucket": ["af-system-file", "juyuan-image-yajada-sa"]
    },
    'al': {"oss_id": ["<OSS_ID>", "<OSS_ID>", "<OSS_ID>"],
           "oss_secret": ["<OSS_SECRET>", "<OSS_SECRET>", "<OSS_SECRET>"],
           "oss_region": ["oss-us-east-1.aliyuncs.com", "oss-us-west-1.aliyuncs.com",
                          "oss-us-east-1.aliyuncs.com"],
           "oss_bucket": ["al-system-file", "am-system-file", "yl-system-file"]
    },
    'ac': {"oss_id": ["<OSS_ID>", "<OSS_ID>"],
           "oss_secret": ["<OSS_SECRET>", "<OSS_SECRET>"],
           "oss_region": ["oss-us-west-1.aliyuncs.com", 'oss-us-east-1.aliyuncs.com'],
           "oss_bucket": ["ac-prod-file", 'cw-system-file']
    },
    'aec': {"oss_id": ["<OSS_ID>", "<OSS_ID>", "<OSS_ID>"],
           "oss_secret": ["<OSS_SECRET>", "<OSS_SECRET>", "<OSS_SECRET>"],
           "oss_region": ["oss-us-east-1.aliyuncs.com", "oss-us-west-1.aliyuncs.com",
                          "oss-us-east-1.aliyuncs.com"],
           "oss_bucket": ["aec-prod-file", "am-system-file", "yec-system-file"]
    },
    'ag': {"oss_id": ["<OSS_ID>", "<OSS_ID>", "<OSS_ID>", "<OSS_ID>"],
           "oss_secret": ["<OSS_SECRET>", "<OSS_SECRET>", "<OSS_SECRET>", "<OSS_SECRET>"],
           "oss_region": ["oss-us-east-1.aliyuncs.com", "oss-us-east-1.aliyuncs.com",
                          "oss-us-west-1.aliyuncs.com", "oss-us-east-1.aliyuncs.com"],
           "oss_bucket": ["ag-system-file", "amapk-system-file", "am-system-file", "y-system-file"]
    },
    'ath': {"oss_id": ["<OSS_ID>"], "oss_secret": ["<OSS_SECRET>"],
           "oss_region": ["oss-ap-southeast-7.aliyuncs.com"], "oss_bucket": ["ath-prod-file"]
    },
    'athl': {"oss_id": ["<OSS_ID>", "<OSS_ID_2>"],
             "oss_secret": ["<OSS_SECRET>", "<OSS_SECRET_2>"],
             "oss_region": ["oss-ap-southeast-7.aliyuncs.com", "oss-ap-southeast-5.aliyuncs.com"],
             "oss_bucket": ["ath-prod-file", 'athl-file']
    },
    'athu': {"oss_id": ["<OSS_ID>", "<OSS_ID_2>"],
             "oss_secret": ["<OSS_SECRET>", "<OSS_SECRET_2>"],
             "oss_region": ["oss-ap-southeast-7.aliyuncs.com", "oss-ap-southeast-5.aliyuncs.com"],
             "oss_bucket": ["ath-prod-file", 'athl-file']
    },
    "at": {
        "oss_id": ["<OSS_ID>"],
        "oss_secret": ["<OSS_SECRET>"],
        "oss_region": ["oss-eu-central-1.aliyuncs.com"],
        "oss_bucket": ["at-prod-file"],
    },
    "an": {
        "oss_id": ["<OSS_ID>"],
        "oss_secret": ["<OSS_SECRET>"],
        "oss_region": ["oss-eu-central-1.aliyuncs.com"],
        "oss_bucket": ["an-prod-file"]
    }
}

class DataFetcher:
    def __init__(self, max_workers=10):
        from concurrent.futures import ThreadPoolExecutor
        from queue import Queue
        import requests
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.q = Queue(maxsize=max_workers)
        for _ in range(self.max_workers):
            session = requests.Session()
            self.q.put(session)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器时关闭线程池"""
        if self.executor:
            self.executor.shutdown(wait=True)
        self.executor = None
        self.q = None

    def _is_valid_url(self, url):
        from urllib.parse import urlparse
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def _down_url_data(self, q, url, retries=3):
        import traceback
        session = q.get()
        try:
            if (url is None) or ("" == url):
                return None
            if isinstance(url, str) and self._is_valid_url(url):
                for _ in range(retries):
                    try:
                        response = session.get(url, stream=True)
                        if response.status_code == 200:
                            block_size = 1024
                            data = b''.join(chunk for chunk in response.iter_content(block_size))
                            return data
                    except Exception as e:
                        logger.warning(f"下载失败，尝试重试 ({retries - _} 次剩余): {url}, 错误: {e} \n{traceback.format_exc()}")
            else:
                logger.warning(f"url格式异常,url{url}")
                return None
            logger.warning(f"下载失败，已超过最大重试次数: {url}")
            return None
        finally:
            q.put(session)

    def fetch_df(self, df, url_column, suffixes='_data', tqdm_desc='Downloading'):
        if url_column not in df.columns:
            raise ValueError(f"列 {url_column} 不存在于DataFrame中")

        new_column_name = url_column + suffixes
        df[new_column_name] = None

        url_with_index = df[[url_column]].reset_index().values.tolist()

        with tqdm(total=len(df), desc=tqdm_desc) as pbar:
            futures = {self.executor.submit(self._down_url_data, self.q, url): (idx, url) for idx, url in
                       url_with_index}
            for future in as_completed(futures):
                idx, url = futures[future]
                try:
                    data = future.result()
                    if data is not None:
                        df.at[idx, new_column_name] = data.decode()
                except Exception as exc:
                    logger.warning(f"下载URL {url} 时发生未知错误: {exc}")
                finally:
                    pbar.update(1)


def fetch_url_data(df, url_column, suffixes='_data', max_workers=10):
    """
    根据指定的url下载相应的数据
    :param df:
    :param url_column:
    :param suffixes:
    :param max_workers:
    :return:
    """
    from queue import Queue
    def _is_valid_url(url):
        from urllib.parse import urlparse
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def _down_url_data(q, url, retries=3):  # 下载单个URL的内容，支持重试。
        session = q.get()
        try:
            if (url is None) or (np.nan == url) or (isinstance(url, str) and (url.strip() == "")):
                return None
            if isinstance(url, str) and _is_valid_url(url):  # 检查URL是否是字符串并且有效
                for _ in range(retries):
                    try:
                        response = session.get(url, stream=True)
                        if response.status_code == 200:
                            block_size = 1024  # 1 Kibibyte
                            data = b''.join(data_chunk for data_chunk in response.iter_content(block_size))
                            return data
                    except Exception as e:
                        logger.warning(f"下载失败，尝试重试 ({retries - _} 次剩余): {url}, 错误: {e} \n{traceback.format_exc()}")
                        time.sleep(3)
                logger.warning(f"下载失败，已超过最大重试次数: {url}")
                return None
            else:
                logger.warning(f"url格式异常,url{url}")
                return None
        finally:
            q.put(session)

    # 校验输入数据的有效性
    assert len(df) > 0, '数据为空'
    assert df.index.is_unique, '输入数据的index有重复值，请优先确保数据格式的正确性'
    if url_column not in df.columns:
        raise ValueError(f"列 {url_column} 不存在于DataFrame中")

    # 根据任务大小优化线程池的初始化大小
    df_len = len(df)
    if df_len == 0:
        return None
    if df_len < max_workers:
        max_workers = df_len

    new_column_name = url_column + suffixes
    df[new_column_name] = None
    url_with_index = df[[url_column]].reset_index().values.tolist()
    session_q = None
    executor = None
    try:
        session_q = Queue(maxsize=max_workers)
        executor = ThreadPoolExecutor(max_workers=max_workers)
        for _ in range(max_workers):
            session = requests.Session()
            session_q.put(session)
        futures = {executor.submit(_down_url_data, session_q, url): (idx, url) for idx, url in url_with_index}
        with tqdm(total=len(df), desc='fetch_http') as pbar:
            for future in as_completed(futures):
                idx, url = futures[future]
                try:
                    data = future.result()
                    if data is not None:
                        df.at[idx, new_column_name] = data.decode()
                except Exception as exc:
                    logger.warning(f"下载URL {url} 时发生未知错误: {exc}")
                finally:
                    pbar.update(1)

    finally:
        if executor is not None:
            executor.shutdown(wait=True)
            executor = None
        if session_q is not None:
            for _ in range(max_workers):
                session = session_q.get()
                session.close()
            session_q = None

class OssDataFecther:
    def __init__(self, max_workers, sys_name):
        oss_ids = oss_conf[sys_name]["oss_id"]
        oss_secrets = oss_conf[sys_name]["oss_secret"]
        oss_regions = oss_conf[sys_name]["oss_region"]
        oss_buckets = oss_conf[sys_name]["oss_bucket"]
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.bucket_list = [
            oss2.Bucket(oss2.Auth(oss_ids[i], oss_secrets[i]), oss_regions[i], oss_buckets[i], connect_timeout=10) for i
            in
            range(len(oss_ids))]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=True)

    def oss_fetch_task(self, url):
        if url is None or url.strip() == '':
            return None
        file_name = url.split('/')[-1]
        for i in range(len(self.bucket_list)):
            try:
                bucket = self.bucket_list[i]
                if url.endswith(".jpg"): # 如果为图片则不解码直接返回数据流
                    return bucket.get_object(file_name).read()
                else: # 如果是字符串，则采用字符串解码
                    return bucket.get_object(file_name).read().decode()
            except:
                pass
        logger.warning(f"{url}，当前配置的oss信息无法正确获取，请确认oss相应的配置信息")
        return None

    def fetch(self, df: pd.DataFrame, url_colums: str, suffix="_data",tqdm_disable=True,tqdm_desc=None):
        assert len(df) > 0, '数据为空'
        assert df.index.is_unique, '输入数据的index有重复值，请优先确保数据格式的正确性'

        idx_url_map = df[[url_colums]].reset_index().values.tolist()
        futures = {self.executor.submit(self.oss_fetch_task, url): idx for idx, url in idx_url_map}

        data_col_name = url_colums + suffix
        df.loc[:, data_col_name] = ''
        for future in tqdm(as_completed(futures), total=df.shape[0],disable=tqdm_disable,desc=tqdm_desc):
            idx = futures[future]
            data = future.result()
            df.at[idx, data_col_name] = data


def fetch_oss_data(df, data_col_name, sys_name, worker_num=13,tqdm_disable=False,tqdm_desc='fetch oss data'):
    """
    根据国家简称获取oss地址，然后下载相应行的数据
    :param df:
    :param data_col_name: 待下载的列
    :param sys_name: 系统名称
    :param worker_num: oss连接池默认:10
    :param tqdm_disable: 是否显示答应进度条
    :param tqdm_desc: 进度条描述
    :return:
    """
    with OssDataFecther(worker_num, sys_name) as fetcher:
        fetcher.fetch(df, data_col_name,tqdm_disable=tqdm_disable,tqdm_desc=tqdm_desc)
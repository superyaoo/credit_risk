import json
import os
import subprocess
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm
from pathlib import Path
import time

from toollib.unversal import service_prot_mapping, data_of_dir
import gc
import logging
logger = logging.getLogger(__name__)

def feauture_lib_taskV2(row, url):
    try:
        req_obj = row['request_params']
        if req_obj is None:
            pass
        if isinstance(req_obj, str):
            req_obj = json.loads(req_obj)

        request_params = {}
        request_params['base_info'] = req_obj['base_info']
        ds = req_obj['data_sources']
        if "app_" in url:
            request_params['data_sources'] = {'applist_data': ds['applist_data']}
        elif "sms_" in url:
            request_params['data_sources'] = {'sms_data': ds['sms_data']}
        elif "call_" in url:
            request_params['data_sources'] = {'calllog_data': ds['calllog_data'], 'contact_list': ds['contact_list']}
        elif "loan_" in url:
            request_params['data_sources'] = {'installments_data_a': ds['installments_data_a'],
                                              'order_data_a': ds['order_data_a'],
                                              'contract_data_a': ds['contract_data_a'],
                                              'user_info_1': ds['user_info_1'],
                                              }
        elif "dev_" in url:
            if '_base_v1' in url:
                request_params['data_sources'] = {'device_info': ds['device_info']}
            if '_base_v2' in url:
                request_params['data_sources'] = {'device_list': ds['device_list']}
        elif "cross_" in url:
            request_params['data_sources'] = {'sms_data': ds['sms_data'],
                                              'calllog_data': ds['calllog_data'],
                                              'applist_data': ds['applist_data'],
                                              'contact_list': ds['contact_list']}
        else:
            request_params = req_obj

        retries = 0
        max_retries = 3
        while True:
            rep = requests.post(url, json=request_params)
            retries = retries + 1
            if rep.status_code == 200 :
                break
            elif retries >=max_retries:
                raise requests.exceptions.RequestException(f"订单{row['app_order_id']}，已经重试{retries}次，接口服务返回:{rep.status_code}")

        rep_obj = {'app_order_id': request_params['base_info']['order_id']}
        rep_json = rep.json()

        if rep_json['code'] == 200:
            rep_obj.update(rep_json['data'])
            return rep_obj
        else:
            logger.warning(f"{row['app_order_id']},后台计算异常，返回异常code:{rep_json['code']}")
    except:
        logger.warning(f"{row['app_order_id']},异常{traceback.format_exc()}")


def request_featurelib(df, model_name, ip='127.0.0.1', uri=None, max_workers=20, tqdm_desc='requesting',
                       tqdm_disable=False):
    """
    基于前面封装的封装url请求数据获取特征
    df 含有app_order_id 和 request_params 这两列数据的pandas
    model_name 待请求的特征模块
    :param df: 符合格式规则的带 request_params 对象的数据
    :param max_workers: 并发数量，默认20
    :param model_name 待访问的模块名称
    :param uri 访问服务的uri数据，若为空，则根据配置的默认端口 和ip组成url
    :param max_workers 并发量
    """
    task_list = []
    if uri is None:
        ser_name = "_".join(model_name.split("_")[0:2])
        port = service_prot_mapping[ser_name]
        url = f'http://{ip}:{port}/api/{model_name}'
    else:
        url = f"{uri}/api/{model_name}"
    if not tqdm_disable:
        logger.info(f"该模块对应的 url为:{url}")

    for idx, row in df.iterrows():
        task_list.append((idx, row, url))
    with ThreadPoolExecutor(max_workers=max_workers) as executor, \
            tqdm(total=len(df), desc=tqdm_desc, disable=tqdm_disable) as pbar:
        futures = {executor.submit(feauture_lib_taskV2, row, url): (idx, row, url) for
                   idx, row, url in task_list}
        result = []
        for future in as_completed(futures):
            data = future.result()
            if data:
                result.append(data)
            pbar.update(1)
        df = pd.DataFrame(result)
        return df


def request_featurelibs(df, model_names, max_workers=10, port=1030,
                        featurelib_path='~/featurelib/',
                        python_bin='~/miniconda3/envs/py310/bin/'):
    """
    批量跑特征：
    1、根据传入的模块名称解析其中的服务名称，并会自动启停服务
    2、基于服务名和模块名称进行启停服务
    3、调用服务输出特征
    :param df 含有app_order_id 和 request_params 这两列数据的pandas
    :param model_names 待请求的特征模块 列表
    :param max_workers: 性能参数，根
    :param max_workers 并发量
    :param featurelib_path: 各模块对应得缓存文件目录得后缀，各个文件保存得缓存的名称为 model_names+dir_prefix
    :param python_bin: 各模块对应得缓存文件目录得后缀，各个文件保存得缓存的名称为 model_names+dir_prefix
    return dict : 模块名及其对应的特征列表，eg:
            {'app_mx_base_v1':app_mx_base_v1_df,'app_mx_base_v1':app_mx_base_v1_df  }
    """
    feature_maps = parse_mode_name(model_names)
    logger.info(f"任务列表:{feature_maps}")
    result = {}
    for ser_name, model_name_list in feature_maps.items():
        start_ser(ser_name, workers=max_workers, port=port, featurelib_path=featurelib_path,
                  python_bin=python_bin)
        uri = f'http://127.0.0.1:{port}'
        client_workers = max_workers + 5
        for model_name in model_name_list:
            try:
                feature_df = request_featurelib(df, model_name, uri=uri, max_workers=client_workers,
                                                tqdm_desc=model_name)
                result[model_name] = feature_df
            except:
                logger.warning(f"{model_name}计算失败\n{traceback.format_exc()}")
                continue
        stop_ser(port)
    return result


def batch_request_featurelib(file_paths, model_name, uri=None, max_workers=20,
                             load_func=pd.read_parquet, save_dir=None):
    """
    扫描文件下中批量下载的文件块，批量回溯模型特征
    :param file_paths: 可以直接读取和加载的文件块,如['/home/longxiaolei/am_a4/raw_dir/req_data_2024-06-03.parquet','/home/longxiaolei/am_a4/raw_dir/req_data_2024-06-10.parquet']
    :param model_name: 待下载的列
    :param max_workers: 并发数量，默认20
    :param load_func: 加载数据的方式，默认采用parquet的方式加载
    :param save_dir:数据保存的文件夹，默认用model_name
    """
    save_dir = save_dir if save_dir else model_name
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    saved_files = data_of_dir(save_dir, '.parquet')
    saved_files = [x.split("/")[-1] for x in saved_files]
    for file_path in tqdm(file_paths, desc=model_name):
        file_name = file_path.split("/")[-1]
        if file_name in saved_files:
            continue
        req_df_part = load_func(file_path)
        df_new = request_featurelib(req_df_part, model_name, uri=uri, tqdm_desc=file_name,
                                    max_workers=max_workers, tqdm_disable=True)
        df_new.to_parquet(save_dir / f"{file_name}")
        time.sleep(1)
        del req_df_part
        del df_new
        gc.collect()


def batch_request_featurelib2(req_files, model_names, port=10012, max_workers=20, dir_prefix='_chunck',
                              featurelib_path='~/featurelib/',
                              python_bin='~/miniconda3/envs/py310/bin/',
                              load_func=pd.read_parquet):
    """
    扫描文件下中批量下载的文件块，批量回溯模型特征
    :param req_files: 可以直接读取和加载的文件块,如['/home/longxiaolei/am_a4/raw_dir/req_data_2024-06-03.parquet','/home/longxiaolei/am_a4/raw_dir/req_data_2024-06-10.parquet']
    :param model_names: 待计算得特征模块
    :param max_workers: 并发数量，默认20
    :param port: 服务启动端口
    :param load_func: 加载数据的方式，默认采用parquet的方式加载
    :param dir_prefix: 各模块对应得缓存文件目录得后缀，各个文件保存得缓存的名称为 model_names+dir_prefix
    :param featurelib_path: 各模块对应得缓存文件目录得后缀，各个文件保存得缓存的名称为 model_names+dir_prefix
    :param python_bin: 各模块对应得缓存文件目录得后缀，各个文件保存得缓存的名称为 model_names+dir_prefix
    """
    feature_maps = parse_mode_name(model_names)
    logger.info(f"生成任务映射表:\n{feature_maps}")
    for ser_name, model_name_list in feature_maps.items():
        start_ser(ser_name, workers=max_workers, port=port, featurelib_path=featurelib_path, python_bin=python_bin)
        uri = f'http://127.0.0.1:{port}'
        client_workers = max_workers + 5
        for model_name in model_name_list:
            save_dir = Path(f'{model_name}{dir_prefix}')
            save_dir.mkdir(exist_ok=True)
            for req_file in tqdm(req_files, desc=model_name):
                file_name = req_file.split("/")[-1]
                try:
                    load_func(save_dir / file_name)  # 加载模块数据判断是否又异常
                except:  # 如果加载失败，则启动跑数任务
                    req_df_part = load_func(req_file)
                    df_new = request_featurelib(req_df_part, model_name, uri=uri, max_workers=client_workers,
                                                tqdm_disable=True)
                    df_new.to_parquet(save_dir / f"{file_name}")
                    del req_df_part
                    del df_new
                    time.sleep(1)
                    gc.collect()
        stop_ser(port)
        logger.info(f"关闭服务{ser_name}...")


def parse_mode_name(model_names):
    serv_names = ['_'.join(model_name.split('_')[:2]) for model_name in model_names]
    serv_map = list(zip(serv_names, model_names))
    feature_maps = defaultdict(list)
    for k, v in serv_map:
        feature_maps[k].append(v)
    feature_maps = dict(feature_maps)
    return feature_maps


def start_ser(serv, workers=20, port=10080, featurelib_path='~/featurelib/',
              python_bin='~/miniconda3/envs/py310/bin/'):
    """
    根据参数启动featurelib服务。
    :param serv: 服务名称 如：loan_th
    :param workers: 并发数量
    :param port:服务启动的端口
    :param featurelib_path:feautrelib的代码所在目录
    :param python_bin: python环境的bin目录，只是针对miniconda环境进行了测试，其他环境，请自行修改
    """
    stop_ser(port)
    serv_full = f'{serv}_service'
    featurelib_path = os.path.expanduser(featurelib_path)
    python_bin_dir = os.path.expanduser(python_bin)
    python_path = os.path.join(python_bin_dir, 'python')
    gunicorn_path = os.path.join(python_bin_dir, 'gunicorn')

    # start_cmd = f"{python_path} {gunicorn_path} -w {workers} --threads 1 --timeout 60 --max-requests 1000 --max-requests-jitter 100 --chdir feature_service/{serv} -D {serv_full}:app -b 0.0.0.0:{port} --access-logfile=logs/access.log --error-logfile=logs/error.log"
    start_cmd = f"{python_path} {gunicorn_path} -w {workers} --threads 1 --chdir feature_service/{serv} -D {serv_full}:app -b 0.0.0.0:{port} --access-logfile=logs/access.log --error-logfile=logs/error.log"
    subprocess.run(start_cmd, shell=True, cwd=featurelib_path, check=True)
    n=0
    while n <= 6:
        if port_is_occupied(port):
            logger.info(f'{port} 服务已经启动')
            result = subprocess.run(f"ps -ef | grep '0.0.0.0:{port}'", text=True, shell=True,
                                    capture_output=True).stdout.splitlines()[0:1]
            logger.info("\n".join(result))
            break
        else:
            logger.info(f"{port} 服务启动中。。。。")
            time.sleep(0.5)
            n+=1


def stop_ser(port):
    """根据端口号kill 服务"""
    result= subprocess.run(["pkill", "-f", f"0.0.0.0:{port}"], check=False)
    # 检查返回码
    if result.returncode in [0,1] :
        n=0
        while n<=6:
            if port_is_occupied(port):
                logger.info(f"{port} 正在释放，请等待.....")
                subprocess.run(["pkill", "-f", f"0.0.0.0:{port}"], check=False)
                time.sleep(0.5)
                n+=1
            else:
                logger.info(f"{port} 已经被成功释放")
                return
    else:
        raise EnvironmentError(f'权限不足，无法杀死进程,返回码: {result.returncode}\n {result.stderr}')

def port_is_occupied(port):
    result = subprocess.run(["netstat", "-tuln"], stdout=subprocess.PIPE, text=True)
    if f":{port}" in result.stdout:
        return True
    else:
        return False


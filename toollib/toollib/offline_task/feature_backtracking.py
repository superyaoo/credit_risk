import warnings
warnings.filterwarnings('ignore')
import toollib as tl
import polars as pl
import pandas as pd
import os
import time
import argparse
from datetime import datetime,timedelta
import yaml

def get_modules(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['models']

def backtrack(feature_maps, file_paths, args):
    for ser_name,model_list in feature_maps.items():
            print(f"开始计算{ser_name}的服务")
            tl.start_ser(ser_name,port=args.port,workers=args.max_workers,featurelib_path=args.featurelib_path,python_bin=args.python_bin)
            uri = f'http://127.0.0.1:{args.port}'
            for model in model_list:
                for file in file_paths:
                    file_name = file.split('/')[-1]
                    save_dir = os.path.join(args.work_dir, args.country, model)
                    os.makedirs(save_dir,exist_ok=True)
                    save_path = os.path.join(save_dir, file_name)
                    if args.write_mode == 'incr':
                        try:
                            pl.read_parquet(save_path)
                            print(f"{save_path}已存在")
                        except:
                            req_df = pd.read_parquet(file)
                            result = tl.request_featurelib(req_df,model,uri=uri,max_workers=(args.max_workers+5),tqdm_desc=f'{file_name}')
                            result.to_parquet(save_path,compression='zstd')
                    elif args.write_mode == 'ovr':
                        req_df = pd.read_parquet(file)
                        result = tl.request_featurelib(req_df,model,uri=uri,max_workers=(args.max_workers+5),tqdm_desc=f'{file_name}')
                        result.to_parquet(save_path,compression='zstd')
            print(f'关闭{ser_name}')
            tl.stop_ser(args.port)
            time.sleep(2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, default='/home/model')
    parser.add_argument('--country', type=str, default='ec')
    parser.add_argument('--config_path', type=str, default='config/ec_old.yaml')
    parser.add_argument('--max_workers', type=int, default=40)
    parser.add_argument('--port', type=int, default=10081)
    parser.add_argument('--python_bin', type=str, default='/home/model/miniconda3/envs/py310/bin/')
    parser.add_argument('--featurelib_path', type=str, default='/home/model/featurelib/')
    parser.add_argument('--start_date', type=str, default='2024-07-01')
    parser.add_argument('--end_date', type=str, default=(datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d'))
    parser.add_argument('--user_type', type=str, choices=['new','old'], default='new')
    parser.add_argument('--write_mode', type=str, choices=['incr','ovr'], default='incr')
    args = parser.parse_args()
    
    print(f"开始回溯特征==========================>")
    req_dir = f'{args.work_dir}/{args.country}/req_dir'
    if args.user_type == 'new':
        file_paths = tl.data_of_dir(req_dir, contains_flags=['_0_'], start_time=args.start_date, end_time=args.end_date)
    elif args.user_type == 'old':
        file_paths = tl.data_of_dir(req_dir, contains_flags=['_1_', '_2_'], start_time=args.start_date, end_time=args.end_date)
    print(f"待运行的数据有{len(file_paths)}块==========================>")

    tl.stop_ser(args.port)
    print(f"服务和特征模块的运行关系：")
    modules = get_modules(args.config_path)
    feature_maps = tl.parse_mode_name(model_names=modules)
    for k,v in feature_maps.items():
        print(k,v)
    time.sleep(2)
    backtrack(feature_maps, file_paths, args)

if __name__ == '__main__':
    main()

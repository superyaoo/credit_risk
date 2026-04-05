#!/usr/bin/env python
# coding: utf-8

# # 导包

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import toollib as tl
import polars as pl
import pandas as pd
from datetime import datetime,timedelta
import time
import os
import gc
import logging
logger = logging.getLogger(__name__) 

# In[ ]:


sys_name = 'af'
work_dir = '/home/model/af'
req_save_dir = os.path.join(work_dir, 'req_dir')
feature_save_dir = work_dir
sample_save_path = os.path.join(work_dir,'sample.parquet')

user,passwd = 'etl','Vrrbpf6PjYStx4+v'

models = [
    'fdc_id_base_v1', 'fdc_id_base_v2',
    "app_id_base_v1","app_id_base_v2","app_id_gcate_v1","app_id_gcate_v2",'app_id_loan_v1',"app_id_loan_v2",
    "app_id_comp_v1",'app_id_cate_v1',"app_id_cashloan_v1",'app_id_bcate_v1','app_id_base_v3','app_id_gcate_v3',
    'app_id_loan_v3','app_id_bcate_v2','app_id_cashloan_v2',
    'sms_id_base_v1','sms_id_base_v2',
    'loan_id_base_v2','loan_id_base_v3_2','loan_id_base_v4',
    'call_id_base0_v1', 'call_id_base0_v2',
    'cross_id_base_v1',
    'dev_id_base_v1',
    'ui_id_base_v2'
]

max_workers = 30
port = 1209

featurelib_path = '~/featurelib/'
python_bin='~/miniconda3/envs/py310/bin/'

write_mode = 'increament' # increament 'over_witre' 覆盖写还是增量写
tqdm_disable = True

new_old_user_status= [0,1,2] # [0 ,1,2]
sample_start_time = datetime.now() - timedelta(days=14)
sample_start_time = sample_start_time.strftime('%Y-%m-%d')
sample_start_time = '2024-10-01'
# sample_start_time = '2025-03-05'
sample_end_time = datetime.now() - timedelta(days=1)
sample_end_time = sample_end_time.strftime('%Y-%m-%d')

logger.info(f'sys_name:{sys_name}')
logger.info(f'word_dir:{work_dir}')
logger.info(f'sample_save_path:{sample_save_path}')
logger.info(f'req_dir:{req_save_dir}')
logger.info(f'new_old_user_status:{new_old_user_status}')
logger.info(f'write_mode:{write_mode}')
logger.info(f'max_workers:{max_workers}')
logger.info(f'起止时间:{sample_start_time} ~ {sample_end_time}')
logger.info(f'关闭进度条显示：{tqdm_disable}')
logger.info(f'待回溯得特征：{models}')


# # 查询并生成样本信息

# In[ ]:


loan_df = tl.get_loans(sys_name,user,passwd,start_date='2024-10-01',end_date=sample_end_time)


# In[ ]:


def gen_file_name(row):
    new_old_user_status = row['new_old_user_status']
    loan_date = row['loan_time'].strftime("%Y-%m-%d")
    return f'df_{new_old_user_status}_{loan_date}.parquet'
loan_df['file_name'] = loan_df.apply(gen_file_name,axis=1)
logger.info(f"file_name样式：\n{loan_df[['new_old_user_status','loan_time','file_name']].sample(5)}")


# In[ ]:


os.makedirs(work_dir,exist_ok=True)


# In[ ]:


sample_df = loan_df[(loan_df['extension_count']==0)&(loan_df.device_type=='android')&(loan_df['installment_num']==1)]
sample_df = sample_df.sort_values(by='loan_time')
sample_df = sample_df.drop_duplicates(subset='app_order_id')
logger.info(f"文件大小:{sample_df.shape},订单枚举:{sample_df['app_order_id'].nunique()},min_loan_time:{sample_df['loan_time'].min()},max_loan_time:{sample_df['loan_time'].max()}")
sample_df.to_parquet(sample_save_path,compression='zstd')
logger.info(f"新老用户统计信息:\n{sample_df['new_old_user_status'].value_counts()}")
logger.info(f"sample_df 已经保存: {sample_save_path}")


# # 加载基本的样本信息

# In[ ]:


sample_df = pd.read_parquet(sample_save_path)
logger.info(f"文件大小:{sample_df.shape},订单枚举:{sample_df['app_order_id'].nunique()},min_loan_time:{sample_df['loan_time'].min()},max_loan_time:{sample_df['loan_time'].max()}")
logger.info(f"新老用户统计信息:\n{sample_df['new_old_user_status'].value_counts()}")


# In[ ]:


sample_df = sample_df[sample_df['new_old_user_status'].isin(new_old_user_status)]
file_names = sorted(sample_df[(sample_df['loan_time']>=sample_start_time)&(sample_df['loan_time']<sample_end_time)]['file_name'].unique())
logger.info(f"待下载的文件有{len(file_names)}")


# In[ ]:


os.makedirs(req_save_dir, exist_ok=True)
def fetch_request_params(file_name,save_path):
    orderlist = sample_df[sample_df['file_name']==file_name]['app_order_id'].to_list()
    logger.info(f"{save_path},{len(orderlist)},开始下载：==============================>")
    req_df = tl.get_sample_req(orderlist,sys_name,user,passwd,tqdm_disable=tqdm_disable)
    req_df.to_parquet(save_path,compression='zstd')
    logger.info(f"{save_path},{len(orderlist)},下载完成：<==============================\n")

for file_name in file_names:    
    req_df =None
    orderlist = None
    save_path = os.path.join(req_save_dir, file_name)
    if write_mode == 'over_witre': 
        fetch_request_params(file_name,save_path)
    elif  write_mode == 'increament':
        try:
            req_df = pl.read_parquet(save_path)
            logger.info(f"{save_path}已经正常下载! {req_df.shape}")
        except:
            fetch_request_params(file_name,save_path)
        finally:
            del req_df
            del orderlist
            gc.collect()
    else:
        raise ValueError(f'write_mode 参数必须是 over_witre 或 increament')


# # 回溯特征

# ## 批量运算普通任务

# In[ ]:


user_type =  [ f'_{i}_'  for i in new_old_user_status ]
file_paths = tl.data_of_dir(req_save_dir, user_type, start_time=sample_start_time, end_time=sample_end_time)
logger.info(f"开始跑回溯特征==========================>")
logger.info(f"待运行的数据有{len(file_paths)}块==========================>")


# In[ ]:


tl.stop_ser(port)
logger.info(f"服务和特征模块的运行关系：")
feature_maps = tl.parse_mode_name(model_names=models)
for k,v in feature_maps.items():
    logger.info(f"{k}:{v}")
time.sleep(2)


# In[ ]:


def calc_features(file,model,uri,max_workers,file_name,tqdm_disable,save_path):
    req_df = pd.read_parquet(file)
    logger.info(f'开始计算{file} {model} {len(req_df)}====>')
    result = tl.request_featurelib(req_df,model,uri=uri,max_workers=(max_workers+5),tqdm_desc=f'{file_name}',tqdm_disable=tqdm_disable)
    result.to_parquet(save_path,compression='zstd')
    logger.info(f'完成计算{file} {model} {len(req_df)},<====\n保存路径{save_path}\n')

for ser_name,model_list in feature_maps.items():
    logger.info(f"开始计算{ser_name}的服务")
    tl.start_ser(ser_name,port=port,workers=max_workers,featurelib_path=featurelib_path,python_bin=python_bin)
    uri = f'http://127.0.0.1:{port}'
    for model in model_list:
        model_feature_save_dir = os.path.join(feature_save_dir,model)
        os.makedirs(model_feature_save_dir,exist_ok=True)
        for file in file_paths:
            file_name = file.split('/')[-1]
            model_feature_path = os.path.join(model_feature_save_dir, file_name)
            if write_mode == 'increament':
                try:
                    tmp = pl.read_parquet(model_feature_path)
                    logger.info(f"{model_feature_path}已存在,无需计算! {tmp.shape}")
                except:
                    calc_features(file,model,uri,max_workers,file_name,tqdm_disable,model_feature_path)
            elif write_mode == 'over_witre':
                calc_features(file,model,uri,max_workers,file_name,tqdm_disable,model_feature_path)
    logger.info(f'关闭{ser_name}')
    tl.stop_ser(port)
    time.sleep(2)




# In[ ]:
# ## 独立计算任务

# In[ ]:


models = [
     'sms_id_loan_v1'
]
max_workers = 8

tl.stop_ser(port)
logger.info(f"服务和特征模块的运行关系：")
feature_maps = tl.parse_mode_name(model_names=models)
for k,v in feature_maps.items():
    logger.info(f"{k}:{v}")
time.sleep(2)

for ser_name,model_list in feature_maps.items():
    logger.info(f"开始计算{ser_name}的服务")
    tl.start_ser(ser_name,port=port,workers=max_workers,featurelib_path=featurelib_path,python_bin=python_bin)
    uri = f'http://127.0.0.1:{port}'
    for model in model_list:
        model_feature_save_dir = os.path.join(feature_save_dir,model)
        os.makedirs(model_feature_save_dir,exist_ok=True)
        for file in file_paths:
            file_name = file.split('/')[-1]
            model_feature_path = os.path.join(model_feature_save_dir, file_name)
            if write_mode == 'increament':
                try:
                    tmp = pl.read_parquet(model_feature_path)
                    logger.info(f"{model_feature_path}已存在,无需计算! {tmp.shape}")
                except:
                    calc_features(file,model,uri,max_workers,file_name,tqdm_disable,model_feature_path)
            elif write_mode == 'over_witre':
                calc_features(file,model,uri,max_workers,file_name,tqdm_disable,model_feature_path)
    logger.info(f'关闭{ser_name}')
    tl.stop_ser(port)
    time.sleep(2)




# In[ ]:


"""
jupyter nbconvert --to script req_and_fea_af.ipynb
conda activate py310
python req_and_fea_af.py
chmod -R +777 /home/model/af/
"""


# In[ ]:





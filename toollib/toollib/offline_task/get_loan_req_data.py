import warnings
warnings.filterwarnings('ignore')
import toollib as tl
import polars as pl
from datetime import datetime,timedelta
import argparse
import os

def gen_file_name(row):
    new_old_user_status = row['new_old_user_status']
    loan_date = row['loan_time'].strftime("%Y-%m-%d")
    return f'df_{new_old_user_status}_{loan_date}.parquet'

def get_req_data(sample_df, args):
    file_names = sorted(sample_df[(sample_df['loan_time']>=args.start_date)&(sample_df['loan_time']<args.end_date)]['file_name'].unique())
    req_dir = f'{args.work_dir}/{args.country}/req_dir'
    print(f"开始下载{len(file_names)}个文件==============================>")
    print(f'保存路径为：{req_dir}')
    os.makedirs(req_dir,exist_ok=True)
    for file_name in file_names:    
        req_df =None
        orderlist = None
        if args.write_mode == 'ovr':
            orderlist = sample_df[sample_df['file_name']==file_name]['app_order_id'].to_list()
            print(file_name,len(orderlist),'开始下载：==============================>')
            req_df = tl.get_sample_req(orderlist,args.country,args.user,args.passwd,args.dblink)
            req_df.to_parquet(os.path.join(req_dir,file_name),compression='zstd')
        elif  args.write_mode == 'incr':
            try:
                pl.read_parquet(os.path.join(req_dir,file_name))
                print(f"{file_name}已经正常下载")
            except:
                orderlist = sample_df[sample_df['file_name']==file_name]['app_order_id'].to_list()
                print(file_name,len(orderlist),'开始下载：==============================>')
                req_df = tl.get_sample_req(orderlist,args.country,args.user,args.passwd,args.dblink)
                req_df.to_parquet(os.path.join(req_dir,file_name),compression='zstd')
        else:
            raise ValueError(f'write_mode 参数必须是 ovr 或 incr')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, default='/home/model')
    parser.add_argument('--country', type=str, default='ec')
    parser.add_argument('--user', type=str, default='etl')
    parser.add_argument('--passwd', type=str, default='Vrrbpf6PjYStx4+v')
    parser.add_argument('--dblink', type=str, default='outer')
    parser.add_argument('--start_date', type=str, default='2024-07-01')
    parser.add_argument('--end_date', type=str, default=(datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d'))
    parser.add_argument('--write_mode', type=str, choices=['ovr','incr'], default='incr')
    args = parser.parse_args()
    
    os.makedirs(f'{args.work_dir}/{args.country}',exist_ok=True)
    loan_df = tl.get_loans(args.country,
                           args.user,
                           args.passwd,
                           dblink=args.dblink,
                           start_date=args.start_date,
                           end_date=args.end_date)
    loan_df['file_name'] = loan_df.apply(gen_file_name,axis=1)
    sample_df = loan_df[(loan_df['extension_count']==0)&(loan_df['installment_num']==1)&(loan_df['device_type']=='android')]
    sample_df.to_parquet(f'{args.work_dir}/{args.country}/sample_df.parquet',compression='zstd')
    get_req_data(sample_df, args)

if __name__ == '__main__':
    main()
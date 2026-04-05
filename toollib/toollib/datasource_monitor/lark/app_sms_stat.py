import json
import pandas as pd
import toollib as tl


def get_basic_data_stat(start_date,sys,acq_channel_list):
    sample_df = pd.read_parquet(f'/home/model/{sys}/sample.parquet')
    sample_df = sample_df[(sample_df['apply_date'] >= start_date)]
    sample_df = sample_df[(sample_df['new_old_user_status'] == 0) & (sample_df['extension_count'] == 0) & (sample_df['status'] != -1) & (sample_df['acq_channel'].isin(acq_channel_list))]
    apply_week_dist = sample_df['apply_week'].unique()
    apply_week_dist.sort()
    sample_df = sample_df[sample_df['apply_week'].isin(apply_week_dist[-9:-1])]
    request_list = []
    file_list = tl.data_of_dir(f'/home/model/{sys}/req_dir', ['_0_'], start_time=start_date)
    for file in file_list:
        request_list.append(pd.read_parquet(file))
    request_df = pd.concat(request_list)
    df = pd.merge(sample_df[['app_order_id', 'acq_channel', 'apply_week']], request_df[['app_order_id', 'request_params']], on=['app_order_id'])
    df['applist_data'] = df['request_params'].apply(lambda x: json.loads(x)['data_sources']['applist_data'])
    df['sms_data'] = df['request_params'].apply(lambda x: json.loads(x)['data_sources']['sms_data'])

    def get_order_stat(x):
        app_df = pd.DataFrame(x['applist_data'])
        app_cnt = len(app_df)
        notsys_app_cnt = len(app_df[app_df['isSystem'] == 0])
        sms_cnt = len(x['sms_data'])
        return pd.Series({'app_cnt': app_cnt, 'notsys_app_cnt': notsys_app_cnt, 'sms_cnt': sms_cnt})

    df[['app_cnt', 'notsys_app_cnt', 'sms_cnt']] = df.apply(get_order_stat, axis=1)
    df_group = df.groupby(['acq_channel', 'apply_week']).agg({'app_order_id': 'count', 'app_cnt': 'mean', 'notsys_app_cnt': 'mean', 'sms_cnt': 'mean'}).reset_index()
    df_group.insert(loc=0, column='项目', value=sys)
    df_group['app_cnt'] = df_group['app_cnt'].round(0)
    df_group['notsys_app_cnt'] = df_group['notsys_app_cnt'].round(0)
    df_group['sms_cnt'] = df_group['sms_cnt'].round(0)
    df_group = df_group.rename(columns={
        'acq_channel': '渠道名',
        'apply_week': '申请周',
        'app_order_id': '新客申请订单数',
        'app_cnt': '平均app个数',
        'notsys_app_cnt': '平均非系统app个数',
        'sms_cnt': '平均短信数'
    })
    return df_group

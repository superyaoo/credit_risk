import json
from toollib.unversal import *
from toollib.data_fetcher import *

import logging
logger = logging.getLogger(__name__)


def query_order_info(order_list, sys_name, user, passwd, dblink='inner', fetch_url=False, tqdm_disable=False):
    """
    根据订单号和国家名称在A系统中查询相应的信息
    :param order_list:
    :param sys_name: 系统名称
    :param user:
    :param passwd:
    :param dblink:
    :param fetch_url:
    :param tqdm_disable: 是否关闭进度条显示
    :return:
    """
    sys_dict = get_sys_info(sys_name, dblink)
    country_id = sys_dict['country_id']
    country_abbr = sys_dict['country_abbr']
    merchant_id = sys_dict['merchant_id']
    database = sys_dict['db']

    mysql_rule = get_dbcon(sys_name, user, passwd, dblink=dblink)


    order_id_list = list(set(order_list))
    logger.info(f"共计输入{len(order_list)}条数据，清理重复数据后共{len(order_id_list)}条数据")
    order_id_list = list2sqlstr(order_id_list)

    # af系统需要剔除外部的重复调用的数据
    req_record_tbl = f"(select * from {database}.t_risk_req_record where biz_type !=4 )" if sys_name == 'af' else f"{database}.t_risk_req_record"

    exc_sql = f"""
               select
                    "{country_id}" as country_code,
                    "{country_abbr.lower()}" as country_abbr,
                    "{merchant_id}" as merchant_id,
                    case when a.device_type='ios' then 1
                         when a.device_type='android' then 0 end as device_type,
                    a.app_order_id as order_id,
                    a.phone_number as phone,
                    a.id_card_number as nid,
                    a.user_id as app_user_id,
                    a.app_track_id as track_id,
                    a.acq_channel,
                    a.apply_time,
                    cast(jSON_EXTRACT(work_info, '$.workYears') as SIGNED)  as workYears,
                    cast(jSON_EXTRACT(work_info, '$.salaryPayFrequency') as SIGNED)  as salaryPayFrequency,
                    cast(jSON_EXTRACT(personal_info, '$.email') as char)  as email,
                    b.sms_records_url,
                    b.app_list_url,
                    b.call_records_url,
                    b.calendar_records_url,     
                    b.device_info,
                    c.tx_id,
                    c.req_data,
                    cr.term,
                    cr.days_per_term
               from (select * from {database}.t_app_order where app_order_id in {order_id_list}) a
               inner  join {database}.t_app_track b on a.app_track_id = b.id
               inner  join {req_record_tbl} c on a.app_order_id = c.biz_id
               left join {database}.t_contract cr on cr.app_order_id= a.app_order_id
               where c.tx_id is not null
    """

    data_info = pd.read_sql(exc_sql, con=mysql_rule, dtype={"order_id": str})

    assert len(data_info) > 0, '输入的订单号无法匹配到符合业务逻辑的订单，请检查订单号！'
    logger.info(f"成功查询到进件相关信息,共计{data_info.shape},对应的用户数量{data_info['app_user_id'].nunique()}")

    sample_user_id_list = data_info.app_user_id.unique().tolist()
    logger.info(f"剔除异常订单后，还剩余{len(data_info)}订单，对应{len(sample_user_id_list)}个用户")
    sample_user_id_list = list2sqlstr(sample_user_id_list)

    app_user_data = pd.read_sql(
        f"""select user_id as app_user_id,acq_channel ,source,create_time as register_time 
        from {database}.t_app_user where user_id in {sample_user_id_list}"""
        ,con=mysql_rule
    )

    customer_data = pd.read_sql(
        f""" select user_id as app_user_id,acq_channel,create_time as fill_time,JSON_EXTRACT(extra_info,'$.lineAccount') as lineAccount 
        from {database}.t_customer where user_id in {sample_user_id_list}"""
        ,con=mysql_rule
    )

    data_info = data_info.merge(app_user_data,on=['app_user_id','acq_channel'])
    data_info = data_info[data_info['register_time']<data_info['apply_time']].sort_values(by=['order_id','register_time'],ascending=True)
    data_info.drop_duplicates(subset=['order_id'],keep='last',inplace=True)

    data_info = data_info.merge(customer_data,on=['app_user_id','acq_channel'])
    data_info = data_info[data_info['fill_time']<data_info['apply_time']].sort_values(by=['order_id','fill_time'],ascending=True)
    data_info.drop_duplicates(subset=['order_id'],keep='last',inplace=True)

    installment_data = pd.read_sql(
        f"""select id,user_id,acq_channel,app_order_id,contract_no,repayment_date,installment_num,installment_amount,principal,
        interest,cut_interest,service_fee,management_fee,overdue_interest,overdue_days,penalty,extension_fee,repaid_principal,
        repaid_interest,repaid_cut_interest,repaid_service_fee,repaid_management_fee,repaid_overdue_interest,repaid_penalty,
        discount_amount,waive_amount,settlement_type,extension_count,new_old_user_status,status,create_time,update_time,__op,
        settlement_time from {database}.t_installment where user_id in {sample_user_id_list}""",
        con=mysql_rule).sort_values(by=['create_time'], ascending=True)
    logger.info(f"基于user_id关联的放款单,共计{installment_data.shape}")

    order_data = pd.read_sql(f"""select * from {database}.t_app_order where user_id in {sample_user_id_list}""",
                             con=mysql_rule).sort_values(by=['create_time'], ascending=True)
    logger.info(f"基于user_id关联的进件订单,共计{order_data.shape}")

    contract_data = pd.read_sql(
        f"""select * from {database}.t_contract where user_id in {sample_user_id_list}""",
        con=mysql_rule).sort_values(by=['create_time'], ascending=True)
    logger.info(f"基于user_id关联的合同单,共计{contract_data.shape}")

    device_list_data = pd.read_sql(
        f"""select user_id, device_info, create_time,update_time from {database}.t_app_track where user_id in {sample_user_id_list} order by create_time""",
        con=mysql_rule).sort_values(by=['create_time'], ascending=True)
    logger.info(f"基于user_id关联设备数据,共计{device_list_data.shape}")

    order_data.fillna(np.nan, inplace=True)

    def get_user_info(df):
        detail = df['req_data']
        request_time = None
        try:
            prod = json.loads(detail)
            user_info = prod.get('applyInfo', {})
            user_info.setdefault('productInfo', {})
            user_info.setdefault('userInfo', {})
            request_time = prod.get('requestTime', None) or df['apply_time'].strftime('%Y-%m-%d %H:%M:%S')

            user_info['productInfo']['createTime'] = str(df['apply_time'])
            user_info['productInfo']['term'] = None if (df['term'] is None) or (np.isnan(df['term'])) else int(
                df['term'])
            user_info['productInfo']['daysPerTerm'] = None if (df['days_per_term'] is None) or (
                np.isnan(df['days_per_term'])) else int(df['days_per_term'])
            user_info['userInfo']['userRecord']['registerTime'] = str(df['register_time'])
            user_info['userInfo']['userRecord']['createTime'] = str(df['fill_time'])
            user_info['userInfo']['userRecord']['workYears'] = df['workYears']
            user_info['userInfo']['userRecord']['salaryPayFrequency'] = df['salaryPayFrequency']
            user_info['userInfo']['userRecord']['email'] = eval(df['email'])
            user_info['userInfo']['userRecord']['lineAccount'] = eval(df['lineAccount']) if df['lineAccount'] and df[
                'lineAccount'] != 'null' else ''
            device_info = prod['userHidden']['deviceInfo']
            return user_info, device_info, request_time
        except:
            logger.info(f"order_id:{df['order_id']} 解析失败，请检查数据\n{traceback.format_exc()}")
            return None, None, request_time

    data_info[['user_info', 'device_info', 'real_apply_time']] = data_info.apply(get_user_info, axis=1,
                                                                                 result_type='expand')

    #  根据入参下载数据
    if fetch_url:
        fetch_oss_data(data_info, 'app_list_url', sys_name,  tqdm_disable=tqdm_disable, tqdm_desc='app')
        fetch_oss_data(data_info, 'sms_records_url', sys_name,  tqdm_disable=tqdm_disable, tqdm_desc='sms')
        fetch_oss_data(data_info, 'call_records_url', sys_name,  tqdm_disable=tqdm_disable, tqdm_desc='call')
        fetch_oss_data(data_info, 'calendar_records_url', sys_name,  tqdm_disable=tqdm_disable, tqdm_desc='calendar')
    else:
        data_info['app_list_url_data'] = None
        data_info['sms_records_url_data'] = None
        data_info['call_records_url_data'] = None
        data_info['calendar_records_url_data'] = None
    return data_info, installment_data, order_data, contract_data, device_list_data


def preprocess_order_data_a(app_order_id, order_data_a):
    """
    预处理order_data_a数据，仅回溯时使用。
    :param app_order_id:申请订单号
    :param order_data_a:申请订单号
    :return:
    """

    app_order_id = int(app_order_id)
    if len(order_data_a) == 0:
        return order_data_a
    else:
        # df_order_data_a = pd.DataFrame(order_data_a)
        # df_order_data_a = df_order_data_a[df_order_data_a['app_order_id'] <= app_order_id]
        # df_order_data_a.sort_values(by = 'app_order_id',inplace=True)
        # df_order_data_a.loc[df_order_data_a['app_order_id'] == app_order_id, 'status'] = 20
        # df_order_data_a.fillna(-999,inplace=True)
        # order_data_a_new = df_order_data_a.to_dict(orient = 'records')
        order_data_a_new = [item for item in order_data_a if item['app_order_id'] <= app_order_id]
        for i in range(len(order_data_a_new)):
            if order_data_a_new[i]['app_order_id'] == app_order_id:
                order_data_a_new[i]['status'] = 20
        return order_data_a_new


def preprocess_installments_data_a(installments_data_a, contract_data_a, apply_time, app_order_id):
    """
    预处理installments_data_a数据，仅回溯时使用。
    :param installments_data_a:原始数据
    :param contract_data_a:合同表
    :param apply_time:申请时间
    :param app_order_id:申请订单号
    :return:
    """

    app_order_id = int(app_order_id)

    if len(installments_data_a) == 0:
        return installments_data_a

    else:
        df_installments_data_a = pd.DataFrame(installments_data_a)
        df_contract_data_a = pd.DataFrame(contract_data_a)

        df_installments_data_a = df_installments_data_a[df_installments_data_a['app_order_id'] <= app_order_id]
        # df_installments_data_a = df_installments_data_a[df_installments_data_a['status'] != -1]

        df_installments_data_a.sort_values(by='app_order_id', inplace=True)
        df_installments_data_a.index = range(df_installments_data_a.shape[0])
        df_installments_data_a = pd.merge(df_installments_data_a,
                                          df_contract_data_a[['app_order_id', 'settlement_time']], on='app_order_id',
                                          how='left', suffixes=('_drop', ''))

        filtered_data = df_installments_data_a[df_installments_data_a['settlement_time'] > apply_time]
        # 找到重复的 app_order_id，并获取对应的 extension_count 最大值
        duplicate_mask = filtered_data['app_order_id'].duplicated(keep=False)
        max_extension_count = filtered_data.groupby('app_order_id')['extension_count'].transform('max')
        # 更新 status 为 1 的条件
        condition = (duplicate_mask) & (filtered_data['extension_count'] == max_extension_count.values)
        condition1 = ~duplicate_mask

        # 将满足条件的列更新数值
        filtered_data.loc[condition, 'status'] = 1
        filtered_data.loc[condition1, 'status'] = 1
        filtered_data.loc[condition, 'settlement_type'] = 0
        filtered_data.loc[condition1, 'settlement_type'] = 0
        filtered_data.loc[condition, 'extension_fee'] = 0.0
        filtered_data.loc[condition1, 'extension_fee'] = 0.0
        df_installments_data_a.loc[df_installments_data_a.index.isin(filtered_data.index), 'status'] = filtered_data[
            'status']
        df_installments_data_a.loc[df_installments_data_a.index.isin(filtered_data.index), 'settlement_type'] = \
            filtered_data['settlement_type']
        df_installments_data_a.loc[df_installments_data_a.index.isin(filtered_data.index), 'extension_fee'] = \
            filtered_data['extension_fee']

        df_installments_data_a.loc[df_installments_data_a['settlement_time'] > apply_time, 'repaid_principal'] = 0.0
        df_installments_data_a.loc[df_installments_data_a['settlement_time'] > apply_time, 'repaid_interest'] = 0.0
        # df_installments_data_a.loc[df_installments_data_a['settlement_time'] > apply_time, 'repaid_cut_interest'] = 0.0
        df_installments_data_a.loc[df_installments_data_a['settlement_time'] > apply_time, 'repaid_service_fee'] = 0.0
        df_installments_data_a.loc[
            df_installments_data_a['settlement_time'] > apply_time, 'repaid_management_fee'] = 0.0
        df_installments_data_a.loc[
            df_installments_data_a['settlement_time'] > apply_time, 'repaid_overdue_interest'] = 0.0
        df_installments_data_a.loc[df_installments_data_a['settlement_time'] > apply_time, 'repaid_penalty'] = 0.0

        df_installments_data_a['overdue_interest'] = 0.0
        df_installments_data_a['overdue_days'] = 0.0
        # df_installments_data_a.fillna(-999, inplace=True)
        # df_installments_data_a['update_time'] = df_installments_data_a['settlement_time']

        installments_data_a_new = df_installments_data_a.to_dict(orient='records')
        # installments_data_a_new = [item for item in installments_data_a if item['app_order_id'] <= app_order_id]

        return installments_data_a_new


def assemble_req_data(data_info, installment_data, order_data, contract_data, device_list_data, fdc_data=None, tqdm_disable=False):
    """
    将查询后的订单数据拼接成跟线上一致的请求格式
    :param data_info:
    :param installment_data:
    :param order_data:
    :param contract_data:
    :param device_list_data:
    :param tqdm_disable:
    :return pd.DataFrame:
    """
    req_data_list = []
    logger.info(f"开始组装数据,共计{len(data_info)}条数据")
    for i, v in tqdm(data_info.iterrows(), desc='assemble data', disable=tqdm_disable):
        main_user_id = str(v['app_user_id'])
        main_order_id = str(v['order_id'])
        main_apply_time = v['apply_time']  # 实际为req表的请求时间一般大约订单表的使劲，所以会包含本单
        main_nid, main_phone = str(v['nid']), str(v['phone'])
        try:
            main_tx_id = str(v['tx_id'])
            req_obj = {}
            apply_time = str(v['real_apply_time'])
            # 兼容source异常的情况
            source = v.get('source', -1)
            if source is None:
                source = -999
            source = int(source)
            base_info = {'country_code': v['country_code'],
                         'merchant_id': v['merchant_id'],
                         'country_abbr': v['country_abbr'],
                         'tx_id': str(v['tx_id']),
                         'order_id': str(v['order_id']),
                         'phone': main_phone,
                         'nid': main_nid,
                         'source': source,
                         'acq_channel': v['acq_channel'],
                         'device_type': str(v['device_type']),
                         'app_user_id': str(v['app_user_id']),
                         'apply_time': apply_time,
                         'track_id': str(v['track_id'])}

            user_id = v['app_user_id']

            req_obj['base_info'] = base_info

            order_data['apply_time'] = order_data.apply_time.apply(lambda x: None if pd.isna(x) else  str(x) )
            order_data['create_time'] = order_data.create_time.apply(lambda x: None if pd.isna(x) else  str(x))
            order_data['update_time'] = order_data.update_time.apply(lambda x: None if pd.isna(x) else  str(x))

            installment_data['create_time'] = installment_data.create_time.apply(lambda x: None if pd.isna(x) else  str(x))
            installment_data['update_time'] = installment_data.update_time.apply(lambda x: None if pd.isna(x) else  str(x))
            installment_data['repayment_date'] = installment_data.repayment_date.apply(lambda x: None if pd.isna(x) else  str(x))
            installment_data['settlement_time'] = installment_data.settlement_time.apply(
                lambda x: None if pd.isna(x) else str(x))

            contract_data['settlement_time'] = contract_data.settlement_time.apply(lambda x: None if pd.isna(x) else  str(x))
            contract_data['create_time'] = contract_data.create_time.apply(lambda x: None if pd.isna(x) else  str(x))
            contract_data['update_time'] = contract_data.update_time.apply(lambda x: None if pd.isna(x) else  str(x))
            contract_data['activation_time'] = contract_data.activation_time.apply(lambda x: None if pd.isna(x) else  str(x))

            device_list_data['update_time'] = device_list_data.update_time.apply(lambda x: None if pd.isna(x) else  str(x) )
            device_list_data['day_inter'] = (
                    pd.to_datetime(apply_time) - pd.to_datetime(device_list_data['update_time'])).dt.days

            installment_data_a = installment_data[
                (installment_data.user_id == user_id) & (installment_data.create_time < apply_time)]
            order_data_a = order_data[
                (order_data.user_id == user_id) & (order_data.apply_time <= apply_time)]  # 包含等号，需要同一批次审核单
            # contract_data_a = contract_data[
            #     (contract_data.user_id == user_id) & (contract_data.activation_time <= apply_time)]
            contract_data_a = contract_data[
                (contract_data.user_id == user_id) & (contract_data.create_time <= apply_time)]
            device_list_data_a = device_list_data[
                (device_list_data.user_id == user_id) & (device_list_data.day_inter <= 180) & (
                        apply_time >= device_list_data.update_time)]

            user_info = v['user_info']
            device_info = v['device_info']
            installment_data_a = json.loads(installment_data_a.to_json(orient='records'))
            order_data_a = json.loads(order_data_a.to_json(orient='records'))
            contract_data_a = json.loads(contract_data_a.to_json(orient='records'))
            device_list_data_a = json.loads(device_list_data_a.to_json(orient='records'))

            if len(installment_data_a) == 0:
                installment_data_a = []
            if len(order_data_a) == 0:
                order_data_a = []
            if len(contract_data_a) == 0:
                contract_data_a = []
            if len(device_list_data_a) == 0:
                device_list_data_a = []

            applist_data = [] if v['app_list_url_data'] is None else json.loads(v['app_list_url_data'])
            sms_data = [] if v['sms_records_url_data'] is None else json.loads(v['sms_records_url_data'])
            calllog_data = [] if v['call_records_url_data'] is None else json.loads(v['call_records_url_data'])
            calendar_data = [] if v['calendar_records_url_data'] is None else json.loads(v['calendar_records_url_data'])
            try:
                contact_list = user_info['userInfo']['userContactInfo']
            except:
                contact_list = []

            str_apply_time = str(main_apply_time)
            order_data_a = preprocess_order_data_a(main_order_id, order_data_a)
            installment_data_a = preprocess_installments_data_a(installment_data_a, contract_data_a, str_apply_time,
                                                                main_order_id)
            req_obj['data_sources'] = {'user_info_1': user_info,
                                       'contact_list': contact_list,
                                       'device_info': device_info,
                                       'order_data_a': order_data_a,
                                       'installments_data_a': installment_data_a,
                                       'contract_data_a': contract_data_a,
                                       'applist_data': applist_data,
                                       'sms_data': sms_data,
                                       'calllog_data': calllog_data,
                                       'calendar_data': calendar_data,
                                       'device_list': device_list_data_a}
            if fdc_data is not None:
                order_fdc_data = fdc_data[fdc_data['order_id'] == main_order_id]
                if not order_fdc_data.empty:
                    req_obj['data_sources'].update({'fdc_id_inquiryV5': order_fdc_data['data'].iloc[0]})
            req_data_list.append(
                [main_nid, main_phone, main_user_id, main_order_id, main_tx_id, main_apply_time,
                 json.dumps(req_obj, ensure_ascii=False)])
        except:
            logger.info(f"订单{main_order_id},封装时发生异常:\n{traceback.format_exc()}")
            continue
    rs = pd.DataFrame(req_data_list,
                      columns=['nid', 'phone', 'app_user_id', 'app_order_id', 'tx_id', 'apply_time', 'request_params'])
    return rs


def get_fdc_data(data_info, user, passwd, dblink='inner', tqdm_disable=False):
    fdc_data_list = []
    id_card_list = data_info['nid'].tolist()
    for batch in tqdm(range(0, len(id_card_list), 1000), desc='get fdc data', disable=tqdm_disable):
        with DBConnection('af', user, passwd, dblink=dblink) as sql_con:
            id_cards = "','".join(id_card_list[batch:batch+1000])
            id_cards = f"'{id_cards}'"
            fdc_sql = f"""
                SELECT 
                    create_time AS time_stamp, 
                    es_id AS nid, 
                    result AS data
                FROM risk_dup.t_risk_third_log
                WHERE acq_channel = 'totem'
                AND es_id IN ({id_cards})
                UNION ALL
                SELECT 
                    time_stamp, 
                    id_card AS nid, 
                    data
                FROM warehouse.dwd_kafka_fdc_persistent_storage_id
                WHERE id_card IN ({id_cards});
                """
            df = pd.read_sql(fdc_sql, sql_con)
            df['nid'] = df['nid'].astype('str')
            df['time_stamp'] = df['time_stamp'].astype('str')
            fdc_data_list.append(df)
    fdc_data = pd.concat(fdc_data_list)
    data = pd.merge(fdc_data, data_info[['nid', 'order_id', 'apply_time']])
    data['time_stamp'] = pd.to_datetime(data['time_stamp'])
    data['apply_time'] = pd.to_datetime(data['apply_time'])
    data['time_diff'] = (data['apply_time'] - data['time_stamp']).abs()
    data = data.sort_values('time_diff').drop_duplicates('order_id')
    data = data.sort_values('time_stamp', ascending=False).drop_duplicates('order_id')
    data['data'] = data['data'].apply(lambda x: json.loads(x))
    return data

def get_sample_req(order_list, sys_name, user, passwd, dblink='inner', tqdm_disable=False):
    """
    获取建模样本的回溯数据信息
    :param order_list: 需要查询信息的订单app_order_id,可以是int64也可以是字符串列表
    :param sys_name: 系统名称
    :param user: tidb的个人账号
    :param passwd: tidb的个人密码
    :return:  {'app_user_id', 'app_order_id', 'apply_time', 'req_obj'} 其中req对象
    """
    data_info, installment_data, order_data, contract_data, device_list_data = query_order_info(order_list,
                                                                                                sys_name, user,
                                                                                                passwd, dblink=dblink,
                                                                                                fetch_url=True, tqdm_disable=tqdm_disable)
    fdc_data = None
    if sys_name == 'af':
        fdc_data = get_fdc_data(data_info, user, passwd, dblink=dblink, tqdm_disable=tqdm_disable)
    req_df = assemble_req_data(data_info, installment_data, order_data, contract_data, device_list_data, fdc_data, tqdm_disable=tqdm_disable)
    return req_df


# def fetch_sample_reqs(order_df, country_abbr, user, passwd, dblink='inner', splite_col='loan_date', save_dir='req_dir'):
#     """
#     根据传入order_df将数据下载到指定的目录
#     """
#     order_df['save_file'] = 'df_' + order_df[splite_col] + '.parquet'
#     save_files = order_df['save_file'].unique()
#     save_dir = Path(save_dir)
#     for save_file in save_files:
#         print(f"开始下载{save_file}的数据===================================>")
#         df = order_df[order_df['save_file'] == save_file]
#         order_list = df['app_order_id'].unique().tolist()
#         req_df = get_sample_req(order_list, country_abbr, user, passwd, dblink=dblink)
#         req_df.to_parquet(save_dir / save_files, compression='zstd')


# def loan_flag(order_list, country_abbr, user, passwd, dblink='inner'):
#     """
#     获取订单号的贷后标签
#     :param order_list: 待查询的定号的列表
#     :param country_abbr: 国家简称
#     :param user: tidb账号
#     :param passwd: tidb密码
#     :return:
#     """
#     # if country_abbr in ['th','id']:
#     #     db_ip, db_port, merchant_id, database = doris_con[dblink][country_abbr]
#     # else:
#     #     db_ip, db_port, merchant_id, database = tidb_conf[dblink][country_abbr]
#     db_ip, db_port, merchant_id, database = doris_con[dblink][country_abbr]
#     country_id, time_delt = country_info[country_abbr]
#
#     order_id_list = list(pd.unique(order_list))
#     print(f"共计输入{len(order_list)}条数据，清理重复数据后共{len(order_id_list)}条数据")
#     if len(order_id_list) == 1:  # 解决只有1条数据会导致格式异常报错的问题
#         order_id_list.extend(order_id_list)
#     order_id_list = tuple(order_id_list)
#     mysql_rule = get_dbcon(country_abbr, user, passwd, dblink)
#     # sql = f"""select * from {database}.t_installment where status != -1 and app_order_id in {tuple(order_id_list)} and extension_count=0"""
#     sql = f"""select * from {database}.t_installment where status != -1 and app_order_id in {tuple(order_id_list)} and extension_count=0"""
#     loan_df = pd.read_sql(sql, con=mysql_rule)
#     orders_query_result = loan_df['app_order_id'].nunique()
#     print(
#         f'输入总数{len(order_list)}条数据，去重后{len(order_id_list)}条数据,installment表中查询到{len(loan_df)}条数据,对应{orders_query_result}订单的数据')
#
#     # 获取当前的北京时间并将其转化成本地时间以便进行后续计算
#     current_time, china_time, = country_now(country_abbr)
#     current_time = current_time.normalize()
#     time_max = pd.to_datetime('2090-12-12')
#
#     loan_df['calc_time'] = china_time
#     loan_df['repayment_date'] = pd.to_datetime(loan_df['repayment_date'])
#
#     loan_df['finish_time'] = pd.to_datetime(np.where(loan_df['status'] == 2, loan_df['update_time'], None))
#     loan_df['finish_date'] = pd.to_datetime(
#         np.where(loan_df['status'] == 2, loan_df['update_time'], time_max)).normalize()
#
#     loan_df['agr_pd0'] = np.where((current_time - loan_df['repayment_date']).dt.days >= 0, 1, 0)
#     loan_df['def_pd0'] = np.where((loan_df['agr_pd0'] == 1) & \
#                                   ((loan_df['finish_date'] - loan_df['repayment_date']).dt.days >= 0), 1, 0)
#     loan_df['agr_pd0_amt'] = np.where(loan_df['agr_pd0'] == 1, loan_df['principal'] - loan_df['cut_interest'], 0)
#     loan_df['def_pd0_amt'] = np.where(loan_df['def_pd0'] == 1, loan_df['principal'] - loan_df['cut_interest'], 0)
#
#     loan_df['agr_cpd'] = loan_df['agr_pd0']
#     loan_df['def_cpd'] = np.where((loan_df['agr_cpd'] == 1) & (loan_df['status'] != 2), 1, 0)
#     loan_df['agr_cpd_amt'] = np.where(loan_df['agr_cpd'] == 1, loan_df['principal'] - loan_df['cut_interest'], 0)
#     loan_df['def_cpd_amt'] = np.where(loan_df['def_cpd'] == 1, loan_df['principal'] - loan_df['cut_interest'], 0)
#
#     loan_df['agr_pd1'] = np.where((current_time - loan_df['repayment_date']).dt.days >= 1, 1, 0)
#     loan_df['def_pd1'] = np.where((loan_df['agr_pd1'] == 1) & \
#                                   ((loan_df['finish_date'] - loan_df['repayment_date']).dt.days >= 1), 1, 0)
#     loan_df['agr_pd1_amt'] = np.where(loan_df['agr_pd1'] == 1, loan_df['principal'] - loan_df['cut_interest'], 0)
#     loan_df['def_pd1_amt'] = np.where(loan_df['def_pd1'] == 1, loan_df['principal'] - loan_df['cut_interest'], 0)
#
#     loan_df['agr_pd3'] = np.where((current_time - loan_df['repayment_date']).dt.days >= 3, 1, 0)
#     loan_df['def_pd3'] = np.where((loan_df['agr_pd3'] == 1) & \
#                                   ((loan_df['finish_date'] - loan_df['repayment_date']).dt.days >= 3), 1, 0)
#     loan_df['agr_pd3_amt'] = np.where(loan_df['agr_pd3'] == 1, loan_df['principal'] - loan_df['cut_interest'], 0)
#     loan_df['def_pd3_amt'] = np.where(loan_df['def_pd3'] == 1, loan_df['principal'] - loan_df['cut_interest'], 0)
#
#     loan_df['agr_pd4'] = np.where((current_time - loan_df['repayment_date']).dt.days >= 4, 1, 0)
#     loan_df['def_pd4'] = np.where((loan_df['agr_pd4'] == 1) & \
#                                   ((loan_df['finish_date'] - loan_df['repayment_date']).dt.days >= 4), 1, 0)
#     loan_df['agr_pd4_amt'] = np.where(loan_df['agr_pd4'] == 1, loan_df['principal'] - loan_df['cut_interest'], 0)
#     loan_df['def_pd4_amt'] = np.where(loan_df['def_pd4'] == 1, loan_df['principal'] - loan_df['cut_interest'], 0)
#
#     loan_df['agr_pd7'] = np.where((current_time - loan_df['repayment_date']).dt.days >= 7, 1, 0)
#     loan_df['def_pd7'] = np.where((loan_df['agr_pd7'] == 1) & \
#                                   ((loan_df['finish_date'] - loan_df['repayment_date']).dt.days >= 7), 1, 0)
#     loan_df['agr_pd7_amt'] = np.where(loan_df['agr_pd7'] == 1, loan_df['principal'] - loan_df['cut_interest'], 0)
#     loan_df['def_pd7_amt'] = np.where(loan_df['def_pd7'] == 1, loan_df['principal'] - loan_df['cut_interest'], 0)
#
#     loan_df['agr_pd14'] = np.where((current_time - loan_df['repayment_date']).dt.days >= 14, 1, 0)
#     loan_df['def_pd14'] = np.where((loan_df['agr_pd14'] == 1) & \
#                                    ((loan_df['finish_date'] - loan_df['repayment_date']).dt.days >= 14), 1, 0)
#     loan_df['agr_pd14_amt'] = np.where(loan_df['agr_pd14'] == 1, loan_df['principal'] - loan_df['cut_interest'], 0)
#     loan_df['def_pd14_amt'] = np.where(loan_df['def_pd14'] == 1, loan_df['principal'] - loan_df['cut_interest'], 0)
#
#     loan_df['repaid'] = loan_df['extension_fee'] + loan_df['repaid_principal'] + loan_df['repaid_interest'] + \
#                         loan_df['repaid_service_fee'] + loan_df['repaid_management_fee'] + loan_df[
#                             'repaid_overdue_interest'] + \
#                         loan_df['repaid_penalty'] - loan_df['waive_amount']
#
#     loan_df['payout'] = np.where(loan_df['extension_count'] == 0, loan_df['principal'] - loan_df['cut_interest'], 0)
#     loan_df['profit'] = np.where(loan_df['agr_pd7'] == 1, loan_df['repaid'] - loan_df['payout'], 0)
#
#     loan_df['original'] = np.where(loan_df['extension_count'] == 0, 1, 0)
#     loan_df['is_extension'] = np.where(loan_df['settlement_type'] == 2, 1, 0)
#
#     loan_df = loan_df[(loan_df['original'] == 1) & (loan_df['installment_num'] == 1)]
#
#     return loan_df[
#         ['app_order_id', 'calc_time', 'agr_pd0', 'def_pd0', 'agr_pd1', 'def_pd1', 'agr_pd3', 'def_pd3', 'agr_pd4',
#          'def_pd4', 'agr_pd7', 'def_pd7', 'agr_pd14', 'def_pd14', 'agr_cpd', 'agr_cpd_amt', 'def_cpd_amt']]


def get_loans(sys_name, user, passwd, dblink='inner', start_date='2022-01-01', end_date='2999-12-01',
              new_old_user_status=None,db_con=None):
    """
    根据国家查询所有的installment表中的样本信息
    :param sys_name: 系统名称 如 mx
    :param user:数据库用户名称
    :param passwd:<PASSWORD>:用户密闭
    :param dblink:使用内网还是外网的ip，在jupyter服务器使用内网，其他都用外网
    :param start_date:起始时间，放款时间
    :param end_date: 截至时间，放款时间
    :param new_old_user_status : 新老用户标签，同业务库中 new_old_user_status 的含义
    :param db_con : 可复用的数据库连接
    """

    if new_old_user_status is None:
        new_old_user_status = [0, 1, 2]

    sys_info = get_sys_info(sys_name,dblink)
    # sql_con = get_dbcon(sys_name, user, passwd, dblink)

    country_abbr = sys_info['country_abbr']
    db = sys_info['db']

    new_old_user_status = list2sqlstr(new_old_user_status)

    with DBConnection(sys_name, user, passwd, dblink=dblink) as sql_con:
        if sys_name == 'af':
            loan_sql = f"""
            with a1 as (
                select
                     a.app_order_id,a.id_card_number,a.user_name,a.phone_number,a.device_type,a.apply_time,a.product_info,
                     a.personal_info,a.work_info,a.contact_info,a.bank_account_info,a.ktp_info,a.face_recognition_info,
                     u.source,u.create_time,row_number() over (partition by a.id order by u.create_time desc) rn
                from {db}.t_app_order a
                left join (select * from {db}.t_app_user) u on a.acq_channel = u.acq_channel and a.user_id = u.user_id and a.apply_time>= u.create_time
            ),app_base_info  as (
                select
                    a.app_order_id,a.id_card_number,a.user_name,a.phone_number,a.device_type,a.apply_time,a.product_info,
                     a.personal_info,a.work_info,a.contact_info,a.bank_account_info,a.ktp_info,a.face_recognition_info,
                     a.source,a.create_time,cr.term,cr.days_per_term,r.biz_id, tx_id
               from  (select * from a1 where rn=1) a
                 left join {db}.t_contract cr on cr.app_order_id= a.app_order_id
                 left join (select distinct biz_id,tx_id from {db}.t_risk_req_record where biz_type =1 ) r on a.app_order_id= r.biz_id
            )select
                 a.*,b.id_card_number,b.phone_number,b.user_name,b.device_type,b.apply_time,b.personal_info,b.work_info,b.contact_info,b.bank_account_info,
                        b.ktp_info,b.face_recognition_info,b.term,b.days_per_term,b.tx_id,b.source
            from {db}.t_installment a
            left join app_base_info  b on b.app_order_id = a.app_order_id
            where a.status !=-1 and b.tx_id is not null and a.new_old_user_status in {new_old_user_status} and a.create_time >= '{start_date} 00:00:00' and a.create_time < '{end_date} 00:00:00'
            """
        else:
            loan_sql = f"""
                with a1 as (
                    select
                        a.app_order_id,a.id_card_number,a.user_name,a.phone_number,a.device_type,a.apply_time,a.product_info,
                        a.personal_info,a.work_info,a.contact_info,a.bank_account_info,a.ktp_info,a.face_recognition_info,
                        u.source,u.create_time,row_number() over (partition by a.id order by u.create_time desc) rn
                    from {db}.t_app_order a
                    left join (select * from {db}.t_app_user) u on a.acq_channel = u.acq_channel and a.user_id = u.user_id and a.apply_time>= u.create_time
                ),app_base_info  as (
                    select
                        a.app_order_id,a.id_card_number,a.user_name,a.phone_number,a.device_type,a.apply_time,a.product_info,
                        a.personal_info,a.work_info,a.contact_info,a.bank_account_info,a.ktp_info,a.face_recognition_info,
                        a.source,a.create_time,cr.term,cr.days_per_term,r.biz_id, tx_id
                from  (select * from a1 where rn=1) a
                    left join {db}.t_contract cr on cr.app_order_id= a.app_order_id
                    left join  {db}.t_risk_req_record r on a.app_order_id= r.biz_id
                )select
                    a.*,b.id_card_number,b.phone_number,b.user_name,b.device_type,b.apply_time,b.personal_info,b.work_info,b.contact_info,b.bank_account_info,
                            b.ktp_info,b.face_recognition_info,b.term,b.days_per_term,b.tx_id,b.source
                from {db}.t_installment a
                left join app_base_info  b on b.app_order_id = a.app_order_id
                where a.status !=-1 and b.tx_id is not null and a.new_old_user_status in {new_old_user_status} and a.create_time >= '{start_date} 00:00:00' and a.create_time < '{end_date} 00:00:00'
            """
        logger.info(f"sql:\n{loan_sql}")

        loan_df = pd.read_sql(loan_sql, sql_con, dtype={'id': 'str', 'app_order_id': 'str', 'tx_id': 'str'})
    assert len(loan_df) > 0, f"sql查询结果为空，请检查入参。。。。。"

    condition = loan_df['phone_number'].isin(['4444555505', '4444555508', '4444555509'])
    loan_df = loan_df[~condition]
    logger.info(f'清理后订单数:{loan_df.shape}' )
    logger.info(f'订单后枚举值:{loan_df.app_order_id.nunique()}')
    loan_df['loan_time'] = pd.to_datetime(loan_df['create_time'])
    loan_df = gen_loan_flag(loan_df, sys_name)
    return loan_df


def gen_loan_flag(loan_df, sys_name):
    """
    根据国家信息，计算订单的标签
    """
    current_time, china_time, = local_time(sys_name)
    current_time = current_time.normalize()
    time_max = pd.to_datetime('2090-12-12')
    loan_df['query_time'] = china_time
    loan_df['repayment_date'] = pd.to_datetime(loan_df['repayment_date'])
    loan_df['loan_month'] = loan_df['create_time'].dt.strftime('%Y-%m')
    loan_df['apply_month'] = loan_df['apply_time'].dt.strftime('%Y-%m')
    loan_df['repayment_month'] = loan_df['repayment_date'].dt.strftime('%Y-%m')
    loan_df['loan_week'] = loan_df['create_time'].map(week_start_day)
    loan_df['apply_week'] = loan_df['apply_time'].map(week_start_day)
    loan_df['repayment_week'] = loan_df['repayment_date'].map(week_start_day)
    loan_df['loan_date'] = loan_df['create_time'].dt.strftime('%Y-%m-%d')
    loan_df['apply_date'] = loan_df['apply_time'].dt.strftime('%Y-%m-%d')
    loan_df['finish_time'] = pd.to_datetime(np.where(loan_df['status'] == 2, loan_df['update_time'], None))
    loan_df['finish_date'] = pd.to_datetime(
        np.where(loan_df['status'] == 2, loan_df['update_time'], time_max)).normalize()
    loan_df['agr_pd0'] = np.where((current_time - loan_df['repayment_date']).dt.days >= 1, 1, 0)
    loan_df['def_pd0'] = np.where((loan_df['agr_pd0'] == 1) & \
                                  ((loan_df['finish_date'] - loan_df['repayment_date']).dt.days >= 1), 1, 0)
    loan_df['agr_pd0_amt'] = np.where(loan_df['agr_pd0'] == 1, loan_df['principal'] - loan_df['cut_interest'], 0)
    loan_df['def_pd0_amt'] = np.where(loan_df['def_pd0'] == 1, loan_df['principal'] - loan_df['cut_interest'], 0)
    loan_df['agr_cpd'] = loan_df['agr_pd0']
    loan_df['def_cpd'] = np.where((loan_df['agr_cpd'] == 1) & (loan_df['status'] != 2), 1, 0)
    loan_df['agr_cpd_amt'] = np.where(loan_df['agr_cpd'] == 1, loan_df['principal'] - loan_df['cut_interest'], 0)
    loan_df['def_cpd_amt'] = np.where(loan_df['def_cpd'] == 1, loan_df['principal'] - loan_df['cut_interest'], 0)
    loan_df['agr_pd1'] = np.where((current_time - loan_df['repayment_date']).dt.days >= 2, 1, 0)
    loan_df['def_pd1'] = np.where((loan_df['agr_pd1'] == 1) & \
                                  ((loan_df['finish_date'] - loan_df['repayment_date']).dt.days >= 2), 1, 0)
    loan_df['agr_pd1_amt'] = np.where(loan_df['agr_pd1'] == 1, loan_df['principal'] - loan_df['cut_interest'], 0)
    loan_df['def_pd1_amt'] = np.where(loan_df['def_pd1'] == 1, loan_df['principal'] - loan_df['cut_interest'], 0)
    loan_df['agr_pd3'] = np.where((current_time - loan_df['repayment_date']).dt.days >= 4, 1, 0)
    loan_df['def_pd3'] = np.where((loan_df['agr_pd3'] == 1) & \
                                  ((loan_df['finish_date'] - loan_df['repayment_date']).dt.days >= 4), 1, 0)
    loan_df['agr_pd3_amt'] = np.where(loan_df['agr_pd3'] == 1, loan_df['principal'] - loan_df['cut_interest'], 0)
    loan_df['def_pd3_amt'] = np.where(loan_df['def_pd3'] == 1, loan_df['principal'] - loan_df['cut_interest'], 0)
    loan_df['agr_pd4'] = np.where((current_time - loan_df['repayment_date']).dt.days >= 5, 1, 0)
    loan_df['def_pd4'] = np.where((loan_df['agr_pd4'] == 1) & \
                                  ((loan_df['finish_date'] - loan_df['repayment_date']).dt.days >= 5), 1, 0)
    loan_df['agr_pd4_amt'] = np.where(loan_df['agr_pd4'] == 1, loan_df['principal'] - loan_df['cut_interest'], 0)
    loan_df['def_pd4_amt'] = np.where(loan_df['def_pd4'] == 1, loan_df['principal'] - loan_df['cut_interest'], 0)
    loan_df['agr_pd7'] = np.where((current_time - loan_df['repayment_date']).dt.days >= 8, 1, 0)
    loan_df['def_pd7'] = np.where((loan_df['agr_pd7'] == 1) & \
                                  ((loan_df['finish_date'] - loan_df['repayment_date']).dt.days >= 8), 1, 0)
    loan_df['agr_pd7_amt'] = np.where(loan_df['agr_pd7'] == 1, loan_df['principal'] - loan_df['cut_interest'], 0)
    loan_df['def_pd7_amt'] = np.where(loan_df['def_pd7'] == 1, loan_df['principal'] - loan_df['cut_interest'], 0)
    loan_df['agr_pd14'] = np.where((current_time - loan_df['repayment_date']).dt.days >= 15, 1, 0)
    loan_df['def_pd14'] = np.where((loan_df['agr_pd14'] == 1) & \
                                   ((loan_df['finish_date'] - loan_df['repayment_date']).dt.days >= 15), 1, 0)
    loan_df['agr_pd14_amt'] = np.where(loan_df['agr_pd14'] == 1, loan_df['principal'] - loan_df['cut_interest'], 0)
    loan_df['def_pd14_amt'] = np.where(loan_df['def_pd14'] == 1, loan_df['principal'] - loan_df['cut_interest'], 0)
    loan_df['repaid'] = loan_df['extension_fee'] + loan_df['repaid_principal'] + loan_df['repaid_interest'] + \
                        loan_df['repaid_service_fee'] + loan_df['repaid_management_fee'] + loan_df[
                            'repaid_overdue_interest'] + loan_df['repaid_penalty'] - loan_df['waive_amount']
    loan_df['payout'] = np.where(loan_df['extension_count'] == 0, loan_df['principal'] - loan_df['cut_interest'], 0)
    loan_df['profit'] = np.where(loan_df['agr_pd7'] == 1, loan_df['repaid'] - loan_df['payout'], 0)
    loan_df['original'] = np.where(loan_df['extension_count'] == 0, 1, 0)
    loan_df['is_extension'] = np.where(loan_df['settlement_type'] == 2, 1, 0)

    # 添加ltv 结果的计算逻辑
    loan_df = loan_df.sort_values('create_time', ascending=True)
    sample_df = loan_df.copy()
    ltv_df = sample_df[['id', 'user_id', 'create_time']].merge(loan_df, on='user_id', how='left')
    condition = (ltv_df['create_time_y'] >= ltv_df['create_time_x']) & \
                ((ltv_df['create_time_y'] - ltv_df['create_time_x']).dt.days <= 30)
    ltv_df = ltv_df[condition]
    ltv_tag = ltv_df.groupby(['id_x'])[['repaid', 'payout', 'profit']].sum().reset_index()
    ltv_tag['def_ltv'] = np.where(ltv_tag['profit'] < 0, 1, 0)
    ltv_tag = ltv_tag.rename(columns={
        "repaid": "repaid_ltv",
        "payout": "payout_ltv",
        "profit": "profit_ltv",
        "id_x": 'id'
    })
    loan_df = loan_df.merge(ltv_tag, on='id')
    return loan_df


def loan_stats(loan_df, groupd='loan_month'):
    return loan_df.groupby(groupd).agg({
        'def_pd1': 'sum',
        'agr_pd1': 'sum',
        'def_pd1_amt': 'sum',
        'agr_pd1_amt': 'sum',
        'def_pd7': 'sum',
        'agr_pd7': 'sum',
        'def_pd7_amt': 'sum',
        'agr_pd7_amt': 'sum',
        'phone_number': 'nunique'
    }).assign(
        pd1=lambda x: x['def_pd1'] / x['agr_pd1'],
        pd1_amt=lambda x: x['def_pd1_amt'] / x['agr_pd1_amt'],
        pd7=lambda x: x['def_pd7'] / x['agr_pd7'],
        pd7_amt=lambda x: x['def_pd7_amt'] / x['agr_pd7_amt']
    ).reset_index()


def get_sample_module_data(order_list, sys_name, module_name, user, passwd, dblink='inner'):
    """
    基于app_order_id查询订单module的模块的数据，并全部解析出来
    :param order_list: 订单列表
    :param sys_name: 系统名称
    :param module_name: 待查询的模块名称
    :param user: dories账号
    :param passwd: dories密码
    :param dblink: 内外网标记
    """
    dories_con = get_dbcon(sys_name, user, passwd, dblink=dblink)
    sys_info = get_sys_info(sys_name, dblink=dblink)
    merchant_id = sys_info['merchant_id']
    order_chunks = chunk_list(order_list, 1000)
    chunk_datas = []
    for chunk in order_chunks:
        orders = list2sqlstr(chunk)
        sql = f"""select b.biz_id as app_order_id,a.request_time,module_name,a.result from
                risk_offline.t_risk_module_record_decision_{merchant_id.lower()} a
                    inner join ath_system.t_risk_req_record b on a.tx_id = b.tx_id
        where b.biz_id in {orders} and module_name = '{module_name}' 
        """
        data_df = pd.read_sql(sql, con=dories_con, dtype={'app_order_id': str})
        data_df = data_df.reset_index(drop=True)
        explode_df = pd.json_normalize(data_df['result'].map(json.loads))
        data_df = data_df.drop(columns=['result'])
        chunk_df = pd.concat([ data_df,explode_df], axis=1)
        chunk_datas.append(chunk_df)
    return pd.concat(chunk_datas,axis=0).reset_index(drop=True)

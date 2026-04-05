import json

from toollib.unversal import *
from toollib.asystem_env.sample_util import gen_loan_flag
import logging

logger = logging.getLogger(__name__)


def get_score_doris(
    sys_name,
    user,
    passwd,
    module_name, 
    score_field,
    new_old_user_status=[0],
    start_date='2024-10-15', 
    end_date='2030-12-01', 
                     dblink='inner'):
    
    sys_dict = get_sys_info(sys_name, dblink)
    merchant_id = sys_dict['merchant_id']
    db = sys_dict['db']
    dories_con = get_dbcon(sys_name, user, passwd, dblink=dblink)

    if len(new_old_user_status) <= 1:
        new_old_user_status = new_old_user_status + new_old_user_status
    """
    根据线上的放款订单匹配线上的评分对应的数据
    """
    sql = f"""
        select r.biz_id, r.tx_id, o.status, o.new_old_user_status, d.result
        from {db}.t_risk_req_record r
        join {db}.t_app_order o
            on r.biz_id = o.app_order_id
            and r.create_time >= '{start_date}'
            and r.create_time < '{end_date}'
            and o.create_time >= '{start_date}'
            and o.create_time < '{end_date}'
        join risk_offline.t_risk_module_record_decision_{merchant_id.lower()} d
            on d.tx_id = r.tx_id
            and d.module_name = '{module_name}'
            and d.create_time >= '{start_date}'
            and d.create_time < '{end_date}'
        where o.new_old_user_status in {tuple(new_old_user_status)}
    """
    
    def get_score_value(row, f):
        # 从reuslt 中提取模型的评分字段
        tx_id = row['tx_id']
        try:
            obj = json.loads(row['result'])
            return obj[f]
        except:
            print(f"tx_id={tx_id}解析失败，请检查result中的数据")
            return -999
    
    print(sql)
    df = pd.read_sql(sql, dories_con)
    if not df.empty:
        df[score_field] = df.apply(lambda x: get_score_value(x, score_field), axis=1)
    return df

def loan_score_doris2(sys_name, user, passwd, module_name, score_field, new_old_user_status=[0],
                       start_date='2024-10-15', end_date='2030-12-01', dblink='inner'):
    """
    根据线上的放款订单匹配线上的评分对应的数据
    :param sys_name: 系统名称
    :param user: dories账号
    :param passwd: dories密码
    :param module_name: 模型的模块名称（对应于在appolo上配置的模型的key）
    :param score_field: 模型上线对应的字段名称（模型接口算出来的评分字段名）
    :param new_old_user_status: 订单类型限制，只能传入list，跟生产的  new_old_user_status 同含义。
    :param start_date: 限制查询的开始时间，默认值2024-10-15
    :param end_date: 限制查询得结束时间，默认值2030-12-01
    """

    def get_score_value(row, f):
        # 从reuslt 中提取模型的评分字段
        tx_id = row['tx_id']
        try:
            obj = json.loads(row['result'])
            return obj[f]
        except:
            print(f"tx_id={tx_id}解析失败，请检查result中的数据")
            return -999

    sys_dict = get_sys_info(sys_name, dblink)
    merchant_id = sys_dict['merchant_id']
    db = sys_dict['db']

    if len(new_old_user_status) <= 1:
        new_old_user_status = new_old_user_status + new_old_user_status
    # dories_con = get_dbcon(sys_name, user, passwd, dblink=dblink)

    t_risk_req_record_tbl = f"{db}.t_risk_req_record" if sys_name != 'af' else " (select * from af_system.t_risk_req_record where biz_type =1 ) "
    with DBConnection(sys_name, user, passwd, dblink=dblink) as dories_con:
        loan_sql = f"""
            with app_base_info as (
                select
                a.app_order_id,a.id_card_number,a.user_name,a.phone_number,a.device_type,a.apply_time,a.product_info,c.tx_id,
                a.personal_info,a.work_info,a.contact_info,a.bank_account_info,a.ktp_info,a.face_recognition_info,cr.term,
                cr.days_per_term,u.source,row_number() over (partition by a.app_order_id order by u.create_time desc) rn
                from {db}.t_app_order a
                left join {db}.t_contract cr on cr.app_order_id= a.app_order_id
                left join {db}.t_app_user u on a.acq_channel = u.acq_channel and a.user_id = u.user_id  and a.apply_time >= u.create_time
                left  join {t_risk_req_record_tbl} c on a.app_order_id = c.biz_id
            )select a.*,b.id_card_number,b.phone_number,b.user_name,b.device_type,b.apply_time,b.personal_info,b.work_info,b.contact_info,b.bank_account_info,
                    b.ktp_info,b.face_recognition_info,b.source,b.term,b.days_per_term,c.acq_channel_name,b.tx_id,ro.result
                    from {db}.t_installment a
            inner join app_base_info b on a.app_order_id = b.app_order_id
            left join {db}.t_acq_channel c on a.acq_channel = c.acq_channel
            inner join (select tx_id,result from risk_offline.t_risk_module_record_decision_{merchant_id.lower()} 
                            where module_name = '{module_name}') ro on ro.tx_id = b.tx_id
            where a.status !=-1 and b.tx_id is not null and b.rn=1 and a.extension_count =0 and a.create_time >= '{start_date} 00:00:00' and a.create_time < '{end_date} 00:00:00'
                and a.new_old_user_status in {tuple(new_old_user_status)}
            """
        logger.info(f"sql:\n{loan_sql}")
        loan_df = pd.read_sql(loan_sql, dories_con, dtype={'app_order_id': 'str'})
    loan_df[score_field] = loan_df.apply(lambda x: get_score_value(x, score_field), axis=1)
    return loan_df

def apply_score_doris(sys_name, user, passwd, module_name, score_field, new_old_user_status=[0],
                      start_date='2024-10-15', end_date='2030-12-01', dblink='inner'):
    def get_score_value(row, f):
        # 从reuslt 中提取模型的评分字段
        tx_id = row['tx_id']
        try:
            obj = json.loads(row['result'])
            return obj[f]
        except:
            print(f"tx_id={tx_id}解析失败，请检查result中的数据")
            return -999

    sys_dict = get_sys_info(sys_name, dblink)
    db = sys_dict['db']
    merchant_id = sys_dict['merchant_id']

    if len(new_old_user_status) <= 1:
        new_old_user_status = new_old_user_status + new_old_user_status
        
    
    t_risk_req_record_tbl = f"{db}.t_risk_req_record" if sys_name != 'af' else " (select * from af_system.t_risk_req_record where biz_type =1 ) "    
    
    with DBConnection(sys_name, user, passwd, dblink=dblink) as dories_con:
        apply_sql = f"""
            select *
            from risk_offline.t_risk_module_record_decision_{merchant_id.lower()} d
            join {t_risk_req_record_tbl} r 
                on d.tx_id = r.tx_id
            join {db}.t_app_order a 
                on r.biz_id = a.app_order_id
            where request_time >= '{start_date}'
            and request_time < '{end_date}'
            and module_name = '{module_name}'
            and a.new_old_user_status in {tuple(new_old_user_status)}
        """
        logger.info(f"sql:\n{apply_sql}")
        apply_df = pd.read_sql(apply_sql, dories_con)
    apply_df[score_field] = apply_df.apply(lambda x: get_score_value(x, score_field), axis=1)
    return apply_df

def loan_score_doris(sys_name, user, passwd, module_name, score_field, new_old_user_status=[0],
                      start_date='2024-10-15', end_date='2030-12-01', dblink='inner'):
    """
    根据线上的放款订单匹配线上的评分对应的数据
    :param sys_name: 系统名称
    :param user: dories账号
    :param passwd: dories密码
    :param module_name: 模型的模块名称（对应于在appolo上配置的模型的key）
    :param score_field: 模型上线对应的字段名称（模型接口算出来的评分字段名）
    :param new_old_user_status: 订单类型限制，只能传入list，跟生产的  new_old_user_status 同含义。
    :param start_date: 限制查询的开始时间，默认值2024-10-15
    :param end_date: 限制查询得结束时间，默认值2030-12-01
    """

    sys_dict = get_sys_info(sys_name, dblink)
    merchant_id = sys_dict['merchant_id']
    db = sys_dict['db']

    dories_con = get_dbcon(sys_name, user, passwd, dblink=dblink)

    if len(new_old_user_status) <= 1:
        new_old_user_status = new_old_user_status + new_old_user_status

    loan_sql = f"""
    with app_base_info as (
        select
         a.app_order_id,a.id_card_number,a.user_name,a.phone_number,a.device_type,a.apply_time,a.product_info,c.tx_id,
         a.personal_info,a.work_info,a.contact_info,a.bank_account_info,a.ktp_info,a.face_recognition_info,cr.term,
         cr.days_per_term,u.source,row_number() over (partition by a.app_order_id order by u.create_time desc) rn
        from {db}.t_app_order a
        left join {db}.t_contract cr on cr.app_order_id= a.app_order_id
        left join {db}.t_app_user u on a.acq_channel = u.acq_channel and a.user_id = u.user_id  and a.apply_time >= u.create_time
        left  join {db}.t_risk_req_record c on a.app_order_id = c.biz_id
    )select a.*,b.id_card_number,b.phone_number,b.user_name,b.device_type,b.apply_time,b.personal_info,b.work_info,b.contact_info,b.bank_account_info,
            b.ktp_info,b.face_recognition_info,b.source,b.term,b.days_per_term,c.acq_channel_name,b.tx_id
            from {db}.t_installment a
    inner join app_base_info b on a.app_order_id = b.app_order_id
    left join {db}.t_acq_channel c on a.acq_channel = c.acq_channel
    where a.status !=-1 and b.tx_id is not null and b.rn=1 and a.extension_count =0 and a.create_time >= '{start_date} 00:00:00' and a.create_time < '{end_date} 00:00:00'
        and a.new_old_user_status in {tuple(new_old_user_status)}
    """
    loan_df = pd.read_sql(loan_sql, dories_con)

    loan_df = gen_loan_flag(loan_df, sys_name)
    print(f"完成installment的订单查询:{loan_df.shape}")

    module_df = model_api_logs(sys_name, dories_con, merchant_id, module_name, score_field, start_date, end_date)
    print(f"完成module数据的查询:{module_df.shape}")
    data_df = loan_df.merge(module_df, on='tx_id')
    print(f"完成数据匹配:{data_df.shape}")
    return data_df


def model_api_logs(sys_name, dories_con, merchant_id, module_name, score_field, start_date, end_date):
    """查询并解析模型的线上调用数据。
    亚洲:只能查询到决策节点的数据(dories中做了授信和决策)
    拉美:只能通过hive.dwd和模块的名称查询到所有的模型调用日志。
    """

    def get_score_value(row, f):
        # 从reuslt 中提取模型的评分字段
        tx_id = row['tx_id']
        try:
            obj = json.loads(row['result'])
            return obj[f]
        except:
            print(f"tx_id={tx_id}解析失败，请检查result中的数据")
            return -999

    # if country_abbr in ['id', 'th']:
    #     module_sql = f"""select tx_id,request_time,module_name,result from risk_offline.t_risk_module_record_decision_{merchant_id.lower()}
    #     where module_name = '{module_name}' and request_time>= '{start_date} 00:00:00' and request_time<'{end_date} 00:00:00'
    #     """
    # else:
    #     module_sql = f"""select tx_id,request_time,module_name,result from hive.dwd.dwd_risk_module_record_id
    #     where merchant_id = '{merchant_id}'
    #     and module_name = '{module_name}'
    #     and dt>= '{start_date}' and dt<'{end_date}'
    #     """
    module_sql = f"""select tx_id,request_time,module_name,result from risk_offline.t_risk_module_record_decision_{merchant_id.lower()} 
    where module_name = '{module_name}' and request_time>= '{start_date} 00:00:00' and request_time<'{end_date} 00:00:00'
    """
    module_df = pd.read_sql(module_sql, dories_con)

    module_df[score_field] = module_df.apply(lambda x: get_score_value(x, score_field), axis=1)
    return module_df


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
        auc = roc_auc_score(df_new[target], df_new[feature])
        auc = auc if auc > 0.5 else 1 - auc
        return auc

    dti['auc_cum'] = dti['bin_code'].map(_cum_calc_auc)
    dti.rename({feature_bin: 'bin'}, axis=1, inplace=True)
    dti.insert(0, "target", [target] * dti.shape[0])
    dti.insert(0, "feature", [feature] * dti.shape[0])
    return dti[['feature', 'target', 'bin', 'bad', 'total', 'total_rate', 'brate', 'brate_cum', 'lift', 'ks', 'iv',
                'auc_cum'] + mean_cols]


def get_days(start='2024-01-01', end='2025-01-01'):
    start_date = datetime.strptime(start, '%Y-%m-%d')
    end_date = datetime.strptime(end, '%Y-%m-%d')
    result = []
    current_date = start_date
    while current_date <= end_date:
        result.append(current_date.strftime('%Y-%m-%d'))  # 格式化输出日期
        current_date += timedelta(days=1)  # 增加一天
    return result


def get_hive_module_info(module_name, dt, sample_df):
    hql = f"""
    select tx_id,result from hive.dwd.dwd_risk_module_record_id
          where module_name ='{module_name}' and dt = '{dt}'
    """
    hive_df = pd.read_sql(hql, doris_con, dtype={'tx_id': 'str'})
    hive_df = hive_df.merge(sample_df[['tx_id', 'file_name', 'app_order_id']], on='tx_id')
    result_df = []
    for i, row in hive_df.iterrows():
        rs = {'app_order_id': row['app_order_id']}
        json_obj = json.loads(row['result'])
        rs.update(json_obj)
        result_df.append(rs)
    result_df = pd.DataFrame(result_df)
    result_df = result_df.fillna(-999)
    return result_df


def get_hive_module_infos(module_name, dts, sample_df):
    rs_df = None
    for dt in tqdm(dts):
        hql = f"""
        select tx_id,result from hive.dwd.dwd_risk_module_record_id
              where module_name ='{module_name}' and dt = '{dt}'
        """
        hive_df = pd.read_sql(hql, doris_con, dtype={'tx_id': 'str'})
        hive_df = hive_df.merge(sample_df[['tx_id', 'file_name', 'app_order_id']], on='tx_id')
        result_df = []
        for i, row in hive_df.iterrows():
            rs = {'app_order_id': row['app_order_id']}
            json_obj = json.loads(row['result'])
            rs.update(json_obj)
            result_df.append(rs)
        result_df = pd.DataFrame(result_df)
        if rs_df is None:
            rs_df = result_df
        else:
            rs_df = pd.concat([rs_df, result_df], axis=0)
    return rs_df


def get_doris_module_infos(module_name, merchant_id, dts, sample_df):
    rs_df = None
    for dt in tqdm(dts):
        start_date = dt
        end_date = datetime.strptime(dt, '%Y-%m-%d')
        end_date += timedelta(days=1)
        end = end_date.strftime('%Y-%m-%d')
        dql = f"""
        select tx_id,result from risk_offline.t_risk_module_record_decision_{merchant_id.lower()}
              where module_name ='{module_name}' and create_time >= '{dt}' and create_time<'{end}'
        """
        doris_df = pd.read_sql(dql, doris_con, dtype={'tx_id': 'str'})
        doris_df = doris_df.merge(sample_df[['tx_id', 'file_name', 'app_order_id']], on='tx_id')
        result_df = []
        for i, row in doris_df.iterrows():
            rs = {'app_order_id': row['app_order_id']}
            json_obj = json.loads(row['result'])
            rs.update(json_obj)
            result_df.append(rs)
        result_df = pd.DataFrame(result_df)
        if rs_df is None:
            rs_df = result_df
        else:
            rs_df = pd.concat([rs_df, result_df], axis=0)
    return rs_df

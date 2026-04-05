import json
import traceback

from toollib.unversal import *


def pase_json_dict(df, json_column, columns):
    def _query_dict(data, key):
        nodes = key.strip("$.").split(".")
        result = data
        for node in nodes:
            if isinstance(result, dict) and node in result:
                result = result[node]
            else:
                return None
        return result

    def _query_json(json_str, keys):
        try:
            obj = json.loads(json_str)
            if isinstance(keys, str) and len(keys) > 0:
                return _query_dict(obj, keys)
            if isinstance(keys, list) and len(keys) > 0:
                result_arr = []
                for query in keys:
                    result_arr.append(_query_dict(obj, query))
                return tuple(result_arr)
        except Exception as e:
            print(f'解析发生医生:{e}')
            return None

    df[columns] = df.apply(lambda x: _query_json(x[json_column], columns), axis=1, result_type='expand')
    return df


def query_score_online(country_abbr, module_name, user, passwd, parse_cols=None, start_date='2024-10-01',
                       dblink='inner'):
    """
    模型上线后，直接查询线上订单的模块的评分字段
    :param country_abbr: 国家简称
    :param module_name: 模块名
    :param user: 用户名
    :param passwd: 密码
    :parse_cols list[str]: 需要提取的字段名称列表
    :start_date : 数据开始的时间
    :return:
    """
    sql_con = get_dbcon(country_abbr, user, passwd, dblink)
    try:
        host, port, merchant_id, db = tidb_conf[dblink][country_abbr]
        sql = f"""
        select tx_id,module_name,result from risk_offline.t_risk_module_record_{merchant_id}_0  where module_name = '{module_name}' and create_time >= '{start_date}'
        union all 
        select tx_id,module_name,result from risk_offline.t_risk_module_record_{merchant_id}_1  where module_name = '{module_name}' and create_time >= '{start_date}'
        """
        module_df = pd.read_sql(sql, sql_con)

        tx_ids = module_df['tx_id'].astype(str).to_list()
        if len(tx_ids) == 1:
            tx_ids = tx_ids + tx_ids

        req_df = pd.read_sql(f"""
        select tx_id,biz_id as app_order_id from {db}.t_risk_req_record where tx_id in {tuple(tx_ids)}
        """, sql_con)

        module_df = module_df.merge(req_df, on='tx_id')

        if parse_cols == None:
            return module_df
        else:
            return pase_json_dict(module_df, 'result', parse_cols)
    except:
        print(f"查询报错{traceback.format_exc()}")
    finally:
        sql_con.dispose()


def query_score_online2(country_abbr, module_name, user, passwd, start_date='2024-10-01', dblink='inner'):
    sql_con = get_dbcon(country_abbr, user, passwd, dblink)
    try:
        host, port, merchant_id, db = tidb_conf[dblink][country_abbr]
        sql = f"""
        select tx_id,module_name,result from risk_offline.t_risk_module_record_{merchant_id}_0  where module_name = '{module_name}' and create_time >= '{start_date}'
        union all 
        select tx_id,module_name,result from risk_offline.t_risk_module_record_{merchant_id}_1  where module_name = '{module_name}' and create_time >= '{start_date}'
        """
        module_df = pd.read_sql(sql, sql_con)

        tx_ids = module_df['tx_id'].astype(str).to_list()
        if len(tx_ids) == 1:
            tx_ids = tx_ids + tx_ids

        req_df = pd.read_sql(f"""
        select tx_id,biz_id as app_order_id from {db}.t_risk_req_record where tx_id in {tuple(tx_ids)}
        """, sql_con)

        module_df = module_df.merge(req_df, on='tx_id')

        return module_df

    except:
        print(f"查询报错{traceback.format_exc()}")
    finally:
        sql_con.dispose()

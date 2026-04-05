import os
import json
import math
import yaml
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from io import BytesIO
from typing import Optional, Dict
from collections import defaultdict
from datetime import datetime, timedelta
import os, json, yaml
# import toollib as tl
from toollib.model_monitor.online_tools_dories import apply_score_doris

import time

start_time = time.time()

class FeishuBase:
    def __init__(self, app_id: str, app_secret: str):
        self.app_id = app_id
        self.app_secret = app_secret
        self.base_url = "https://open.feishu.cn/open-apis"
        self.token = self._get_tenant_access_token()

    def _get_tenant_access_token(self) -> str:
        url = f"{self.base_url}/auth/v3/tenant_access_token/internal"
        payload = {"app_id": self.app_id, "app_secret": self.app_secret}
        response = requests.post(url, json=payload)
        return response.json()["tenant_access_token"]


class FeishuTableBuilder(FeishuBase):
    def build_table_from_template(self, file_token: str, folder_token: str, new_name: str):
        url = f"{self.base_url}/drive/v1/files/{file_token}/copy"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        data = {"type": "sheet", "folder_token": folder_token, "name": new_name}
        response = requests.post(url=url, headers=headers, data=json.dumps(data))
        if response.status_code != 200:
            raise Exception(f"创建表格失败: {response.json()}")
        return response.json()

    def write_df_to_table(self, 
                          file_token: str, 
                          sheet_token: str,
                          left_position: str,
                          right_position: str,
                          df: pd.DataFrame, 
                          ):
        
        url = f"https://open.feishu.cn/open-apis/sheets/v2/spreadsheets/{file_token}/values"
        
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        range_str = f"{sheet_token}!{left_position}:{right_position}"
        data = {
            "valueRange": {
                "range": range_str,
                "values": df.values.tolist()
            }
        }
        
        response = requests.put(
            url=url,
            headers=headers,
            data=json.dumps(data)
        )
        print(response.json())
        if response.status_code != 200:
            raise Exception(f"写入表格失败: {response.json()}")
        return response.json()

def load_env():
    env_path = '.env'
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key.strip()] = value.strip()

class ModelReport:
    @staticmethod
    def cut_bins(x, n_bins=10, method='freq'):
        data = np.array(x, copy=True, dtype=np.float64)
        data = data[~np.isnan(data)]
    
        if len(data) == 0:
            return [-np.inf, np.inf]
    
        if method == 'freq':
            sorted_data = np.sort(data)
            indices = np.linspace(0, len(sorted_data) - 1, n_bins + 1, dtype=int)
            bin_edges = sorted_data[indices]
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < n_bins + 1:
                bin_edges = np.insert(bin_edges, 0, -np.inf)
            else:
                bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            return bin_edges.tolist()
    
        elif method == 'dist':
            min_v, max_v = np.min(data), np.max(data)
            bin_edges = np.linspace(min_v, max_v, n_bins + 1)
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            return bin_edges.tolist()
    
        else:
            raise ValueError("method must be 'freq' or 'dist'")

    @staticmethod
    def make_bin(y_score, bound, num_fillna=-999, cate_fillna=""):
        data = np.asarray(y_score)
        if pd.api.types.is_numeric_dtype(y_score):
            bound = np.asarray(bound)
            data[np.isnan(data)] = num_fillna
            bound_max = np.max(bound)
            data[data > bound_max] = np.max(bound)
            indices = np.digitize(data, bound, right=True)
            return bound[indices]
        else:
            data = pd.Series(data).fillna(cate_fillna)
            keys = bound.keys()
            data = data.map(lambda x: x if x in keys else cate_fillna)
            return np.asarray(pd.Series(data).map(bound).fillna(0))

    @staticmethod
    def calc_psi(x_true, x_pred, num_bins=10, method='freq', lamb=0.001):
        bound = ModelReport.cut_bins(x_true, n_bins=num_bins, method=method)
        x_base_bins = np.array(ModelReport.make_bin(x_true, bound))
        x_pred_bins = np.array(ModelReport.make_bin(x_pred, bound))
        n_base, n_pred = len(x_true), len(x_pred)
        b_bins = np.unique(np.concatenate((x_base_bins, x_pred_bins)))
        psi = sum([
            ((x_base_bins == b).sum() / n_base - (x_pred_bins == b).sum() / n_pred)
            * math.log(((x_base_bins == b).sum() / n_base + lamb) / ((x_pred_bins == b).sum() / n_pred + lamb))
            for b in b_bins
        ])
        return psi

    @staticmethod
    def parse_result_column(df):
        if isinstance(df['result'].iloc[0], str):
            parsed = df['result'].apply(json.loads)
        else:
            parsed = df['result']
        return pd.DataFrame(parsed.tolist())

    @staticmethod
    def compute_psi(df_hist, df_now):
        df_hist_parsed = ModelReport.parse_result_column(df_hist)
        df_now_parsed = ModelReport.parse_result_column(df_now)
        features = list(set(df_hist_parsed.columns) & set(df_now_parsed.columns))
        psi_list = []
        for feat in features:
            psi_val = ModelReport.calc_psi(df_hist_parsed[feat], df_now_parsed[feat])
            psi_list.append((feat, psi_val))
        return pd.DataFrame(psi_list, columns=['feature', 'psi'])

def send_to_feishu_group(webhook_url: str, msg: str):
    headers = {"Content-Type": "application/json"}
    data = {
        "msg_type": "text",
        "content": {
            "text": msg
        }
    }
    response = requests.post(webhook_url, headers=headers, json=data)
    if response.status_code != 200:
        print(f"Webhook 发送失败: {response.text}")
    else:
        print("已通过 Webhook 发送到飞书群。")

def run_psi_monitoring_from_template(
    yaml_path: str,
    template_file_token: str,
    folder_token: str,
    country_sheet_name_map: Dict[str, str],
    sheet_token_map: Dict[str, str],  
    start_cell: str = "A2"
):
    load_env()
    APP_ID = os.getenv('APP_ID')
    APP_SECRET = os.getenv('APP_SECRET')
    USER = os.getenv('DORIES_USER')
    PASSWD = os.getenv('DORIES_PASSWD')
    webhook_url = os.getenv('FEISHU_WEBHOOK')

    today = datetime.today().date()
    t1 = today - timedelta(days=1)
    t4 = today - timedelta(days=4)
    today_str = today.strftime("%Y-%m-%d")

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    country_sheets = defaultdict(list)

    # 复制模板，生成新文件
    builder = FeishuTableBuilder(APP_ID, APP_SECRET)
    new_name = f"PSI日报_{today_str}"
    print(f"正在创建新表格：{new_name}...")
    copy_result = builder.build_table_from_template(
        file_token=template_file_token,
        folder_token=folder_token,
        new_name=new_name
    )
    new_file_token = copy_result["data"]["file"]["token"]
    print(f"新表格创建成功，file_token: {new_file_token}")
         
    alerts = []  # 用于收集超阈值的警告信息
    PSI_THRESHOLD = 0.1

    model_type_to_status = {
    '新客': 0,
    '半新': 2,
    '老客': 1
    }
    # Step 2: 收集每个国家 PSI 数据
    for model in config['models']:
        sys_name = model['sys_name']
        module_name = model['model_name']
        score_field = model['score_field']
        model_type = model.get('model_type', '')

        user_status = model_type_to_status.get(model_type)
        
        df_hist = apply_score_doris(
            sys_name, USER, PASSWD, module_name, score_field,
            new_old_user_status=[user_status],
            start_date=str(t4), end_date=str(t1), dblink='inner'
        )
        df_now = apply_score_doris(
            sys_name, USER, PASSWD, module_name, score_field,
            new_old_user_status=[user_status],
            start_date=str(t1), end_date=str(today), dblink='inner'
        )
        psi_df = ModelReport.compute_psi(df_hist, df_now)
        psi_df['model'] = f"{module_name}_{score_field}"
        psi_df['model_type'] = model_type
        psi_df['hist_sample_cnt'] = len(df_hist)
        psi_df['yest_sample_cnt'] = len(df_now)
        psi_df['sys_name'] = sys_name

        # 触发告警判断：仅看 _score特征
        for _, row in psi_df.iterrows():
            feature_name = row['feature']
            psi_val = row['psi']
            if feature_name.endswith('_score') and psi_val > PSI_THRESHOLD:
                alerts.append(
                    f"🚨 模型 [{sys_name} - {module_name} - {model_type}] 的特征 `{feature_name}` PSI={psi_val:.3f} 超过阈值({PSI_THRESHOLD})"
                )
    
        country_sheets[sys_name].append(psi_df)


    # Step 3: 写入新表格中每个 sheet
    for sys_name, dfs in country_sheets.items():
        sheet_name = country_sheet_name_map.get(sys_name)
        sheet_token = sheet_token_map.get(sys_name)

        if not sheet_name or not sheet_token:
            print(f"[警告] {sys_name} 缺少 sheet 映射或 token，跳过")
            continue

        df_to_write = pd.concat(dfs)[[
            'sys_name', 'model', 'model_type', 'feature',
            'psi', 'hist_sample_cnt', 'yest_sample_cnt'
        ]]
        
        # 自动计算写入区域
        left_position = start_cell
        start_col_letter = ''.join(filter(str.isalpha, start_cell))
        start_row = int(''.join(filter(str.isdigit, start_cell)))
        num_cols = df_to_write.shape[1]
        num_rows = df_to_write.shape[0]

        def col_to_letter(n):
            result = ''
            while n > 0:
                n, r = divmod(n - 1, 26)
                result = chr(65 + r) + result
            return result

        start_col_idx = ord(start_col_letter.upper()) - ord('A') + 1
        end_col_letter = col_to_letter(start_col_idx + num_cols - 1)
        end_row = 5000
        right_position = f"{end_col_letter}{end_row}"

        # 写入
        builder.write_df_to_table(
            file_token=new_file_token,
            sheet_token=sheet_token,
            left_position=left_position,
            right_position=right_position,
            df=df_to_write  
        )

        print(f"{sys_name} 数据已写入 sheet `{sheet_name}`")
        
    # Step 4: 群通知
    print(f"所有数据已写入新表格：{new_name}") 
    if webhook_url:
        msg = f"📊PSI日报完成：{new_name}\n📎 链接：https://lf.feishu.cn/sheets/{new_file_token}"
        if alerts:
            alert_msg = "\n\n⚠️ 以下特征 PSI 超阈值：\n" + "\n".join(alerts)
        else:
            alert_msg = "\n\n✅ 所有模型特征 PSI 正常，无异常偏移。"
    
        msg += alert_msg
        send_to_feishu_group(webhook_url, msg)

current_dir = Path(__file__).resolve().parent 
run_psi_monitoring_from_template(
    yaml_path = current_dir / "monitor_config.yaml",
    template_file_token="Gk3psuGlVhiiwEtwkqYlISUQgTf",
    folder_token="GgAJfVlaOl93SUdIFg0loOwwgZg",
    country_sheet_name_map={
        "af": "af",
        "ath": "ath",
        "athl": "athl",
    },
    sheet_token_map={ 
        "af": "52b826",
        "ath": "gyeL1Z",
        "athl": "jX2j3U",
    },
    start_cell="A2"
)

end_time = time.time()
duration = end_time - start_time

print(f"\n✅ 脚本运行时间：{duration:.2f} 秒")
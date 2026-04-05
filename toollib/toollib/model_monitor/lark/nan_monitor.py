from feishu_apis import FeishuTableBuilder, FeishuXlsxExporter
from toollib.model_monitor.online_tools_dories import get_score_dories
from datetime import datetime, timedelta
import toollib as tl
import time
import requests
import os
import pandas as pd
import argparse
import yaml
def load_env():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, '.env')
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key.strip()] = value.strip()

load_env()
APP_ID = os.getenv('APP_ID')
APP_SECRET = os.getenv('APP_SECRET')
USER=os.getenv('DORIES_USER')
PASSWD=os.getenv('DORIES_PASSWD')

def main():
    parser = argparse.ArgumentParser(description='模型异常值监控报告生成工具')
    parser.add_argument('--conf_path', required=True, help='模型监控配置文件的路径')
    parser.add_argument('--upload_folder_token', required=True, help='上传文件夹的 token')
    parser.add_argument('--template_file_token', required=True, help='模板表格的token')
    parser.add_argument('--template_sheet_token', required=True, help='模板表格sheet页的token')
    parser.add_argument('--dblink', default='outer', help='数据库链接')
    parser.add_argument('--webhook_url', required=True, help='Webhook URL')
    args = parser.parse_args()
    
    current_date = datetime.now().strftime('%Y%m%d')
    exporter = FeishuXlsxExporter(APP_ID, APP_SECRET)
    table_builder = FeishuTableBuilder(APP_ID, APP_SECRET)
    with open(args.conf_path, 'r') as f:
        config = yaml.safe_load(f)
    conf_list = config['models']
    model_reports = []
    
    for model_info in conf_list:
        sys_name = model_info['sys_name']
        module_name = model_info['model_name']
        print(f'get {module_name} report')
        score_field = model_info['score_field']
        model_type = model_info['model_type']
        if model_type == '新客':
            new_old_user_status = [0]
        else:
            new_old_user_status = [1, 2]
        label = model_info['label']
        
        if args.dblink == 'outer':
            dblink = 'outer'
        else:
            dblink = 'inner' if sys_name not in ['ath', 'af'] else 'outer'
        local_date = tl.country_now(sys_name)[0]
        start_date = (local_date - timedelta(days=1)).strftime('%Y-%m-%d')
        end_date = local_date.strftime('%Y-%m-%d')
        data = get_score_dories(sys_name,
                            USER, 
                            PASSWD,
                            module_name,
                            score_field,
                            new_old_user_status,
                            start_date=start_date,
                            end_date=end_date,
                            dblink=dblink)
        if len(data) == 0:
            num_nan = 0
            nan_rate = 0
        else:
            data[score_field] = pd.to_numeric(data[score_field])
            num_nan = data[data[score_field] < 0].shape[0]
            nan_rate = num_nan / len(data)
        report_df = pd.DataFrame({
            'model_name': [module_name],
            'date': [start_date],
            'total_cnt': [len(data)],
            'nan_cnt': [num_nan],
            'nan_rate': [nan_rate]
        })
        model_reports.append(report_df)
        print(report_df)
    report_for_all_models = pd.concat(model_reports)
    report_name = f'model_nan_report_{current_date}.xlsx'
    res = table_builder.build_table_from_template(file_token=args.template_file_token, 
                                                  folder_token=args.upload_folder_token, 
                                                  new_name=report_name)
    report_file_token = res['data']['file']['token']
    table_builder.write_df_to_table(file_token=report_file_token, 
                                    sheet_token=args.template_sheet_token,
                                    left_position='A2', 
                                    right_position='E100',
                                    df=report_for_all_models)
    file_url = f"https://nzd66m35vu.larksuite.com/sheets/{report_file_token}"

    current_date = datetime.now().strftime('%Y-%m-%d')

    elements = [
        {
            "tag": "div",
            "text": {
                "content": f"**报告生成时间**: {current_date}",
                "tag": "lark_md"
            }
        }
    ]
    
    elements.append({
        "tag": "action",
        "actions": [
            {
                "tag": "button",
                "text": {
                    "content": "查看完整报告",
                    "tag": "plain_text"
                },
                "url": file_url,
                "type": "default"
            }
        ]
    })

    message = {
        "msg_type": "interactive",
        "card": {
            "config": {
                "wide_screen_mode": True
            },
            "header": {
                "title": {
                    "content": f"⚠ 模型异常值监控报告",
                    "tag": "plain_text"
                },
                "template": "blue"
            },
            "elements": elements
        }
    }

    response = requests.post(args.webhook_url, json=message)
    print(response.json())

if __name__ == '__main__':
    main()

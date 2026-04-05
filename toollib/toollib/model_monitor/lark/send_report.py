from feishu_apis import FeishuTableBuilder, FeishuXlsxExporter
from model_report import get_online_data, get_date_ranges, calc_sample_period, write_df_to_template, ModelReport
from datetime import datetime
import time
import requests
import os
import pandas as pd
import argparse
import yaml
from datetime import timedelta

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


monitor_sample_config = {
    'week': {
        'n_period': 4,
        'interval_days': 7,
        'sample_period': 50,
        'right_position': 'BX1500',
        'sheet_id': {
            'athl': 'jg4cNe',
            'athl_ios': 'JcCxGG',
            'ath': '0xbjdf',
            'af': 'n3EJxs',
        }
    },
    'month': {
        'n_period': 3,
        'interval_days': 30,
        'sample_period': 120,
        'right_position': 'BD1500',
        'sheet_id': {
            'athl': 'yKpYS9',
            'athl_ios': 'OsMlVt',
            'ath': '1opYiu',
            'af': 'u8v8Sv',
        }
    }
}

def main():
    parser = argparse.ArgumentParser(description='模型监控报告生成工具')
    parser.add_argument('--monitor_sample', required=True, default='week', help='监控样本类型')
    parser.add_argument('--conf_path', required=True, help='模型监控配置文件的路径')
    parser.add_argument('--upload_folder_token', required=True, help='上传文件夹的 token')
    parser.add_argument('--template_file_token', required=True, help='模板表格的token')
    parser.add_argument('--dblink', default='outer', help='数据库链接')
    parser.add_argument('--webhook_url', required=True, help='Webhook URL')
    args = parser.parse_args()
    
    
    exporter = FeishuXlsxExporter(APP_ID, APP_SECRET)
    # table_builder = FeishuTableBuilder(APP_ID, APP_SECRET)
    model_report_builder = ModelReport()
    with open(args.conf_path, 'r') as f:
        config = yaml.safe_load(f)
    conf_list = config['models']
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, 'reports')
    os.makedirs(save_dir, exist_ok=True)
    
    current_date = datetime.now().strftime('%Y%m%d')
    report_name = f'model_report_{current_date}_{args.monitor_sample}.xlsx'
    
    model_reports = {}
    for sheet_name in monitor_sample_config[args.monitor_sample]['sheet_id']:
        model_reports[sheet_name] = []
    for model_info in conf_list:
        sys_name = model_info['sys_name']
        module_name = model_info['model_name']
        print(f'get {module_name} report')
        score_field = model_info['score_field']
        model_type = model_info['model_type']
        app_type = model_info.get('app_type', 'android')
        if model_type == '新客':
            new_old_user_status = [0]
        elif model_type == '老客':
            new_old_user_status = [1]
        elif model_type == '半新':
            new_old_user_status = [2]
        label = model_info['label']
        
        if args.dblink == 'outer':
            dblink = 'outer'
        else:
            dblink = 'inner'
        performance_days = int(label.replace('def_pd', ''))
        start_date, end_date = calc_sample_period(performance_days, 
                                                  monitor_sample_config[args.monitor_sample]['sample_period'])
        sample_periods = get_date_ranges(end_date, 
                                         monitor_sample_config[args.monitor_sample]['n_period'], 
                                         monitor_sample_config[args.monitor_sample]['interval_days'])
        dfs = []
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        
        data = get_online_data(sys_name,
                                USER,
                                PASSWD, 
                                module_name, 
                                score_field, 
                                new_old_user_status=new_old_user_status,
                                app_type=app_type,
                                start_date=start_date, 
                                end_date=end_date, 
                                dblink=dblink)
        for date_period in sample_periods:
            df = data[data['repayment_date'] >= date_period[0]]
            df = df[df['repayment_date'] < date_period[1]]
            dfs.append(df)
        if len(dfs) == 0:
            print(f'{module_name} has no data')
            continue
        if len(dfs[0]) <= 50:
            print(f'{module_name} data is too less')
            continue
        for df in dfs:
            df[score_field] = pd.to_numeric(df[score_field])
            df = df[df[score_field]>=0]
        bins = 5
        if len(dfs[0]) >= 1000:
            bins = 10
        print(dfs[0])
        cut_bins = model_report_builder.get_cut_bins(dfs[0], score_field, bins=bins)
        data[score_field] = pd.to_numeric(data[score_field])
        model_report = model_report_builder.get_concat_report(sys_name,
                                                              data, 
                                                              sample_periods,
                                                              model_type,
                                                              cut_bins,
                                                              score_field,
                                                              label)
        if sys_name == 'athl' and app_type == 'ios':
            sys_name = 'athl_ios'
        model_reports[sys_name].append(model_report)
    table_builder = FeishuTableBuilder(APP_ID, APP_SECRET)
    res = table_builder.build_table_from_template(file_token=args.template_file_token, folder_token=args.upload_folder_token, new_name=report_name)
    report_file_token = res['data']['file']['token']
    for sys_name in model_reports:
        print(sys_name, len(model_reports[sys_name]))
        if len(model_reports[sys_name]) == 0:
            print(f'{sys_name} has no model report')
            continue
        report_for_all_models = pd.concat(model_reports[sys_name])
        table_builder.write_df_to_table(file_token=report_file_token, 
                                        sheet_token=monitor_sample_config[args.monitor_sample]['sheet_id'][sys_name],
                                        left_position='B4',
                                        right_position=monitor_sample_config[args.monitor_sample]['right_position'],
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
                    "content": f"📊 模型监控报告({args.monitor_sample})",
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

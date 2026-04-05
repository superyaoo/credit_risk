import pandas as pd
import numpy as np
import json
import yaml
from datetime import datetime, timedelta
import logging
import sys
import os
import requests  # 添加这个导入用于发送消息
#sys.path.append('/home/zengjunyao/notebook')
import toollib as tl
from typing import List, Dict, Union

sys.path.append('/home/zengjunyao/notebook/toollib/toollib/model_monitor/lark/')
from model_report import get_online_data, get_date_ranges, calc_sample_period, write_df_to_template, ModelReport
from feishu_apis import (
    FeishuTableBuilder, 
    FeishuFileUploader,
    FeishuBase  # 添加这个基类的导入
)
# ============= 配置区域 =============
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# # 飞书报警webhook配置
#FEISHU_WEBHOOK1 = "https://open.larksuite.com/open-apis/bot/v2/hook/39117e89-0053-46e9-ba22-1e59191333ae"  # 替换为实际的webhook地址(报警)
                                            
# APP_ID = "your_app_id"  # 替换为实际的应用ID
# APP_SECRET = "your_app_secret"  # 替换为实际的应用密钥
def load_env():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, 'toollib', 'toollib', '.env')
    #env_path = os.path.join(script_dir, '.env')
    try:
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"错误：找不到环境变量文件 {env_path}")
    except Exception as e:
        print(f"加载环境变量时出错：{e}")

load_env()
APP_ID = os.getenv('APP_ID')
APP_SECRET = os.getenv('APP_SECRET')
USER=os.getenv('DORIES_USER')
PASSWD=os.getenv('DORIES_PASSWD')
# # lark文档配置
FEISHU_WEBHOOK = "https://open.larksuite.com/open-apis/bot/v2/hook/39117e89-0053-46e9-ba22-1e59191333ae"  # 飞书群机器人webhook
#TEMPLATE_TOKEN = "I3mWsez7chvMiMtl24ElTCZKgIe"  # 飞书文档模板的token副本
#FOLDER_TOKEN = "MLeWfbZ26lht84d9Bpju85LQsZe"     #模型监控文件夹

TEMPLATE_TOKEN = "S3T7s01sLhtEuktQ7twlpDb0g9e" #文件模板token
FOLDER_TOKEN = "P4Clfl5zDlqwHcduKsYlDGEDg1d"  # 新文件夹token

# 报告输出路径
REPORT_OUTPUT_DIR = "model_psi_reports"
if not os.path.exists(REPORT_OUTPUT_DIR):
    os.makedirs(REPORT_OUTPUT_DIR)
    
class FeishuMessenger(FeishuBase):
    """扩展FeishuBase类来处理消息发送"""
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        
    def send_message(self, message: str):
        """发送文本消息到飞书群"""
        try:
            headers = {'Content-Type': 'application/json'}
            payload = {
                "msg_type": "text",
                "content": {
                    "text": message
                }
            }
            response = requests.post(
                self.webhook_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"发送飞书消息失败: {str(e)}")
            raise

def load_config(config_path: str = 'monitor_config_new.yaml') -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def calculate_psi(expected, actual, buckettype='bins', buckets=10):
    """计算 PSI 值"""
    if buckettype == 'bins':
        try:
            bins = np.histogram_bin_edges(np.concatenate([expected, actual]), bins=buckets)
            expected_perc = np.histogram(expected, bins=bins, density=True)[0]
            actual_perc = np.histogram(actual, bins=bins, density=True)[0]
        except Exception as e:
            logger.error(f"PSI分箱出错: {e}")
            return np.nan
    else:
        expected_perc = expected.value_counts(normalize=True).sort_index()
        actual_perc = actual.value_counts(normalize=True).sort_index()
    
    expected_perc = np.where(expected_perc == 0, 1e-6, expected_perc)
    actual_perc = np.where(actual_perc == 0, 1e-6, actual_perc)
    psi_value = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
    return psi_value

def parse_result_column(df):
    """将result列的json字符串解析为字典"""
    return df['result'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)


def get_customer_type_cn(new_old_user_status: List[int]) -> str:
    """将客群类型转换为中文"""
    if new_old_user_status == [0]:
        return "新客"
    elif new_old_user_status == [1]:
        return "半新"
    elif new_old_user_status == [2]:
        return "纯老"
    elif new_old_user_status == [1, 2]:
        return "半新老客"
    else:
        return f"未知客群({new_old_user_status})"


def generate_feishu_report(df: pd.DataFrame, today: datetime.date) -> str:
    """生成飞书文档报告"""
    try:
        # 初始化飞书表格构建器
        table_builder = FeishuTableBuilder(APP_ID, APP_SECRET)
        
        # 从模板创建新表格
        new_file_name = f"模型PSI监控日报_{today}"
        result = table_builder.build_table_from_template(
            file_token=TEMPLATE_TOKEN,
            folder_token=FOLDER_TOKEN,
            new_name=new_file_name
        )
        
        if result.get("code") == 0:
            new_file_token = result["data"]["file_token"]
            sheet_token = "Sheet1"  # 根据实际模板sheet名称调整
            
            # 写入数据
            # 假设模板中A1:J1是表头，从A2开始写入数据
            table_builder.write_df_to_table(
                file_token=new_file_token,
                sheet_token=sheet_token,
                left_position="A2",  # 数据起始位置
                right_position=f"J{len(df)+2}",  # 数据结束位置
                df=df
            )
            
            # 生成文档URL
            doc_url = f"https://bytedance.feishu.cn/sheets/{new_file_token}"
            return doc_url
            
    except Exception as e:
        logger.error(f"生成飞书报告失败: {str(e)}")
        raise

def upload_excel_to_feishu(file_path: str) -> str:
    """上传Excel文件到飞书云文档"""
    try:
        uploader = FeishuFileUploader(APP_ID, APP_SECRET)
        result = uploader.upload_file(
            file_path=file_path,
            parent_node=FOLDER_TOKEN,
            file_name=os.path.basename(file_path)
        )
        
        if result.get("code") == 0:
            file_token = result["data"]["file_token"]
            return f"https://bytedance.feishu.cn/file/{file_token}"
        else:
            raise Exception(f"上传文件失败: {result}")
            
    except Exception as e:
        logger.error(f"上传Excel文件失败: {str(e)}")
        raise



def monitor_single_model(sys_name: str, user: str, passwd: str, 
                        module_name: str, score_field: str, 
                        customer_type: str,
                        threshold: float = 0.1) -> pd.DataFrame:
    """监控单个模型并返回结果DataFrame"""
    try:
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        three_days_ago = today - timedelta(days=3)

        # 根据客户类型设置new_old_user_status
        # new_old_user_status = [0] if customer_type == 'new_customer' else [1, 2]
        # 根据客户类型设置new_old_user_status
        if customer_type == 'new_customer':
            new_old_user_status = [0]  # 新客
        elif customer_type == 'semi_new_customer':
            new_old_user_status = [1]  # 半新
        elif customer_type == 'old_customer':
            new_old_user_status = [2]  # 老客
        else:
            new_old_user_status = [1, 2]  # 默认半新+老客
            
        customer_type_cn = get_customer_type_cn(new_old_user_status)
        logger.info(f"开始获取模型 {module_name} 的数据，客群类型: {customer_type}")

        # 获取数据
        data_today = tl.apply_score_doris(
            sys_name=sys_name,
            user=user,
            passwd=passwd,
            module_name=module_name,
            score_field=score_field,
            new_old_user_status=new_old_user_status,
            start_date=yesterday,
            end_date=today
        )
        
        data_three_days = tl.apply_score_doris(
            sys_name=sys_name,
            user=user,
            passwd=passwd,
            module_name=module_name,
            score_field=score_field,
            new_old_user_status=new_old_user_status,
            start_date=three_days_ago,
            end_date=yesterday
        )

        logger.info(f"模型 {module_name} 获取数据完成：今日数据 {len(data_today)} 条，前三天数据 {len(data_three_days)} 条")

        # 解析result列
        today_results = parse_result_column(data_today)
        three_days_results = parse_result_column(data_three_days)

        if len(today_results) == 0:
            logger.error(f"模型 {module_name} 当天数据为空")
            return pd.DataFrame()

        feature_names = list(today_results.iloc[0].keys())
        psi_detail = []
        
        for feat in feature_names:
            today_feat = today_results.apply(lambda x: x.get(feat, np.nan))
            three_days_feat = three_days_results.apply(lambda x: x.get(feat, np.nan))
            
            today_feat = today_feat.dropna()
            three_days_feat = three_days_feat.dropna()
            
            if len(today_feat) == 0 or len(three_days_feat) == 0:
                psi = np.nan
            else:
                psi = calculate_psi(three_days_feat.values, today_feat.values)
            
            # 修改这里的字典键名，确保与后面的columns_order一致
            psi_detail.append({
                "country": sys_name,                          # 国家
                "model": module_name,                    # 模型名称
                "score_field": score_field,                   # 评分字段
                "user_type": customer_type_cn,            # 客群类型（中文）
                "feature": feat,                         # 特征名
                "psi": psi,                            # PSI值
                "yesterday_sample_count": len(today_feat),    # 昨日样本数
                "three_days_sample_count": len(three_days_feat),  # 前三天样本数
                "yesterday_mean": today_feat.mean() if len(today_feat) > 0 else np.nan,  # 昨日均值
                "three_days_mean": three_days_feat.mean() if len(three_days_feat) > 0 else np.nan,  # 前三天均值
            })

        result_df = pd.DataFrame(psi_detail)
        return result_df
      
        
    #     # 检查并记录超过阈值的特征
    #     alert_feats = result_df[result_df['PSI'] > threshold]['特征名'].tolist()
    #     if alert_feats:
    #         alert_message = (
    #             f"⚠️ PSI监控告警 ⚠️\n"
    #             f"国家: {sys_name}\n"
    #             f"模型: {module_name}\n"
    #             f"评分字段: {score_field}\n"
    #             f"客群类型: {'新客' if customer_type == 'new_customer' else '老客'}\n"
    #             f"以下特征PSI超过阈值({threshold}):\n"
    #             f"{', '.join(alert_feats)}"
    #         )
    #         send_alert_to_group(alert_message)
    #         logger.warning(alert_message)


    except Exception as e:
        error_message = f"监控模型 {module_name} (客群: {customer_type}) 时发生错误: {str(e)}"
        # logger.error(error_message)
        # send_alert_to_group(f"❌ PSI监控错误 ❌\n{error_message}")
        return pd.DataFrame()






def monitor_all_models():
    """监控所有配置的模型并生成统一报告"""
    try:
        # 初始化飞书相关的类
        table_builder = FeishuTableBuilder(APP_ID, APP_SECRET)
        file_uploader = FeishuFileUploader(APP_ID, APP_SECRET)
        messenger = FeishuMessenger(FEISHU_WEBHOOK)
        #加载配置
        config = load_config()
        common_config = config['common']
        countries_config = config['countries']

        logger.info("开始执行所有模型的监控")
        
        all_results = []
        for country, country_config in countries_config.items():
            logger.info(f"开始监控国家: {country}")
            
            for customer_type, models in country_config.items():
                for model in models:
                    module_name = model['module_name']
                    
                    # 处理单个或多个score_field
                    score_fields = model.get('score_fields', [model.get('score_field')])
                    if not isinstance(score_fields, list):
                        score_fields = [score_fields]
                    
                    for score_field in score_fields:
                        result_df = monitor_single_model(
                            sys_name=country,
                            user=common_config['user'],
                            passwd=common_config['passwd'],
                            module_name=module_name,
                            score_field=score_field,
                            customer_type=customer_type,
                            threshold=common_config['threshold']
                        )
                        
                        if not result_df.empty:
                            all_results.append(result_df)
        
        if all_results:
            # 合并所有结果
            final_df = pd.concat(all_results, ignore_index=True)
            logger.info(f"实际的列名: {final_df.columns.tolist()}")
            
            # 设置列的显示顺序
            columns_order = [
                "country",                  # 国家
                "model",               # 模型名称
                "score_field",              # 评分字段
                "user_type",            # 客群类型
                "feature",             # 特征名
                "psi",                # PSI值
                "yesterday_sample_count",    # 昨日样本数
                "three_days_sample_count",   # 前三天样本数
                "yesterday_mean",            # 昨日均值
                "three_days_mean"           # 前三天均值
            ]
            
            final_df = final_df[columns_order]
            
           # 生成Excel报告
            today = datetime.now().date()
            report_file = os.path.join(REPORT_OUTPUT_DIR, f"all_models_psi_report_{today}.xlsx")
            
            # 创建Excel报告
            with pd.ExcelWriter(report_file, engine='openpyxl') as writer:
                final_df.to_excel(writer, sheet_name='Summary', index=False)
                for country in countries_config.keys():
                    country_df = final_df[final_df['country'] == country]
                    if not country_df.empty:
                        country_df.to_excel(writer, sheet_name=country, index=False)
            
            # 上传Excel到飞书
            excel_result = file_uploader.upload_file(
                file_path=report_file,
                parent_node=FOLDER_TOKEN,
                file_name=os.path.basename(report_file)
            )
            logger.info(f"Excel上传结果: {excel_result}")  # 添加日志
            if not excel_result.get('code') == 0:  # 添加错误检查
                raise Exception(f"上传Excel文件失败: {excel_result}")
            excel_url = f"https://bytedance.feishu.cn/file/{excel_result['data']['file_token']}"
                   
            # 创建飞书在线表格
            table_result = table_builder.build_table_from_template(
                file_token=TEMPLATE_TOKEN,
                folder_token=FOLDER_TOKEN,
                new_name=f"模型PSI监控日报_{today}"
            )
            logger.info(f"创建表格结果: {table_result}")  # 添加日志
            if not table_result.get('code') == 0:  # 添加错误检查
                raise Exception(f"创建飞书表格失败: {table_result}")
            if table_result.get("code") == 0:
                new_file_token = table_result["data"]["file_token"]
                # 写入数据
                table_builder.write_df_to_table(
                    file_token=new_file_token,
                    sheet_token="Sheet1",  # 根据实际模板sheet名称调整
                    left_position="A2",
                    right_position=f"J{len(final_df)+2}",
                    df=final_df
                )
                doc_url = f"https://bytedance.feishu.cn/sheets/{new_file_token}"
                
                # 检查模型分PSI告警
                model_score_psi = final_df[final_df['feature'] == final_df['score_field']]
                alert_models = model_score_psi[model_score_psi['psi'] > 0.1]
                
                if not alert_models.empty:
                    # 生成告警消息
                    alert_message = "⚠️ 模型分PSI监控告警 ⚠️\n\n"
                    
                    for country in alert_models['country'].unique():
                        country_alerts = alert_models[alert_models['country'] == country]
                        if not country_alerts.empty:
                            alert_message += f"国家: {country}\n"
                            
                            for _, row in country_alerts.iterrows():
                                alert_message += (
                                    f"- 模型: {row['model']}\n"
                                    f"  评分字段: {row['score_field']}\n"
                                    f"  客群: {row['user_type']}\n"
                                    f"  PSI值: {row['psi']:.4f}\n"
                                    f"  样本数(昨日/前三天): {row['yesterday_sample_count']}/{row['three_days_sample_count']}\n"
                                    f"  均值(昨日/前三天): {row['yesterday_mean']:.2f}/{row['three_days_mean']:.2f}\n\n"
                                )
                    
                    alert_message += (
                        f"\n📊 详细报告:\n"
                        f"报告日期: {today}\n"
                        f"监控模型总数: {final_df['model'].nunique()}\n"
                        f"告警模型数: {len(alert_models)}\n"
                        f"📑 在线表格: {doc_url}\n"
                        f"📥 Excel下载: {excel_url}\n"
                    )
                    
                    # 发送告警消息
                    messenger.send_message(alert_message)
                else:
                    # 发送正常报告
                    normal_message = (
                        f"📊 模型PSI监控日报 ({today})\n"
                        f"监控模型数: {final_df['model'].nunique()}\n"
                        f"所有模型分PSI正常\n"
                        f"📑 在线表格: {doc_url}\n"
                        f"📥 Excel下载: {excel_url}\n"
                    )
                    messenger.send_message(normal_message)
                
                logger.info("报告生成完成并已发送到飞书群")
            
    except Exception as e:
        logger.error(f"监控过程发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        monitor_all_models()
        logger.info("所有模型监控执行完成")
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
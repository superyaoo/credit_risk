import pandas as pd
import numpy as np
import json
import yaml
from datetime import datetime, timedelta
import logging
import sys
import os
import requests  
import schedule  
import time
import toollib as tl
from typing import List, Dict, Union
from tl.model_report import get_online_data, get_date_ranges, calc_sample_period, write_df_to_template, ModelReport
from tl.model_monitor.lark.feishu_apis import (
    FeishuTableBuilder, 
    FeishuFileUploader,
    FeishuBase  
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
        logger.info(f"成功加载环境变量文件: {env_path}")
    except FileNotFoundError:
        logger.warning(f"找不到环境变量文件 {env_path}，将使用系统环境变量")
    except Exception as e:
        logger.error(f"加载环境变量时出错：{e}")

load_env()
APP_ID = os.getenv('APP_ID')
APP_SECRET = os.getenv('APP_SECRET')
USER=os.getenv('DORIES_USER')
PASSWD=os.getenv('DORIES_PASSWD')


# # lark文档配置
FEISHU_WEBHOOK = "https://open.larksuite.com/open-apis/bot/v2/hook/39117e89-0053-46e9-ba22-1e59191333ae"  # 飞书群机器人webhook - 监控群
ALERT_WEBHOOK = "https://open.larksuite.com/open-apis/bot/v2/hook/39117e89-0053-46e9-ba22-1e59191333ae"  # 飞书群机器人webhook - 报警群（请替换为实际的报警群webhook）
#TEMPLATE_TOKEN = "I3mWsez7chvMiMtl24ElTCZKgIe"  # 飞书文档模板的token副本
#FOLDER_TOKEN = "MLeWfbZ26lht84d9Bpju85LQsZe"     #模型监控文件夹

TEMPLATE_TOKEN = "S3T7s01sLhtEuktQ7twlpDb0g9e" #文件模板token
FOLDER_TOKEN = "P4Clfl5zDlqwHcduKsYlDGEDg1d"  # 新文件夹token

# Sheet名称配置（使用sheet名称而不是token）
SHEET_NAMES = {
    "汇总": "0dWKJw",
    "ath": "1jKlKU", 
    "athl": "2NblaB",
    "af": "3JIbDu"
}

# 检查环境变量是否正确加载
logger.info(f"APP_ID: {'已设置' if APP_ID else '未设置'}")
logger.info(f"APP_SECRET: {'已设置' if APP_SECRET else '未设置'}")
logger.info(f"USER: {'已设置' if USER else '未设置'}")
logger.info(f"PASSWD: {'已设置' if PASSWD else '未设置'}")
logger.info(f"FEISHU_WEBHOOK: {'已设置' if FEISHU_WEBHOOK else '未设置'}")

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
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"配置文件 {config_path} 不存在")
        raise
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        raise

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
            new_file_token = result["data"]["file"]["token"]
            sheet_token = "Sheet1"  # 根据实际模板sheet名称调整
            
            # 写入数据
               # 自动计算写入区域
            start_cell = "A2"
            left_position = start_cell
            start_col_letter = ''.join(filter(str.isalpha, start_cell))
            start_row = int(''.join(filter(str.isdigit, start_cell)))
            num_cols = df.shape[1]
            num_rows = df.shape[0]

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
            # 假设模板中A1:J1是表头，从A2开始写入数据
            table_builder.write_df_to_table(
                file_token=new_file_token,
                sheet_token=sheet_token,
                left_position= start_cell,  # 数据起始位置
                right_position=right_position,  # 数据结束位置
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
            file_token = result["data"]["file"]["token"]
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
        monitor_messenger = FeishuMessenger(FEISHU_WEBHOOK)  # 监控群消息发送器
        alert_messenger = FeishuMessenger(ALERT_WEBHOOK)      # 报警群消息发送器
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
            #excel_url = f"https://bytedance.feishu.cn/file/{excel_result['data']['file_token']}"
                   
            # 创建飞书在线表格
            today = datetime.now().date()
            table_result = table_builder.build_table_from_template(
                file_token=TEMPLATE_TOKEN,
                folder_token=FOLDER_TOKEN,
                new_name=f"模型PSI监控日报_{today}"
            )
            logger.info(f"创建表格结果: {table_result}")  # 添加日志
            if not table_result.get('code') == 0:  # 添加错误检查
                raise Exception(f"创建飞书表格失败: {table_result}")
            if table_result.get("code") == 0:
                new_file_token = table_result["data"]["file"]["token"]
                
                # 写入汇总数据到汇总sheet
                
                logger.info(f"开始写入汇总数据，数据行数: {len(final_df)}")
                        # 自动计算写入区域
                start_cell = "A2"
                left_position = start_cell
                start_col_letter = ''.join(filter(str.isalpha, start_cell))
                start_row = int(''.join(filter(str.isdigit, start_cell)))
                num_cols = final_df.shape[1]
                num_rows = final_df.shape[0]
        
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
                # 汇总
                summary_write_result = table_builder.write_df_to_table(
                    file_token=new_file_token,
                    sheet_token= "0dWKJw",
                    left_position=left_position,
                    right_position=right_position,
                    df=final_df
                )
                logger.info(f"写入汇总sheet结果: {summary_write_result}")
                
                SHEET_COUNTRY = {
                    "ath": "1jKlKU", 
                    "athl": "2NblaB",
                    "af": "3JIbDu"
                }
                # 按国家分别写入数据到对应sheet
                for country in countries_config.keys():
                    print(country)
                    country_df = final_df[final_df['country'] == country]
                    sheet_token = SHEET_COUNTRY.get(country)
                    print(sheet_token)
                    if not country_df.empty and country in SHEET_NAMES:
                        logger.info(f"开始写入{country}数据，数据行数: {len(country_df)}")
                                # 自动计算写入区域
                        start_cell = "A2"
                        left_position = start_cell
                        start_col_letter = ''.join(filter(str.isalpha, start_cell))
                        start_row = int(''.join(filter(str.isdigit, start_cell)))
                        num_cols = country_df.shape[1]
                        num_rows = country_df.shape[0]
                
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
        
                        country_write_result = table_builder.write_df_to_table(
                            file_token=new_file_token,
                            sheet_token=sheet_token,
                            left_position=left_position,
                            right_position=right_position,
                            df=country_df
                        )
                        logger.info(f"写入{country} sheet结果: {country_write_result}")
                
                doc_url = f"https://bytedance.feishu.cn/sheets/{new_file_token}"
                
                # 检查模型分PSI告警
                logger.info(f"开始检查PSI告警，总数据行数: {len(final_df)}")
                logger.info(f"特征列唯一值: {final_df['feature'].unique()}")
                logger.info(f"评分字段列唯一值: {final_df['score_field'].unique()}")
                
                # 筛选出模型分（特征名等于评分字段的行）
                model_score_psi = final_df[final_df['feature'] == final_df['score_field']]
                logger.info(f"模型分数据行数: {len(model_score_psi)}")
                
                if len(model_score_psi) > 0:
                    logger.info(f"模型分PSI值范围: {model_score_psi['psi'].min():.4f} - {model_score_psi['psi'].max():.4f}")
                    alert_models = model_score_psi[model_score_psi['psi'] > 0.1]
                    logger.info(f"超过阈值的模型数: {len(alert_models)}")
                    
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
                        today = datetime.now().date()
                        alert_message += (
                            f"\n📊 详细报告:\n"
                            f"报告日期: {today}\n"
                            f"监控模型总数: {final_df['model'].nunique()}\n"
                            f"告警模型数: {len(alert_models)}\n"
                            f"📑 在线表格: {doc_url}\n"
                            #f"📥 Excel下载: {excel_url}\n"
                        )
                        
                        # 发送告警消息到报警群
                        logger.info("发送告警消息到报警群")
                        try:
                            alert_messenger.send_message(alert_message)
                            logger.info("告警消息发送成功")
                        except Exception as e:
                            logger.error(f"发送告警消息失败: {e}")
                    else:
                        # 发送正常报告到监控群
                        today = datetime.now().date()
                        normal_message = (
                            f"📊 模型PSI监控日报 ({today})\n"
                            f"监控模型数: {final_df['model'].nunique()}\n"
                            f"所有模型分PSI正常\n"
                            f"📑 在线表格: {doc_url}\n"
                           #f"📥 Excel下载: {excel_url}\n"
                        )
                        logger.info("发送正常报告到监控群")
                        try:
                            monitor_messenger.send_message(normal_message)
                            logger.info("正常报告发送成功")
                        except Exception as e:
                            logger.error(f"发送正常报告失败: {e}")
                else:
                    # 没有找到模型分数据
                    logger.warning("没有找到模型分数据，可能所有特征都不是评分字段")
                    today = datetime.now().date()
                    no_data_message = (
                        f"📊 模型PSI监控日报 ({today})\n"
                        f"监控模型数: {final_df['model'].nunique()}\n"
                        f"⚠️ 警告: 没有找到模型分数据\n"
                        f"📑 在线表格: {doc_url}\n"
                        #f"📥 Excel下载: {excel_url}\n"
                    )
                    try:
                        monitor_messenger.send_message(no_data_message)
                        logger.info("无数据警告消息发送成功")
                    except Exception as e:
                        logger.error(f"发送无数据警告消息失败: {e}")
                
                logger.info("报告生成完成并已发送到飞书群")
            
    except Exception as e:
        logger.error(f"监控过程发生错误: {str(e)}")
        raise

def test_alert_function():
    """测试报警功能"""
    try:
        logger.info("开始测试报警功能")
        
        # 初始化飞书消息发送器
        messenger = FeishuMessenger(FEISHU_WEBHOOK)
        
        # 发送测试消息
        test_message = "🧪 PSI监控报警功能测试\n\n这是一个测试消息，用于验证报警功能是否正常工作。"
        
        logger.info("发送测试消息到飞书群")
        result = messenger.send_message(test_message)
        logger.info(f"测试消息发送结果: {result}")
        
        return True
    except Exception as e:
        logger.error(f"测试报警功能失败: {e}")
        return False

def run_scheduled_monitor():
    """定时运行监控任务"""
    try:
        logger.info("开始执行定时监控任务")
        monitor_all_models()
        logger.info("定时监控任务执行完成")
    except Exception as e:
        logger.error(f"定时监控任务执行失败: {e}")
        # 发送错误通知到监控群
        try:
            error_messenger = FeishuMessenger(FEISHU_WEBHOOK)
            error_message = f"❌ PSI监控任务执行失败\n\n错误信息: {str(e)}\n\n时间: {datetime.now()}"
            error_messenger.send_message(error_message)
        except Exception as notify_error:
            logger.error(f"发送错误通知失败: {notify_error}")

def start_scheduler():
    """启动定时任务调度器"""
    logger.info("启动定时任务调度器")
    
    # 设置每天上午10:00执行监控任务
    schedule.every().day.at("15:00").do(run_scheduled_monitor)
    
    logger.info("定时任务已设置: 每天15:00执行PSI监控")
    
    # 运行调度器
    while True:
        schedule.run_pending()
        time.sleep(60)  # 每分钟检查一次

if __name__ == "__main__":
    import sys
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            # 运行测试
            success = test_alert_function()
            if success:
                logger.info("报警功能测试成功")
            else:
                logger.error("报警功能测试失败")
        elif sys.argv[1] == "schedule":
            # 启动定时任务
            logger.info("启动定时任务模式")
            start_scheduler()
        else:
            logger.error(f"未知参数: {sys.argv[1]}")
            logger.info("可用参数: test (测试), schedule (定时任务), 无参数 (立即执行)")
    else:
        # 立即运行监控
        try:
            monitor_all_models()
            logger.info("所有模型监控执行完成")
        except Exception as e:
            logger.error(f"程序执行失败: {str(e)}")
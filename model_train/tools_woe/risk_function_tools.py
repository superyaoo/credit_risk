import numpy as np
import pandas as pd
import pytz
import time
import datetime
import re
import hashlib
import requests
import json
import sys
sys.path.append(r"/euler/public/tools")
from datetime_tools import timestamp_to_strftime,get_week_date


def cal_loan_new_cols(df_loan,target_time,timezone_str,order_type=0):
    '''
    用途:1）时间等字段进行格式转换，2）新增到期、结清、逾期等关键指标，例如:t+1,t+3等字段；
    df_loan 为基础数据;
    target_time为观测时间点;
    timezone_str 为时区 例如'Asia/Kolkata' #印度 加尔各答； 'America/Bogota' #哥伦比亚 波哥大
    order_type 为订单状态，状态为0，看所有订单数据，状态为1，仅看展期数据,默认为0
    '''
    #### 订单真实到期时间-部分订单进行了展期,到期时间进行了更改
    df_loan['deadline_time']=pd.to_datetime(df_loan.deadline.apply(lambda x:timestamp_to_strftime(x,timezone_str,str_formate = "%Y-%m-%d")))## 展期后到期时间
    #### 订单初始到期时间
    df_loan['Fdeadline_time']=pd.to_datetime(df_loan.firstDeadline.apply(lambda x:timestamp_to_strftime(x,timezone_str,str_formate = "%Y-%m-%d")))## 正常到期时间
    df_loan['Fdeadline_month']=df_loan.firstDeadline.apply(lambda x:timestamp_to_strftime(x,timezone_str,str_formate = "%Y-%m"))## 正常到期时间
    df_loan['due_week_start_date']=df_loan.Fdeadline_time.apply(lambda x:get_week_date(x)[0])### 获取今年的第几个自然周
    df_loan['due_week']=df_loan.Fdeadline_time.apply(lambda x:str(get_week_date(x)[0])+'('+str(x.isocalendar()[1])+'周'+')')### 获取自然周和周的开始时间拼接
    
    
    ##判断订单是否为展期（展期状态  展期状态存在取消的情况 故用最后到期时间大于正常到期时间） 1为展期
    df_loan['is_extension']=df_loan.apply(lambda x:1 if x['deadline_time']>x['Fdeadline_time'] else 0,axis=1)
    ## 当订单状态为FINISH 时，为结清时间，当订单为dunning时 为逾期时间 ,当订单为在贷时 为订单最后一次操作时间
    df_loan['last_update_time']=pd.to_datetime(df_loan.flowInfo.apply(lambda x:timestamp_to_strftime(int(re.findall(r"date.:.(\d{13})",str(x))[-1]),timezone_str,str_formate = "%Y-%m-%d")))
    #### 观测时间  为了计算逾期天数及到期天数
    df_loan['current_date']=target_time
    #### 当订单出现展期时，将该订单状态更新为FINISH 对应还款金额更改为合同金额，只对整体量级有影响 不影响后续监控指标(后期展期订单单独设置监控表) 展期订单状态直接使用status即可
    df_loan['status_new']=df_loan.apply(lambda x:'FINISH' if x['is_extension']==1 else x['status'],axis=1) ## 更改展期订单状态
    #### 取flowIngo中进行展期的第一次操作时间 【会有部分订单出现打款一段时间后展期成功后 在提示打款失败的情况 故需提取最后一次打款时间，并显示展期时间大于打款时间】
    df_loan['update_extension_time']=pd.to_datetime(df_loan.flowInfo.apply(lambda x:[item_z['date'] for item_z in x if (item_z['note']=='展期成功,重置状态为LOAN_SUCCESS') & (item_z['date']>=[item['date'] for item in x if '打款成功' in item['note']][-1])])\
    .apply(lambda x:timestamp_to_strftime(np.sort(list(x))[0],timezone_str,str_formate = "%Y-%m-%d") if len(list(x))>0 else '2030-01-01')  )
    ##### 初始逻辑【状态为status_new字段】：最大逾期天数 当状态为dunning时  使用观测时间减去初始到期时间  当状态为finish时且不为展期订单 使用最后的结清时间减去初始到期时间  展期默认逾期为0 
    #df_loan['his_overdue_days']=df_loan.apply(lambda x:x['current_date']-x['Fdeadline_time'] if x['status_new']=='DUNNING' else (x['last_update_time']-x['Fdeadline_time'] if (x['status_new']=='FINISH') & (x['is_extension']==0)  else datetime.timedelta(days=0)) ,axis=1).astype('timedelta64[D]')
    ##### 二版逾期逻辑【状态为status_new字段】:历史最大逾期天数 当状态为dunning时，使用观测时间减去初始到期时间  当状态为finish时且状态不为展期，使用最后的结清时间减去初始到期时间 
    ##### 当状态为finish 且为展期订单的，历史最大逾期天数为首次进行展期时间减去首次到期时间的差
    df_loan['his_overdue_days']=df_loan.apply(lambda x:x['current_date']-x['Fdeadline_time'] if x['status_new']=='DUNNING' \
                                                    else (x['last_update_time']-x['Fdeadline_time'] if (x['status_new']=='FINISH') & (x['is_extension']==0) \
                                                          else (x['update_extension_time']-x['Fdeadline_time'] if x['is_extension']==1 else datetime.timedelta(days=0))) ,axis=1).dt.days
    # ##### 定义展期订单最大逾期天数  当状态为dunning时  使用观测时间减去真实到期时间  当状态为finish时 使用最后的结清时间减去真实到期时间 状态为loan_success 默认逾期0
    df_loan['extension_his_overdue_days']=df_loan.apply(lambda x:x['current_date']-x['deadline_time'] if x['status']=='DUNNING' else (x['last_update_time']-x['deadline_time'] if x['status']=='FINISH'  else datetime.timedelta(days=0)) ,axis=1).dt.days
    # #### 到期天数---------
    # ##### 按初始到期时间计算到期天数--展期订单认为正常到期且结清
    df_loan['dq_days']=(df_loan['current_date']-df_loan['Fdeadline_time']).dt.days
    # #### 按最后真实到期时间计算到期天数
    df_loan['extension_dq_days']=(df_loan['current_date']-df_loan['deadline_time']).dt.days
    df_loan=df_loan.drop('flowInfo',axis=1)
    if order_type==0:
        df_loan['totalPaidAmount_new']=df_loan.apply(lambda x:x['amount'] if x['is_extension']==1 else (x['amount'] if (x['is_extension']==0) & (x['totalPaidAmount']>x['amount']) else x['totalPaidAmount']),axis=1) ##更改展期订单还款金
        # ##########################以下指标适用于正常逾期结清订单(展期订单归为结清订单)######################################################
        # #### 统计T+1---15分母 新增到期订单 到期0天字段 应所有订单据为已到期数据故不用新增资源
        df_loan['dq_days_3_0']=df_loan.dq_days.apply(lambda x: 1 if x>-3 else 0)
        df_loan['dq_days_2_0']=df_loan.dq_days.apply(lambda x: 1 if x>-2 else 0)
        df_loan['dq_days_1_0']=df_loan.dq_days.apply(lambda x: 1 if x>-1 else 0)
        df_loan['dq_days_0']=df_loan.dq_days.apply(lambda x: 1 if x>0 else 0)
        df_loan['dq_days_1']=df_loan.dq_days.apply(lambda x: 1 if x>1 else 0)
        df_loan['dq_days_3']=df_loan.dq_days.apply(lambda x: 1 if x>3 else 0)
        df_loan['dq_days_7']=df_loan.dq_days.apply(lambda x: 1 if x>7 else 0)
        df_loan['dq_days_15']=df_loan.dq_days.apply(lambda x: 1 if x>15 else 0)

        # #### 统计T+0---15分子 新增到结清订单
        df_loan['finish_days_3_0']=df_loan.apply(lambda x: 1 if x['dq_days']>-3 and x['status_new']=='FINISH' and x['his_overdue_days']<=-3  else 0,axis=1)
        df_loan['finish_days_2_0']=df_loan.apply(lambda x: 1 if x['dq_days']>-2 and x['status_new']=='FINISH' and x['his_overdue_days']<=-2  else 0,axis=1)
        df_loan['finish_days_1_0']=df_loan.apply(lambda x: 1 if x['dq_days']>-1 and x['status_new']=='FINISH' and x['his_overdue_days']<=-1  else 0,axis=1)
        df_loan['finish_days_0']=df_loan.apply(lambda x: 1 if x['dq_days']>0 and x['status_new']=='FINISH' and x['his_overdue_days']<=0  else 0,axis=1)
        df_loan['finish_days_1']=df_loan.apply(lambda x: 1 if x['dq_days']>1 and x['status_new']=='FINISH' and x['his_overdue_days']<=1  else 0,axis=1)
        df_loan['finish_days_3']=df_loan.apply(lambda x: 1 if x['dq_days']>3 and x['status_new']=='FINISH' and x['his_overdue_days']<=3  else 0,axis=1)
        df_loan['finish_days_7']=df_loan.apply(lambda x: 1 if x['dq_days']>7 and x['status_new']=='FINISH' and x['his_overdue_days']<=7  else 0,axis=1)
        df_loan['finish_days_15']=df_loan.apply(lambda x: 1 if x['dq_days']>15 and x['status_new']=='FINISH' and x['his_overdue_days']<=15  else 0,axis=1)


        #### 统计T+1---15分母 新增到期合同金额  到期0天字段 应所有订单据为已到期数据故不用新增资源
        df_loan['amount_days_3_0']=df_loan.apply(lambda x: x['amount'] if x['dq_days']>-3  else 0,axis=1)
        df_loan['amount_days_2_0']=df_loan.apply(lambda x: x['amount'] if x['dq_days']>-2  else 0,axis=1)
        df_loan['amount_days_1_0']=df_loan.apply(lambda x: x['amount'] if x['dq_days']>-1  else 0,axis=1)
        df_loan['amount_days_0']=df_loan.apply(lambda x: x['amount'] if x['dq_days']>0  else 0,axis=1)
        df_loan['amount_days_1']=df_loan.apply(lambda x: x['amount'] if x['dq_days']>1  else 0,axis=1)
        df_loan['amount_days_3']=df_loan.apply(lambda x: x['amount'] if x['dq_days']>3  else 0,axis=1)
        df_loan['amount_days_7']=df_loan.apply(lambda x: x['amount'] if x['dq_days']>7  else 0,axis=1)
        df_loan['amount_days_15']=df_loan.apply(lambda x: x['amount'] if x['dq_days']>15  else 0,axis=1)

        #### 统计T+0---15 新增到期还款金额  ，因逾期订单会有部分还款，所以逾期状态的订单使用最后更新时间+0至15小于等于到期时间为在此区间还款金额

        df_loan['totalPaidAmount_days_3_0']=df_loan.apply(lambda x: x['totalPaidAmount_new'] if (x['dq_days']>-3) and ((x['status_new']=='FINISH' and x['his_overdue_days']<=-3) or (x['status_new']=='DUNNING' and x['last_update_time']+datetime.timedelta(days=3)<=x['Fdeadline_time'])) else 0,axis=1)
        df_loan['totalPaidAmount_days_2_0']=df_loan.apply(lambda x: x['totalPaidAmount_new'] if (x['dq_days']>-2) and ((x['status_new']=='FINISH' and x['his_overdue_days']<=-2) or (x['status_new']=='DUNNING' and x['last_update_time']+datetime.timedelta(days=2)<=x['Fdeadline_time'])) else 0,axis=1)
        df_loan['totalPaidAmount_days_1_0']=df_loan.apply(lambda x: x['totalPaidAmount_new'] if (x['dq_days']>-1) and ((x['status_new']=='FINISH' and x['his_overdue_days']<=-1) or (x['status_new']=='DUNNING' and x['last_update_time']+datetime.timedelta(days=1)<=x['Fdeadline_time'])) else 0,axis=1)
        df_loan['totalPaidAmount_days_0']=df_loan.apply(lambda x: x['totalPaidAmount_new'] if (x['dq_days']>0) and ((x['status_new']=='FINISH' and x['his_overdue_days']<=0) or (x['status_new']=='DUNNING' and x['last_update_time']<=x['Fdeadline_time'])) else 0,axis=1)
        df_loan['totalPaidAmount_days_1']=df_loan.apply(lambda x: x['totalPaidAmount_new'] if (x['dq_days']>1) and ((x['status_new']=='FINISH' and x['his_overdue_days']<=1) or (x['status_new']=='DUNNING' and x['last_update_time']-datetime.timedelta(days=1)<=x['Fdeadline_time'])) else 0,axis=1)
        df_loan['totalPaidAmount_days_3']=df_loan.apply(lambda x: x['totalPaidAmount_new'] if (x['dq_days']>3) and ((x['status_new']=='FINISH' and x['his_overdue_days']<=3) or (x['status_new']=='DUNNING' and x['last_update_time']-datetime.timedelta(days=3)<=x['Fdeadline_time'])) else 0,axis=1)
        df_loan['totalPaidAmount_days_7']=df_loan.apply(lambda x: x['totalPaidAmount_new'] if (x['dq_days']>7) and ((x['status_new']=='FINISH' and x['his_overdue_days']<=7) or (x['status_new']=='DUNNING' and x['last_update_time']-datetime.timedelta(days=7)<=x['Fdeadline_time'])) else 0,axis=1)
        df_loan['totalPaidAmount_days_15']=df_loan.apply(lambda x: x['totalPaidAmount_new'] if (x['dq_days']>15) and ((x['status_new']=='FINISH' and x['his_overdue_days']<=15) or (x['status_new']=='DUNNING' and x['last_update_time']-datetime.timedelta(days=15)<=x['Fdeadline_time'])) else 0,axis=1)

        df_loan['finish_days_0_1']=df_loan.apply(lambda x: 1 if x['dq_days']>1 and x['status_new']=='FINISH' and x['his_overdue_days']<=0  else 0,axis=1)
        df_loan['finish_days_0_3']=df_loan.apply(lambda x: 1 if x['dq_days']>3 and x['status_new']=='FINISH' and x['his_overdue_days']<=0  else 0,axis=1)
        df_loan['finish_days_0_7']=df_loan.apply(lambda x: 1 if x['dq_days']>7 and x['status_new']=='FINISH' and x['his_overdue_days']<=0  else 0,axis=1)
        df_loan['finish_days_0_15']=df_loan.apply(lambda x: 1 if x['dq_days']>15 and x['status_new']=='FINISH' and x['his_overdue_days']<=0  else 0,axis=1)
        df_result=df_loan.copy()
    else:
        base_extension_df=df_loan[df_loan['is_extension']==1]
        base_extension_df['totalPaidAmount_new']=base_extension_df.apply(lambda x:x['amount'] if x['totalPaidAmount']>x['amount'] else x['totalPaidAmount'],axis=1)

        # ### 统计T+1---15分母 新增到期订单 到期0天字段 应所有订单据为已到期数据故不用新增资源
        base_extension_df['dq_days_3_0']=base_extension_df.extension_dq_days.apply(lambda x: 1 if x>-3 else 0)
        base_extension_df['dq_days_2_0']=base_extension_df.extension_dq_days.apply(lambda x: 1 if x>-2 else 0)
        base_extension_df['dq_days_1_0']=base_extension_df.extension_dq_days.apply(lambda x: 1 if x>-1 else 0)
        base_extension_df['dq_days_0']=base_extension_df.extension_dq_days.apply(lambda x: 1 if x>0 else 0)
        base_extension_df['dq_days_1']=base_extension_df.extension_dq_days.apply(lambda x: 1 if x>1 else 0)
        base_extension_df['dq_days_3']=base_extension_df.extension_dq_days.apply(lambda x: 1 if x>3 else 0)
        base_extension_df['dq_days_7']=base_extension_df.extension_dq_days.apply(lambda x: 1 if x>7 else 0)
        base_extension_df['dq_days_15']=base_extension_df.extension_dq_days.apply(lambda x: 1 if x>15 else 0)

        #### 统计T+0---15分子 新增到结清订单
        base_extension_df['finish_days_3_0']=base_extension_df.apply(lambda x: 1 if x['extension_dq_days']>-3 and x['status']=='FINISH' and x['extension_his_overdue_days']<=-3  else 0,axis=1)
        base_extension_df['finish_days_2_0']=base_extension_df.apply(lambda x: 1 if x['extension_dq_days']>-2 and x['status']=='FINISH' and x['extension_his_overdue_days']<=-2  else 0,axis=1)
        base_extension_df['finish_days_1_0']=base_extension_df.apply(lambda x: 1 if x['extension_dq_days']>-1 and x['status']=='FINISH' and x['extension_his_overdue_days']<=-1  else 0,axis=1)
        base_extension_df['finish_days_0']=base_extension_df.apply(lambda x: 1 if x['extension_dq_days']>0 and x['status']=='FINISH' and x['extension_his_overdue_days']<=0  else 0,axis=1)
        base_extension_df['finish_days_1']=base_extension_df.apply(lambda x: 1 if x['extension_dq_days']>1 and x['status']=='FINISH' and x['extension_his_overdue_days']<=1  else 0,axis=1)
        base_extension_df['finish_days_3']=base_extension_df.apply(lambda x: 1 if x['extension_dq_days']>3 and x['status']=='FINISH' and x['extension_his_overdue_days']<=3  else 0,axis=1)
        base_extension_df['finish_days_7']=base_extension_df.apply(lambda x: 1 if x['extension_dq_days']>7 and x['status']=='FINISH' and x['extension_his_overdue_days']<=7  else 0,axis=1)
        base_extension_df['finish_days_15']=base_extension_df.apply(lambda x: 1 if x['extension_dq_days']>15 and x['status']=='FINISH' and x['extension_his_overdue_days']<=15  else 0,axis=1)

        #### 统计T+1---15分母 新增到期合同金额  到期0天字段 应所有订单据为已到期数据故不用新增资源
        base_extension_df['amount_days_3_0']=base_extension_df.apply(lambda x: x['amount'] if x['extension_dq_days']>-3  else 0,axis=1)
        base_extension_df['amount_days_2_0']=base_extension_df.apply(lambda x: x['amount'] if x['extension_dq_days']>-2  else 0,axis=1)
        base_extension_df['amount_days_1_0']=base_extension_df.apply(lambda x: x['amount'] if x['extension_dq_days']>-1  else 0,axis=1)
        base_extension_df['amount_days_0']=base_extension_df.apply(lambda x: x['amount'] if x['extension_dq_days']>0  else 0,axis=1)
        base_extension_df['amount_days_1']=base_extension_df.apply(lambda x: x['amount'] if x['extension_dq_days']>1  else 0,axis=1)
        base_extension_df['amount_days_3']=base_extension_df.apply(lambda x:  x['amount'] if x['extension_dq_days']>3  else 0,axis=1)
        base_extension_df['amount_days_7']=base_extension_df.apply(lambda x:  x['amount'] if x['extension_dq_days']>7  else 0,axis=1)
        base_extension_df['amount_days_15']=base_extension_df.apply(lambda x:  x['amount'] if x['extension_dq_days']>15  else 0,axis=1)

        ##### 统计T+0---15 新增到期还款金额  ，因逾期订单会有部分还款，所以逾期状态的订单使用最后更新时间+0至15小于等于到期时间为在此区间还款金额
        base_extension_df['totalPaidAmount_days_3_0']=base_extension_df.apply(lambda x: x['totalPaidAmount_new'] if (x['extension_dq_days']>-3) and ((x['status']=='FINISH' and x['extension_his_overdue_days']<=-3) or (x['status']=='DUNNING' and x['last_update_time']+datetime.timedelta(days=3)<=x['deadline_time'])) else 0,axis=1)
        base_extension_df['totalPaidAmount_days_2_0']=base_extension_df.apply(lambda x: x['totalPaidAmount_new'] if (x['extension_dq_days']>-2) and ((x['status']=='FINISH' and x['extension_his_overdue_days']<=-2) or (x['status']=='DUNNING' and x['last_update_time']+datetime.timedelta(days=2)<=x['deadline_time'])) else 0,axis=1)
        base_extension_df['totalPaidAmount_days_1_0']=base_extension_df.apply(lambda x: x['totalPaidAmount_new'] if (x['extension_dq_days']>-1) and ((x['status']=='FINISH' and x['extension_his_overdue_days']<=-1) or (x['status']=='DUNNING' and x['last_update_time']+datetime.timedelta(days=1)<=x['deadline_time'])) else 0,axis=1)
        base_extension_df['totalPaidAmount_days_0']=base_extension_df.apply(lambda x: x['totalPaidAmount_new'] if (x['extension_dq_days']>0) and ((x['status']=='FINISH' and x['extension_his_overdue_days']<=0) or (x['status']=='DUNNING' and x['last_update_time']<=x['deadline_time'])) else 0,axis=1)
        base_extension_df['totalPaidAmount_days_1']=base_extension_df.apply(lambda x: x['totalPaidAmount_new'] if (x['extension_dq_days']>1) and ((x['status']=='FINISH' and x['extension_his_overdue_days']<=1) or (x['status']=='DUNNING' and x['last_update_time']-datetime.timedelta(days=1)<=x['deadline_time'])) else 0,axis=1)
        base_extension_df['totalPaidAmount_days_3']=base_extension_df.apply(lambda x: x['totalPaidAmount_new'] if (x['extension_dq_days']>3) and ((x['status']=='FINISH' and x['extension_his_overdue_days']<=3) or (x['status']=='DUNNING' and x['last_update_time']-datetime.timedelta(days=3)<=x['deadline_time'])) else 0,axis=1)
        base_extension_df['totalPaidAmount_days_7']=base_extension_df.apply(lambda x: x['totalPaidAmount_new'] if (x['extension_dq_days']>7) and ((x['status']=='FINISH' and x['extension_his_overdue_days']<=7) or (x['status']=='DUNNING' and x['last_update_time']-datetime.timedelta(days=7)<=x['deadline_time'])) else 0,axis=1)
        base_extension_df['totalPaidAmount_days_15']=base_extension_df.apply(lambda x: x['totalPaidAmount_new'] if (x['extension_dq_days']>15) and ((x['status']=='FINISH' and x['extension_his_overdue_days']<=15) or (x['status']=='DUNNING' and x['last_update_time']-datetime.timedelta(days=15)<=x['deadline_time'])) else 0,axis=1)


        #### 统计T+1-15到期时，未发生逾期的结清订单
        base_extension_df['finish_days_0_1']=base_extension_df.apply(lambda x: 1 if x['extension_dq_days']>1 and x['status']=='FINISH' and x['extension_his_overdue_days']<=0  else 0,axis=1)
        base_extension_df['finish_days_0_3']=base_extension_df.apply(lambda x: 1 if x['extension_dq_days']>3 and x['status']=='FINISH' and x['extension_his_overdue_days']<=0  else 0,axis=1)
        base_extension_df['finish_days_0_7']=base_extension_df.apply(lambda x: 1 if x['extension_dq_days']>7 and x['status']=='FINISH' and x['extension_his_overdue_days']<=0  else 0,axis=1)
        base_extension_df['finish_days_0_15']=base_extension_df.apply(lambda x: 1 if x['extension_dq_days']>15 and x['status']=='FINISH' and x['extension_his_overdue_days']<=0  else 0,axis=1)
        df_result=base_extension_df.copy()
        
    return df_result


#

def risk_target(df_risk,cols,order_type=0):

    '''
    参数解释:df_risk 为基础数据，cols为分组字段
    用途:贷后指标统计函数，与上面的cal_loan_new_cols() 函数相匹配
    '''
    df_risk=df_risk.groupby(cols).agg({'phone':'nunique','id':'nunique','amount':'sum','actualAmount':'sum','adminAmount':'sum','is_extension':'sum','dq_days_3_0':'sum','dq_days_2_0':'sum',\
                                   'dq_days_1_0':'sum','dq_days_0':'sum','dq_days_1':'sum','dq_days_3':'sum','dq_days_7':'sum','dq_days_15':'sum','finish_days_3_0':'sum',\
                                   'finish_days_2_0':'sum','finish_days_1_0':'sum','finish_days_0':'sum','finish_days_1':'sum', 'finish_days_3':'sum','finish_days_7':'sum',\
                                   'finish_days_15':'sum','amount_days_3_0':'sum','amount_days_2_0':'sum','amount_days_1_0':'sum','amount_days_0':'sum','amount_days_1':'sum',\
                                   'amount_days_3':'sum','amount_days_7':'sum','amount_days_15':'sum','totalPaidAmount_days_3_0':'sum','totalPaidAmount_days_2_0':'sum',\
                                   'totalPaidAmount_days_1_0':'sum','totalPaidAmount_days_0':'sum','totalPaidAmount_days_1':'sum','totalPaidAmount_days_3':'sum',\
                                   'totalPaidAmount_days_7':'sum','totalPaidAmount_days_15':'sum','finish_days_0_1':'sum','finish_days_0_3':'sum','finish_days_0_7':'sum',\
                                   'finish_days_0_15':'sum'}).reset_index()
    

    df_risk['amount_avg']=round(df_risk['amount']/df_risk['id']) ### 件均
    df_risk['admin_rate']=round(df_risk['adminAmount']/df_risk['amount'],3) ## 平均服务费率
    df_risk['extension_rate']=round(df_risk['is_extension']/df_risk['id'],3) ##展期率
    if order_type==1: 
        df_risk['extension_due_rate']=round(df_risk['dq_days_3_0']/df_risk['id'],3) ##展期率

    ## 金额坏账率 -----------------------------------------------
    df_risk['bad_amount_days_3_0']=round(1-df_risk['totalPaidAmount_days_3_0']/df_risk['amount_days_3_0'],3)
    df_risk['bad_amount_days_2_0']=round(1-df_risk['totalPaidAmount_days_2_0']/df_risk['amount_days_2_0'],3)
    df_risk['bad_amount_days_1_0']=round(1-df_risk['totalPaidAmount_days_1_0']/df_risk['amount_days_1_0'],3)
    df_risk['bad_amount_days_0']=round(1-df_risk['totalPaidAmount_days_0']/df_risk['amount_days_0'],3)
    df_risk['bad_amount_days_1']=round(1-df_risk['totalPaidAmount_days_1']/df_risk['amount_days_1'],3)
    df_risk['bad_amount_days_3']=round(1-df_risk['totalPaidAmount_days_3']/df_risk['amount_days_3'],3)
    df_risk['bad_amount_days_7']=round(1-df_risk['totalPaidAmount_days_7']/df_risk['amount_days_7'],3)
    df_risk['bad_amount_days_15']=round(1-df_risk['totalPaidAmount_days_15']/df_risk['amount_days_15'],3)

    ### 订单坏账率 --------------------------------------------
    df_risk['bad_order_days_3_0']=round(1-df_risk['finish_days_3_0']/df_risk['dq_days_3_0'],3)
    df_risk['bad_order_days_2_0']=round(1-df_risk['finish_days_2_0']/df_risk['dq_days_2_0'],3)
    df_risk['bad_order_days_1_0']=round(1-df_risk['finish_days_1_0']/df_risk['dq_days_1_0'],3)
    df_risk['bad_order_days_0']=round(1-df_risk['finish_days_0']/df_risk['dq_days_0'],3)
    df_risk['bad_order_days_1']=round(1-df_risk['finish_days_1']/df_risk['dq_days_1'],3)
    df_risk['bad_order_days_3']=round(1-df_risk['finish_days_3']/df_risk['dq_days_3'],3)
    df_risk['bad_order_days_7']=round(1-df_risk['finish_days_7']/df_risk['dq_days_7'],3)
    df_risk['bad_order_days_15']=round(1-df_risk['finish_days_15']/df_risk['dq_days_15'],3)



    ## 订单催回率 ---------------------------------------------------
    df_risk['recall_rate_1']=round((df_risk['finish_days_1']-df_risk['finish_days_0_1'])/(df_risk['dq_days_1']-df_risk['finish_days_0_1']),3)
    df_risk['recall_rate_3']=round((df_risk['finish_days_3']-df_risk['finish_days_0_3'])/(df_risk['dq_days_3']-df_risk['finish_days_0_3']),3)
    df_risk['recall_rate_7']=round((df_risk['finish_days_7']-df_risk['finish_days_0_7'])/(df_risk['dq_days_7']-df_risk['finish_days_0_7']),3)
    df_risk['recall_rate_15']=round((df_risk['finish_days_15']-df_risk['finish_days_0_15'])/(df_risk['dq_days_15']-df_risk['finish_days_0_15']),3)


    df_risk_res=df_risk.fillna('/')

    return df_risk_res


def cal_apply_target(df_apply,cols):
    '''
    参数解释:apply_df 为基础数据，cols为分组字段
    用途:新增拒贷、通过、卡单等关键指标，并以订单、用户进行统计
    '''
    ### 新增拒绝、卡单、放款、取消、通过指标
    df_apply['reject_num']=df_apply.status.apply(lambda x:1 if x=='ROBOT_DENY' else 0) ### 拒绝量
    df_apply['pending_num']=df_apply.status.apply(lambda x:1 if x=='PENDING' else 0) ### 卡单量
    df_apply['loan_num']=df_apply.status.apply(lambda x:1 if x in ['FINISH','DUNNING','LOAN_SUCCESS'] else 0) ### 交易量 
    df_apply['cancel_num']=df_apply.status.apply(lambda x:1 if x not in ['FINISH','DUNNING','LOAN_SUCCESS','PENDING','ROBOT_DENY'] else 0) ## 取消交易量
    df_apply['pass_num']=df_apply.status.apply(lambda x:1 if x not in ['PENDING','ROBOT_DENY'] else 0) ### 通过量
    
    #### 统计订单申请指标
    df_apply_order=df_apply.groupby(cols).agg({'id':'count','pending_num':'sum','reject_num':'sum','pass_num':'sum','loan_num':'sum','cancel_num':'sum'}).reset_index()
    df_apply_order['pending_rate']=round(df_apply_order['pending_num']/df_apply_order['id'],4)### 卡单率
    df_apply_order['reject_rate']=round(df_apply_order['reject_num']/df_apply_order['id'],4)### 拒绝率
    df_apply_order['pass_rate']=round(df_apply_order['pass_num']/df_apply_order['id'],4)### 通过率
    df_apply_order['loan_rate']=round(df_apply_order['loan_num']/df_apply_order['id'],4)### 交易率
    df_apply_order['cancel_rate']=round(df_apply_order['cancel_num']/df_apply_order['id'],4)### 交易取消率
    
    #### 统计用户拒贷率 需去重统计
    df_apply_phone=df_apply[cols+['phone','reject_num','pending_num','loan_num','cancel_num','pass_num']].drop_duplicates()
    df_apply_phone=df_apply_phone.groupby(cols).agg({'phone':'nunique','reject_num':'sum','pass_num':'sum','loan_num':'sum'}).reset_index().rename(columns={'reject_num':'phone_reject_num','pass_num':'phone_pass_num','loan_num':'phone_loan_num'})
    
    df_apply_phone['phone_reject_rate']=round(df_apply_phone['phone_reject_num']/df_apply_phone['phone'],4)### 拒绝率
    df_apply_phone['phone_pass_rate']=round(df_apply_phone['phone_pass_num']/df_apply_phone['phone'],4)### 通过率
    df_apply_phone['phone_loan_rate']=round(df_apply_phone['phone_loan_num']/df_apply_phone['phone'],4)### 交易率
    df_apply_rate=pd.merge(df_apply_order,df_apply_phone,how='left',on=cols)
    return df_apply_rate


# +
#调用示例
# group_list = ['cmRiskStrategyName',"apply_bin"]
# columns=["apply_bin"]
# show_list=['order_id','order_rate','phone','phone_rate','pass_order_rate','pass_phone_rate','pass_order','pass_phone','loan_order','loan_phone','amount','actualAmount','avg_amount_order']
# cal_apply_pass_rate(df_apply_firstloan,group_list = group_list,columns=columns,show_list=show_list)
def cal_apply_pass_rate(df_apply_in,group_list,columns,show_list=['order_id','order_rate','phone','phone_rate','pass_order','pass_phone','pass_order_rate','pass_phone_rate']):
    '''
    用途:统计分组的订单/用户申请数量,占比,拒贷率、通过率等 show_list=['order_id','order_rate','phone','phone_rate','pass_order_rate','pass_phone_rate','pass_order','pass_phone','loan_order','loan_phone','amount','actualAmount','avg_amount_order']；
    df_apply:申请订单数据集；
    group_list: 分组统计的字段list，对应pd.pivot_table的index参数;
    columns:如果为[]则不转置，如果不为空，则转置，对应pd.pivot_table的columns参数;
    show_list:返回的字段,默认为['order_id','order_rate','phone','phone_rate','pass_order_rate','pass_phone_rate'],对应pd.pivot_table的values参数;
    return:pivot_table后的df
    '''
    df_apply = df_apply_in.copy()
    df_apply.rename(columns = {"id":"order_id"},inplace =True)
    loan_status = ["LOAN_SUCCESS","FINISH","DUNNING"]
    reject_status = ["ROBOT_DENY","FINAL_DENY"]
    df_pass = df_apply[~df_apply.status.isin(reject_status)]
    df_loan = df_apply[df_apply.status.isin(loan_status)]

    for item in columns:  ## 交集字段，并保持原group_list的顺序
        if item not in group_list:
            group_list.append(item) 
    group_dict = {'order_id': pd.Series.nunique, 'phone': pd.Series.nunique}

    if len(columns) == 0:
        #申请
        df_group_01 = pd.pivot_table(df_apply, values=['order_id','phone'], index = group_list, columns=[] ,margins=True,aggfunc = pd.Series.nunique)
        total_order_id = df_apply['order_id'].nunique()
        total_phone = df_apply['phone'].nunique()
        df_group_01['order_rate'] = df_group_01.apply(lambda x: round(x['order_id'] / total_order_id,4) if total_order_id != 0 else 0, axis=1)
        df_group_01['phone_rate'] = df_group_01.apply(lambda x: round(x['phone'] / total_phone,4) if total_phone != 0 else 0, axis=1)
        #通过
        df_group_02 = pd.pivot_table(df_pass, values=['order_id','phone'], index = group_list, columns=[] ,margins=True,aggfunc = pd.Series.nunique)
        df_group_02.rename(columns = {"order_id":"pass_order","phone":"pass_phone"},inplace =True)
        #交易
        df_group_03 = pd.pivot_table(df_loan, values=['order_id','phone'], index = group_list, columns=[] ,margins=True,aggfunc = pd.Series.nunique)
        df_group_03.rename(columns = {"order_id":"loan_order","phone":"loan_phone"},inplace =True)
        df_group_04 = pd.pivot_table(df_loan, values=['amount','actualAmount'], index = group_list, columns=[] ,margins=True,aggfunc = 'sum')
        #拼接
        df_final = pd.merge(df_group_01,df_group_02,on = group_list,how ="left")
        df_final = pd.merge(df_final,df_group_03,on = group_list,how ="left")
        df_final = pd.merge(df_final,df_group_04,on = group_list,how ="left")
        df_final['pass_order_rate'] = df_final.apply(lambda x: round(x['pass_order'] / x["order_id"],4) if x["order_id"] != 0 else 0, axis=1)
        df_final['pass_phone_rate'] = df_final.apply(lambda x: round(x['pass_phone'] / x["phone"],4) if x["phone"] != 0 else 0, axis=1)
        df_final['avg_amount_order'] = df_final.apply(lambda x: round(x['amount'] / x["loan_order"],4) if x["loan_order"] != 0 else 0, axis=1)
        df_final.reset_index(inplace=True)
        df_final = df_final[group_list + show_list]
    else:
        #申请
        df_apply[columns]=df_apply[columns].applymap(lambda x:'NAN' if pd.isna(x) else x)
        df_group_00 = pd.pivot_table(df_apply, values=['order_id','phone'], index = columns, columns=[],aggfunc = pd.Series.nunique).reset_index()  #转置的小计
        df_group_00.rename(columns={'order_id':'order_id_subtotal','phone':'phone_subtotal'}, inplace=True)
        df_group_01 = pd.pivot_table(df_apply, values=['order_id','phone'], index = group_list, columns=[],aggfunc = pd.Series.nunique).reset_index()
        df_group_01 = pd.merge(df_group_01,df_group_00,on = columns,how ="left")
        df_group_01['order_rate'] = df_group_01.apply(lambda x: round(x['order_id'] / x['order_id_subtotal'],4) if x['order_id_subtotal'] != 0 else 0, axis=1)  #每个转置的分箱的订单占比，A区间的订单百分比，小计是100%
        df_group_01['phone_rate'] = df_group_01.apply(lambda x: round(x['phone'] / x['phone_subtotal'],4) if x['phone_subtotal'] != 0 else 0, axis=1)           #每个转置的分箱的手机号占比，A区间的手机号百分比，小计是100%

        #通过--------------------------------------------
        ### 细分组别通过的订单及用户数
        df_group_02 = pd.pivot_table(df_pass, values=['order_id','phone'], index = group_list, columns=[],aggfunc = pd.Series.nunique).reset_index() 
        df_group_02.rename(columns = {"order_id":"pass_order","phone":"pass_phone"},inplace =True)

        ### 粗分组【整体】通过订单及用户数
        df_group_02_total = pd.pivot_table(df_pass, values=['order_id','phone'], index = columns, columns=[],aggfunc = pd.Series.nunique).reset_index() 
        df_group_02_total.rename(columns = {"order_id":"pass_order","phone":"pass_phone"},inplace =True)

        #交易----------------------------------------------
        ### 细分组别交易的订单及用户数
        df_group_03 = pd.pivot_table(df_loan, values=['order_id','phone'], index = group_list, columns=[],aggfunc = pd.Series.nunique).reset_index() 
        df_group_03.rename(columns = {"order_id":"loan_order","phone":"loan_phone"},inplace =True)

        ### 粗分组【整体】交易订单及用户数
        df_group_03_total = pd.pivot_table(df_loan, values=['order_id','phone'], index = columns, columns=[],aggfunc = pd.Series.nunique).reset_index() 
        df_group_03_total.rename(columns = {"order_id":"loan_order","phone":"loan_phone"},inplace =True)


        df_group_04 = pd.pivot_table(df_loan, values=['amount','actualAmount'], index = group_list, columns=[],aggfunc = 'sum').reset_index() 
        df_group_04_total = pd.pivot_table(df_loan, values=['amount','actualAmount'], index = columns, columns=[],aggfunc = 'sum').reset_index() 
        #拼接
        df_final = pd.merge(df_group_01,df_group_02,on = group_list,how ="left")
        df_final = pd.merge(df_final,df_group_03,on = group_list,how ="left")
        df_final = pd.merge(df_final,df_group_04,on = group_list,how ="left")
        group_list_type=df_final[group_list[0]].dtype### 判读数据类型

        ##拼接total数据
        df_final_total = pd.merge(df_group_00,df_group_02_total,on = columns,how ="left")
        df_final_total = pd.merge(df_final_total,df_group_03_total,on = columns,how ="left")
        df_final_total = pd.merge(df_final_total,df_group_04_total,on = columns,how ="left")
        df_final_total['order_id']=df_final_total['order_id_subtotal']
        df_final_total['phone']=df_final_total['phone_subtotal']
        df_final_total['order_rate']=df_final_total.apply(lambda x: round(x['order_id'] / x['order_id_subtotal'],4) if x['order_id_subtotal'] != 0 else 0, axis=1)
        df_final_total['phone_rate']=df_final_total.apply(lambda x: round(x['phone'] / x['phone_subtotal'],4) if x['phone_subtotal'] != 0 else 0, axis=1)
        df_final_total[group_list[0]]='1ALL'
        df_final=pd.concat([df_final,df_final_total]).fillna(0)
        df_final=df_final.reset_index().drop('index',axis=1)
        df_final['pass_order_rate'] = df_final.apply(lambda x: round(x['pass_order'] / x["order_id"],4) if x["order_id"] != 0 else 0, axis=1)
        df_final['pass_phone_rate'] = df_final.apply(lambda x: round(x['pass_phone'] / x["phone"],4) if x["phone"] != 0 else 0, axis=1)
        df_final['avg_amount_order'] = df_final.apply(lambda x: round(x['amount'] / x["loan_order"],4) if x["loan_order"] != 0 else 0, axis=1)

    #     #转置,变换columns的level，更新订单通过率，手机号通过率，订单件均；并按照show_list排序列
        df_final = pd.pivot_table(df_final, values=['order_id','order_rate','phone','phone_rate','pass_order','pass_order_rate','pass_phone','pass_phone_rate','loan_order','loan_phone','amount','actualAmount','avg_amount_order'], index=group_list[0] , columns = columns,fill_value=0)
        if group_list_type=='object':###为字符串时进行数据排序 ,因时间日期也会判断为object 故只有字母的数据进行单独排序
            index_type=any(char.isalpha() for char in df_final.index[0]) # 判断是否含字母
            if index_type==True:
                df_final = df_final.sort_values(by=group_list[0],ascending=False)
        df_final.columns = df_final.columns.swaplevel(0,1)
    #     #del df_final[('total')]
        pivot_columns_len=len(df_final.columns.levels)
        if pivot_columns_len>2:
            df_final.columns=df_final.columns.swaplevel(1,2)
            df_final.sort_index(axis=1, level=[0,1], inplace=True,ascending=[True,True])
            df_final=df_final.reindex(columns = show_list,level=2)
        else:
            df_final.sort_index(axis=1, level=[0], inplace=True,ascending=[True])
            df_final=df_final.reindex(columns = show_list,level=1)


#         df_final.loc['total'] = df_final.iloc[:-1].sum(axis=0, skipna=False)
#         df_final.columns = df_final.columns.swaplevel(0,1)
#         del df_final[('total')]
#         columns_list = df_apply[columns[0]].unique().tolist()
#         for columns_i in columns_list:
#             df_final[(columns_i,'pass_order_rate')] = df_final.apply(lambda x: round(x[(columns_i,'pass_order')] / x[(columns_i,"order_id")],4) if x[(columns_i,"order_id")] != 0 else 0, axis=1)
#             df_final[(columns_i,'pass_phone_rate')] = df_final.apply(lambda x: round(x[(columns_i,'pass_phone')] / x[(columns_i,"phone")],4) if x[(columns_i,"phone")] != 0 else 0, axis=1)
#             df_final[(columns_i,'avg_amount_order')] = df_final.apply(lambda x: round(x[(columns_i,'amount')] / x[(columns_i,"loan_order")],4) if x[(columns_i,"loan_order")] != 0 else 0, axis=1)
#         df_final.sort_index(axis=1, level=[0], inplace=True,ascending=[True])
#         df_final = df_final.reindex(columns = show_list, level=1)
        
    return df_final.fillna(0)


# -

#调用示例
# group_list = ['cmRiskStrategyName',"due_bin"]
# columns=["due_bin"]
# show_list=['order_id','order_rate','phone','phone_rate','amount','amount_rate','extension_order','extension_rate','bad_amount_rate_2_0','bad_amount_rate_1_0','bad_amount_rate_0','bad_amount_rate_3','bad_amount_rate_7','bad_order_rate_2_0','bad_order_rate_1_0','bad_order_rate_0','bad_order_rate_3','bad_order_rate_7','recall_order_rate_2_0','recall_order_rate_3','recall_order_rate_7']
# cal_due_risk_rate(df_due_firstloan,group_list = group_list,columns=columns,show_list=show_list)
def cal_due_risk_rate(df_due_in,group_list,columns,show_list=['order_id','order_rate','phone','phone_rate','amount','amount_rate','extension_order','extension_rate','bad_amount_rate_2_0','bad_amount_rate_1_0','bad_amount_rate_0','bad_amount_rate_3','bad_amount_rate_7']):
    '''
    用途:统计分组的订单/用户申请数量,占比,拒贷率、通过率等 
    show_list=['order_id','order_rate','phone','phone_rate','amount','amount_rate','extension_order','extension_rate','avg_amount_order','bad_amount_rate_2_0','bad_amount_rate_1_0','bad_amount_rate_0','bad_amount_rate_3','bad_amount_rate_7'
            ,'bad_order_rate_2_0','bad_order_rate_1_0','bad_order_rate_0','bad_order_rate_3','bad_order_rate_7','recall_order_rate_2_0','recall_order_rate_3','recall_order_rate_7','actualAmount','totalPaidAmount']
    df_due:交易到期订单数据集；
    group_list: 分组统计的字段list，对应pd.pivot_table的index参数;
    columns:如果为[]则不转置，如果不为空，则转置，对应pd.pivot_table的columns参数;
    show_list:返回的字段,对应pd.pivot_table的values参数;
    return:pivot_table后的df
    '''
    df_due = df_due_in.copy()
    df_due.rename(columns = {"id":"order_id"},inplace =True)
    df_extension = df_due[df_due.is_extension == 1]

    for item in columns:  ## 交集字段，并保持原group_list的顺序
        if item not in group_list:
            group_list.append(item) 
    group_dict = {'order_id': pd.Series.nunique, 'phone': pd.Series.nunique,'amount':'sum','actualAmount':'sum','totalPaidAmount':'sum','totalPaidAmount_days_2_0':'sum','amount_days_2_0':'sum','totalPaidAmount_days_1_0':'sum','amount_days_1_0':'sum'
                 ,'totalPaidAmount_days_0':'sum','amount_days_0':'sum','totalPaidAmount_days_3':'sum','amount_days_3':'sum','totalPaidAmount_days_7':'sum','amount_days_7':'sum'
                 ,'finish_days_2_0':'sum','dq_days_2_0':'sum','finish_days_1_0':'sum','dq_days_1_0':'sum','finish_days_0':'sum','dq_days_0':'sum','finish_days_3':'sum','dq_days_3':'sum','finish_days_7':'sum','dq_days_7':'sum'
                 ,'finish_days_1':'sum','finish_days_0_1':'sum','dq_days_1':'sum','finish_days_0_1':'sum'
                 ,'finish_days_0_3':'sum','finish_days_0_7':'sum'
                 }

    if len(columns) == 0:
        #到期
        df_group_01 = pd.pivot_table(df_due, values=list(group_dict.keys()), index = group_list, columns=[] ,margins=True,aggfunc = group_dict)
        total_order_id = df_due['order_id'].nunique()
        total_phone = df_due['phone'].nunique()
        total_amount = df_due['amount'].sum()
        df_group_01['order_rate'] = df_group_01.apply(lambda x: round(x['order_id'] / total_order_id,4) if total_order_id != 0 else 0, axis=1)
        df_group_01['phone_rate'] = df_group_01.apply(lambda x: round(x['phone'] / total_phone,4) if total_phone != 0 else 0, axis=1)
        df_group_01['amount_rate'] = df_group_01.apply(lambda x: round(x['amount'] / total_amount,4) if total_amount != 0 else 0, axis=1)
        #展期
        df_group_02 = pd.pivot_table(df_extension, values=['order_id'], index = group_list, columns=[] ,margins=True,aggfunc = pd.Series.nunique)
        df_group_02.rename(columns = {"order_id":"extension_order"},inplace =True)
        #拼接
        df_final = pd.merge(df_group_01,df_group_02,on = group_list,how ="left")
        #rate：bad_rate，extension_rate
        df_final['extension_rate'] = df_final.apply(lambda x: round(x['extension_order'] / x["order_id"],4) if x["order_id"] != 0 else 0, axis=1)
        df_final['avg_amount_order'] = df_final.apply(lambda x: round(x['amount'] / x["order_id"],2) if x["order_id"] != 0 else 0, axis=1)
        df_final['bad_amount_rate_2_0'] = df_final.apply(lambda x: round(1-x['totalPaidAmount_days_2_0']/x['amount_days_2_0'],4) if x["amount_days_2_0"] != 0 else 0, axis=1)
        df_final['bad_amount_rate_1_0'] = df_final.apply(lambda x: round(1-x['totalPaidAmount_days_1_0']/x['amount_days_1_0'],4) if x["amount_days_1_0"] != 0 else 0, axis=1)
        df_final['bad_amount_rate_0'] = df_final.apply(lambda x: round(1-x['totalPaidAmount_days_0']/x['amount_days_0'],4) if x["amount_days_0"] != 0 else 0, axis=1)
        df_final['bad_amount_rate_3'] = df_final.apply(lambda x: round(1-x['totalPaidAmount_days_3']/x['amount_days_3'],4) if x["amount_days_3"] != 0 else 0, axis=1)
        df_final['bad_amount_rate_7'] = df_final.apply(lambda x: round(1-x['totalPaidAmount_days_7']/x['amount_days_7'],4) if x["amount_days_7"] != 0 else 0, axis=1)

        df_final['bad_order_rate_2_0'] = df_final.apply(lambda x: round(1-x['finish_days_2_0']/x['dq_days_2_0'],4) if x["dq_days_2_0"] != 0 else 0, axis=1)
        df_final['bad_order_rate_1_0'] = df_final.apply(lambda x: round(1-x['finish_days_1_0']/x['dq_days_1_0'],4) if x["dq_days_1_0"] != 0 else 0, axis=1)
        df_final['bad_order_rate_0'] = df_final.apply(lambda x: round(1-x['finish_days_0']/x['dq_days_0'],4) if x["dq_days_0"] != 0 else 0, axis=1)
        df_final['bad_order_rate_3'] = df_final.apply(lambda x: round(1-x['finish_days_3']/x['dq_days_3'],4) if x["dq_days_3"] != 0 else 0, axis=1)
        df_final['bad_order_rate_7'] = df_final.apply(lambda x: round(1-x['finish_days_7']/x['dq_days_7'],4) if x["dq_days_7"] != 0 else 0, axis=1)

        df_final['recall_order_rate_2_0']=df_final.apply(lambda x: round((x['finish_days_0']-x['finish_days_2_0'])/(x['dq_days_0']-x['finish_days_2_0']),4) if (x['dq_days_0']-x['finish_days_2_0']) != 0 else 0, axis=1)
        df_final['recall_order_rate_3']=df_final.apply(lambda x: round((x['finish_days_3']-x['finish_days_0_3'])/(x['dq_days_3']-x['finish_days_0_3']),4) if (x['dq_days_3']-x['finish_days_0_3']) != 0 else 0, axis=1)
        df_final['recall_order_rate_7']=df_final.apply(lambda x: round((x['finish_days_7']-x['finish_days_0_7'])/(x['dq_days_7']-x['finish_days_0_7']),4) if (x['dq_days_7']-x['finish_days_0_7']) != 0 else 0, axis=1)

        df_final.reset_index(inplace=True)
        df_final = df_final[group_list + show_list]
    else:
        #到期
        df_due[columns]=df_due[columns].applymap(lambda x:'NAN' if pd.isna(x) else x)
        df_group_00 = pd.pivot_table(df_due, values=list(group_dict.keys()), index = columns, columns=[],aggfunc = group_dict).reset_index()  #转置的小计
        df_group_00['order_id_subtotal']=df_group_00['order_id']
        df_group_00['phone_subtotal']=df_group_00['phone']
        df_group_00['amount_subtotal']=df_group_00['amount']
        df_group_00[group_list[0]]='1All'
        df_group_01 = pd.pivot_table(df_due, values=list(group_dict.keys()), index = group_list, columns=[],aggfunc =group_dict).reset_index()
        df_group_01 = pd.merge(df_group_01,df_group_00[columns+["order_id_subtotal","phone_subtotal","amount_subtotal"]],on = columns,how ="left")
        group_list_type=df_group_01[group_list[0]].dtype### 判读数据类型
        df_group_01=pd.concat([df_group_01,df_group_00])
        df_group_01['order_rate'] = df_group_01.apply(lambda x: round(x['order_id'] / x['order_id_subtotal'],4) if x['order_id_subtotal'] != 0 else 0, axis=1)  #每个转置的分箱的订单占比，A区间的订单百分比，小计是100%
        df_group_01['phone_rate'] = df_group_01.apply(lambda x: round(x['phone'] / x['phone_subtotal'],4) if x['phone_subtotal'] != 0 else 0, axis=1)           #每个转置的分箱的手机号占比，A区间的手机号百分比，小计是100%
        df_group_01['amount_rate'] = df_group_01.apply(lambda x: round(x['amount'] / x['amount_subtotal'],4) if x['amount_subtotal'] != 0 else 0, axis=1)

        #展期
        df_group_02 = pd.pivot_table(df_extension, values=['order_id'], index =group_list, columns=[],aggfunc = pd.Series.nunique).reset_index() 
        df_group_03 = pd.pivot_table(df_extension, values=['order_id'], index =columns, columns=[],aggfunc = pd.Series.nunique).reset_index() 
        df_group_03[group_list[0]]='1All'
        df_group_02=pd.concat([df_group_02,df_group_03])
        df_group_02.rename(columns = {"order_id":"extension_order"},inplace =True)
        
        #拼接
        df_final = pd.merge(df_group_01,df_group_02[group_list+["extension_order"]],on = group_list,how ="left")
        df_final.reset_index(inplace=True)
        df_final=df_final.drop('index',axis=1)
        df_final['extension_rate'] = df_final.apply(lambda x: round(x['extension_order'] / x["order_id"],4) if x["order_id"] != 0 else 0, axis=1)
        df_final['avg_amount_order'] = df_final.apply(lambda x: round(x['amount'] / x["order_id"],2) if x["order_id"] != 0 else 0, axis=1)
        df_final['bad_amount_rate_2_0'] = df_final.apply(lambda x: round(1-x['totalPaidAmount_days_2_0']/x['amount_days_2_0'],4) if x["amount_days_2_0"] != 0 else 0, axis=1)
        df_final['bad_amount_rate_1_0'] = df_final.apply(lambda x: round(1-x['totalPaidAmount_days_1_0']/x['amount_days_1_0'],4) if x["amount_days_1_0"] != 0 else 0, axis=1)
        df_final['bad_amount_rate_0'] = df_final.apply(lambda x: round(1-x['totalPaidAmount_days_0']/x['amount_days_0'],4) if x["amount_days_0"] != 0 else 0, axis=1)
        df_final['bad_amount_rate_3'] = df_final.apply(lambda x: round(1-x['totalPaidAmount_days_3']/x['amount_days_3'],4) if x["amount_days_3"] != 0 else 0, axis=1)
        df_final['bad_amount_rate_7'] = df_final.apply(lambda x: round(1-x['totalPaidAmount_days_7']/x['amount_days_7'],4) if x["amount_days_7"] != 0 else 0, axis=1)

        df_final['bad_order_rate_2_0'] = df_final.apply(lambda x: round(1-x['finish_days_2_0']/x['dq_days_2_0'],4) if x["dq_days_2_0"] != 0 else 0, axis=1)
        df_final['bad_order_rate_1_0'] = df_final.apply(lambda x: round(1-x['finish_days_1_0']/x['dq_days_1_0'],4) if x["dq_days_1_0"] != 0 else 0, axis=1)
        df_final['bad_order_rate_0'] = df_final.apply(lambda x: round(1-x['finish_days_0']/x['dq_days_0'],4) if x["dq_days_0"] != 0 else 0, axis=1)
        df_final['bad_order_rate_3'] = df_final.apply(lambda x: round(1-x['finish_days_3']/x['dq_days_3'],4) if x["dq_days_3"] != 0 else 0, axis=1)
        df_final['bad_order_rate_7'] = df_final.apply(lambda x: round(1-x['finish_days_7']/x['dq_days_7'],4) if x["dq_days_7"] != 0 else 0, axis=1)

        df_final['recall_order_rate_2_0']=df_final.apply(lambda x: round((x['finish_days_0']-x['finish_days_2_0'])/(x['dq_days_0']-x['finish_days_2_0']),4) if (x['dq_days_0']-x['finish_days_2_0']) != 0 else 0, axis=1)
        df_final['recall_order_rate_3']=df_final.apply(lambda x: round((x['finish_days_3']-x['finish_days_0_3'])/(x['dq_days_3']-x['finish_days_0_3']),4) if (x['dq_days_3']-x['finish_days_0_3']) != 0 else 0, axis=1)
        df_final['recall_order_rate_7']=df_final.apply(lambda x: round((x['finish_days_7']-x['finish_days_0_7'])/(x['dq_days_7']-x['finish_days_0_7']),4) if (x['dq_days_7']-x['finish_days_0_7']) != 0 else 0, axis=1)


        #转置,变换columns的level，更新订单通过率，手机号通过率，订单件均；并按照show_list排序列
        df_final = pd.pivot_table(df_final, values=list(df_final.columns.drop(group_list)), index=group_list[0] , columns = columns).fillna(0)
        if group_list_type=='object':###为字符串时进行数据排序 ,因时间日期也会判断为object 故只有字母的数据进行单独排序
            index_type=any(char.isalpha() for char in df_final.index[0]) # 判断是否含字母
            if index_type==True:
                df_final = df_final.sort_values(by=group_list[0],ascending=False)
        df_final.columns = df_final.columns.swaplevel(0,1)
    #     #del df_final[('total')]
        pivot_columns_len=len(df_final.columns.levels)
        if pivot_columns_len>2:
            df_final.columns=df_final.columns.swaplevel(1,2)
            df_final.sort_index(axis=1, level=[0,1], inplace=True,ascending=[True,True])
            df_final=df_final.reindex(columns = show_list,level=2)
        else:
            df_final.sort_index(axis=1, level=[0], inplace=True,ascending=[True])
            df_final=df_final.reindex(columns = show_list,level=1)
        
    return df_final.fillna(0)


#调用示例
# col_list_file = "balck_data/col_ol_list.csv"
# df_col_list = pd.read_csv(col_list_file)
#df_due_firstloan_zz["hit_black"] = df_due_firstloan_zz.apply(lambda x: co_hit_black(x,df_col_list),axis =1)
def co_hit_black(df,df_col_list):
    '''
    df: 数据表，必须含phone和idcard 和created_time[datetime格式]
    df_col_list:读取co的csv数据
    return:是否命中,逻辑判断了入黑时间
    '''
    phone_md5 = hashlib.md5(df.phone.encode()).hexdigest()
    idcard_md5 = hashlib.md5(df.idCard.encode()).hexdigest()
    formatted_date = df.created_time.strftime("%Y-%m-%d")
    if (phone_md5 in df_col_list['phone'].values) | (idcard_md5 in df_col_list['idcard'].values):
        df_tmp = df_col_list[(df_col_list['phone'] == phone_md5) | (df_col_list['idcard'] == idcard_md5)]
        if formatted_date > df_tmp['plan_time'].min():
            return 1
        else:
            return 0
    else:
        return 0


def rule_clean(rule_phone_df):
    '''
    参数解释:rule_phone_df 为数据集,主要进入字段为ruleResults 除该字段外不能包含其他的字典数据
    用途:将规则编号、名称、values、是否通过字段提取并转置
    备注:因将订单的所有规则转置为行信息，订单会出现重复，统计时需注意去重
    '''
    ##### 规则命中数据表 后续只需获取需要展示的时间节点即可
    group_pattern="'ruleId'|'groupId': '(\w+)'"
    rule_pattern="'groupId'|'ruleId': '(\w+)'"
    pass_pattern= r"'pass': (\w+)"
    Id_pattern=r"ruleId|groupId"

    ### 因在提取pass 数据时发现 部分模型规则中values字段中也有pass数据，故将values字段进行剔除
    rule_phone_df['ruleResults_1']=rule_phone_df.ruleResults.apply(lambda x:re.sub(r"'values': .*?}", '', str(x)))
    rule_phone_df['rule_code']=rule_phone_df.ruleResults.apply(lambda x:list(x.keys()))  ## 获取规则code
    rule_phone_df['rule_type']=rule_phone_df.ruleResults.apply(lambda x:re.findall(Id_pattern,str(x))) ### 区分规则组与子规则标识
    rule_phone_df['rule_name']=rule_phone_df.ruleResults.apply(lambda x:re.findall(rule_pattern,str(x))) ## 获取子规则名称
    #rule_phone_df['group_name']=rule_phone_df.ruleResults.apply(lambda x:re.findall(group_pattern,str(x))) ## 获取规则组 名称
    rule_phone_df['pass_type']=rule_phone_df.ruleResults_1.apply(lambda x:re.findall(pass_pattern,str(x))) ## 获取是否通过 

    rule_phone_df=rule_phone_df.drop(['ruleResults','ruleResults_1'],axis=1).reset_index().drop('index',axis=1)
    rule_phone_dfs = pd.concat([rule_phone_df[col].explode() for col in rule_phone_df.columns], axis=1)
    rule_phone_dfs['reject_id']=rule_phone_dfs.pass_type.apply(lambda x:1 if x=='False' else 0)
    return rule_phone_dfs


def apply_rejection(df,subset_cols,cols):
    '''
    参数解释:df 为数据集，subset_cols 为主要观测的分组，cols 为subset_cols的合计组别
    用途:统计分组/整体对应的订单/用户拒贷率、订单/用户占比
    '''

    non_intersect_cols =list(set(subset_cols) - set(cols))[0]## 非交集字段
    intersect_cols = list(set(subset_cols) & set(cols)) ## 交集字段
    #### 分箱统计
    apply_res=df.groupby(subset_cols).agg({'phone':'nunique','id':'nunique','reject_id':'sum'}).reset_index()
    apply_reject_phone=df[df['reject_id']==1].groupby(subset_cols).agg({'phone':'nunique'}).reset_index().rename(columns={'phone':'reject_phone'})### 拒绝用户数
    apply_res=pd.merge(apply_res,apply_reject_phone,how='left',on=subset_cols)
   
     #### 合计统计
    apply_res_total=df.groupby(cols).agg({'phone':'nunique','id':'nunique','reject_id':'sum'}).reset_index()
    apply_reject_phone_total=df[df['reject_id']==1].groupby(cols).agg({'phone':'nunique'}).reset_index().rename(columns={'phone':'reject_phone'})### 拒绝用户数
    apply_res_total=pd.merge(apply_res_total,apply_reject_phone_total,how='left',on=cols)
    apply_res_total[non_intersect_cols]='合计'

    apply_result=pd.concat([apply_res,apply_res_total])
    apply_result=pd.merge(apply_res,apply_res_total[intersect_cols+['phone','id']].rename(columns={'phone':'total_phone','id':'total_id'})).fillna(0)

    apply_result['reject_id_rate']=round(apply_result['reject_id']/apply_result['id'],3)# 每个组别内的拒贷率 订单维度
    apply_result['reject_phone_rate']=round(apply_result['reject_phone']/apply_result['phone'],3) # 每个组别内的拒贷率 用户维度
    apply_result['total_reject_id_rate']=round(apply_result['reject_id']/apply_result['total_id'],3) ### 每个组的整体拒贷率  订单维度
    apply_result['total_reject_phone_rate']=round(apply_result['reject_phone']/apply_result['total_phone'],3) ### 每个组的整体拒贷率  用户维度
    
    apply_result['id_rate']=round(apply_result['id']/apply_result['total_id'],3)## 每个组进入订单占整体订单的百分比  订单维度
    apply_result['phone_rate']=round(apply_result['phone']/apply_result['total_phone'],3)### 每个组进入用户占整体用户的百分比  用户维度
    return apply_result


# +
#####印度首贷规则，提取对应values数据信息
rule_data_dict={}
rule_data_dict['CHECK_USER_DUNNING_PHONE']='逾期订单'
rule_data_dict['CHECK_USER_DUNNING_KTP']='逾期订单'
rule_data_dict['CHECK_USER_DUNNING_BANKCARD']='逾期订单'
rule_data_dict['CHECK_USER_DUNNING_DEV']='逾期订单'
rule_data_dict['AGE_CHECK']='年龄'
rule_data_dict['APP_NO_SYSTEM_COUNT_CHECK']='limit'
rule_data_dict['CUSTOMER_LOAN_LIMIT']='当前在贷数'
rule_data_dict['KTP_ORDER_COUNT']='订单数'
rule_data_dict['BANKCARD_CUR_LOAN_COUNT']='在贷笔数'
rule_data_dict['DEVICE_CUR_COUNT']='在贷笔数'
rule_data_dict['DEVICE_MOBILE_COUNT']='手机号数量'
rule_data_dict['BANKCARD_MOBILE_COUNT']='手机号数量'
rule_data_dict['KTP_MOBILE_COUNT']='手机号数量'
rule_data_dict['ORDER_FINISH_INTERVAL_SMALL']='放款时间与结清时间相差天数'
rule_data_dict['ORDER_OVERDUE_FINISH_CHECK']='逾期大于0天结清订单'
rule_data_dict['CONTACT_OVERDUE_COUNT_CHECK']='通讯录逾期贷款人数'
rule_data_dict['CONTACT_LOAN_COUNT_CHECK']='通讯录申请贷款人数'
rule_data_dict['JOINT_FACE_LOAN_COUNT_CHECK']='在贷数'
rule_data_dict['JOINT_FACE_LOAN_COUNT_CUSTOMER_CHECK']='在贷数'
rule_data_dict['JOINT_FACE_PHONE_COUNT_CHECK']='手机数'
rule_data_dict['JOINT_FACE_DUNNING_COUNT_CHECK']='逾期数'
rule_data_dict['CONTACT_COUNT_CHECK']='通讯录人数'
rule_data_dict['SMS_SIMILARITY_CHECK']='相似度达到数量'
rule_data_dict['CONTACT_SIMILARITY_CHECK']='相似度达到数量'
rule_data_dict['PRE_OVERDUE_DAY']='逾期天数'

def rule_values(x,rule_name):
    for code in x.values():
        if code.get('ruleId') == rule_name:
            if rule_name in list(rule_data_dict.keys()):
                rule_value=rule_data_dict[rule_name]
                res=re.findall(f"'{rule_value}': (\w+)",str(code['values']))
                if len(res)>0:
                    return res[0]
                else:
                    return -998
            else:
                return code.get('pass')


# -
def page_numbers(dateStr,country):
    # 定义请求的数据
    data = {
    "countryId": country,
    "dateStr": dateStr,
    "pageNo": 1,
    "pageSize": 1000
    }
    # 将数据转换为 JSON 格式
    json_data = json.dumps(data)

    # 设置请求头部
    headers = {
    "Content-Type": "application/json"
    }
    # 发送 POST 请求
    response = requests.post("http://147.139.197.212:20888/api/app/query_pre_check_page", data=json_data, headers=headers)

    # 获取响应结果
    query_pre_check_page_dict = response.json()

    totalPages=query_pre_check_page_dict.get("page",{}).get("totalPages")
    return totalPages


def flow_data(dateStr,country):
    totalPages=page_numbers(dateStr,country)
    dfs=[]
    for pageNo in range(1,totalPages+1):
        data = {
        "countryId":country,
        "dateStr": dateStr,
        "pageNo": pageNo,
        "pageSize": 1000
        }
        # 将数据转换为 JSON 格式
        json_data = json.dumps(data)

        # 设置请求头部
        headers = {
        "Content-Type": "application/json"
        }
        # 发送 POST 请求
        response = requests.post("http://147.139.197.212:20888/api/app/query_pre_check_page", data=json_data, headers=headers)

        # 获取响应结果
        query_pre_check_page_dict = response.json()
        df=query_pre_check_page_dict.get("page",{}).get("content")
        dfs.extend(df)
    return dfs

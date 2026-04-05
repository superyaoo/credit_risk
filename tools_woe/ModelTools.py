# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 14:38:56 2019

@author: 神毒术士
"""
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ks_2samp
import hashlib
import os
def cal_ks(y_pred,y_true):
    '''
    KS 计算
    '''
    return ks_2samp(y_pred[y_true==1], y_pred[y_true!=1]).statistic



def MD5(original_str):
    '''
    输入 :  要加密的字符串
    输出 : (16位小写,16位大写,32位小写,32位大写)
    '''
    m = hashlib.md5()
    m.update(original_str.encode('utf-8'))
    dest_32=m.hexdigest()
    dest_16=dest_32[8:-8]
    dest_32_up=dest_32.upper()
    dest_16_up=dest_16.upper()
    dest=(dest_16,dest_16_up,dest_32,dest_32_up)
    return dest

def saveWoe(woe_dict,variables,header=None,processMiss=True,addVar=True):
    woe_result=[]
    for col in variables:
        temp = woe_dict.get(col,np.nan)
        if temp is not None:
            value=temp.copy()
            if addVar:
                value['variable']=col
            else:
                header = True
            if header is None:
                value.to_csv('temp.csv',header=None,index=False)
            else:
                value.to_csv('temp.csv',index=False)
            temp=pd.read_csv('temp.csv',header=None)
            col_len=len(temp.columns)
            empty=pd.DataFrame([[ np.nan for index in range(col_len)]],columns=list(range(col_len)))
            woe_result.append(temp)
            woe_result.append(empty)
            woe_result.append(empty)
        else:
            print('error column %s'%(col))
    woe_result=pd.concat(woe_result)
    if processMiss:
        woe_result=woe_result.replace('NAN','缺失')
    if addVar:
        woe_result.columns=['variable_cut','bad', 'good', 'badRate', 'Total', 'good_pct', 'bad_pct',
           'Total_pct', 'Odds', 'WOE', 'IV','variable_name']
        woe_result=woe_result[['variable_name','variable_cut','bad', 'good', 'badRate', 'Total', 'good_pct', 'bad_pct',
           'Total_pct', 'Odds', 'WOE', 'IV']]
    else:
        woe_result.columns=['variable_cut','bad', 'good', 'badRate', 'Total', 'good_pct', 'bad_pct',
           'Total_pct', 'Odds', 'WOE', 'IV']
        woe_result=woe_result[['variable_cut','bad', 'good', 'badRate', 'Total', 'good_pct', 'bad_pct',
           'Total_pct', 'Odds', 'WOE', 'IV']]
    os.system('rm temp.csv')
    return woe_result

def saveTrainValidWoe(train_woes,valid_woes,variables,header=None,processMiss=True,addVar=True):
    train_woe_result=[]
    valid_woe_result=[]
    for col in variables:
        temp = train_woes.get(col,np.nan)
        tmp = valid_woes.get(col,np.nan)
        if temp is not None and tmp is not None:
            value=temp.copy()
            value1=tmp.copy()
            if addVar:
                value['variable']=col
                value1['variable']=col
            else:
                header = True
            if header is None:
                value.to_csv('temp.csv',header=None,index=False)
                value1.to_csv('temp1.csv',header=None,index=False)
            else:
                value.to_csv('temp.csv',index=False)
                value1.to_csv('temp1.csv',index=False)
            temp=pd.read_csv('temp.csv',header=None)
            tmp=pd.read_csv('temp1.csv',header=None)
            col_len=len(temp.columns)
            empty=pd.DataFrame([[ np.nan for index in range(col_len)]],columns=list(range(col_len)))
            train_woe_result.append(temp)
            valid_woe_result.append(tmp)
            shape=temp.shape[0]-tmp.shape[0]
            if shape==0:
                train_empty_num=2
                valid_empty_num=2
            elif shape>0:
                train_empty_num=2
                valid_empty_num=train_empty_num+shape
            else:
                valid_empty_num=2
                train_empty_num=valid_empty_num-shape
            for index in range(train_empty_num):
                train_woe_result.append(empty)
            for index in range(valid_empty_num):
                valid_woe_result.append(empty)
        else:
            print('error column %s'%(col))
    train_woe_result=pd.concat(train_woe_result)
    valid_woe_result=pd.concat(valid_woe_result)
    if processMiss:
        train_woe_result=train_woe_result.replace('NAN','缺失')
        valid_woe_result=valid_woe_result.replace('NAN','缺失')
    if addVar:
        train_woe_result.columns=['variable_cut','bad', 'good', 'badRate', 'Total', 'good_pct', 'bad_pct',
           'Total_pct', 'Odds', 'WOE', 'IV','variable_name']
        train_woe_result=train_woe_result[['variable_name','variable_cut','bad', 'good', 'badRate', 'Total', 'good_pct', 'bad_pct',
           'Total_pct', 'Odds', 'WOE', 'IV']]
        valid_woe_result.columns=['variable_cut','bad', 'good', 'badRate', 'Total', 'good_pct', 'bad_pct',
           'Total_pct', 'Odds', 'WOE', 'IV','variable_name']
        valid_woe_result=valid_woe_result[['variable_name','variable_cut','bad', 'good', 'badRate', 'Total', 'good_pct', 'bad_pct',
           'Total_pct', 'Odds', 'WOE', 'IV']]
    else:
        train_woe_result.columns=['variable_cut','bad', 'good', 'badRate', 'Total', 'good_pct', 'bad_pct',
           'Total_pct', 'Odds', 'WOE', 'IV']
        train_woe_result=train_woe_result[['variable_cut','bad', 'good', 'badRate', 'Total', 'good_pct', 'bad_pct',
           'Total_pct', 'Odds', 'WOE', 'IV']]
        valid_woe_result.columns=['variable_cut','bad', 'good', 'badRate', 'Total', 'good_pct', 'bad_pct',
           'Total_pct', 'Odds', 'WOE', 'IV']
        valid_woe_result=valid_woe_result[['variable_cut','bad', 'good', 'badRate', 'Total', 'good_pct', 'bad_pct',
           'Total_pct', 'Odds', 'WOE', 'IV']]
    os.system('rm temp.csv')
    os.system('rm temp1.csv')
    return train_woe_result,valid_woe_result





def heatmap(df,columns,savefile=None,cmap='GnBu'):
    '''
    df 数据集
    columns 列名
    savefile 保存图片路径
    cmap 颜色样式
    '''
    df_corr=df[columns].corr()
    plt.subplots(figsize=(9,9))
    sns.heatmap(df_corr,annot=True,vmax=1,vmin=-1,square=True,cmap=cmap)
    if savefile is not None:
        plt.savefig(savefile,bbox_inches='tight')
    plt.show()

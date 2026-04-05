# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 19:07:43 2019
@author: 神毒术士
"""
import pandas as pd
import numpy as np
from .BinsWoe import getBins,getWoe,getScoreBins
class CsiTools:
    def getPSI(self,train_data,test_data,valid_data,bin_num=10,weight=None,invalid_ftr=['target'],processMissing=True,negativeMissing=True):
        
        #数据集合并
        if test_data is not None:
            train=pd.concat([train_data,test_data])
        else:
            train=train_data.copy()
        valid=valid_data.copy()
        flag='TrainValidFlag'
        train[flag]='mdl'
        valid[flag]='vld'
        
        #数据集权重处理
        if weight is not None:
            if type(weight)==str:
                train=train.rename(columns={weight:'weight'})
                valid=valid.rename(columns={weight:'weight'})
            else:
                raise '输入样本无法找到权重'
        else:
            train['weight']=train.apply(lambda x: 1,axis=1)
            valid['weight']=valid.apply(lambda x: 1,axis=1)
                 
        train_sum=train['weight'].sum()
        valid_sum=valid['weight'].sum()
        data=pd.concat([train,valid])

        data['mdl']=data.apply(lambda x: x['weight'] if x[flag]=='mdl' else 0,axis=1)
        data['vld']=data.apply(lambda x: x['weight'] if x[flag]=='vld' else 0,axis=1)
        columns=data.columns
        psi=[]
        psi_dict={}
        for column in columns:
            if column not in [flag,'mdl','vld','weight'] and column not in invalid_ftr:
                if processMissing:
                    if negativeMissing:
                        missing_data=data[(data[column].isnull())|(data[column]<0)][[column,'mdl','vld']].reset_index(drop=True)
                        normal_data=data[data[column]>=0][[column,'mdl','vld']].reset_index(drop=True)
                    else:
                        missing_data=data[data[column].isnull()][[column,'mdl','vld']].reset_index(drop=True)
                        normal_data=data[data[column]==data[column]][[column,'mdl','vld']].reset_index(drop=True)
                    if normal_data.shape[0]>0:
                        bins,right=getBins(normal_data,column,bin_num)
                        if negativeMissing:
                            if bins[1]!=0 and min(normal_data[column])>0:
                                bins=[0]+bins[1:]
                            else:
                                bins=[-0.001]+bins[1:]   
                        normal_cuts=pd.DataFrame(pd.cut(normal_data[column],bins=bins,right=right))
                        normal_cuts.columns=[column]
                        normal_data=pd.concat([normal_cuts,normal_data[['mdl','vld']]],axis=1)
                    else:
                        normal_data=pd.DataFrame(columns=[column,'mdl','vld'])
                    
                    if missing_data.shape[0]>0:
                        missing_data[column]=missing_data[column].apply(lambda x: 'NAN')
                        resul_data=pd.concat([missing_data,normal_data])
                    else:
                        resul_data=normal_data
                    temp=resul_data.groupby(column)['mdl','vld'].sum().reset_index()
                    temp['mdl_pct']=temp['mdl'].apply(lambda x: x/train_sum)
                    temp['vld_pct']=temp['vld'].apply(lambda x: x/valid_sum)
                    temp['csi']=temp.apply(lambda x: getWoe(x.mdl,train_sum,x.vld,valid_sum)*(x.mdl/train_sum-x.vld/valid_sum),axis=1)
                    temp=temp[(temp['mdl']>0)|(temp['vld']>0)].reset_index(drop=True)
                    psi.append([column,temp['csi'].sum()])
                    psi_dict[column]=temp
                else:
                    bins,right=getBins(data,column,bin_num)
                    cuts=pd.DataFrame(pd.cut(data[column],bins=bins,right=right))
                    cuts.columns=[column]
                    resul_data=pd.concat([cuts,data[['mdl','vld']]],axis=1)
                    temp=resul_data.groupby(column)['mdl','vld'].sum().reset_index()
                    temp['mdl_pct']=temp['mdl'].apply(lambda x: x/train_sum)
                    temp['vld_pct']=temp['vld'].apply(lambda x: x/valid_sum)
                    temp['csi']=temp.apply(lambda x: getWoe(x.mdl,train_sum,x.vld,valid_sum)*(x.mdl/train_sum-x.vld/valid_sum),axis=1)
                    temp=temp[(temp['mdl']>0)|(temp['vld']>0)].reset_index(drop=True)
                    psi.append([column,temp['csi'].sum()])
                    psi_dict[column]=temp        
        psi=pd.DataFrame(psi,columns=['feature','CSI'])
        psi.sort_values('CSI',ascending=False).reset_index(drop=True)
        return psi,psi_dict
    def getScorePSI(self,train_data,test_data,valid_data,bin_num=10,weight=None,invalid_ftr=['target'],negativeMissing=True):
        #数据集合并
        if test_data is not None:
            train=pd.concat([train_data,test_data])
        else:
            train=train_data
        valid=valid_data
        flag='TrainValidFlag'
        train[flag]='mdl'
        valid[flag]='vld'
        
        #数据集权重处理
        if weight is not None:
            if type(weight)==str:
                train=train.rename(columns={weight:'weight'})
                valid=valid.rename(columns={weight:'weight'})
            else:
                raise '输入样本无法找到权重'
        else:
            train['weight']=train.apply(lambda x: 1,axis=1)
            valid['weight']=valid.apply(lambda x: 1,axis=1)
        
        train_sum=train['weight'].sum()
        valid_sum=valid['weight'].sum()
        data=pd.concat([train,valid])

        data['mdl']=data.apply(lambda x: x['weight'] if x[flag]=='mdl' else 0,axis=1)
        data['vld']=data.apply(lambda x: x['weight'] if x[flag]=='vld' else 0,axis=1)
        columns=data.columns
        psi=[]
        psi_dict={}
        for column in columns:
            if column not in [flag,'mdl','vld','weight'] and column not in invalid_ftr:
                if negativeMissing:
                    missing_data=data[(data[column].isnull())|(data[column]<0)][[column,'mdl','vld']].reset_index(drop=True)
                    normal_data=data[data[column]>=0][[column,'mdl','vld']].reset_index(drop=True)
                else:
                    missing_data=data[data[column].isnull()][[column,'mdl','vld']].reset_index(drop=True)
                    normal_data=data[data[column]==data[column]][[column,'mdl','vld']].reset_index(drop=True)
                bins,right=getScoreBins(normal_data,column,bin_num)
                if bins[1]!=0 and min(normal_data[column])>0:
                    bins=[0]+bins[1:]
                else:
                    bins=[-0.001]+bins[1:]   
                normal_cuts=pd.DataFrame(pd.cut(normal_data[column],bins=bins,right=right))
                normal_cuts.columns=[column]
                normal_data=pd.concat([normal_cuts,normal_data[['mdl','vld']]],axis=1)
                
                if missing_data.shape[0]>0:
                    missing_data[column]=missing_data[column].apply(lambda x: 'NAN')
                    resul_data=pd.concat([missing_data,normal_data])
                else:
                    resul_data=normal_data
                temp=resul_data.groupby(column)['mdl','vld'].sum().reset_index()
                temp['mdl_pct']=temp['mdl'].apply(lambda x: x/train_sum)
                temp['vld_pct']=temp['vld'].apply(lambda x: x/valid_sum)
                temp['csi']=temp.apply(lambda x: getWoe(x.mdl,train_sum,x.vld,valid_sum)*(x.mdl/train_sum-x.vld/valid_sum),axis=1)
                temp=temp[(temp['mdl']>0)|(temp['vld']>0)].reset_index(drop=True)
                psi.append([column,temp['csi'].sum()])
                psi_dict[column]=temp
        psi=pd.DataFrame(psi,columns=['feature','CSI'])
        psi.sort_values('CSI',ascending=False).reset_index(drop=True)
        return psi,psi_dict
    
    def getScoreBox(self,train_data,bin_num=10,weight=None,invalid_ftr=['target'],negativeMissing=True):
            #数据集合并
            train=train_data
            #数据集权重处理
            if weight is not None:
                if type(weight)==str:
                    train=train.rename(columns={weight:'weight'})
                else:
                    raise '输入样本无法找到权重'
            else:
                train['weight']=train.apply(lambda x: 1,axis=1)
            data_sum=train['weight'].sum()
            data=train
            columns=data.columns
            result_dict={}
            for column in columns:
                if column not in ['weight'] and column not in invalid_ftr:
                    if negativeMissing:
                        missing_data=data[(data[column].isnull())|(data[column]<0)][[column,'weight']].reset_index(drop=True)
                        normal_data=data[data[column]>=0][[column,'weight']].reset_index(drop=True)
                    else:
                        missing_data=data[data[column].isnull()][[column,'weight']].reset_index(drop=True)
                        normal_data=data[data[column]==data[column]][[column,'weight']].reset_index(drop=True)
                    bins,right=getScoreBins(normal_data,column,bin_num)
                    if bins[1]!=0 and min(normal_data[column])>0:
                        bins=[0]+bins[1:]
                    else:
                        bins=[-0.001]+bins[1:]   
                    normal_cuts=pd.DataFrame(pd.cut(normal_data[column],bins=bins,right=right))
                    normal_cuts.columns=[column]
                    normal_data=pd.concat([normal_cuts,normal_data[['weight']]],axis=1)
                    
                    if missing_data.shape[0]>0:
                        missing_data[column]=missing_data[column].apply(lambda x: 'NAN')
                        resul_data=pd.concat([missing_data,normal_data])
                    else:
                        resul_data=normal_data
                    temp=resul_data.groupby(column)['weight'].sum().reset_index()
                    temp.rename(columns={'weight':'num'},inplace=True)
                    temp['num_pct']=temp['num'].apply(lambda x:x/data_sum)
                    result_dict[column]=temp
            return result_dict
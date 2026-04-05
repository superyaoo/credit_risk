# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 17:00:41 2020

@author: 神毒术士
"""
import pandas as pd
import numpy as np
from .BinsWoe import getBins,getWoe,getScoreBins
import multiprocessing
import sys

class CsiToolsWithMultiProgress:
    def __init__(self,process=True,processes=8):
        self.__cpu_count=multiprocessing.cpu_count()
        self.__platform=sys.platform
        if process:
            if self.__platform in ['linux','darwin']:
                self.__process=True
                self.__processes=min(processes,self.__cpu_count)
        else:
            self.__process=False
            self.__processes=processes
    
    def dataProcess(self,data,column,bin_num,processMissing,negativeMissing):
        '''
        processMissing   是否对缺失值单独分箱
        negativeMissing  是否将负值判断为缺失
        '''
        if data[column].dtype in [float,int]:
            if processMissing:
                if negativeMissing:
                    missing_data=data[(data[column].isnull())|(data[column]<0)][[column,'mdl','vld']].reset_index(drop=True)
                    normal_data=data[data[column]>=0][[column,'mdl','vld']].reset_index(drop=True)
                else:
                    missing_data=data[data[column].isnull()][[column,'mdl','vld']].reset_index(drop=True)
                    normal_data=data[data[column]==data[column]][[column,'mdl','vld']].reset_index(drop=True)
                if not normal_data.empty:
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

                if not missing_data.empty:
                    missing_data[column]=missing_data[column].apply(lambda x: 'NAN')
                    resul_data=pd.concat([missing_data,normal_data])
                else:
                    resul_data=normal_data
            else:
                bins,right=getBins(data,column,bin_num)
                cuts=pd.DataFrame(pd.cut(data[column],bins=bins,right=right))
                cuts.columns=[column]
                resul_data=pd.concat([cuts,data[['mdl','vld']]],axis=1)

        else:
            resul_data=data[[column,'mdl','vld']]
            if processMissing:
                resul_data[column]=resul_data[column].fillna('NAN')
            else:
                resul_data=resul_data[resul_data[column]==resul_data[column]].reset_index(drop=True)
            resul_data[column]=resul_data[column].astype(str)
        return resul_data
    def calculationPsi(self,data,column,bin_num,processMissing,negativeMissing,train_sum,valid_sum):
        resul_data=self.dataProcess(data,column,bin_num,processMissing,negativeMissing)
        #temp=resul_data.groupby(column)['mdl','vld'].sum().reset_index()
        temp=resul_data.groupby(column)[['mdl','vld']].sum().reset_index()
        temp['mdl_pct']=temp['mdl'].apply(lambda x: x/train_sum)
        temp['vld_pct']=temp['vld'].apply(lambda x: x/valid_sum)
        temp['csi']=temp.apply(lambda x: getWoe(x.mdl,train_sum,x.vld,valid_sum)*(x.mdl/train_sum-x.vld/valid_sum),axis=1)
        temp=temp[(temp['mdl']>0)|(temp['vld']>0)].reset_index(drop=True)
        return temp
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
        if self.__process:
            pool = multiprocessing.Pool(processes=self.__processes)
            executeResults={} 
            for column in columns:
                if column not in [flag,'mdl','vld','weight'] and column not in invalid_ftr:
                     executeResults[column]=pool.apply_async(func=self.calculationPsi,args=(data,column,bin_num,processMissing,negativeMissing,train_sum,valid_sum))
            pool.close()
            pool.join()
            for column,value in executeResults.items():
                temp=value.get()
                psi_dict[column]=temp
                psi.append([column,temp['csi'].sum()])
        else:
            for column in columns:
                if column not in [flag,'mdl','vld','weight'] and column not in invalid_ftr:
                    temp=self.calculationPsi(data,column,bin_num,processMissing,negativeMissing,train_sum,valid_sum)
                    psi_dict[column]=temp
                    psi.append([column,temp['csi'].sum()])
        psi=pd.DataFrame(psi,columns=['feature','CSI'])
        psi.sort_values('CSI',ascending=False).reset_index(drop=True)
        return psi,psi_dict

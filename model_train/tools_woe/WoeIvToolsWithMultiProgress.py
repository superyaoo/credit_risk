# -*- coding: utf-8 -*-
"""
Author shendushushi
date 2020-09-04
"""
import pandas as pd
import numpy as np
from sklearn.utils.multiclass import type_of_target
from .BinsWoe import getBins,getWoe,weightProcessing,getIvWoe
import multiprocessing
import sys

class WoeIvToolsWithMultiProgress:
    def __init__(self,process=True,processes=10):
        self.__cpu_count=multiprocessing.cpu_count()
        self.__platform=sys.platform
        if process:
            if self.__platform in ['linux','darwin']:
                self.__process=True
                self.__processes=min(processes,self.__cpu_count)
        else:
            self.__process=False
            self.__processes=processes
                
    def partitionData(self,data,column,processMissing,negativeMissing):
        '''
        processMissing   是否对缺失值单独分箱
        negativeMissing  是否将负值判断为缺失
        '''
        # 正常数值类型特征
        # 类型特征
        if data[column].dtype in [float,int]:
            if processMissing:
                if negativeMissing:
                    missing_data=data[(data[column].isnull())|(data[column]<0)][[column,'good','bad']].reset_index(drop=True)
                    normal_data=data[data[column]>=0][[column,'good','bad']].reset_index(drop=True)
                else:
                    missing_data=data[data[column].isnull()][[column,'good','bad']].reset_index(drop=True)
                    normal_data=data[data[column]==data[column]][[column,'good','bad']].reset_index(drop=True)
                return normal_data,missing_data
            else:
                return data[[column,'good','bad']],pd.DataFrame(columns=[column,'good','bad'])
        else:
            df=data[[column,'good','bad']]
            if processMissing:
                df[column]=data[column].fillna('NAN')
            else:
                df=df[df[column]==df[column]].reset_index(drop=True)
            df[column]=df[column].astype(str)
            return df,pd.DataFrame(columns=[column,'good','bad'])
        
    def numericalWoeMethod(self,data,column,bin_num,processMissing,negativeMissing,good_sum,bad_sum,bins,right):
        normal_data,missing_data=self.partitionData(data,column,processMissing,negativeMissing)
        if normal_data[column].dtype in [float,int]:
            if not normal_data.empty:
                if bins is None:
                    bins,right=getBins(normal_data,column,bin_num)
                    if negativeMissing:
                        if bins[1]!=0 and min(normal_data[column])>0:
                            bins=[0]+bins[1:]
                        else:
                            bins=[-0.001]+bins[1:]
                #正常数据分箱
                normal_cuts=pd.DataFrame(pd.cut(normal_data[column],bins=bins,right=right))
                normal_cuts.columns=[column]
                normal_cuts=pd.concat([normal_cuts,normal_data[['good','bad']]],axis=1)
                resul_normal_data=getIvWoe(normal_cuts,column,good_sum,bad_sum)
            else:
                resul_normal_data=pd.DataFrame(columns=[column,'bad','good','badRate','Total','good_pct','bad_pct','Odds','WOE','IV'])
            if not missing_data.empty:
                missing_data[column]=missing_data[column].apply(lambda x: 'NAN')
                resul_missing_data=getIvWoe(missing_data,column,good_sum,bad_sum)
                resul_data=pd.concat([resul_missing_data,resul_normal_data])  
            else:
                resul_data=resul_normal_data
        else:
            resul_data=getIvWoe(normal_data,column,good_sum,bad_sum)
        resul_data=resul_data.reset_index(drop=True)
        return resul_data,bins,right
    
    def getTrainIV(self,dataset,bin_num=10,target='target',weight='weight',good_event=1,processMissing=True,negativeMissing=True,bins_dict={},right_dict={}):
        '''
        processMissing   是否对缺失值单独分箱
        negativeMissing  是否将负值判断为缺失
        ''' 
        #权重处理
        data=weightProcessing(dataset,weight=weight,target=target,good_event=good_event)
        '''
        样本标签
        样本标签的值必须为 二值型
        '''
        y_type = type_of_target(data[target])
        if y_type not in ['binary']:
            raise ValueError('Label type must be binary')
            
        data['good']=data.apply(lambda x: x.weight if x[target]==good_event else 0,axis=1)
        data['bad']=data.apply(lambda x: x.weight if x[target]!=good_event else 0,axis=1)
        #好样本数量
        good_sum=data['good'].sum()
        #坏样本数量
        bad_sum=data['bad'].sum()
        iv=[]
        woes={}
        if self.__process:
            pool = multiprocessing.Pool(processes=self.__processes)
            executeResults={}
            for column in data.columns:
                if column not in [target,'weight','good','bad']:
                    bins = bins_dict.get(column)
                    right = right_dict.get(column)
                    executeResults[column]=pool.apply_async(func=self.numericalWoeMethod,args=(data,column,bin_num,processMissing,negativeMissing,good_sum,bad_sum,bins,right))
            pool.close()
            pool.join()
            for column,value in executeResults.items():
                #print("column",column)
                #print("value",value)
                resul_data,bins,right=value.get()
                woes[column]=resul_data
                iv.append([column,resul_data['IV'].sum()])
                bins_dict[column]=bins
                right_dict[column]=right
        else:
            for column in data.columns:
                if column not in [target,'weight','good','bad']:
                    bins=bins_dict.get(column)
                    right=right_dict.get(column)
                    resul_data,bins,right=self.numericalWoeMethod(data,column,processMissing,negativeMissing,good_sum,bad_sum,bins,right)
                    iv.append([column,resul_data['IV'].sum()])
                    bins_dict[column]=bins
                    right_dict[column]=right
        iv=pd.DataFrame(iv,columns=['feature','IV'])
        iv=iv.sort_values('IV',ascending=False).reset_index(drop=True)
        return iv,woes,bins_dict,right_dict
    def getIV(self,dataset,bin_num=10,target='target',weight='weight',good_event=1,processMissing=True,negativeMissing=True):
        iv,woes,bins_dict,right_dict=self.getTrainIV(dataset,bin_num,target,weight,good_event,processMissing,negativeMissing,bins_dict={},right_dict={})
        return iv,woes  
    
    def getTrainValidIV(self,train,test,valid,weight=None,bin_num=10,target='target',good_event=1,consistent=True,processMissing=True,negativeMissing=True):
        '''
        train,test,valid DataFrame,其中变量要做过二元化
        target 标签列
        good_event 好标签对应的值
        consistent 训练集合验证集的变量分箱是否保持一致
        processMissing 是否对缺失值单独处理
        negativeMissing 负值是否划分到缺失值中
        
        ''' 
        if test is not None:
            train_data=pd.concat([train,test])
        else:
            train_data=train.copy()
        valid_data=valid.copy()
        iv_train,train_woes,bins_dict,right_dict = self.getTrainIV(dataset=train_data,bin_num=bin_num,target=target,weight=weight,good_event=good_event,processMissing=processMissing,negativeMissing=negativeMissing,bins_dict={},right_dict={})                                            
        if consistent:
            iv_valid,valid_woes,bins_dict,right_dict =self.getTrainIV(dataset=valid_data,bin_num=bin_num,target=target,weight=weight,good_event=good_event,processMissing=processMissing,negativeMissing=negativeMissing,bins_dict=bins_dict,right_dict=right_dict)                                            
        else:
            iv_valid,valid_woes,bins_dict,right_dict =self.getTrainIV(dataset=valid_data,bin_num=bin_num,target=target,weight=weight,good_event=good_event,processMissing=processMissing,negativeMissing=negativeMissing,bins_dict={},right_dict={})                                            
        iv_train.rename(columns={'IV':'iv_train'},inplace=True)
        iv_valid.rename(columns={'IV':'iv_valid'},inplace=True)
        iv=pd.merge(iv_train,iv_valid,on='feature',how='left')
        iv.sort_values('iv_train',ascending=False).reset_index(drop=True)
        return iv,train_woes,valid_woes
# -*- coding: utf-8 -*-
"""
Author shendushushi
date 2019-07-24
"""
import pandas as pd
import numpy as np
from sklearn.utils.multiclass import type_of_target
from .BinsWoe import getBins,getWoe,weightProcessing,getIvWoe
class WoeIvTools:
    def getIV(self,dataset,bin_num=10,target='target',weight='weight',good_event=1,processMissing=True,negativeMissing=True):
        '''
        negativeMissing  是否将负值判断为缺失
        processMissing   是否对缺失值单独分箱
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
        for column in data.columns:
            if column not in [target,'weight','good','bad']:
                if processMissing:
                    if negativeMissing:
                        missing_data=data[(data[column].isnull())|(data[column]<0)][[column,'good','bad']].reset_index(drop=True)
                        normal_data=data[data[column]>=0][[column,'good','bad']].reset_index(drop=True)
                    else:
                        missing_data=data[data[column].isnull()][[column,'good','bad']].reset_index(drop=True)
                        normal_data=data[data[column]==data[column]][[column,'good','bad']].reset_index(drop=True)
                    if normal_data.shape[0]>0:
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
                    
                    if missing_data.shape[0]>0:
                        missing_data[column]=missing_data[column].apply(lambda x: 'NAN')
                        resul_missing_data=getIvWoe(missing_data,column,good_sum,bad_sum)
                        if resul_normal_data.shape[0]>0:
                            resul_data=pd.concat([resul_missing_data,resul_normal_data])
                        else:
                            resul_data=resul_missing_data
                    else:
                        resul_data=resul_normal_data
                    iv.append([column,resul_data['IV'].sum()])
                    woes[column]=resul_data
                else:
                    bins,right=getBins(data,column,bin_num)
                    cuts=pd.DataFrame(pd.cut(data[column],bins=bins,right=right))
                    cuts.columns=[column]
                    cuts=pd.concat([cuts,data[['good','bad']]],axis=1)
                    resul_data=getIvWoe(cuts,column,good_sum,bad_sum)
                    iv.append([column,resul_data['IV'].sum()])
                    woes[column]=resul_data     
        iv=pd.DataFrame(iv,columns=['feature','IV'])
        iv=iv.sort_values('IV',ascending=False).reset_index(drop=True)
        return iv,woes
    def TrainValidIV(self,train_data,valid_data,bin_num=10,target='target',weight='weight',good_event=1,processMissing=True,negativeMissing=True):
        
        #训练集 权重处理
        train=weightProcessing(train_data,weight=weight,target=target,good_event=good_event)
        
        #验证集权重处理
        valid=weightProcessing(valid_data,weight=weight,target=target,good_event=good_event)
        
        #标签必须是 二值型
        train_y_type = type_of_target(train[target])
        valid_y_type = type_of_target(valid[target])
        if train_y_type not in ['binary']:
            raise ValueError('Train label type must be binary')
        if valid_y_type not in ['binary']:
            raise ValueError('Valid label type must be binary')
        
        
        train['good']=train.apply(lambda x: x.weight if x[target]==good_event else 0,axis=1)
        train['bad']=train.apply(lambda x: x.weight if x[target]!=good_event else 0,axis=1)
        #好样本数量
        train_good_sum=train['good'].sum()
        #坏样本数量
        train_bad_sum=train['bad'].sum()
        

        valid['good']=valid.apply(lambda x: x.weight if x[target]==good_event else 0,axis=1)
        valid['bad']=valid.apply(lambda x: x.weight if x[target]!=good_event else 0,axis=1)
        #好样本数量
        valid_good_sum=valid['good'].sum()
        #坏样本数量
        valid_bad_sum=valid['bad'].sum()
        
        train_iv=[]
        train_woes={}
        valid_iv=[]
        valid_woes={}
        
        for column in train.columns:
            if column not in [target,'weight','good','bad']:
                if processMissing:
                    if negativeMissing:
                        missing_train=train[(train[column].isnull())|(train[column]<0)][[column,'good','bad']].reset_index(drop=True)
                        normal_train=train[train[column]>=0][[column,'good','bad']].reset_index(drop=True)
                        
                        missing_valid=valid[(valid[column].isnull())|(valid[column]<0)][[column,'good','bad']].reset_index(drop=True)
                        normal_valid=valid[valid[column]>=0][[column,'good','bad']].reset_index(drop=True)
                    else:
                        missing_train=train[train[column].isnull()][[column,'good','bad']].reset_index(drop=True)
                        normal_train=train[train[column]==train[column]][[column,'good','bad']].reset_index(drop=True)
                        
                        missing_valid=valid[valid[column].isnull()][[column,'good','bad']].reset_index(drop=True)
                        normal_valid=valid[valid[column]==valid[column]][[column,'good','bad']].reset_index(drop=True)
                    if normal_train.shape[0]>0:
                        bins,right=getBins(normal_train,column,bin_num)
                        if negativeMissing:
                            if bins[1]!=0 and min(normal_train[column])>0:
                                bins=[0]+bins[1:]
                            else:
                                bins=[-0.001]+bins[1:]

                        #训练集分箱
                        #print("column:",column)
                        train_cuts=pd.DataFrame(pd.cut(normal_train[column],bins=bins,right=right))
                        train_cuts.columns=[column]
                        train_cuts=pd.concat([train_cuts,normal_train[['good','bad']]],axis=1)
                        resul_normal_train=getIvWoe(train_cuts,column,train_good_sum,train_bad_sum)
                    else:
                        resul_normal_train=pd.DataFrame(columns=[column,'bad','good','badRate','Total','good_pct','bad_pct','Odds','WOE','IV'])
                    if missing_train.shape[0]>0:
                        missing_train[column]=missing_train[column].apply(lambda x: 'NAN')
                        resul_missing_train=getIvWoe(missing_train,column,train_good_sum,train_bad_sum)
                        if resul_normal_train.shape[0]>0:
                            resul_train=pd.concat([resul_missing_train,resul_normal_train])
                        else:
                            resul_train=resul_missing_train
                    else:
                        resul_train=resul_normal_train
                    train_iv.append([column,resul_train['IV'].sum()])
                    train_woes[column]=resul_train

                    #验证集分箱
                    if normal_train.shape[0]>0:
                        if normal_valid.shape[0]>0:
                            valid_cuts=pd.DataFrame(pd.cut(normal_valid[column],bins=bins,right=right))
                            valid_cuts.columns=[column]
                            valid_cuts=pd.concat([valid_cuts,normal_valid[['good','bad']]],axis=1)
                            resul_normal_valid=getIvWoe(valid_cuts,column,valid_good_sum,valid_bad_sum)
                        else:
                            resul_normal_valid=pd.DataFrame(columns=[column,'bad','good','badRate','Total','good_pct','bad_pct','Odds','WOE','IV'])
                    else:
                        if normal_valid.shape[0]>0:
                            bins,right=getBins(normal_valid,column,bin_num)
                            valid_cuts=pd.DataFrame(pd.cut(normal_valid[column],bins=bins,right=right))
                            valid_cuts.columns=[column]
                            valid_cuts=pd.concat([valid_cuts,normal_valid[['good','bad']]],axis=1)
                            resul_normal_valid=getIvWoe(valid_cuts,column,valid_good_sum,valid_bad_sum)
                        else:
                            resul_normal_valid=pd.DataFrame(columns=[column,'bad','good','badRate','Total','good_pct','bad_pct','Odds','WOE','IV'])
                    if missing_valid.shape[0]>0:
                        missing_valid[column]=missing_valid[column].apply(lambda x: 'NAN')
                        resul_missing_valid=getIvWoe(missing_valid,column,valid_good_sum,valid_bad_sum)
                        if resul_normal_valid.shape[0]>0:
                            resul_valid=pd.concat([resul_missing_valid,resul_normal_valid])
                        else:
                            resul_valid=resul_missing_valid
                    else:
                        resul_valid=resul_normal_valid
                    valid_iv.append([column,resul_valid['IV'].sum()])
                    valid_woes[column]=resul_valid
                else:
                    bins,right=getBins(train,column,bin_num)
                    
                    cuts=pd.DataFrame(pd.cut(train[column],bins=bins,right=right))
                    cuts.columns=[column]
                    cuts=pd.concat([cuts,train[['good','bad']]],axis=1)
                    resul_train=getIvWoe(cuts,column,train_good_sum,train_bad_sum)
                    
                    train_iv.append([column,resul_train['IV'].sum()])
                    train_woes[column]=resul_train
                    
                    cuts=pd.DataFrame(pd.cut(valid[column],bins=bins,right=right))
                    cuts.columns=[column]
                    cuts=pd.concat([cuts,valid[['good','bad']]],axis=1)
                    resul_valid=getIvWoe(cuts,column,valid_good_sum,valid_bad_sum)
                    
                    valid_iv.append([column,resul_valid['IV'].sum()])
                    valid_woes[column]=resul_valid
                    
        train_iv=pd.DataFrame(train_iv,columns=['feature','train_iv'])
        valid_iv=pd.DataFrame(valid_iv,columns=['feature','valid_iv'])
        ivs=pd.merge(train_iv,valid_iv,on='feature')
        ivs=ivs.sort_values('train_iv',ascending=False).reset_index(drop=True)
        return ivs,train_woes,valid_woes
    
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
            train_data=train
        valid_data=valid
        if consistent:
            iv,train_woes,valid_woes=self.TrainValidIV(train_data,valid_data,bin_num=bin_num,target=target,weight=weight,good_event=good_event,processMissing=processMissing,negativeMissing=negativeMissing)
        else:
            train_ivs,train_woes=self.getIV(dataset=train_data,weight=weight,target=target,bin_num=bin_num,good_event=good_event,processMissing=processMissing,negativeMissing=negativeMissing)
            valid_ivs,valid_woes=self.getIV(dataset=valid_data,weight=weight,target=target,bin_num=bin_num,good_event=good_event,processMissing=processMissing,negativeMissing=negativeMissing)
            train_ivs.rename(columns={'IV':'iv_train'},inplace=True)
            valid_ivs.rename(columns={'IV':'iv_valid'},inplace=True)
            iv=pd.merge(train_ivs,valid_ivs,on='feature',how='left')
            iv.sort_values('iv_train',ascending=False).reset_index(drop=True)
        return iv,train_woes,valid_woes
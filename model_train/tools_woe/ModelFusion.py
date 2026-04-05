# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 16:47:49 2019
@author: 神毒术士
"""
import pandas as pd
from scipy.stats import ks_2samp
from catboost import CatBoostRegressor
import multiprocessing
class ModelFusion:
    def cal_ks(self,y_pred,y_true):
        return ks_2samp(y_pred[y_true==1], y_pred[y_true!=1]).statistic
    def getResult2(self,model,x_train,y_train,x_test,y_test,x_valid,y_valid):
        model.fit(x_train,y_train,logging_level='Silent')
        predprob = pd.DataFrame(model.predict(x_train), columns = ['predprob'])
        predprob['ytrue'] = y_train.values
        train_ks=self.cal_ks(predprob['predprob'],predprob['ytrue'])
        predprob1 = pd.DataFrame(model.predict(x_test), columns = ['predprob'])
        predprob1['ytrue'] = y_test.values
        test_ks=self.cal_ks(predprob1['predprob'],predprob1['ytrue'])
        predprob2 = pd.DataFrame(model.predict(x_valid), columns = ['predprob'])
        predprob2['ytrue'] = y_valid.values
        valid_ks=self.cal_ks(predprob2['predprob'],predprob2['ytrue'])
        return [train_ks,test_ks,valid_ks]
    def Combination(self,s,num):
        if len(s)<num:
            raise '无法满足条件'
        if num==1:
            return [[x] for x in s]
        else:
            result=[]
            for index in range(len(s)-num+1):
                a=s[index:index+1]
                b=s[index+1:]
                temp=self.Combination(b,num-1)
                for x in temp:
                    result.append(a+x)
            return result
        
    def fusionModel(self,train,test,valid,target='target',most_score_num=None,least_score_num=None,processes=10):
        x_train = train.drop([target],axis=1).reset_index(drop=True)
        x_test =  test.drop([target],axis=1).reset_index(drop=True)
        x_valid = valid.drop([target],axis=1).reset_index(drop=True)
        y_train = train[[target]].reset_index(drop=True)
        y_test = test[[target]].reset_index(drop=True)
        y_valid = valid[[target]].reset_index(drop=True)
        
        all_column=list(x_train.columns)
        if len(all_column)<2:
            raise '评分数量不足'
        if most_score_num is None:
            most_score_num = len(all_column)
        if least_score_num is None:
            least_score_num = 2
        pool = multiprocessing.Pool(processes=processes)
        executeResults={}
        result={}
        for num in range(least_score_num,most_score_num+1):
            ks_result=[]
            columns=self.Combination(all_column,num)
            for cols in columns:
                model=CatBoostRegressor(iterations=500,depth=6,learning_rate=0.03,loss_function='RMSE',logging_level='Silent')
                executeResults[tuple(cols)] = pool.apply_async(func=self.getResult2,args=(model,x_train[cols],y_train,x_test[cols],y_test,x_valid[cols],y_valid))
            pool.close()
            pool.join()
            for cols,value in executeResults.items():
                temp=value.get()
                ks_result.append(list(cols)+temp)
            ks_result=pd.DataFrame(ks_result,columns=['score%d'%(x+1) for x in range(num)]+['train_ks','test_ks','valid_ks'])
            result[num]=ks_result
        return result



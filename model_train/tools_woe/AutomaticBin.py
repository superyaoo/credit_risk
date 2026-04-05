# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 09:32:29 2019
@author: 神毒术士
"""
import multiprocessing
from .WoeIvtools import WoeIvTools
def Monotonic(woes):
    '''
    强制WOE单调
    '''
    all_num=len(woes)
    result=[]
    for index in range(all_num-1):
        result.append(woes[index+1]-woes[index])
    Increment=[x for x in result if x>=0]
    Decrement=[x for x in result if x<0]
    Increment_num=len(Increment)
    Decrement_num=len(Decrement)
    if Increment_num==all_num-1 or Decrement_num==all_num-1:
        return 0
    else:
        if Increment_num>=Decrement_num:
            return -sum(Decrement)
        else:
            return sum(Increment)

def continuous(woes):
    '''
    woe 不要求单调,要求不出现倒挂
    '''
    all_num=len(woes)
    num=0
    for index in range(all_num-1):
        if woes[index+1]*woes[index]<0:
            num+=1
    if num>1 or num==0:
        return False
    else:
        return True
        
    
def forcedContinuousBinning(dataset,col,target='target',weight='weight',good_event=1,max_bin=20,min_bin=4,processMissing=True,negativeMissing=True):
    woeIvTools=WoeIvTools()
    res={}
    #分箱数量逐渐减少,直到最后woe连续
    for bin_num in range(max_bin,min_bin-1,-1):
        if weight is None:
            ivs,woes=woeIvTools.getIV(dataset[[col,target]],target=target,weight=None,bin_num=bin_num,good_event=good_event,processMissing=processMissing,negativeMissing=negativeMissing)
        elif type(weight)==str:
            ivs,woes=woeIvTools.getIV(dataset[[col,target,weight]],target=target,weight=weight,bin_num=bin_num,good_event=good_event,processMissing=processMissing,negativeMissing=negativeMissing)
        else:
            ivs,woes=woeIvTools.getIV(dataset[[col,target]],target=target,weight=weight,bin_num=bin_num,good_event=good_event,processMissing=processMissing,negativeMissing=negativeMissing)
        temp=woes[col]
        temp.rename(columns={'bad':'r Low','good':'r High','good_pct':'% High','bad_pct':'% Low','Total_pct':'% Total','IV':'I.V.','WOE':'WoE'},inplace=True)
        temp=temp[[col,'r High','r Low','Total','% High','% Low','% Total','Odds','I.V.','WoE']]
        result=Monotonic(list(temp['WoE']))
        if result==0:
            Flag=False
            break
        else:
            temp=woes[col]
            temp.rename(columns={'bad':'r Low','good':'r High','good_pct':'% High','bad_pct':'% Low','Total_pct':'% Total','IV':'I.V.','WOE':'WoE'},inplace=True)
            temp=temp[[col,'r High','r Low','Total','% High','% Low','% Total','Odds','I.V.','WoE']]
            if result not in res.keys():
                res[result]=temp
            Flag=True
    if len(res)>0 and Flag:
        key=min(list(res.keys()))
        temp=res[key]
    return temp    

def automaticBin(dataset,target='target',weight='weight',good_event=1,max_bin=20,min_bin=4,processes=10,processMissing=True,negativeMissing=True):
    if processes>1:
        pool = multiprocessing.Pool(processes=processes)
        bad_column=[]
        executeResults={}
        result={}
        cols=list(dataset.columns)
        cols.remove(target)
        if weight is not None and type(weight)==str:
            cols.remove(weight)
        for col in cols:
            executeResults[col] = pool.apply_async(func=forcedContinuousBinning,args=(dataset,col,target,weight,good_event,max_bin,min_bin,processMissing,negativeMissing))
        pool.close()
        pool.join()
        for key,value in executeResults.items():
            try:
                temp=value.get()
                result[key]=temp
                if processMissing:
                    tmp=temp[temp[key]!='NAN']
                else:
                    tmp=temp
                if Monotonic(list(tmp['WoE']))!=0 and not continuous(list(tmp['WoE'])):
                    bad_column.append(key)
            except:
                print('error columns %s'%(key))
                bad_column.append(key)
        return result,bad_column
    else:
        bad_column=[]
        executeResults={}
        result={}
        cols=list(dataset.columns)
        cols.remove(target)
        if weight is not None and type(weight)==str:
            cols.remove(weight)
        for col in cols:
            executeResults[col]=forcedContinuousBinning(dataset,col,target,weight,good_event,max_bin,min_bin,processMissing,negativeMissing)
        for key,value in executeResults.items():
            try:
                temp=value
                result[key]=temp
                if processMissing:
                    tmp=temp[temp[key]!='NAN']
                else:
                    tmp=temp
                if Monotonic(list(tmp['WoE']))!=0 and not continuous(list(tmp['WoE'])):
                    bad_column.append(key)
            except:
                print('error columns %s'%(key))
                bad_column.append(key)
        return result,bad_column
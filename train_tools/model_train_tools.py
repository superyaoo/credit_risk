import sys
import time
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import roc_auc_score,accuracy_score,classification_report
from scipy.stats import ks_2samp
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from datetime import datetime
from sklearn.preprocessing import LabelBinarizer
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn import metrics
#from matplotlib import pyplot as plt
#from sklearn_pandas import DataFrameMapper

import sys
sys.path.append('/euler/public/')
import gc

#缺失值：常量
DEFAULT = -999

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

import warnings
warnings.filterwarnings("ignore")
from openpyxl import Workbook,load_workbook

def get_train_test(data,target='target', test_size=0.3, random_state=2018):
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)
    y_train = train[target]
    X_train = train.drop([target],axis=1)
    y_test = test[target]
    X_test = test.drop([target],axis=1)
    return train, test, X_train, y_train

def modelWithCv(model, x_array, y_array, cv=5):
    model.fit(x_array, y_array)
    
    dtrain_predictions = model.predict(x_array)
    dtrain_predprob = model.predict_proba(x_array)[:,1]
    
    print("--AUC Score (Train): %f" % roc_auc_score(y_array, dtrain_predprob))
    print("--ACC Score (Train): %f" % accuracy_score(y_array, dtrain_predictions))
    print ("class metrics:")
    print (classification_report(y_array, dtrain_predictions))
    
    cv_score = cross_val_score(model,x_array, y_array, cv=cv, scoring = 'roc_auc')
    print("--CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" %(np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))

def KS_ROC_ALL(predprob_train,predprob_test,predprob_valid):
    plt.figure(figsize=(17.5,21))
    plt.subplots_adjust(wspace =0.1,hspace =0.3)
    # 训练集
    plt.subplot(3,2,1) 
    df,sub=ks_sub(predprob_train)
    plt.plot(df.probability, df.bad_cum_rate, "g-", linewidth=1)  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(df.probability, df.good_cum_rate, "b-", linewidth=1)
    x_abline = df['probability'][df['ks'] == df['ks'].max()] 
    y_abline1 = df['bad_cum_rate'][df['ks'] == df['ks'].max()]
    y_abline2 = df['good_cum_rate'][df['ks'] == df['ks'].max()]
    plt.fill_between(x_abline, y_abline1, y_abline2, color = "red",linewidth=2)
    plt.legend(title=sub,loc='lower right')
    plt.xlabel("Train probability") #X轴标签
    plt.ylabel("Cumulative percentage(%)")  # Y轴标签
    plt.title('Train KS')  # 图标题
    plt.subplot(3,2,2)
    FRPS,TPRS,fill,sub=roc_sub(predprob_train)
    plt.plot(FRPS,TPRS,'b',label="roc")
    plt.fill_between(FRPS,TPRS,fill , where=TPRS >fill, facecolor='red' , interpolate=True)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel("Train TPR")  # X轴标签
    plt.ylabel("Train FPR")  # Y轴标签
    plt.title('Train ROC')  # 图标题
    plt.legend(title=sub,loc='best')
    #测试集
    plt.subplot(3,2,3)
    df,sub=ks_sub(predprob_test)
    plt.plot(df.probability, df.bad_cum_rate, "g-", linewidth=1)  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(df.probability, df.good_cum_rate, "b-", linewidth=1)
    x_abline = df['probability'][df['ks'] == df['ks'].max()] 
    y_abline1 = df['bad_cum_rate'][df['ks'] == df['ks'].max()]
    y_abline2 = df['good_cum_rate'][df['ks'] == df['ks'].max()]
    plt.fill_between(x_abline, y_abline1, y_abline2, color = "red",linewidth=2)
    plt.legend(title=sub,loc='lower right')
    plt.xlabel("Test probability")  # X轴标签
    plt.ylabel("Cumulative percentage(%)")  # Y轴标签
    plt.title('Test KS')  # 图标题
    plt.subplot(3,2,4)
    FRPS,TPRS,fill,sub=roc_sub(predprob_test)
    plt.plot(FRPS,TPRS,'b',label="roc")
    plt.fill_between(FRPS,TPRS,fill , where=TPRS >fill, facecolor='lightgreen' , interpolate=True)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel("Test TPR")  # X轴标签
    plt.ylabel("Test FPR")  # Y轴标签
    plt.title('Test ROC')  # 图标题
    plt.legend(title=sub,loc='best')
    #验证集
    plt.subplot(3,2,5)
    df,sub=ks_sub(predprob_valid) 
    plt.plot(df.probability, df.bad_cum_rate, "g-", linewidth=1)  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(df.probability, df.good_cum_rate, "b-", linewidth=1)
    x_abline = df['probability'][df['ks'] == df['ks'].max()] 
    y_abline1 = df['bad_cum_rate'][df['ks'] == df['ks'].max()]
    y_abline2 = df['good_cum_rate'][df['ks'] == df['ks'].max()]
    plt.fill_between(x_abline, y_abline1, y_abline2, color = "red",linewidth=2)
    plt.legend(title=sub,loc='lower right')
    plt.xlabel("Valid probability")  # X轴标签
    plt.ylabel("Cumulative percentage(%)")  # Y轴标签
    plt.title('Valid KS')  # 图标题
    plt.subplot(3,2,6)
    FRPS,TPRS,fill,sub=roc_sub(predprob_valid)
    plt.plot(FRPS,TPRS,'b',label="roc")
    plt.fill_between(FRPS,TPRS,fill , where=TPRS >fill, facecolor='lightgreen' , interpolate=True)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel("Valid TPR")  # X轴标签
    plt.ylabel("Valid FPR")  # Y轴标签
    plt.title('Valid ROC')  # 图标题
    plt.legend(title=sub,loc='best')
    plt.show()
    
    
def ks(df_score, df_good,fig_dir):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    df_score = pd.DataFrame(df_score)
    df_good = pd.DataFrame(df_good) 
    df_score.columns = ['score']
    df_good.columns = ['good']
    df = pd.concat([df_score,df_good],axis=1)
    
    df['bad'] = 1 - df.good
    bin = np.arange(0, 1.001, 0.05)
    df['bucket'] = pd.cut(df.score, bin)  # 根据bin来划分区间
   
    grouped = df.groupby('bucket', as_index=False) # 统计在每个区间的样本量
    agg1 = pd.DataFrame()
    agg1['min_scr'] = grouped.min().score # 取得每个区间的最小值
    agg1['max_scr'] = grouped.max().score
    agg1['bads'] = grouped.sum().bad # 计算每个区间bad的总数量
    agg1['goods'] = grouped.sum().good
    
    agg2 = (agg1.sort_values(['min_scr'])).reset_index(drop=True) # 根据区间最小值排序
    agg2['bad_cum_rate'] = np.round((agg2.bads / df.bad.sum()).cumsum(), 4) # 计算bad样本累计概率
    agg2['good_cum_rate'] = np.round((agg2.goods / df.good.sum()).cumsum(), 4) 
    agg2['ks'] = abs(np.round(((agg2.bads / df.bad.sum()).cumsum() - (agg2.goods / df.good.sum()).cumsum()), 4)) # 计算bad和good累计概率之差的绝对值
    ks = agg2.ks.max()  # 求出ks
    
    plt.figure(figsize=(8, 4))  # 创建绘图对象
    plt.plot(agg2.min_scr, agg2.bad_cum_rate, "g-", linewidth=1)  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(agg2.min_scr, agg2.good_cum_rate, "b-", linewidth=1)
    
    x_abline = agg2['min_scr'][agg2['ks'] == agg2['ks'].max()] # ks最大的min_scr
    y_abline1 = agg2['bad_cum_rate'][agg2['ks'] == agg2['ks'].max()] # ks最大时bad_cum_rate
    y_abline2 = agg2['good_cum_rate'][agg2['ks'] == agg2['ks'].max()]
    plt.fill_between(x_abline, y_abline1, y_abline2, color = "red",linewidth=2)    
    
    sub = "%s%s"%('ks = ',ks)
    plt.legend(title=sub,loc='lower right')
    plt.xlabel("Minimum score")  # X轴标签
    plt.ylabel("Cumulative percentage(%)")  # Y轴标签
    plt.title('KS chart')  # 图标题
    plt.savefig(fig_dir)
    plt.show()  # 显示图

def cal_ks(y_pred,y_true):
    return ks_2samp(y_pred[y_true==1], y_pred[y_true!=1]).statistic
    
def ks_sub(predprob):
    for col in ['predprob','ypred','ytrue']:
        if col not in predprob.columns:
            raise "%s is not in predprob's columns"%(col)
    
    acc = accuracy_score(y_pred=predprob['ypred'], y_true=predprob['ytrue'])
    auc = roc_auc_score(y_score=predprob['predprob'], y_true=predprob['ytrue'])

    df=predprob[['predprob','ytrue']]
    df.columns=['probability','good']
    df['bad']=1-df['good']
    df=df.sort_values('probability',ascending=True)
    df=df.groupby('probability')['good','bad'].sum().reset_index()
    df=df.sort_values('probability',ascending=True)
    df['bad_cum_rate'] = df.bad.cumsum() / df.bad.sum()
    df['good_cum_rate'] = df.good.cumsum()/ df.good.sum()
    df['ks']=np.abs(df['good_cum_rate']-df['bad_cum_rate'])
    ks = df.ks.max()
    sub = "%s%s\n%s%s\n%s%s"%('KS  =  ',np.round(ks,4),'ACC = ',np.round(acc,4),'AUC = ',np.round(auc,4))
    return df,sub    

def roc_sub(predprob):
    df=predprob[['predprob','ytrue']]
    min_predprob=df['predprob'].min()
    if min_predprob>=0:
        min_predprob=0
    else:
        min_predprob=((min_predprob//0.01)+1)*0.01
    TPRS=[]
    FRPS=[]
    fill=[]
    for threshold in np.arange(min_predprob,1.01,0.01):
        df['ypred']=df['predprob'].apply(lambda x:1 if x>=threshold else 0)
        #预测为 1实际为1
        TP=df[(df['ypred']==1)&(df['ytrue']==1)].shape[0]
        #预测为 0实际为1
        FN=df[(df['ypred']==0)&(df['ytrue']==1)].shape[0]
        #预测为 1实际为0
        FP=df[(df['ypred']==1)&(df['ytrue']==0)].shape[0]
        #预测为 0实际为0
        TN=df[(df['ypred']==0)&(df['ytrue']==0)].shape[0]
        TPR=TP/(TP+FN)
        FRP=FP/(FP+TN)
        TPRS.append(TPR)
        FRPS.append(FRP)
        fill.append(0)
    AUC=np.round(roc_auc_score(y_score=df['predprob'], y_true=df['ytrue']),4)
    sub = "%s%s"%('AUC = ',AUC)
    return FRPS,TPRS,fill,sub
    
def auc_ks(model, dataframemapper, trainset, testset=None, target='target', train_only=False, with_dfm=False,types='Test'):
    
    def ks(y_pred,y_true):
        return ks_2samp(y_pred[y_true==1], y_pred[y_true!=1]).statistic
    
    if with_dfm == True:
        if train_only == False:
            predprob_test = pd.DataFrame(model.predict_proba(dataframemapper.transform(testset))[:,1], columns = ['predprob'])
            predprob_test['ytrue'] = testset[target].values
            predprob_test['ypred'] = model.predict(dataframemapper.transform(testset))
            acc_test = accuracy_score(y_pred=predprob_test['ypred'], y_true=predprob_test['ytrue'])
            auc_test = roc_auc_score(y_score=predprob_test['predprob'], y_true=predprob_test['ytrue'])
            print('AUC On {} is: {}'.format(types,auc_test))
            print('ACC On {} is: {}'.format(types,acc_test))
            print('KS  On {} is: {}'.format(types,ks(predprob_test['predprob'], predprob_test['ytrue'])))
        
        predprob_train = pd.DataFrame(model.predict_proba(dataframemapper.transform(trainset))[:,1], columns = ['predprob'])
        predprob_train['ytrue'] = trainset[target].values
        predprob_train['ypred'] = model.predict(dataframemapper.transform(trainset))
        acc_train = accuracy_score(y_pred=predprob_train['ypred'], y_true=predprob_train['ytrue'])
        auc_train = roc_auc_score(y_score=predprob_train['predprob'], y_true=predprob_train['ytrue'])
        if types=='Test':
            print('AUC On Train is: {}'.format(auc_train))
            print('ACC On Train is: {}'.format(acc_train))
            print('KS  On Train is: {}'.format(ks(predprob_train['predprob'], predprob_train['ytrue'])))
        else:
            print('AUC On Test is: {}'.format(auc_train))
            print('ACC On Test is: {}'.format(acc_train))
            print('KS  On Test is: {}'.format(ks(predprob_train['predprob'], predprob_train['ytrue'])))
        return predprob_train,predprob_test
    else:
        if train_only == False:
            predprob_test = pd.DataFrame(model.predict_proba(testset.drop([target],axis=1))[:,1], columns = ['predprob'])
            predprob_test['ytrue'] = testset[target].values
            predprob_test['ypred'] = model.predict((testset.drop([target],axis=1)))
            acc_test = accuracy_score(y_pred=predprob_test['ypred'], y_true=predprob_test['ytrue'])
            auc_test = roc_auc_score(y_score=predprob_test['predprob'], y_true=predprob_test['ytrue'])
            print('AUC On {} is: {}'.format(types,auc_test))
            print('ACC On {} is: {}'.format(types,acc_test))
            print('KS  On {} is: {}'.format(types,ks(predprob_test['predprob'], predprob_test['ytrue'])))

        predprob_train = pd.DataFrame(model.predict_proba(trainset.drop([target],axis=1))[:,1], columns = ['predprob'])
        predprob_train['ytrue'] = trainset[target].values
        predprob_train['ypred'] = model.predict((trainset.drop([target],axis=1)))
        acc_train = accuracy_score(y_pred=predprob_train['ypred'], y_true=predprob_train['ytrue'])
        auc_train = roc_auc_score(y_score=predprob_train['predprob'], y_true=predprob_train['ytrue'])
        if types=='Test':
            print('AUC On Train is: {}'.format(auc_train))
            print('ACC On Train is: {}'.format(acc_train))
            print('KS  On Train is: {}'.format(ks(predprob_train['predprob'], predprob_train['ytrue'])))
        else:
            print('AUC On Test is: {}'.format(auc_train))
            print('ACC On Test is: {}'.format(acc_train))
            print('KS  On Test is: {}'.format(ks(predprob_train['predprob'], predprob_train['ytrue'])))
#         print(predprob_train.shape)
#         print(predprob_test.shape)
        return predprob_train,predprob_test

def get_feature_importance(model, dataset, importance_type='split'):
    if importance_type == 'split':
        feature_importance = model.feature_importances_
        importance = pd.DataFrame(dataset.columns, feature_importance,columns=['feature'])
        importance.sort_index(ascending=False,inplace=True)
        return importance
    elif importance_type == 'gain':
        feature_importance = model.booster_.feature_importance(importance_type=importance_type)
        importance = pd.DataFrame(dataset.columns, feature_importance,columns=['feature'])
        importance.sort_index(ascending=False,inplace=True)
        return importance
#print (pd.sort_values(importance['feature_importance']))

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print('Model with rank: {}'.format(i))
            print('Mean valiation score: {0:.3f} (std: {1:.3f})'.format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print('Parameters: {0}'.format(results['params'][candidate]))
            print('')
            
def getKSresult(model,x_train,y_train,x_test,y_test,x_valid,y_valid):
    def getPredprob(model,x,y):
        predprob=pd.DataFrame(model.predict_proba(x)[:,1],columns=['predprob'])
        ypred=pd.DataFrame(model.predict(x),columns=['ypred'])
        ytrue=pd.DataFrame(y.values,columns=['ytrue'])
        return pd.concat([predprob,ypred,ytrue],axis=1)
    predprob_train = getPredprob(model=model,x=x_train,y=y_train)
    predprob_test = getPredprob(model=model,x=x_test,y=y_test)
    predprob_valid = getPredprob(model=model,x=x_valid,y=y_valid)
    train_ks=cal_ks(predprob_train['predprob'],predprob_train['ytrue'])
    test_ks=cal_ks(predprob_test['predprob'],predprob_test['ytrue'])
    valid_ks=cal_ks(predprob_valid['predprob'],predprob_valid['ytrue'])
    overfitting=(train_ks-test_ks)*100/train_ks
    loss=(train_ks-valid_ks)*100/train_ks
    result=pd.DataFrame([[train_ks,test_ks,valid_ks,overfitting,loss]],columns=['train_ks','test_ks','valid_ks','overfitting','loss'])
    return result,predprob_train,predprob_test,predprob_valid

def get_risk(df1, score, target, bins, weight=None):
    """
    df1: 数据
    score: 分数字段，例：'score'
    target: y标签，且1是好，0是坏
    bins: 分数分箱值
    weight: 权重,可以为空
    """
    if weight == None:
        df1['sample_weight'] = 1
    else:
        df1.rename(columns={weight:'sampe_weight'}, inplace=True)
    df = df1.copy()
    df['cuts'] = pd.cut(df[score],bins=bins, right=False)
    df['good_weight'] = df[target] * df['sample_weight']
    df['bad_weight'] = df['sample_weight'] - df['good_weight']
    grouped = df.groupby('cuts').agg({'good_weight':'sum','bad_weight':'sum'}).reset_index()
    grouped.rename(columns={'good_weight':'good','bad_weight':'bad'},inplace=True)
    grouped['badrate'] = grouped['bad']/(grouped['bad'] + grouped['good'])
    grouped['population_pct'] = (grouped['bad'] + grouped['good'])/(grouped['bad'].sum() + grouped['good'].sum())
    grouped['cumsum_pct'] = grouped['population_pct'].cumsum().iloc[::-1].cumsum().iloc[::-1]
    return grouped[['cuts', 'bad', 'good', 'badrate', 'population_pct', 'cumsum_pct']]

def new_score_pct(df):
    df['total'] = df['bad'] + df['good']
    for i in range(df.shape[0]):
        df.loc[i,'cum_bad'] = df.loc[i:]['bad'].sum()
        df.loc[i,'cum_total'] = df.loc[i:]['total'].sum()
    df['badrate_acc'] = df['cum_bad']/df['cum_total']
    df_new =df[['cuts', 'bad', 'good','total','badrate', 'badrate_acc', 'population_pct', 'cumsum_pct']]
    return df_new

def emptyData(length):
    return pd.DataFrame([np.nan for i in range(length)])

def SaveToExcel(df_list:list, row_col_list:list, excel_file, sheet_name):
    """
    excel_file：要写入的表格文件
    sheet_name：写入指定的sheet页
    df_list: 需要写入excel的df列表，
    row_col_list：df列表对应的写入位置
    例： df_tmp = df_list[0]，row_col_list[0]= [1,1],代表从第1行第1列开始，写入df_tmp的内容
         df_tmp2 = df_list[1], row_col_list[0]= [10,1],代表从第10行第1列开始，写入df_tmp2的内容
    """
    with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a') as writer:
        book = writer.book
        try:
            book.remove(book[sheet_name])
        except KeyError as e:
            for i in range(len(df_list)):
                df_list[i].to_excel(writer, sheet_name=sheet_name, index=False, startrow=row_col_list[i][0], startcol= row_col_list[i][1])
        for i in range(len(df_list)):
                df_list[i].to_excel(writer, sheet_name=sheet_name, index=False, startrow=row_col_list[i][0], startcol= row_col_list[i][1])
        writer.save()

        
def get_TrainTestValid_IV(data_list, columns_list, feature_importance):
    """
    data_list: 数据集，依照训练、测试、验证1、验证2...排序
    columns_list: 入模变量列表
    feature_importance： 模型变量重要性
    
    输出1：入模变量在各数据集上的iv，并依照重要性排序
    输出2：按照数据集顺序输出变量pattern
    """
    
    iv = pd.DataFrame(columns_list, columns=['feature'])
    woe_result_list = []
    for i in range(len(data_list))[1:]:
        iv_tmp,train_woe_tmp,valid_woe_tmp=woetools.getTrainValidIV(
        train = data_list[0][['target']+columns_list],
        test = None,
        valid = data_list[i][['target']+columns_list],
        weight = None,
        bin_num = 10,
        target = 'target',
        good_event = 1,
        consistent=True,
        processMissing=True,
        negativeMissing=True,)

        train_woe_result,valid_woe_result=saveTrainValidWoe(
        train_woes=train_woe_tmp,
        valid_woes=valid_woe_tmp,
        variables= feature_importance['feature'].to_list(),
        header=None,
        processMiss=True,
        addVar=True,
        )
        train_woe_result = train_woe_result.reset_index(drop=True)
        valid_woe_result = valid_woe_result.reset_index(drop=True)

        if i == 1:
            iv = iv.merge(iv_tmp,on='feature', how='left')
            iv.rename(columns={'iv_valid':'iv_test'}, inplace=True)
            woe_result_list.append(train_woe_result)
            woe_result_list.append(valid_woe_result)
        else:
            col_name = 'iv_valid' + str(i-1)
            iv = iv.merge(iv_tmp[['feature','iv_valid']],on='feature', how='left')
            iv.rename(columns={'iv_valid':col_name}, inplace=True)
            woe_result_list.append(valid_woe_result)
    if len(data_list) == 2:     
        iv['iv_ratio'] = iv['iv_test']/iv['iv_train']
    else:
        iv['iv_ratio'] = iv['iv_valid1']/iv['iv_train']
    iv_tmp1 = feature_importance.merge(iv, on='feature', how='inner')
    iv_tmp2 = iv[~iv['feature'].isin(list(iv_tmp1['feature']))]
    iv_final = pd.concat([iv_tmp1, iv_tmp2], axis=0)
    valid_col_list =['iv_valid%s'%str(i) for i in range(1,len(data_list[2:])+1)]
    iv_final = iv_final[['imp','feature', 'iv_train', 'iv_test'] + valid_col_list + ['iv_ratio']]
    iv_final.fillna(0, inplace=True)
    iv_final.reset_index(drop=True, inplace=True)
    iv_final.rename(columns={'iv_valid1':'iv_valid'}, inplace=True)
    
    woe_result_list_new  = []
    for woe in woe_result_list:
        woe_result_list_new.append(woe)
        woe_result_list_new.append(emptyData(len(train_woe_result)))
    woe_result = pd.concat(woe_result_list_new, axis=1)
    return iv_final, woe_result
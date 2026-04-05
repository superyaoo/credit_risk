import pandas as pd
import numpy as np
def getBins(data,column,bin_num):
    '''
    分箱函数
    data: 原始数据集
    column 需要分箱的列
    bin_num 分箱数
    '''
    try:
        rp_data,bins = pd.qcut(list(data[column]),bin_num,duplicates='drop',retbins=True)
        if len(list(bins))>2:
            bins=[-np.inf]+list(bins[1:-1])+[np.inf]
            right=True
        else:
            start=bins[0]
            end=bins[1]
            temp=data[(data[column]>start)&(data[column]<end)]
            if temp.shape[0]>0:
                rp_data,bins = pd.qcut(list(temp[column]),bin_num-2,duplicates='drop',retbins=True)
                bins=[-np.inf]+[start]+list(bins)+[end]+[np.inf]
                right=True
            else:
                bins=[-np.inf]+list(bins)+[np.inf]
                right=True
    except:
        rp_data,bins = pd.qcut(list(data[column]),1,duplicates='drop',retbins=True)
        bins=[-np.inf]+[list(bins)[0]]+[np.inf]
        right=False
    bins=sorted(set(bins))
    return bins,right
def getWoe(A,A_sum,B,B_sum,defaults=1):
    '''
    除数为0情况处理
    '''
    A_pct=A/A_sum
    B_pct=B/B_sum
    if B_pct==0:
        woe=defaults
    else:
        if A_pct==0:
            woe=-defaults
        else:
            woe=np.log(A_pct/B_pct)
    return woe

def weightProcessing(dataset,weight=None,target='target',good_event=1):
    '''
    好坏权重: 默认 [好,坏]=[1,1]
    如果好坏权重不一样,请带上权重列,并指定列名
    '''
    data=dataset
    if weight is None:
        data['weight']=data.apply(lambda x: 1,axis=1)
    elif type(weight)==list:
        if len(weight)!=2:
            data['weight']=data.apply(lambda x: 1,axis=1)
        else: 
            data['weight']=data[target].apply(lambda x: weight[0] if x==good_event else weight[1])
    elif type(weight)==str:
        if weight!='weight':
            data=data.rename(columns={weight:'weight'})
    data=data.reset_index(drop=True)
    return data
def getIvWoe(result,column,good_sum,bad_sum):
    temp=pd.DataFrame(columns=[column,'bad','good','badRate','Total','good_pct','bad_pct','Odds','WOE','IV'])
    if result.shape[0]>0:
        #print("result")
        #print(result)
        #temp=result.groupby(column)['bad','good'].sum().reset_index()
        temp=result.groupby(column)[['bad','good']].sum().reset_index()
        temp=temp[(temp['bad']>0)|(temp['good']>0)]
        if temp.shape[0]>0:
            temp['badRate']=temp.apply(lambda x: x.bad/(x.bad+x.good),axis=1)
            temp['Total']=temp.apply(lambda x: x.bad+x.good,axis=1)
            temp['good_pct']=temp['good'].apply(lambda x: x/good_sum)
            temp['bad_pct']=temp['bad'].apply(lambda x:x/bad_sum)
            temp['Total_pct']=temp['Total'].apply(lambda x: x/(good_sum+bad_sum))
            temp['Odds']=temp.apply(lambda x: x.good/x.bad if x.bad>0 else np.nan,axis=1)
            temp['WOE']=temp.apply(lambda x: getWoe(x.good,good_sum,x.bad,bad_sum),axis=1)
            temp['IV']=temp.apply(lambda x:(x.good/good_sum-x.bad/bad_sum)*x.WOE,axis=1)
            temp=temp.replace(np.inf,np.nan).replace(-np.inf,np.nan)
    return temp



def scoreBoxSingle(dataset,cut=10,score='score',target='target',weight=None,good_event=1):
    
    '''
    评分分箱 函数
    需要 [评分:score,标签:target,权重:weight] 字段
    good_event target 中 好的值   
    '''
    
    def process(df):
        df['badrate']=df.apply(lambda x: x.bad/(x.good+x.bad) if x.good+x.bad>0 else np.nan,axis=1)
        return df
    data=weightProcessing(dataset,weight=weight,target=target,good_event=good_event) 
    max_score=data[score].max()
    min_score=data[score].min()
    x=range((int(min_score)//cut+1)*cut,(int(max_score)//cut+1)*cut,cut)
    bins=[-np.inf]+list(x)+[np.inf]
    cuts=pd.DataFrame(pd.cut(data[score],bins=bins,right=False))
    cuts.columns=['cuts']
    data=pd.concat([data,cuts],axis=1)
    data['good']=data.apply(lambda x: x.weight if x[target]==good_event else 0,axis=1)
    data['bad']=data.apply(lambda x: x.weight if x[target]!=good_event else 0,axis=1)
    good_sum=data['good'].sum()
    bad_sum=data['bad'].sum()
    temp=data.groupby(['cuts'])['bad','good'].sum().reset_index()
    for col in ['bad','good']:
        temp[col]=temp[col].fillna(0)
    temp=process(temp)
    temp['population_pct']=temp.apply(lambda x:(x['good']+x['bad'])/(good_sum+bad_sum),axis=1)
    temp['cumsum_pct']=temp['population_pct'].cumsum()
    return temp   

def scoreBox(dataset,cut=10,score='score',flag='flag',train_flag='mdl',valid_flag='vld',target='target',weight=None,good_event=1):
        '''
        dataset 数据集(含训练集和验证集)
        score  评分
        cut    评分分段
        flag   训练集和验证集的字段
        train_flag flg 中训练集值
        valid_flag flg 中验证集值
        target 好坏标签字段
        weight 样本权重
        good_event target 中 好的值   
        ''' 
        #权重处理
        data=weightProcessing(dataset,weight=weight,target=target,good_event=good_event) 
        
        max_score=data[score].max()
        min_score=data[score].min()
        x=range((int(min_score)//cut+1)*cut,(int(max_score)//cut+1)*cut,cut)
        bins=[-np.inf]+list(x)+[np.inf]
        cuts=pd.DataFrame(pd.cut(data[score],bins=bins,right=False))
        cuts.columns=['cuts']
        data=pd.concat([data,cuts],axis=1)
        
        data['good']=data.apply(lambda x: x.weight if x[target]==good_event else 0,axis=1)
        data['bad']=data.apply(lambda x: x.weight if x[target]!=good_event else 0,axis=1)
        
        temp=data.groupby(['cuts',flag])['bad','good'].sum().reset_index()
        mdl=temp[temp[flag]==train_flag][['cuts','good','bad']]
        vld=temp[temp[flag]==valid_flag][['cuts','good','bad']]
        for col in ['bad','good']:
            mdl[col]=mdl[col].fillna(0)
            vld[col]=vld[col].fillna(0)
        def process(df):
            df['badrate']=df.apply(lambda x: x.bad/(x.good+x.bad) if x.good+x.bad>0 else np.nan,axis=1)
            return df
        mdl=process(mdl)
        vld=process(vld)
        return mdl,vld
    
def getScoreBins(data,column,cut=10,dynamicCut=True):
    max_score=data[column].max()
    min_score=max(0,data[column].min())
    x=range((int(min_score)//cut+1)*cut,(int(max_score)//cut+1)*cut,cut)
    if dynamicCut:
        while len(x)<5:
            cut= int(cut/2)
            x=range((int(min_score)//cut+1)*cut,(int(max_score)//cut+1)*cut,cut)
            if cut<2:
                break
        while len(x)>20:
            cut=cut*2
            x=range((int(min_score)//cut+1)*cut,(int(max_score)//cut+1)*cut,cut)
    bins=[0]+list(x)+[np.inf]
    return bins,False

def getWeightBins(dataset,cut=10,score='score',target='target',weight='weight',good_event=1): 
    def process(df):
        df['badrate']=df.apply(lambda x: x.bad/(x.good+x.bad) if x.good+x.bad>0 else np.nan,axis=1)
        return df
    data=weightProcessing(dataset,weight=weight,target=target,good_event=good_event) 
    weight_sum=data['weight'].sum()
    data=data.sort_values(score,ascending=True).reset_index(drop=True)
    data['cumsum']=data['weight'].cumsum()
    weight_bins=[-np.inf]+[weight_sum/cut*(i+1) for i in range(cut-1)]+[np.inf]
    bins=set()
    for index,weight_bin in enumerate(weight_bins[:-2]):
        bins.add(data[(data['cumsum']>=weight_bin)&(data['cumsum']<weight_bins[index+1])][score].max())
    bins=list(bins)
    bins=sorted(bins)
    bins=[-np.inf]+bins+[np.inf]
    cuts=pd.DataFrame(pd.cut(data[score],bins=bins,right=False))
    cuts.columns=['cuts']
    data=pd.concat([data,cuts],axis=1)    
    data['good']=data.apply(lambda x: x.weight if x[target]==good_event else 0,axis=1)
    data['bad']=data.apply(lambda x: x.weight if x[target]!=good_event else 0,axis=1)
    good_sum=data['good'].sum()
    bad_sum=data['bad'].sum()
    temp=data.groupby(['cuts'])['bad','good'].sum().reset_index()
    for col in ['bad','good']:
        temp[col]=temp[col].fillna(0)
    temp=process(temp)
    temp['population_pct']=temp.apply(lambda x:(x['good']+x['bad'])/(good_sum+bad_sum),axis=1)
    temp['cumsum_pct']=temp['population_pct'].cumsum()
    return temp

def badRateSequence(badRate):
    '''
    坏账率是从小到大还是从大到小
    True  从大到小
    False 从小到大
    '''
    all_num=len(badRate)
    result=[]
    for index in range(all_num-1):
        result.append(badRate[index+1]-badRate[index])
    Increment=[x for x in result if x>=0]
    Decrement=[x for x in result if x<0]
    Increment_num=len(Increment)
    Decrement_num=len(Decrement)
    if Decrement_num==all_num-1 or Decrement_num>=Increment_num:
        return True
    return False


def scoreBoxByPct(dataset,cut=10,score='score',target='target',weight='weight',good_event=1,badRateSort=True,is_int=True,process_missing=False): 
    def process(df):
        df['badrate']=df.apply(lambda x: x.bad/(x.good+x.bad) if x.good+x.bad>0 else np.nan,axis=1)
        return df
    df=weightProcessing(dataset,weight=weight,target=target,good_event=good_event) 
    data_missing=df[(df[score]<0)|(df[score].isnull())]
    data=df[df[score]>=0]
    if process_missing:
         weight_sum=df['weight'].sum()
    else:
        weight_sum=data['weight'].sum()
    data=data.sort_values(score,ascending=True).reset_index(drop=True)
    score_temp=score+'_temp'
    if is_int:
        data[score_temp]=data[score].apply(lambda x: int(x+0.5))
    else:
        data[score_temp]=data[score]
    data['cumsum']=data['weight'].cumsum()
    data_normal_weight_sum=data['weight'].sum()
    weight_bins=[-np.inf]+[data_normal_weight_sum/cut*(i+1) for i in range(cut-1)]+[np.inf]
    bins=set()
    for index,weight_bin in enumerate(weight_bins[:-2]):
        bins.add(data[(data['cumsum']>=weight_bin)&(data['cumsum']<weight_bins[index+1])][score_temp].max())
    bins=list(bins)
    bins=sorted(bins)
    bins=[-np.inf]+bins+[np.inf]
    cuts=pd.DataFrame(pd.cut(data[score],bins=bins,right=False))
    cuts.columns=['cuts']
    data=pd.concat([data,cuts],axis=1)
    data['good']=data.apply(lambda x: x.weight if x[target]==good_event else 0,axis=1)
    data['bad']=data.apply(lambda x: x.weight if x[target]!=good_event else 0,axis=1)
    temp=data.groupby(['cuts']).agg({'bad':'sum','good':'sum',score:'median'}).reset_index()
    temp.rename(columns={score:'median'},inplace=True)
    for col in ['bad','good']:
        temp[col]=temp[col].fillna(0)
    temp=process(temp)
    if process_missing:
        if not data_missing.empty:
            bad_sum=data_missing[data_missing[target]!=good_event].weight.sum()
            good_sum=data_missing[data_missing[target]==good_event].weight.sum()
            tmp=pd.DataFrame([['NAN',bad_sum,good_sum]],columns=['cuts','bad','good'])
            tmp=process(tmp)
            tmp['median']=-9999979
            temp=pd.concat([tmp,temp])
            temp=temp[(temp['bad']>0)|(temp['good']>0)]
            temp=temp.reset_index(drop=True)
    if badRateSort:
        if not badRateSequence(list(temp[temp['cuts']!='NAN']['badrate'])):
            temp=temp.sort_values('median',ascending=False).reset_index()
    temp['population_pct']=temp.apply(lambda x:(x['good']+x['bad'])/weight_sum,axis=1)
    temp['cumsum_pct']=temp['population_pct'].iloc[::-1].cumsum().iloc[::-1]
    temp=temp[['cuts','bad','good','badrate','population_pct','cumsum_pct','median']]
    return temp
import numpy as np
import pandas as pd

def filter_features_by_iv_csi(module_df, iv_csi_df, iv_train_threshold=0.01, iv_valid_threshold=0.01, 
                             ratio_threshold=0.7, csi_threshold=0.2):
    """
    Returns:
    filtered_features: 筛选后的特征列表
    """
    
    # 基础IV筛选
    iv_csi_filtered = iv_csi_df[
        (iv_csi_df['iv_train'] > iv_train_threshold) & 
        (iv_csi_df['iv_valid'] > iv_valid_threshold)
    ]
    #print(f"IV筛选后特征数量: {len(iv_filtered)}")
    # IV比率筛选
    iv_csi_filtered = iv_csi_filtered[iv_csi_filtered['ratio'] > ratio_threshold]
    #print(f"IV比率筛选后特征数量: {len(ratio_filtered)}")

    # CSI筛选
    iv_csi_filtered = iv_csi_filtered[iv_csi_filtered['CSI'] < csi_threshold]
    #print(f"CSI筛选后特征数量: {len(csi_filtered)}")
    feature_list_new = iv_csi_filtered['feature'].tolist()
     # 筛选相关性
    sample_new = module_df[feature_list_new].replace({-999: np.nan})
    for col in sample_new.columns:
        sample_new[col] = sample_new[col].fillna(sample_new[col].mode()[0])
    df_corr = sample_new.corr()   
    drop_corr_list = []
    corr_threhold = 0.8   # 阈值
    for i, col in enumerate(df_corr.columns):    
        series_col = df_corr[col][i+1:]     
        for j in range(len(series_col)):     
            if abs(series_col.iloc[j]) >= corr_threhold:
                #print(col, round(series_col.iloc[j], 4), "  ", series_col.index[j])   
                drop_corr_list.append(series_col.index[j])    
    feature_list_new = list(set(feature_list_new).difference(set(drop_corr_list)))
    
    return feature_list_new

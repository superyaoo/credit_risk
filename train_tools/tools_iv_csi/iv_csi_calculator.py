import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import warnings
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

warnings.filterwarnings("ignore")


class IVCSICalculator:
    """IV和CSI计算器类"""
    
    def __init__(self, bin_num: int = 10, good_event: int = 1, process_missing: bool = True, n_jobs: int = 1):
        """
        Parameters:
        bin_num : int, default=10
        good_event : int, default=1
        process_missing : bool, default=True
        n_jobs : int, default=1, 并行处理线程数
        """
        self.bin_num = bin_num
        self.good_event = good_event
        self.process_missing = process_missing
        self.n_jobs = n_jobs
        self._bin_edges_cache = {}    #缓存分箱边界
    
    def _preprocess_data_batch(self, data: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """批量预处理数据，避免重复操作"""
        data_processed = data.copy()
        
        if self.process_missing:
            # 向量化处理，避免逐列apply
            for col in feature_cols:
                # 处理无穷值
                data_processed[col] = data_processed[col].replace([np.inf, -np.inf], np.nan)
                # 向量化处理缺失值和负值
                mask = data_processed[col].isnull() | (data_processed[col] < 0)
                data_processed.loc[mask, col] = -999
        return data_processed
    
    def _get_bin_edges(self, data: pd.DataFrame, feature_col: str) -> Tuple[np.ndarray, bool]:
        """获取分箱边界，支持缓存"""
        cache_key = f"{feature_col}_{self.bin_num}"  #缓存key
        
        if cache_key in self._bin_edges_cache:
            return self._bin_edges_cache[cache_key]  #若缓存中存在则直接返回，避免重复计算
        
        try:
            # 等频分箱
            _, bin_edges = pd.qcut(data[feature_col], q=self.bin_num, duplicates='drop', retbins=True)
            is_qcut = True
        except:
            # 等距分箱
            _, bin_edges = pd.cut(data[feature_col], bins=self.bin_num, duplicates='drop', retbins=True)
            is_qcut = False
        # 缓存结果
        self._bin_edges_cache[cache_key] = (bin_edges, is_qcut)
        return bin_edges, is_qcut
    
    def calculate_iv_woe(self, data: pd.DataFrame, target_col: str, feature_col: str) -> Tuple[float, Dict]:
        """
        计算单个特征的IV和Woe
        data : pd.DataFrame
        target_col : str
        feature_col : str
        Returns:Tuple[float, Dict]
        """
        try:
            if self.process_missing:
                data = data.copy()
                data[feature_col] = data[feature_col].replace([np.inf, -np.inf], np.nan)
                # 向量化处理，避免apply
                mask = data[feature_col].isnull() | (data[feature_col] < 0)
                data.loc[mask, feature_col] = -999
            if len(data) == 0:
                return 0, {}
            # 分箱
            try:
                # 等频分箱
                data['bin'] = pd.qcut(data[feature_col], q=self.bin_num, duplicates='drop')  #等频分箱
            except:
                data['bin'] = pd.cut(data[feature_col], bins=self.bin_num, duplicates='drop')  #等距分箱
            
            # 统计各箱
            bin_stats = data.groupby('bin').agg({
                target_col: ['count', 'sum']
            }).reset_index()
            bin_stats.columns = ['bin', 'total', 'bad']
            bin_stats['good'] = bin_stats['total'] - bin_stats['bad']
            bin_stats = bin_stats[bin_stats['total'] > 0]  #避免除零
            if len(bin_stats) == 0:
                return 0, {}
            
            # 计算WOE和IV
            total_good = bin_stats['good'].sum()
            total_bad = bin_stats['bad'].sum()
            if total_good == 0 or total_bad == 0:
                return 0, {}  #避免除零
            
            bin_stats['good_pct'] = bin_stats['good'] / total_good
            bin_stats['bad_pct'] = bin_stats['bad'] / total_bad
            bin_stats['good_pct'] = bin_stats['good_pct'].replace(0, 0.0001)
            bin_stats['bad_pct'] = bin_stats['bad_pct'].replace(0, 0.0001)
            
            bin_stats['woe'] = np.log(bin_stats['bad_pct'] / bin_stats['good_pct'])
            bin_stats['iv'] = (bin_stats['bad_pct'] - bin_stats['good_pct']) * bin_stats['woe']
            
            total_iv = bin_stats['iv'].sum()
            
            # 构建WOE字典
            woe_dict = {}
            for _, row in bin_stats.iterrows():
                woe_dict[str(row['bin'])] = row['woe']
            return total_iv, woe_dict
        except Exception as e:
            print(f"计算特征 {feature_col} 的IV时出错: {str(e)}")
            return 0, {}
    
    def calculate_csi(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, feature_col: str) -> float:
        """
        计算单个特征的CSI
        Parameters:
        train_data : pd.DataFrame
        valid_data : pd.DataFrame
        feature_col : str
        Returns:float：CSI值
        """
        try:
            # 处理缺失值
            if self.process_missing:
                train_data = train_data.copy()
                valid_data = valid_data.copy()
                train_data[feature_col] = train_data[feature_col].replace([np.inf, -np.inf], np.nan)
                valid_data[feature_col] = valid_data[feature_col].replace([np.inf, -np.inf], np.nan)
                train_data[feature_col] = train_data[feature_col].apply(lambda x:-999 if pd.isnull(x) or x < 0 else x)
                valid_data[feature_col] = valid_data[feature_col].apply(lambda x:-999 if pd.isnull(x) or x < 0 else x)
            if len(train_data) == 0 or len(valid_data) == 0:
                return 0
            
            # 分箱
            try:
                train_data['bin'] = pd.qcut(train_data[feature_col], q=self.bin_num, duplicates='drop')  #等频分箱
                bin_edges = train_data['bin'].cat.categories
            except:
                train_data['bin'] = pd.cut(train_data[feature_col], bins=self.bin_num, duplicates='drop')  #否则等距分箱
                bin_edges = train_data['bin'].cat.categories
            
            try:
                valid_data['bin'] = pd.cut(valid_data[feature_col], bins=bin_edges, duplicates='drop')  #使用训练集的分箱边界
            except:
                return 0
            
            # 统计各箱
            train_dist = train_data['bin'].value_counts(normalize=True).sort_index()
            valid_dist = valid_data['bin'].value_counts(normalize=True).sort_index()
            all_bins = train_dist.index.union(valid_dist.index)
            train_dist = train_dist.reindex(all_bins, fill_value=0)
            valid_dist = valid_dist.reindex(all_bins, fill_value=0)
            
            # 计算CSI
            csi = 0
            for bin_name in all_bins:
                if train_dist[bin_name] > 0:
                    csi += (valid_dist[bin_name] - train_dist[bin_name]) * np.log(valid_dist[bin_name] / train_dist[bin_name])  
            return csi
        except Exception as e:
            print(f"计算特征 {feature_col} 的CSI时出错: {str(e)}")
            return 0
    
    def calculate_train_valid_iv(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, 
                                target_col: str = 'label', feature_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        计算训练集和验证集的IV值
        Parameters:
        train_data : pd.DataFrame
        valid_data : pd.DataFrame
        target_col : str, default='label'
        feature_cols : List[str], optional 如果为None则使用除target_col外的所有列
        Returns:Tuple[pd.DataFrame, Dict, Dict](IV结果DataFrame, 训练集WOE字典, 验证集WOE字典)
        """
        if feature_cols is None:
            feature_cols = [col for col in train_data.columns if col != target_col]
        iv_results = []
        train_woes = {}
        valid_woes = {}
        
        for col in feature_cols:
            print(f"计算特征: {col}")  
            # 计算训练集IV和WOE
            train_iv, train_woe_dict = self.calculate_iv_woe(train_data, target_col, col)
            # 计算验证集IV和WOE
            valid_iv, valid_woe_dict = self.calculate_iv_woe(valid_data, target_col, col)
            # 保存结果
            iv_results.append({
                'feature': col,
                'iv_train': train_iv,
                'iv_valid': valid_iv
            })
            
            # 保存WOE字典
            train_woes[col] = train_woe_dict
            valid_woes[col] = valid_woe_dict
        
        return pd.DataFrame(iv_results), train_woes, valid_woes
    
    def calculate_csi_batch(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, 
                           feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        批量计算CSI值
        Parameters:
        train_data : pd.DataFrame
        valid_data : pd.DataFrame
        feature_cols : List[str], optional
        Returns:pd.DataFrame：CSI结果DataFrame
        """
        if feature_cols is None:
            feature_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
        
        csi_results = []
        
        for col in feature_cols:
            print(f"计算特征CSI: {col}")
            csi_value = self.calculate_csi(train_data, valid_data, col)
            
            csi_results.append({
                'feature': col,
                'CSI': csi_value
            })
        
        return pd.DataFrame(csi_results)
    
    def calculate_iv_csi_batch(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, 
                              target_col: str = 'label', feature_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
        """
        批量计算IV和CSI值
        Returns: Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]
            (IV结果DataFrame, CSI结果DataFrame, 训练集WOE字典, 验证集WOE字典)
        """
        if feature_cols is None:
            feature_cols = [col for col in train_data.columns if col != target_col]
        
        # 计算IV
        iv_df, train_woes, valid_woes = self.calculate_train_valid_iv(
            train_data, valid_data, target_col, feature_cols
        )
        
        # 计算CSI
        csi_df = self.calculate_csi_batch(train_data, valid_data, feature_cols)
        
        return iv_df, csi_df, train_woes, valid_woes
    
    def _calculate_single_feature_iv(self, args) -> Tuple[str, float, Dict]:
        """计算单个特征的IV（用于并行处理）"""
        data, target_col, feature_col = args
        iv, woe_dict = self.calculate_iv_woe(data, target_col, feature_col)
        return feature_col, iv, woe_dict
    
    def _calculate_single_feature_csi(self, args) -> Tuple[str, float]:
        """计算单个特征的CSI（用于并行处理）"""
        train_data, valid_data, feature_col = args
        csi = self.calculate_csi(train_data, valid_data, feature_col)
        return feature_col, csi
    
    def calculate_train_valid_iv_optimized(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, 
                                          target_col: str = 'label', feature_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict, Dict]:
        """优化版批量计算IV - 避免重复分箱，提升计算速度"""
        if feature_cols is None:
            feature_cols = [col for col in train_data.columns if col != target_col]
        
        # 批量预处理数据
        print("批量预处理数据...")
        train_processed = self._preprocess_data_batch(train_data, feature_cols)
        valid_processed = self._preprocess_data_batch(valid_data, feature_cols)
        
        # 准备并行计算参数
        train_args = [(train_processed, target_col, col) for col in feature_cols]
        valid_args = [(valid_processed, target_col, col) for col in feature_cols]
        
        iv_results = []
        train_woes = {}
        valid_woes = {}
        
        if self.n_jobs > 1:
            # 并行计算
            print(f"使用 {self.n_jobs} 个线程并行计算...")
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                # 训练集IV
                train_results = list(executor.map(self._calculate_single_feature_iv, train_args))
                # 验证集IV
                valid_results = list(executor.map(self._calculate_single_feature_iv, valid_args))
        else:
            # 串行计算
            print("串行计算...")
            train_results = [self._calculate_single_feature_iv(args) for args in train_args]
            valid_results = [self._calculate_single_feature_iv(args) for args in valid_args]
        
        # 整理结果
        for (col, train_iv, train_woe_dict), (_, valid_iv, valid_woe_dict) in zip(train_results, valid_results):
            iv_results.append({
                'feature': col,
                'iv_train': train_iv,
                'iv_valid': valid_iv
            })
            train_woes[col] = train_woe_dict
            valid_woes[col] = valid_woe_dict
        
        return pd.DataFrame(iv_results), train_woes, valid_woes
    
    def calculate_csi_batch_optimized(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, 
                                     feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """优化版批量计算CSI - 避免重复分箱，提升计算速度"""
        if feature_cols is None:
            feature_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 批量预处理数据
        print("批量预处理数据...")
        train_processed = self._preprocess_data_batch(train_data, feature_cols)
        valid_processed = self._preprocess_data_batch(valid_data, feature_cols)
        
        # 准备并行计算参数
        csi_args = [(train_processed, valid_processed, col) for col in feature_cols]
        
        csi_results = []
        
        if self.n_jobs > 1:
            # 并行计算
            print(f"使用 {self.n_jobs} 个线程并行计算CSI...")
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                results = list(executor.map(self._calculate_single_feature_csi, csi_args))
        else:
            # 串行计算
            print("串行计算CSI...")
            results = [self._calculate_single_feature_csi(args) for args in csi_args]
        
        # 整理结果
        for col, csi_value in results:
            csi_results.append({
                'feature': col,
                'CSI': csi_value
            })
        
        return pd.DataFrame(csi_results)
    
    def calculate_iv_csi_batch_optimized(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, 
                                        target_col: str = 'label', feature_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
        """优化版批量计算IV和CSI - 避免重复分箱，提升计算速度"""
        if feature_cols is None:
            feature_cols = [col for col in train_data.columns if col != target_col]
        
        print("开始优化版批量计算...")
        
        # 计算IV
        print("计算IV...")
        iv_df, train_woes, valid_woes = self.calculate_train_valid_iv_optimized(
            train_data, valid_data, target_col, feature_cols
        )
        
        # 计算CSI
        print("计算CSI...")
        csi_df = self.calculate_csi_batch_optimized(train_data, valid_data, feature_cols)
        
        print("计算完成！")
        return iv_df, csi_df, train_woes, valid_woes
    
    def calculate_iv_csi_ultra_fast(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, 
                                   target_col: str = 'label', feature_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
        """超高速批量计算IV和CSI - 最小化数据复制和重复计算"""
        if feature_cols is None:
            feature_cols = [col for col in train_data.columns if col != target_col]
        
        print("开始超高速批量计算...")
        
        # 一次性预处理所有数据
        print("批量预处理数据...")
        train_processed = self._preprocess_data_batch(train_data, feature_cols)
        valid_processed = self._preprocess_data_batch(valid_data, feature_cols)
        
        # 并行计算所有特征
        if self.n_jobs > 1:
            print(f"使用 {self.n_jobs} 个线程并行计算...")
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                # 准备参数
                train_args = [(train_processed, target_col, col) for col in feature_cols]
                valid_args = [(valid_processed, target_col, col) for col in feature_cols]
                csi_args = [(train_processed, valid_processed, col) for col in feature_cols]
                
                # 并行计算
                train_results = list(executor.map(self._calculate_single_feature_iv, train_args))
                valid_results = list(executor.map(self._calculate_single_feature_iv, valid_args))
                csi_results = list(executor.map(self._calculate_single_feature_csi, csi_args))
        else:
            print("串行计算...")
            train_results = [self._calculate_single_feature_iv((train_processed, target_col, col)) for col in feature_cols]
            valid_results = [self._calculate_single_feature_iv((valid_processed, target_col, col)) for col in feature_cols]
            csi_results = [self._calculate_single_feature_csi((train_processed, valid_processed, col)) for col in feature_cols]
        
        # 整理结果
        print("整理结果...")
        iv_results = []
        csi_results_list = []
        train_woes = {}
        valid_woes = {}
        
        for (col, train_iv, train_woe_dict), (_, valid_iv, valid_woe_dict), (_, csi_value) in zip(train_results, valid_results, csi_results):
            iv_results.append({
                'feature': col,
                'iv_train': train_iv,
                'iv_valid': valid_iv
            })
            csi_results_list.append({
                'feature': col,
                'CSI': csi_value
            })
            train_woes[col] = train_woe_dict
            valid_woes[col] = valid_woe_dict
        
        iv_df = pd.DataFrame(iv_results)
        csi_df = pd.DataFrame(csi_results_list)
        
        print("计算完成！")
        return iv_df, csi_df, train_woes, valid_woes
    
    def calculate_iv_csi_memory_efficient(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, 
                                        target_col: str = 'label', feature_cols: Optional[List[str]] = None,
                                        batch_size: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
        """内存高效的批量计算IV和CSI - 分批处理大量特征"""
        if feature_cols is None:
            feature_cols = [col for col in train_data.columns if col != target_col]
        
        print(f"开始内存高效批量计算，特征数: {len(feature_cols)}, 批次大小: {batch_size}")
        
        # 一次性预处理数据
        print("批量预处理数据...")
        train_processed = self._preprocess_data_batch(train_data, feature_cols)
        valid_processed = self._preprocess_data_batch(valid_data, feature_cols)
        
        all_iv_results = []
        all_csi_results = []
        all_train_woes = {}
        all_valid_woes = {}
        
        # 分批处理特征
        for i in range(0, len(feature_cols), batch_size):
            batch_features = feature_cols[i:i+batch_size]
            print(f"处理批次 {i//batch_size + 1}/{(len(feature_cols)-1)//batch_size + 1}, 特征数: {len(batch_features)}")
            
            # 并行计算当前批次
            if self.n_jobs > 1:
                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    # 准备参数
                    train_args = [(train_processed, target_col, col) for col in batch_features]
                    valid_args = [(valid_processed, target_col, col) for col in batch_features]
                    csi_args = [(train_processed, valid_processed, col) for col in batch_features]
                    
                    # 并行计算
                    train_results = list(executor.map(self._calculate_single_feature_iv, train_args))
                    valid_results = list(executor.map(self._calculate_single_feature_iv, valid_args))
                    csi_results = list(executor.map(self._calculate_single_feature_csi, csi_args))
            else:
                train_results = [self._calculate_single_feature_iv((train_processed, target_col, col)) for col in batch_features]
                valid_results = [self._calculate_single_feature_iv((valid_processed, target_col, col)) for col in batch_features]
                csi_results = [self._calculate_single_feature_csi((train_processed, valid_processed, col)) for col in batch_features]
            
            # 整理当前批次结果
            for (col, train_iv, train_woe_dict), (_, valid_iv, valid_woe_dict), (_, csi_value) in zip(train_results, valid_results, csi_results):
                all_iv_results.append({
                    'feature': col,
                    'iv_train': train_iv,
                    'iv_valid': valid_iv
                })
                all_csi_results.append({
                    'feature': col,
                    'CSI': csi_value
                })
                all_train_woes[col] = train_woe_dict
                all_valid_woes[col] = valid_woe_dict
            
            # 清理内存
            del train_results, valid_results, csi_results
        
        iv_df = pd.DataFrame(all_iv_results)
        csi_df = pd.DataFrame(all_csi_results)
        
        print("计算完成！")
        return iv_df, csi_df, all_train_woes, all_valid_woes
    
    def clear_cache(self):
        """清理缓存"""
        self._bin_edges_cache.clear()
        print("缓存已清理")
    
    def get_cache_info(self):
        """获取缓存信息"""
        return {
            'bin_edges_cache_size': len(self._bin_edges_cache)
        }

import glob
import threading
from pathlib import Path
from typing import Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
from tqdm import tqdm
import pandas as pd

def read_parquet_file(file_path: Path) -> Union[pd.DataFrame, None]:
    try:
        return pd.read_parquet(file_path)
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {str(e)}")
        return None

def concat_parquet_files(file_list, max_workers=10) -> pd.DataFrame:
    dfs = []
    total_files = len(file_list)
    
    if total_files == 0:
        raise ValueError(f"没有传入任何parquet文件")
    
    dfs_lock = threading.Lock()
    pbar = tqdm(total=total_files, desc=f'正在加载parquet文件')
    
    def process_file(file_path):
        df = read_parquet_file(file_path)
        if df is not None:
            with dfs_lock:
                dfs.append(df)
        pbar.update(1)
        return df is not None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_file, file): file 
                         for file in file_list}
        completed_files = 0
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                if future.result():
                    completed_files += 1
            except Exception as e:
                print(f"处理文件 {file} 时出错: {str(e)}")
    
    pbar.close()
    
    if not dfs:
        raise ValueError(f"没有找到有效的parquet文件")
    
    result = pd.concat(dfs, ignore_index=True)
    print(f"成功合并 {len(dfs)} 个文件，总行数: {len(result)}")
    
    return result
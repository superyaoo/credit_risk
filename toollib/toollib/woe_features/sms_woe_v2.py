import json
import traceback
from pathlib import Path

import pandas as pd
from nltk import word_tokenize
from pythainlp.tokenize import word_tokenize as thai_word_tokenize
from tqdm import tqdm
from collections import defaultdict

from toollib import token_woe_level, batch_load_data
from toollib.unversal import parallel_process, data_of_dir, ngrams, time_trans, COUNTRY_LANGUAGE


def chunk_stats_task(file_path, def_col, agr_col, language, n_gram, data_intervals, chunk_stats_dir, country_abbr,
                     order_base_dir):
    try:
        df = pd.read_parquet(file_path)
        file_name = file_path.split('/')[-1]
        df = df[df[agr_col] == 1]
        if len(df) == 0:
            return

        df = df[['app_order_id', 'apply_time', 'sms_data', 'random_no', def_col]]
        df = df[df['sms_data'].notnull()]

        df['apply_time'] = pd.to_datetime(df['apply_time'])
        df['sms_data'] = df['sms_data'].map(json.loads)
        df = df[df['sms_data'].map(lambda x: len(x) > 0)]
        user_sms = df.explode('sms_data').reset_index(drop=True)

        df = df[['app_order_id', 'apply_time', 'random_no', def_col]]
        df.to_parquet(order_base_dir / file_name, compression='zstd')

        user_sms = pd.concat([user_sms, pd.json_normalize(user_sms['sms_data'])], axis=1)
        user_sms['send_date'] = user_sms['time'].map(lambda x: time_trans(x, country_abbr)).dt.date
        user_sms['apply_date'] = user_sms['apply_time'].dt.date
        user_sms['date_diff'] = (user_sms['apply_date'] - user_sms['send_date']).map(lambda x: x.days)
        user_sms = user_sms[user_sms['body'].notnull()]
        order_sms_count = user_sms.groupby('app_order_id')['sms_data'].count()

        user_sms['words'] = user_sms['body'].map(
            lambda x: thai_word_tokenize(x) if language == 'thai' else word_tokenize(x, language=language))
        user_sms = user_sms.drop(columns=['sms_data'])
        for i in range(1, n_gram + 1):
            gram_col = f'gram{i}'
            user_sms[gram_col] = user_sms['words'].map(lambda x: ngrams(x, i))
            for data_interval in data_intervals:

                words_exploded = user_sms[user_sms['date_diff'] <= data_interval][
                    ['app_order_id', gram_col, def_col]].explode(gram_col)
                word_grouped = words_exploded.groupby(['app_order_id', gram_col])[[def_col]].count().reset_index()
                word_grouped.rename(columns={def_col: 'word_count'}, inplace=True)
                word_grouped = word_grouped.merge(df[['app_order_id', def_col, 'random_no']], on='app_order_id')
                word_grouped['sms_count'] = word_grouped['app_order_id'].map(order_sms_count)
                word_grouped['weight_count'] = word_grouped['word_count'] / word_grouped['sms_count']
                word_stats_info = pd.pivot_table(word_grouped, index=['random_no', gram_col], columns=def_col,
                                                 values=['word_count', 'weight_count', 'app_order_id'],
                                                 aggfunc={'word_count': 'sum', 'weight_count': 'sum',
                                                          'app_order_id': 'count'}, fill_value=0)
                word_stats_info.columns = [f'{k}_{v}' for k, v in word_stats_info.columns]
                word_stats_info = word_stats_info.reset_index()
                for random_no in word_stats_info['random_no'].unique():
                    save_name = f"{gram_col}_{data_interval}d_{random_no}_{file_name}"
                    tmp = word_stats_info[word_stats_info['random_no'] == random_no]
                    tmp.to_parquet(chunk_stats_dir / save_name, compression='zstd')
    except:
        print(f"{file_path} 计算失败,{traceback.format_exc()}")


def chunk_merge_task(task_name, task_files, chunk_merged_dir):
    stats_df = None
    gram_col = task_name.split('_')[0]
    for file in task_files:
        party_df = pd.read_parquet(file)
        if stats_df is None:
            stats_df = party_df
        else:
            stats_df = pd.concat([stats_df, party_df], axis=0)
            stats_df = stats_df.groupby(['random_no', gram_col]).sum().reset_index()
    stats_df.to_parquet(chunk_merged_dir / f'{task_name}.parquet', compression='zstd')


def word_select(df, random_stats_df, random_no, def_col, agr_col, gram_col, date_diff):
    random_order_dict = random_stats_df[random_stats_df['random_no'] == random_no].to_dict(orient='records')[0]
    bad_total = random_order_dict[def_col]
    order_total = random_order_dict[agr_col]
    brate_total = bad_total / order_total
    df['brate'] = df['app_order_id_1'] / (df['app_order_id_1'] + df['app_order_id_0'])
    df['lift'] = df['brate'] / brate_total
    df['weight_count'] = df['weight_count_1'] + df['weight_count_0']
    df['order_pct'] = (df['app_order_id_1'] + df['app_order_id_0']) / order_total

    # 截取权重最高的0.05
    weight_count_1_limit = df['weight_count_1'].quantile(0.95)
    weight_count_0_limit = df['weight_count_0'].quantile(0.95)

    condition1 = (df['lift'] > 1.1) | (df['lift'] < 0.9)
    condition2 = (df['weight_count_1'] > weight_count_1_limit) | (df['weight_count_0'] > weight_count_0_limit)
    condition3 = df['order_pct'] > 0.03
    filter_df = df[condition1 & condition2 & condition3]
    lower_tokens = filter_df.head(5000)[gram_col].to_list()
    upper_tokens = filter_df.tail(5000)[gram_col].to_list()
    most_tokens = filter_df.sort_values('order_pct', ascending=True).tail(5000)[gram_col].to_list()
    select_tokens = set(lower_tokens + upper_tokens + most_tokens)
    filter_df = filter_df[filter_df[gram_col].isin(select_tokens)]

    low_lift_df = filter_df[filter_df['lift'] < 0.9].sort_values('lift')
    up_lift_df = filter_df[filter_df['lift'] > 1.1].sort_values('lift')
    qcut_n = 5
    low_lift_df['bin'], bi = pd.qcut(low_lift_df['lift'], qcut_n, duplicates='drop', retbins=True)
    low_lift_df['woe_level'] = low_lift_df['bin'].cat.codes
    low_lift_df['bin'] = low_lift_df['bin'].astype(str)

    up_lift_df['bin'], bi = pd.qcut(up_lift_df['lift'], qcut_n, duplicates='drop', retbins=True)
    up_lift_df['woe_level'] = up_lift_df['bin'].cat.codes + qcut_n
    up_lift_df['bin'] = up_lift_df['bin'].astype(str)
    token_selected_df = pd.concat([low_lift_df, up_lift_df])
    token_selected_df['random_no'] = random_no
    token_selected_df['gram'] = gram_col
    token_selected_df['token'] = token_selected_df[gram_col]
    token_selected_df['date_diff'] = date_diff
    return token_selected_df[['gram', 'date_diff', 'random_no', 'token', 'lift', 'woe_level']]


def random_merge_select_task(file_name, file_paths, save_dir, random_stats_df, def_col, agr_col):
    df = None
    gram_col = file_name.split('_')[0]
    date_diff = file_name.split('_')[1]
    random_no = file_name.removesuffix('.parquet').split('_')[2]
    for file_path in file_paths:
        part_df = pd.read_parquet(file_path)
        part_df.drop(columns='random_no', inplace=True)
        if df is None:
            df = part_df
        else:
            df = pd.concat([df, part_df], axis=0)
            df = df.groupby(gram_col).sum().reset_index()
    if len(df) > 0:
        data = word_select(df, random_stats_df, random_no, def_col, agr_col, gram_col, date_diff)
        data.to_parquet(save_dir / file_name, compression='zstd')


def order_stats_info(order_info_dir, def_col, agr_col):
    order_info_files = data_of_dir(order_info_dir)
    order_info = None
    for file in tqdm(order_info_files, desc='order_info_merge'):
        part_df = pd.read_parquet(file)
        if order_info is None:
            order_info = part_df
        else:
            order_info = pd.concat([order_info, part_df], axis=0)
    order_info[agr_col] = 1
    random_info = order_info.groupby('random_no').agg({
        def_col: 'sum',
        agr_col: 'sum'
    }).reset_index()
    stats_arr = []
    stats_result = random_info.sum().to_dict()
    stats_result['random_no'] = 'all'
    stats_arr.append(stats_result)
    for no in random_info['random_no'].unique():
        tmp = random_info[random_info['random_no'] != no]
        stats_result = tmp.sum().to_dict()
        stats_result['random_no'] = no
        stats_arr.append(stats_result)
    random_stats_df = pd.DataFrame(stats_arr)
    random_stats_df['random_no'] = random_stats_df['random_no'].astype(str)
    return order_info, random_stats_df


def sms_woe_level2(woe_raw_files, country_abbr, n_gram=2, def_col='def_pd7', agr_col='agr_pd7',
                   data_intervals=None, chunk_task_workers=30, random_merge_workers=10,
                   save_dir='sms_woe'):
    """根据固定格式的文件生成sms的配置文件。每个文件块需要以下几列数据 ['app_order_id','apply_time','sms_data',def_col,agr_col]
        注意: sms_data 以json格式的字符串保存。 def_col 为标签的分子(bad=1，good=0)，agr_col 为标签分母（即到达表现期数为1，未到达表现期数0）
        woe_raw_files:带路径的文件，可以将数据切片放到指定的目录了然后调用本包的 data_of_dir 扫描出来
        country_abbr:国家编码
        n_gram：分词的gram数
        def_col:为标签的分子(bad=1，good=0)
        agr_col:为标签分母（即到达表现期数为1，未到达表现期数0）
        data_intervals:提取标签的时间组合默认[7, 14, 30, 99999]
        chunk_task_workers:处理切片数据的并发进程数量，如果切片的较为碎，建议多开一点打满cpu
        random_merge_workers:后面reduce任务的并发数，拉美的机器如果较为空闲建议开5-10个
    """

    if data_intervals is None:
        data_intervals = [7, 14, 30, 99999]

    language = COUNTRY_LANGUAGE[country_abbr]
    print(f"国家 {country_abbr},分词器语种 {language}")

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    chunk_stats_dir = save_dir / 'chunk_stats_dir'
    chunk_stats_dir.mkdir(exist_ok=True)

    order_info_dir = save_dir / 'order_base_dir'
    order_info_dir.mkdir(exist_ok=True)

    chunk_stats_task_list = [
        (x, def_col, agr_col, language, n_gram, data_intervals, chunk_stats_dir, country_abbr, order_info_dir) for x in
        woe_raw_files]
    parallel_process(chunk_stats_task, chunk_stats_task_list, process_num=chunk_task_workers,
                     tqdm_desc='stats info party')

    order_info, random_stats_df = order_stats_info(order_info_dir, def_col, agr_col)
    order_info.to_parquet(save_dir / 'order_random_no.parquet',compression='zstd')

    chunk_stats_file = data_of_dir(chunk_stats_dir)
    merge_task_map = defaultdict(list)
    for x in chunk_stats_file:
        k = "_".join(x.split('/')[-1].split('_')[0:3])
        merge_task_map[k].append(x)

    chunk_merged_dir = save_dir / 'chunk_merged_dir'
    chunk_merged_dir.mkdir(exist_ok=True)

    chunk_merge_tasks = [(task_name, task_files, chunk_merged_dir) for task_name, task_files in merge_task_map.items()]
    parallel_process(chunk_merge_task, chunk_merge_tasks, process_num=chunk_task_workers, tqdm_desc='chunk merge')

    chunk_merged_files = data_of_dir(chunk_merged_dir)
    random_merged_dir = save_dir / 'random_merged_dir'
    random_merged_dir.mkdir(exist_ok=True)

    task_split_no_mapping = defaultdict(list)
    for file_path in chunk_merged_files:
        file_name = file_path.split('/')[-1]
        random_no = int(file_name.removesuffix('.parquet').split('_')[2])
        task_name = "_".join(file_name.removesuffix('.parquet').split('_')[0:2])
        task_split_no_mapping[task_name].append(random_no)

    random_merge_task_list = []
    for task_head_name, random_no_arr in task_split_no_mapping.items():
        all_file_name = f'{task_head_name}_all.parquet'
        all_file_task = [chunk_merged_dir / f'{task_head_name}_{i}.parquet' for i in random_no_arr]
        random_merge_task_list.append((all_file_name, all_file_task,random_merged_dir, random_stats_df, def_col, agr_col))
        for i in random_no_arr:
            arr = random_no_arr.copy()
            arr.remove(i)
            new_file_name = f'{task_head_name}_{i}.parquet'
            file_task = [chunk_merged_dir / f'{task_head_name}_{i}.parquet' for i in arr]
            random_merge_task_list.append(
                (new_file_name, file_task,random_merged_dir, random_stats_df, def_col, agr_col))
    parallel_process(random_merge_select_task, random_merge_task_list, process_num=random_merge_workers,
                     tqdm_desc='random merged')

    token_leve_files = data_of_dir(random_merged_dir)
    sms_token_level_df = batch_load_data(token_leve_files).reset_index(drop=True)
    sms_token_level_df.to_parquet(save_dir / 'sms_token_level.parquet',compression='zstd')
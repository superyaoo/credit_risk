import gc
import json
import traceback
from collections import Counter
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from nltk.tokenize import word_tokenize
from pythainlp.tokenize import word_tokenize as thai_word_tokenize
import pandas as pd
import numpy as np
from toollib.unversal import parallel_process

from toollib import random_list, gen_random_col, ngrams
from toollib.asystem_env.sample_util import get_sample_req
from pathlib import Path
from tqdm import tqdm

tqdm.pandas()


def sample_split(df, oot_time='2024-11-01'):
    """
    在原有的sample数据集上生成sample_type,依据一下思路进行train，test和oot的划分：
    oot采用>= 某一天的数据，在train和test中剔除oot中的手机号，tain和test基于手机号形成7:3切分
    """
    df['sample_type'] = ''
    df.loc[df['loan_time'] >= oot_time, 'sample_type'] = 'oot'
    df.loc[(df['loan_time'] < oot_time) & \
           (~df['phone_number'].isin(df[df['sample_type'] == 'oot']['phone_number'])), 'sample_type'] = 'train'

    train_phones = list(df[df['sample_type'] == 'train']['phone_number'].unique())
    splite_no_list = random_list(len(train_phones), n_split=10)
    splite_no_list = ['test' if x <= 2 else 'train' for x in splite_no_list]

    map_s = pd.Series(splite_no_list, index=train_phones)

    df['map_result'] = df['phone_number'].map(map_s)

    def gen_flag(row):
        if pd.isnull(row['map_result']):
            return row['sample_type']
        else:
            return row['map_result']

    df['sample_flag'] = df.apply(gen_flag, axis=1)
    rs = df['sample_flag'].values

    df.drop(columns=['sample_type', 'map_result', 'sample_flag'], inplace=True)
    return rs


def sms2bow(sms_list: list[dict], n_gram=3, language='spanish'):
    """
    将一个sms list中的多有body提取出来进行分词计算bow，支持n-gram模式的编码
    :param sms_list: 待编码的短信列表
    :param n_gram: 设置gram的数量
    :param language: word_tokenize 对应的语言
    """
    try:
        if isinstance(sms_list, str):
            sms_list = json.loads(sms_list)

        rs = {'senders':[]}
        for i in range(1, n_gram + 1):
            fname = f'gram{i}'
            rs[fname] = []
        # 基于时间字段排序取最近的3000条短信，如果排序异常直接取
        try:
            sms_list = sorted(sms_list, key=lambda x: x['time'], reverse=True)[:3000]
        except:
            sms_list = sms_list[:3000]

        for sms in sms_list:
            try:
                if language == 'thai':
                    tokens = thai_word_tokenize(sms['body'])
                else:
                    tokens = word_tokenize(sms['body'], language=language)
                rs['senders'].append(sms['src_phone'])
            except:
                tokens = []
            for i in range(1, n_gram + 1):
                fname = f'gram{i}'
                rs[fname].extend(ngrams(tokens, i))
        result = {'sms_count': len(sms_list), 'senders': [(k, v) for k, v in Counter(rs['senders']).items()]}
        for k, v in rs.items():
            result[k] = [(k, v) for k, v in Counter(v).items()]
        return result
    except:
        if sms_list is not None:
            print(f"有数据发生编码异常，请检查结果中为None的部分:\n{traceback.format_exc()}")
        return None


def parallel_sms2bow(sms_datas: list[list], n_gram=3, max_workers=20, body='body', language='spanish',
                     tqdm_desc='sms2bow', tqdm_disable=False):
    """
    将一个sms list中的多有body提取出来进行分词计算bow，支持n-gram模式的编码
    :param sms_datas: 待编码的短信列表
    :param n_gram: 设置gram的数量
    :param max_workers: 开启多进程的数量
    :param body: dict中短信body的字段名称
    :param language: word_tokenize 对应的语言
    :param tqdm_disable: 是否显示进度条
    :param tqdm_desc: 进度条名称
    """
    data_len = len(sms_datas)
    task_list = list(zip(range(data_len), list(sms_datas)))
    max_workers = min(max_workers, len(task_list))  # max_workers if max_workers < l else l
    with ProcessPoolExecutor(max_workers=max_workers) as executor, \
            tqdm(total=len(sms_datas), desc=tqdm_desc, disable=tqdm_disable) as pbar:
        futures = {executor.submit(sms2bow, sms_list, n_gram, language): idx for idx, sms_list in task_list}
        result = list(range(data_len))
        for future in as_completed(futures):
            idx = futures[future]
            data = future.result()
            result[idx] = data
            pbar.update(1)
        return result


def fetch_req_2_dir(df, country_abbr, user, passwd, id_col='app_order_id', split_col='split_no',
                    save_dir='req_dir', file_name_prefix='req_split_', dblink='inner', retain_cols=None):
    """
    将对应表格中的订单，按照split_no切片进行下载，然后保存到指定的目录
    :param df: 待下载的数据表,需要包含指定的两列数据 order_col , split_col
    :param country_abbr: 国家简写
    :param user: tidb 数据库账号
    :param passwd: 数据库密码
    :param id_col: order_id 的列名
    :param split_col: 切片编码 的列名
    :param save_dir: 数据保存的文件夹名称，会在当前的目录新增该文件夹
    :param file_name_prefix: 保存的文件名称的前缀
    :param dblink: 内网还是外网下载
    :param retain_cols: 待保留的df中的列，默认为None不保留
    """
    order_c = df[id_col].nunique()
    assert order_c == df.shape[0], '输入的表格中存在重复的订单数据，请检查输入的表格信息！'
    df[id_col] = df[id_col].astype(str)

    save_dir = Path(".") / save_dir
    save_dir.mkdir(exist_ok=True)

    split_nos = df[split_col].unique()
    split_nos = sorted(split_nos)

    for no in split_nos:
        file_name = f'{file_name_prefix}{no}.parquet'
        save_path = save_dir / file_name
        tmp_df = df[df['split_no'] == no]
        order_list = tmp_df[id_col].tolist()
        print(f"开始下载数据切片{no}的数据,save_path:{save_path}")
        req_df = get_sample_req(order_list, country_abbr, user, passwd, dblink)
        if isinstance(retain_cols, list) and len(retain_cols) > 0:
            req_df = req_df.merge(df[[id_col] + retain_cols], on=id_col, how='left')
        req_df.to_parquet(save_path, compression='zstd')


def token_stats_info(df: pd.DataFrame, n_gram=3, language='spanish', target='def_pd1', sms_col='sms_data',
                     max_workers=20, tqdm_disable=False):
    """
    统计短信分词中的贷后相关的信息，主要统计以下指标，贷后标签对应的，好坏词频，权重词频，命中的好坏订单数目;
    注意： 请确保传入的数据都到达了表现其，target字段无歧义
    :param df : 待下载的数据表,需要包含指定的两列数据 order_col , split_col
    :param n_gram: 需要统计的 gram 数量
    :param language: nltk 分词器的语种
    :param target: 好坏标签
    :param sms_col: sms数据所在的列
    :param body: 短信列表中body字段名，便于适配不同的短息数据
    :param max_workers: >1 则按照 max_workers的数量开启 并发计算bow，为1则单独计算
    :param tqdm_disable 是否打开tqdm进度条
    return list[pd.DataFrame]: 长度为 n_gram的list，分别是 对应的token的统计信息表
    """
    df['target'] = df[target]
    target_unique = df['target'].unique()
    result_unique = np.array([0, 1])
    set_diff = np.setdiff1d(result_unique, target_unique)
    if len(set_diff) > 0:
        print(f"输入的数据中,target的枚举值不全，缺少 {list(set_diff)}的结果")

    if max_workers > 1:
        df['sms_bow'] = parallel_sms2bow(df[sms_col], language=language,  n_gram=n_gram,
                                         max_workers=max_workers, tqdm_disable=tqdm_disable)
    else:
        if tqdm_disable:
            df['sms_bow'] = df[sms_col].map(lambda x: sms2bow(x, n_gram=n_gram, language=language))
        else:
            df['sms_bow'] = df[sms_col].progress_apply(
                lambda x: sms2bow(x, n_gram=n_gram, language=language))
    df = df[df['sms_bow'].notnull()]
    df['sms_count'] = df['sms_bow'].map(lambda x: x['sms_count'])
    result = []
    for n in tqdm(range(1, n_gram + 1), desc='对ngram的数据进行统计', disable=tqdm_disable):
        bow_key = f'gram{n}'
        order_0_all = df[df['target'] == 0]['target'].count()
        order_1_all = df[df['target'] == 1]['target'].count()
        df['token_list'] = df['sms_bow'].map(lambda x: x[bow_key])

        # 列转行
        token_df = df[['token_list', 'target', 'sms_count']].explode('token_list')
        token_df = token_df[token_df['token_list'].notnull()]
        token_df['token'] = token_df['token_list'].map(lambda x: x[0])
        token_df['token_count'] = token_df['token_list'].map(lambda x: x[1])
        token_df['token_count_weight'] = token_df['token_count'] / token_df['sms_count']

        # 透视统计
        token_df = pd.pivot_table(token_df, index='token', columns='target',
                                  values=['token_count', 'token_count_weight', 'token_list'],
                                  aggfunc={'token_count': 'sum', 'token_count_weight': 'sum', 'token_list': 'count'},
                                  fill_value=0)
        token_df.columns = [f'order_{v}' if 'token_list' == k else f'{k}_{v}' for k, v in token_df.columns]

        for n in set_diff:  # 如果枚举值缺少则使用0来填充
            token_df[f'token_count_{n}'] = 0
            token_df[f'token_count_weight_{n}'] = 0
            token_df[f'order_{n}'] = 0

        token_df['order_0_all'] = order_0_all
        token_df['order_1_all'] = order_1_all
        token_df = token_df[['token_count_0', 'token_count_1', 'token_count_weight_0',
                             'token_count_weight_1', 'order_0', 'order_1', 'order_0_all', 'order_1_all']]
        token_df = token_df.sort_values('order_0', ascending=False)
        token_df = token_df.reset_index()
        result.append(token_df)
    return result





def token_woe_level(party_stats_df, select_params=None):
    party_stats_df['count_weight'] = party_stats_df['token_count_weight_0'] + party_stats_df['token_count_weight_1']

    if select_params is None:
        select_params = {
            'order_thresdhold': 0.005,
            'lift_limit': [0.9, 1.1],
            'count_weight_thresdhold': 0.95,
            'word_select_limit': 5000,
            'qcut_bins': 5
        }

    lift_low_limit = select_params['lift_limit'][0]
    lift_up_limit = select_params['lift_limit'][1]
    order_thresdhold = select_params['order_thresdhold']
    count_weight_thresdhold = select_params['count_weight_thresdhold']
    word_select_limit = select_params['word_select_limit']
    qcut_n = select_params['qcut_bins']
    count_weight_limit = party_stats_df['count_weight'].quantile(count_weight_thresdhold)
    order_limit = order_thresdhold * (party_stats_df['order_0_all'] + party_stats_df['order_1_all']).mean()

    condition = party_stats_df['order'] > order_limit  # 该词命中的订单比例>order_thresdhold

    condition = condition & (party_stats_df['count_weight'] > count_weight_limit)

    condition = condition & ((party_stats_df['lift'] < lift_low_limit) | (party_stats_df['lift'] > lift_up_limit))

    filter_df = party_stats_df[condition].sort_values('lift')

    lower_tokens = filter_df.head(word_select_limit)['token'].to_list()

    upper_tokens = filter_df.tail(word_select_limit)['token'].to_list()

    most_tokens = filter_df.sort_values('order', ascending=True).tail(word_select_limit)['token'].to_list()

    select_tokens = set(lower_tokens + upper_tokens + most_tokens)

    filter_df = filter_df[filter_df['token'].isin(select_tokens)]

    low_lift_df = filter_df[filter_df['lift'] < lift_low_limit].sort_values('lift')
    up_lift_df = filter_df[filter_df['lift'] > lift_up_limit].sort_values('lift')

    low_lift_df['bin'],bi = pd.qcut(low_lift_df['lift'], qcut_n,duplicates='drop',retbins=True)
    low_lift_df['woe_level'] = low_lift_df['bin'].cat.codes
    low_lift_df['bin'] = low_lift_df['bin'].astype(str)

    up_lift_df['bin'],bi = pd.qcut(up_lift_df['lift'], qcut_n,duplicates='drop',retbins=True)
    up_lift_df['woe_level'] = up_lift_df['bin'].cat.codes + qcut_n
    up_lift_df['bin'] = up_lift_df['bin'].astype(str)

    token_selected_df = pd.concat([low_lift_df, up_lift_df])

    return token_selected_df[['token', 'lift', 'woe_level','random_no']].reset_index(drop=True)


def sms_woe_level(data, n_grams=3, language='spanish', n_split=20, random_state=42, chunk_base_col='phone',
                  target_col='def_pd1', sms_col='sms_data', id_col='app_order_id', save_dir='sms_woe',
                  file_prefix='split_no_', main_workers=10, sub_workers=3, merge_workers=5, select_params: dict = None):
    """
    对短信数据进行切块,并分析grams模式下的分词的统计信息。并对结果进行缓存。在数据体量不算太大的时候，使用该方法更好，更多环节采用了并发计算的功能。
    注意:传递进来的data数据，需要有以下4列信息: 1、订单号(app_order_id) 2贷后标签（target_col） 3、sms数据列(sms_col)
                        4、随机base列(chunk_base_col:默认是phone，如果没有phone也可以将chunk_base_col 设置为 order_id)
    :param data:pd.DataFrame 待分析的数据
    :param n_grams:int 分词中n_grams的数量
    :param language:str nltk中分词器的语种设置
    :param n_split:int 随机切块的份数
    :param random_state:int 数据切块的随机种子
    :param chunk_base_col:str 数据切块的基础列，针对该列的枚举值设置随机分隔
    :param target_col:str 贷后信息字段，只能使1 和0 的格式
    :param sms_col:str 短信的字段
    :param id_col:str 订单号的字段
    :param save_dir:str 相关结果所在的字段
    :param file_prefix:str 切块文件的名称前缀
    :param main_workers:int 计算切块本身的数据的并发量
    :param sub_workers:int 每个切块在做bow编码的时候的并发量
    :param merge_workers:int 合并计算的时候对应的并发量，应为该逻辑占用的内存较大，容易打满内存，不建议设置太多的并发量
    :param select_params :dict 选词的参数，如果设置为空则采用以下默认参数
    select_params = {
            'order_thresdhold': 0.005,
            'lift_limit': [0.9, 1.1],
            'count_weight_thresdhold': 0.95,
            'word_select_limit': 5000,
            'qcut_bins': 5
        }
    """
    import warnings
    warnings.filterwarnings('ignore')

    # 校验并成成缓存的目录
    save_dir = Path('.') / save_dir
    save_dir.mkdir(exist_ok=True)

    # 为该数据基于手机号生成随机切割编码，为保证数据之间的用户独立性，采用身份唯一标识
    data = gen_random_col(data, chunk_base_col=chunk_base_col, n_split=n_split, random_state=random_state,
                          random_col='random_no')
    # 在缓存目录中保存用户的编码映射信息
    order_split_path = save_dir / 'order_split_no.conf'
    if chunk_base_col == id_col:
        data[[id_col, 'random_no', target_col]].to_parquet(order_split_path, compression='zstd')
    else:
        data[[id_col, chunk_base_col, 'random_no', target_col]].to_parquet(order_split_path, compression='zstd')
    print(f"订单id和分组编码的映射已经保存:{order_split_path}")

    # 数据切块保存的根路径
    only_party_stats_dir = save_dir / 'only_party_stats'
    only_party_stats_dir.mkdir(exist_ok=True)

    random_no = sorted(list(data['random_no'].unique()))
    file_names = [f'{file_prefix}{i}.parquet' for i in random_no]

    gram_name_dict = {f'gram{i}': i for i in range(1, n_grams + 1)}

    task_list = []
    for no in random_no:
        task_list.append((data, no, only_party_stats_dir))

    def calc_task(task_df, party_no, only_party_stats):
        df_part = task_df[task_df['random_no'] == party_no]
        token_info_list = token_stats_info(df_part, language=language, n_gram=n_grams, target=target_col,
                                           sms_col=sms_col, max_workers=sub_workers, tqdm_disable=True)

        for grams_name, i in gram_name_dict.items():
            grams_name = f'gram{i}'
            ind = i - 1
            gram_dir = only_party_stats / grams_name
            gram_dir.mkdir(exist_ok=True)
            file_name = f'{file_prefix}{party_no}.parquet'
            token_info_list[ind].to_parquet(gram_dir / file_name, compression='zstd')

    parallel_process(calc_task, task_list=task_list, process_num=main_workers, tqdm_desc="chunk's word count")

    print(f'已经完成了单个chunk文件的sms分词分析,保存路径:{only_party_stats_dir}')

    # 合并后的数据存储路径
    party_stats_dir = save_dir / 'party_stats'
    party_stats_dir.mkdir(exist_ok=True)

    for grams_name, i in gram_name_dict.items():
        part_stats_result_dir = party_stats_dir / grams_name
        part_stats_result_dir.mkdir(exist_ok=True)
        data_dict = {f: pd.read_parquet(only_party_stats_dir / grams_name / f) for f in file_names}
        split_mapping = {}
        for file_name in file_names:
            arr = file_names.copy()
            arr.remove(file_name)
            split_mapping[file_name] = arr
        split_mapping['split_no_all.parquet'] = file_names.copy()

        def gram_merge_task(splite_file_name, name_list, part_stats_result_dir):
            result = None
            for file_name in name_list:
                if result is None:
                    result = data_dict[file_name]
                else:
                    result = pd.concat([result, data_dict[file_name]], axis=0)
            result = result.groupby('token').agg({
                'token_count_0': sum,
                'token_count_1': sum,
                'token_count_weight_0': sum,
                'token_count_weight_1': sum,
                'order_0': sum,
                'order_1': sum
            }).reset_index()
            random_no = splite_file_name.removeprefix(file_prefix).removesuffix('.parquet')

            try:
                no = int(random_no)
            except:
                no = -999
            part_df = data[data['random_no'] != no]
            result['order_0_all'] = part_df[(part_df[target_col] == 0)][id_col].nunique()
            result['order_1_all'] = part_df[part_df[target_col] == 1][id_col].nunique()
            result['bad_rate_all'] = result['order_1_all'] / (result['order_1_all'] + result['order_0_all'])
            result['order'] = result['order_0'] + result['order_1']
            result['bad_rate'] = result['order_1'] / result['order']
            result['lift'] = result['bad_rate'] / result['bad_rate_all']
            result = result.sort_values('order', ascending=False)

            result['random_no'] = random_no
            result.to_parquet(part_stats_result_dir / splite_file_name, compression='zstd')

        task_list = []
        for splite_file_name, file_name_mapping_list in split_mapping.items():
            task_list.append((splite_file_name, file_name_mapping_list, part_stats_result_dir))

        parallel_process(gram_merge_task, task_list=task_list, process_num=merge_workers,
                         tqdm_desc="chunk's config merge")
    print(f'已经完成了单个chunk文件的分词信息合并操作,保存路径:{party_stats_dir}')
    del data_dict

    # 为file_names中添加全量的配置文件
    file_names.append('split_no_all.parquet')

    token_level_dir = save_dir / 'token_level'
    token_level_dir.mkdir(exist_ok=True)

    for grams_name, i in gram_name_dict.items():
        part_stats_result_dir = party_stats_dir / grams_name

        token_level_gram_dir = token_level_dir / grams_name

        token_level_gram_dir.mkdir(exist_ok=True)

        for file_name in tqdm(file_names, desc=f'{grams_name} woe leve'):
            party_stats_df = pd.read_parquet(part_stats_result_dir / file_name)
            token_woe_df = token_woe_level(party_stats_df, select_params)
            token_woe_df['random_no'] = file_name.replace(file_prefix, '').replace('.parquet', '')
            token_woe_df.to_parquet(token_level_gram_dir / file_name)
    print(f'完成了选词并生成每个词的woe level')

    token_level_lists = []
    for grams_name, i in gram_name_dict.items():
        gram_token_level_dir = token_level_dir / grams_name
        token_level_df = None
        for file in file_names:
            file_path = gram_token_level_dir / file
            if token_level_df is None:
                token_level_df = pd.read_parquet(file_path)
            else:
                token_level_df = pd.concat([token_level_df, pd.read_parquet(file_path)], axis=0)
        token_level_df['gram'] = i
        token_level_lists.append(token_level_df)
    token_levels = pd.concat(token_level_lists, axis=0)
    token_levels.to_parquet(save_dir / 'token_levels.conf', compression='zstd')
    print(f'完成了配置文件的合并梳理')


def applist_woe_level(data, n_split=20, random_state=42, chunk_base_col='phone', target_col='def_pd1'
                      , app_col='applist_data', package_id='app_package', id_col='app_order_id', save_dir='app_woe',
                      file_prefix='split_no_', max_wokers=20, select_params=None):
    import warnings
    warnings.filterwarnings('ignore')

    # 校验并成成缓存的目录
    save_dir = Path('.') / save_dir
    save_dir.mkdir(exist_ok=True)

    # 为该数据基于手机号生成随机切割编码，为保证数据之间的用户独立性，采用身份唯一标识
    data = gen_random_col(data, chunk_base_col=chunk_base_col, n_split=n_split, random_state=random_state,
                          random_col='random_no')
    # 在缓存目录中保存用户的编码映射信息
    order_split_path = save_dir / 'order_split_no.conf'
    if chunk_base_col == id_col:
        data[[id_col, 'random_no', target_col]].to_parquet(order_split_path, compression='zstd')
    else:
        data[[id_col, chunk_base_col, 'random_no', target_col]].to_parquet(order_split_path, compression='zstd')
    print(f"订单id和分组编码的映射已经保存:{order_split_path}")

    # 数据切块保存的根路径
    only_party_stats_dir = save_dir / 'only_party_stats'
    only_party_stats_dir.mkdir(exist_ok=True)

    random_nos = sorted(list(data['random_no'].unique()))
    file_names = [f'{file_prefix}{i}.parquet' for i in random_nos]

    def app_base_stats(random_no, save_dir_path):
        df_part = data[data['random_no'] == random_no]
        user_apps = df_part[[id_col, target_col, app_col, 'random_no']]

        user_apps[app_col] = user_apps[app_col].map(lambda x: json.loads(x) if isinstance(x, str) else x)
        user_apps['app_count'] = user_apps[app_col].map(len)

        user_apps = user_apps[user_apps['app_count'] > 0]
        user_apps['count_weight'] = 1 / user_apps['app_count']

        explode_df = user_apps.explode(app_col)
        explode_df = explode_df[explode_df[app_col].notnull()]
        explode_df['package_id'] = explode_df[app_col].map(lambda x: str(x[package_id]).strip())

        pivot_df = pd.pivot_table(explode_df, index='package_id', columns=target_col, values=[id_col, 'count_weight'],
                                  aggfunc={id_col: 'nunique', 'count_weight': 'sum'}, fill_value=0)
        pivot_df.columns = [f"{a.replace('app_order_id', 'order')}_{b}" for a, b in pivot_df.columns]
        pivot_df['order'] = pivot_df['order_0'] + pivot_df['order_1']
        pivot_df['order_all'] = user_apps[id_col].nunique()
        pivot_df = pivot_df.reset_index()
        pivot_df.to_parquet(save_dir_path / f'{file_prefix}{random_no}.parquet', compression='zstd')

    only_part_task = [(no, only_party_stats_dir) for no in random_nos]
    parallel_process(app_base_stats, only_part_task, max_wokers, "chunck's base stats")

    data_dict = {f_name: pd.read_parquet(only_party_stats_dir / f_name)
                 for f_name in file_names
                 }

    merge_task = {}
    for f_name in file_names:
        tasks = file_names.copy()
        tasks.remove(f_name)
        merge_task[f_name] = tasks
    merge_task[f'{file_prefix}all.parquet'] = file_names.copy()

    def chunck_merge_task(splite_file_name, f_names, result_dir):
        result = None
        for file_name in f_names:
            if result is None:
                result = data_dict[file_name]
            else:
                result = pd.concat([result, data_dict[file_name]], axis=0)

            result = result.groupby('package_id').agg(
                order_0 = ('order_0','sum'),
                order_1 = ('order_1', 'sum'),
                count_weight_0 = ('count_weight_0','sum'),
                count_weight_1 = ('count_weight_1','sum'),
            ).reset_index()

        random_no = splite_file_name.removeprefix(file_prefix).removesuffix('.parquet')
        try:
            no = int(random_no)
        except:
            no = -999
        part_df = data[data['random_no'] != no]
        result['order_0_all'] = part_df[(part_df[target_col] == 0)][id_col].nunique()
        result['order_1_all'] = part_df[part_df[target_col] == 1][id_col].nunique()
        result['bad_rate_all'] = result['order_1_all'] / (result['order_1_all'] + result['order_0_all'])
        result['order'] = result['order_0'] + result['order_1']
        result['bad_rate'] = result['order_1'] / result['order']
        result['lift'] = result['bad_rate'] / result['bad_rate_all']
        result = result.sort_values('order', ascending=False)
        result['random_no'] = random_no
        result.to_parquet(result_dir/splite_file_name)

    party_stats_dir = save_dir / 'party_stats'
    party_stats_dir.mkdir(exist_ok=True)

    party_task = [  (k,v,party_stats_dir) for k,v in merge_task.items()  ]

    parallel_process(chunck_merge_task,party_task, int(max_wokers/2), "chunck merge task")

    file_names.append(f'{file_prefix}all.parquet')
    def select_and_woe_tag(party_stats_df, select_params):
        party_stats_df['count_weight'] = party_stats_df['count_weight_0'] + party_stats_df['count_weight_1']
    
        if select_params is None:
            select_params = {
                'order_thresdhold': 0.002,
                'lift_limit': [0.9, 1.1],
                'app_select_limit': 1000,
                'qcut_bins': 2
            }
    
        lift_low_limit = select_params['lift_limit'][0]
        lift_up_limit = select_params['lift_limit'][1]
        order_thresdhold = select_params['order_thresdhold']
        app_select_limit = select_params['app_select_limit']
        qcut_n = select_params['qcut_bins']
        order_limit = order_thresdhold * (party_stats_df['order_0_all'] + party_stats_df['order_1_all']).mean()
        
        # 筛选条件
        condition = party_stats_df['order'] > order_limit
        condition = condition & ((party_stats_df['lift'] < lift_low_limit) | (party_stats_df['lift'] > lift_up_limit))
        filter_df = party_stats_df[condition].sort_values('lift')
        
        # 选择特征
        lower_apps = filter_df.head(app_select_limit)['package_id'].to_list()
        upper_apps = filter_df.tail(app_select_limit)['package_id'].to_list()
        most_apps = filter_df.sort_values('order', ascending=True).tail(app_select_limit)['package_id'].to_list()
        select_apps = set(lower_apps + upper_apps + most_apps)
        filter_df = filter_df[filter_df['package_id'].isin(select_apps)]
    
        low_lift_df = filter_df[filter_df['lift'] < lift_low_limit].sort_values('lift')
        up_lift_df = filter_df[filter_df['lift'] > lift_up_limit].sort_values('lift')
    
        def safe_qcut(df, value_col, n_bins):
            """安全的分箱函数"""
            if len(df) == 0:
                return pd.Series(dtype='category')
            
            # 获取唯一值的数量
            unique_values = df[value_col].nunique()
            
            # 如果唯一值太少，直接使用唯一值作为分箱
            if unique_values <= n_bins:
                return pd.Categorical(df[value_col].astype(str))
            
            try:
                # 尝试使用qcut进行分箱，允许重复值
                return pd.qcut(df[value_col], n_bins, duplicates='drop')
            except ValueError:
                try:
                    # 如果失败，尝试减少分箱数量
                    return pd.qcut(df[value_col], max(2, n_bins-1), duplicates='drop')
                except ValueError:
                    # 如果还是失败，使用中位数分割
                    median = df[value_col].median()
                    return pd.cut(df[value_col], 
                                bins=[float('-inf'), median, float('inf')],
                                labels=['0', '1'])
    
        # 处理低lift值数据
        if len(low_lift_df) > 0:
            low_lift_df['bin'] = pd.qcut(low_lift_df['lift'], qcut_n, duplicates='drop')
            low_lift_df['woe_level'] = low_lift_df['bin'].cat.codes
            low_lift_df['bin'] = low_lift_df['bin'].astype(str)
        else:
            low_lift_df['woe_level'] = pd.Series(dtype='int64')
            low_lift_df['bin'] = pd.Series(dtype='str')
    
        if len(up_lift_df) > 0:
            up_lift_df['bin'] = pd.qcut(up_lift_df['lift'], qcut_n, duplicates='drop')
            up_lift_df['woe_level'] = up_lift_df['bin'].cat.codes + (qcut_n if len(low_lift_df) > 0 else 0)
            up_lift_df['bin'] = up_lift_df['bin'].astype(str)
        else:
            up_lift_df['woe_level'] = pd.Series(dtype='int64')
            up_lift_df['bin'] = pd.Series(dtype='str')
    
        # 合并结果
        token_selected_df = pd.concat([low_lift_df, up_lift_df])
        if len(token_selected_df) > 0:
            token_selected_df['woe_level'] = token_selected_df['woe_level'].astype(str)
            return token_selected_df[['package_id', 'woe_level', 'lift', 'random_no']]
        else:
            # 如果没有数据，返回空DataFrame但保持正确的列结构
            return pd.DataFrame(columns=['package_id', 'woe_level', 'lift', 'random_no'])


    # def select_and_woe_tag(party_stats_df, select_params):
    #     party_stats_df['count_weight'] = party_stats_df['count_weight_0'] + party_stats_df['count_weight_1']

    #     if select_params is None:
    #         select_params = {
    #             'order_thresdhold': 0.002,
    #             'lift_limit': [0.9, 1.1],
    #             # 'count_weight_thresdhold': 0.9,
    #             'app_select_limit': 1000,
    #             'qcut_bins': 3
    #         }

    #     lift_low_limit = select_params['lift_limit'][0]
    #     lift_up_limit = select_params['lift_limit'][1]
    #     order_thresdhold = select_params['order_thresdhold']
    #     app_select_limit = select_params['app_select_limit']
    #     qcut_n = select_params['qcut_bins']
    #     order_limit = order_thresdhold * (party_stats_df['order_0_all'] + party_stats_df['order_1_all']).mean()
    #     condition = party_stats_df['order'] > order_limit  # 该词命中的订单比例>order_thresdhold
    #     condition = condition & ((party_stats_df['lift'] < lift_low_limit) | (party_stats_df['lift'] > lift_up_limit))
    #     filter_df = party_stats_df[condition].sort_values('lift')
    #     lower_apps = filter_df.head(app_select_limit)['package_id'].to_list()
    #     upper_apps = filter_df.tail(app_select_limit)['package_id'].to_list()
    #     most_apps = filter_df.sort_values('order', ascending=True).tail(app_select_limit)['package_id'].to_list()
    #     select_apps = set(lower_apps + upper_apps + most_apps)
    #     filter_df = filter_df[filter_df['package_id'].isin(select_apps)]

    #     low_lift_df = filter_df[filter_df['lift'] < lift_low_limit].sort_values('lift')
    #     up_lift_df = filter_df[filter_df['lift'] > lift_up_limit].sort_values('lift')

    #     low_lift_df['bin'] = pd.qcut(low_lift_df['lift'], qcut_n)
    #     low_lift_df['woe_level'] = low_lift_df['bin'].cat.codes
    #     low_lift_df['bin'] = low_lift_df['bin'].astype(str)

    #     up_lift_df['bin'] = pd.qcut(up_lift_df['lift'], qcut_n)
    #     up_lift_df['woe_level'] = up_lift_df['bin'].cat.codes + qcut_n
    #     up_lift_df['bin'] = up_lift_df['bin'].astype(str)

    #     token_selected_df = pd.concat([low_lift_df, up_lift_df])
    #     token_selected_df['woe_level'] = token_selected_df['woe_level'].astype(str)
    #     return token_selected_df[['package_id', 'woe_level','lift', 'random_no']]

    conf = None
    for file in tqdm(file_names,desc='select apps'):
        party_stats_df = pd.read_parquet(party_stats_dir / file)
        if conf is None:
            conf = select_and_woe_tag(party_stats_df, select_params)
        else:
            conf = pd.concat([conf, select_and_woe_tag(party_stats_df, select_params)],axis=0)
    conf.to_parquet(save_dir / 'app_woe_level.conf')
    
def get_unique_sender_smslist_rename(smslist):
    unique_smslist = []
    src_phone_names = set()
    for sms in smslist:
        src_phone = sms["src_phone"]
        if src_phone not in src_phone_names:
            src_phone_names.add(src_phone)
            app = {}
            app['app_name'] = src_phone
            app['app_package'] = src_phone
            app['fi_time'] = sms['time']
            app['isSystem'] = 0
            app['lu_time'] = sms['time']
            unique_smslist.append(app)
    return unique_smslist

def smslist_sender_woe_level(data, n_split=20, random_state=42, chunk_base_col='phone', target_col='def_pd0'
                      , sms_col='sms_data', id_col='app_order_id', save_dir='sms_sender_woe',
                      file_prefix='split_no_', max_wokers=20, select_params=None):
    data[sms_col] = data[sms_col].map(lambda x: json.loads(x) if isinstance(x, str) else x)
    data['applist_data'] = data[sms_col].apply(lambda x:get_unique_sender_smslist_rename(x))
    
    applist_woe_level(data, n_split=n_split, random_state=random_state, chunk_base_col=chunk_base_col, target_col=target_col
                      , app_col='applist_data', id_col=id_col, save_dir=save_dir,
                      file_prefix=file_prefix, max_wokers=max_wokers, select_params=select_params)

def applist_cut(applist_data,apply_timestamp,time_cut):
    if time_cut == 10000:
        return applist_data
    day = 1000*60*60*24
    app_cut = applist_data.copy()
    if len(app_cut) == 0:
        return app_cut
    day_start_timestamp = apply_timestamp - day*time_cut
    app_cut = [app for app in app_cut if (app['fi_time'] >= day_start_timestamp)]
    return app_cut

def applist_woe_level_v2(data, n_split=20, random_state=42, chunk_base_col='phone', target_col='def_pd0'
                      , app_col='applist_data', id_col='app_order_id',applytime_col = 'apply_time',save_dir='app_woe_v2',
                      file_prefix='split_no_', max_wokers=20, select_params=None):
    
    day_list = [3, 7, 15, 30, 90, 180]
    data[app_col] = data[app_col].map(lambda x: json.loads(x) if isinstance(x, str) else x)
    data['apply_timestamp'] = data[applytime_col].apply(lambda x:x.timestamp()*1000)

    for time_cut in day_list:
        data['applist_data_new'] = data.apply(lambda x:applist_cut(x[app_col],x['apply_timestamp'],time_cut),axis = 1)
        applist_woe_level(data, n_split=n_split, random_state=random_state, chunk_base_col=chunk_base_col, target_col=target_col
                          , app_col='applist_data_new', id_col=id_col, save_dir=save_dir+'_'+str(time_cut),
                          file_prefix=file_prefix, max_wokers=max_wokers, select_params=select_params)
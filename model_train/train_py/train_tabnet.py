import os
import glob
import pandas as pd
import numpy as np
import torch
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from pytorch_tabnet.tab_model import TabNetClassifier
import warnings

# 忽略不必要的警告
warnings.filterwarnings('ignore')

# ================= 配置区域 =================
# 模块列表
MODULE_LIST = ['device_info', 'app_list']  # 请替换为你实际的 module_list
# 数据路径配置
BASE_PATH = '/home/zengjunyao/notebook/model_link/af'
LABEL_PATH = '/home/mahaocheng/workspace/model_train/model_train_result/th_0_all_1017/label_allsample_merge.pickle'
OUTPUT_DIR = '/home/zengjunyao/notebook/tabnet_result'

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 工具函数 =================

def get_ks_auc(y_true, y_pred, name="Model"):
    """计算并打印 AUC 和 KS"""
    try:
        auc = roc_auc_score(y_true, y_pred)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        ks = max(tpr - fpr)
        print(f"[{name}] AUC: {auc:.5f} | KS: {ks:.5f}")
        return auc, ks
    except Exception as e:
        print(f"[{name}] 计算指标出错: {e}")
        return 0, 0

def concat_parquet_files(file_list):
    """读取并合并parquet文件列表"""
    dfs = []
    for f in file_list:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            print(f"读取文件失败: {f}, error: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

# ================= 1. 数据读取与合并 =================

print(">>> 正在读取标签数据...")
try:
    # 假设 Label 数据有 app_order_id, label, sample_type 列
    label_df = pd.read_pickle(LABEL_PATH)
    label_df = label_df[['app_order_id', 'label', 'sample_type']]
    print(f"Label数据加载完成，形状: {label_df.shape}")
except Exception as e:
    print(f"Label读取失败，请检查路径: {e}")
    exit()

sample_list = []

for module in MODULE_LIST:
    print(f"\n正在处理模块: {module}")
    # 构造文件路径
    module_files = glob.glob(f'{BASE_PATH}/{module}/df_1_*.parquet')
    
    if not module_files:
        print(f"警告: 模块 {module} 未找到文件，跳过。")
        continue
        
    # 读取模块数据
    module_df = concat_parquet_files(module_files)
    
    # 仅保留数值型特征
    # 注意：TabNet 也能处理 Embedding，但为了脚本通用性，这里先只处理数值
    # 且保留 app_order_id 用于后续合并
    if 'app_order_id' not in module_df.columns:
        print(f"错误: 模块 {module} 缺少 app_order_id 列")
        continue

    # 筛选数值列 + ID
    num_cols = module_df.select_dtypes(include=['number']).columns.tolist()
    cols_to_keep = ['app_order_id'] + [c for c in num_cols if c != 'app_order_id']
    module_df = module_df[cols_to_keep]
    
    # 简单的特征筛选：去除方差为0（全是一个值）的列
    # (注：此处替换了你原脚本中依赖外部 calculator 的 IV 筛选)
    nunique = module_df.nunique()
    valid_cols = nunique[nunique > 1].index.tolist()
    if 'app_order_id' not in valid_cols:
        valid_cols.append('app_order_id')
    
    module_df = module_df[valid_cols]
    
    # 这里的 merge 是为了确保每一步数据都对齐，也可以最后再 merge
    # 为了节省内存，这里只保留特征，最后统一 merge
    print(f"模块 {module} 预处理后形状: {module_df.shape}")
    sample_list.append(module_df)

print("\n>>> 正在合并所有模块数据...")
if len(sample_list) > 0:
    # 逐步合并
    all_features = reduce(lambda left, right: pd.merge(left, right, on=['app_order_id'], how='inner'), sample_list)
    # 最后合入标签
    full_data = pd.merge(all_features, label_df, on='app_order_id', how='inner')
else:
    print("没有有效的数据被加载，程序退出。")
    exit()

print(f"全量数据合并完成，形状: {full_data.shape}")

# ================= 2. 数据清洗与分割 =================

print("\n>>> 开始数据清洗 (TabNet 适配)...")

# 填充 sample_type
full_data['sample_type'].fillna('train', inplace=True)

# 替换 inf 为 nan
full_data = full_data.replace([np.inf, -np.inf], np.nan)

# 确定特征列
exclude_cols = ['app_order_id', 'label', 'sample_type']
feature_cols = [c for c in full_data.columns if c not in exclude_cols]

print(f"入模特征数量: {len(feature_cols)}")

# 切分数据集
# 逻辑：先切分出 dataframe，再分别进行填充和标准化，防止数据穿越
train_df = full_data[full_data['sample_type'] == 'train']
oot_df = full_data[full_data['sample_type'] == 'oot']

# 再次从 train 中切分出 train 和 test (验证集)
train_split, test_split = train_test_split(train_df, train_size=0.85, random_state=2023)

# 重置索引
train_split = train_split.reset_index(drop=True)
test_split = test_split.reset_index(drop=True)
oot_df = oot_df.reset_index(drop=True)

print(f"样本分布 -> Train: {train_split.shape[0]}, Test: {test_split.shape[0]}, OOT: {oot_df.shape[0]}")

# ================= 3. 特征工程 (标准化) =================
# TabNet 对数值敏感，必须做 StandardScaler
# 且必须：Fit on Train, Transform on All

print(">>> 正在进行标准化处理...")

# 1. 缺失值处理：神经网络不能有 NaN。使用均值填充。
# 为了效率，我们使用 pandas 的均值填充，并记录 train 的均值
train_means = train_split[feature_cols].mean()

# 使用训练集的均值填充所有数据集
train_split[feature_cols] = train_split[feature_cols].fillna(train_means)
test_split[feature_cols] = test_split[feature_cols].fillna(train_means)
oot_df[feature_cols] = oot_df[feature_cols].fillna(train_means)

# 2. 标准化
scaler = StandardScaler()
# 仅在训练集上 fit
scaler.fit(train_split[feature_cols])

# 转换所有数据集
X_train = scaler.transform(train_split[feature_cols])
X_test = scaler.transform(test_split[feature_cols])
X_oot = scaler.transform(oot_df[feature_cols])

# 准备标签
y_train = train_split['label'].values
y_test = test_split['label'].values
y_oot = oot_df['label'].values

# ================= 4. TabNet 模型训练 =================

print("\n>>> 开始训练 TabNet 模型...")

# 初始化模型
# 针对 740/Cash Loan 业务，特征稀疏且非线性强
clf = TabNetClassifier(
    n_d=16, n_a=16,            # 网络宽度，特征少可调小为 8
    n_steps=3,                 # 决策步数
    gamma=1.3,                 # 特征重用系数
    lambda_sparse=1e-3,        # 稀疏惩罚，越大特征筛选越狠
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2), # 学习率，不收敛可调小
    scheduler_params={"step_size":10, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='entmax',        # entmax 比 sparsemax 更平滑
    verbose=1,
    seed=42
)

# 训练
# max_epochs: 740 业务容易过拟合，建议不要跑太多轮
# patience: 早停机制
clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_name=['train', 'test'],
    eval_metric=['auc'],
    max_epochs=50,             # 建议 50-100
    patience=10,               # 10轮没提升就停止
    batch_size=1024,           # 显存允许越大越好，增加训练稳定性
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)

# ================= 5. 模型评估与保存 =================

print("\n>>> 模型评估中...")

# 获取预测概率 (第2列是正样本概率)
pred_train = clf.predict_proba(X_train)[:, 1]
pred_test = clf.predict_proba(X_test)[:, 1]
pred_oot = clf.predict_proba(X_oot)[:, 1]

# 打印指标
print("-" * 30)
get_ks_auc(y_train, pred_train, "Train Set")
get_ks_auc(y_test, pred_test, "Test Set")
get_ks_auc(y_oot, pred_oot, "OOT Set")
print("-" * 30)

# 保存模型
model_save_path = os.path.join(OUTPUT_DIR, 'tabnet_model.zip')
clf.save_model(model_save_path)
print(f"模型已保存至: {model_save_path}")

# 保存预测结果 (可选，方便后续分析)
result_df = pd.DataFrame({
    'app_order_id': oot_df['app_order_id'],
    'label': y_oot,
    'pred_prob': pred_oot
})
result_save_path = os.path.join(OUTPUT_DIR, 'oot_predictions.csv')
result_df.to_csv(result_save_path, index=False)
print(f"OOT 预测结果已保存至: {result_save_path}")

print("\n>>> 全部流程执行完毕！")
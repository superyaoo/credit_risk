#!/bin/bash

# 切换main分支并拉取最新的代码
cd /home/etl/featurelib
git checkout main
git branch
echo "正在拉取最新的 GitLab 代码..."
git pull origin main || { echo "Git 拉取失败"; exit 1; }


# 定义 Python 解释器路径
PYTHON_EXEC="/home/etl/miniconda3/envs/py310/bin/python"

# 定义主日志目录
LOG_BASE_DIR="/home/model/logs"
mkdir -p "$LOG_BASE_DIR"  # 确保日志主目录存在

# 获取当前日期
DATE=$(date +%Y%m%d)

# 任务列表（路径格式：<类别>/<脚本路径>）
TASKS=(
    "ath:/home/etl/toollib/toollib/model_data/ath/req_and_fea_ath.py"
    "af:/home/etl/toollib/toollib/model_data/af/req_and_fea_af.py"
    "athl:/home/etl/toollib/toollib/model_data/athl/req_and_fea_athl.py"
)

# 遍历任务并按顺序执行
for task in "${TASKS[@]}"; do
    # 解析任务类别和脚本路径
    CATEGORY=$(echo "$task" | cut -d':' -f1)  # 取出 mw / ac / am
    SCRIPT_PATH=$(echo "$task" | cut -d':' -f2)  # 取出对应的 Python 脚本路径
    SCRIPT_NAME=$(basename "$SCRIPT_PATH" .py)  # 获取脚本文件名

    # 定义日志目录（按类别存放）
    LOG_DIR="$LOG_BASE_DIR/$CATEGORY"
    mkdir -p "$LOG_DIR"  # 确保日志目录存在

    # 定义日志文件
    LOG_FILE="$LOG_DIR/${SCRIPT_NAME}_${DATE}.log"

    echo "开始执行任务：$SCRIPT_PATH" | tee -a "$LOG_FILE"
    $PYTHON_EXEC "$SCRIPT_PATH" >> "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        echo "任务 $SCRIPT_PATH 执行成功！" | tee -a "$LOG_FILE"
    else
        echo "任务 $SCRIPT_PATH 失败！请检查日志：$LOG_FILE" | tee -a "$LOG_FILE"
        exit 1  # 如果任务失败，停止执行后续任务
    fi

    echo "--------------------------" | tee -a "$LOG_FILE"
done

echo "所有任务执行完成！ 🎉"
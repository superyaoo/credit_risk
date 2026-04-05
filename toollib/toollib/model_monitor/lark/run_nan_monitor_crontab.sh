#!/bin/bash

DBLINK="inner"
CONF_PATH="/home/mahaocheng/workspace/models/monitor/monitor_config.yaml"
UPLOAD_FOLDER_TOKEN="MLeWfbZ26lht84d9Bpju85LQsZe"
WEBHOOK_URL="https://open.larksuite.com/open-apis/bot/v2/hook/faa2e5bc-9af3-4281-b2e1-c4c269fa14c4"

nohup /home/mahaocheng/app/miniconda3/envs/feature_lib/bin/python nan_monitor.py \
    --conf_path "${CONF_PATH}" \
    --upload_folder_token "${UPLOAD_FOLDER_TOKEN}" \
    --template_file_token "JcMVsnij1hkDaEtW3w6u8ffBstd" \
    --template_sheet_token "c67241" \
    --dblink "${DBLINK}" \
    --webhook_url "${WEBHOOK_URL}"  > /home/mahaocheng/workspace/lark_nan_monitor_log 2>&1
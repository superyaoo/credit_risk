#!/bin/bash

DBLINK="inner"
CONF_PATH="/home/mahaocheng/workspace/models/monitor/monitor_config.yaml"
UPLOAD_FOLDER_TOKEN="MLeWfbZ26lht84d9Bpju85LQsZe"
WEBHOOK_URL="https://open.larksuite.com/open-apis/bot/v2/hook/39117e89-0053-46e9-ba22-1e59191333ae"

python nan_monitor.py \
    --conf_path "${CONF_PATH}" \
    --upload_folder_token "${UPLOAD_FOLDER_TOKEN}" \
    --template_file_token "JcMVsnij1hkDaEtW3w6u8ffBstd" \
    --template_sheet_token "c67241" \
    --dblink "${DBLINK}" \
    --webhook_url "${WEBHOOK_URL}"
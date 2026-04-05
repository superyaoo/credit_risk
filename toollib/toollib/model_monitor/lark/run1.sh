#!/bin/bash

PRODUCT_PERIOD=7
DBLINK="inner"

CONF_PATH="/home/mahaocheng/workspace/models/monitor/monitor_config.yaml"
UPLOAD_FOLDER_TOKEN="MLeWfbZ26lht84d9Bpju85LQsZe"
WEBHOOK_URL="https://open.larksuite.com/open-apis/bot/v2/hook/39117e89-0053-46e9-ba22-1e59191333ae"

python send_report.py \
    --monitor_sample "month" \
    --conf_path "${CONF_PATH}" \
    --upload_folder_token "${UPLOAD_FOLDER_TOKEN}" \
    --template_file_token "E0MYsTUzJhh80mt2iBlu094DsIf" \
    --dblink ${DBLINK} \
    --webhook_url "${WEBHOOK_URL}"


#!/bin/bash

PRODUCT_PERIOD=7
DBLINK="outer"

CONF_PATH="/home/pony/workspace/model/models/monitor/monitor_config1.yaml"
UPLOAD_FOLDER_TOKEN="MLeWfbZ26lht84d9Bpju85LQsZe"
WEBHOOK_URL="https://open.larksuite.com/open-apis/bot/v2/hook/39117e89-0053-46e9-ba22-1e59191333ae"

python send_report.py \
    --monitor_sample "week" \
    --conf_path "${CONF_PATH}" \
    --upload_folder_token "${UPLOAD_FOLDER_TOKEN}" \
    --template_file_token "SVljslA4thYysptkVtduDAq3sgh" \
    --dblink ${DBLINK} \
    --webhook_url "${WEBHOOK_URL}"

python send_report.py \
    --monitor_sample "month" \
    --conf_path "${CONF_PATH}" \
    --upload_folder_token "${UPLOAD_FOLDER_TOKEN}" \
    --template_file_token "E0MYsTUzJhh80mt2iBlu094DsIf" \
    --dblink ${DBLINK} \
    --webhook_url "${WEBHOOK_URL}"

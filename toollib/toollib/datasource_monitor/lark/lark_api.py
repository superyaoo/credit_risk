# -*- coding: utf-8 -*-
import os
import requests
import json
from requests_toolbelt import MultipartEncoder


def get_tenant_access_token(app_id, app_secret):
    url = 'https://open.larksuite.com/open-apis/auth/v3/tenant_access_token/internal/'
    headers = {'Content-Type': 'application/json'}
    data = {
        'app_id': app_id,
        'app_secret': app_secret
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json().get('tenant_access_token')


def upload_file(token, file_name, file_path):
    url = 'https://open.larksuite.com/open-apis/drive/v1/files/upload_all'

    form = {'file_name': file_name,
            'parent_type': 'explorer',
            'parent_node': 'VnuCfivljlnwUfdfp3ClYtpqgOh',
            'size': str(os.path.getsize(file_path)),
            'file': (open(file_path, 'rb'))}
    multi_form = MultipartEncoder(form)

    headers = {
        'Authorization': f'Bearer {token}', 'Content-Type': multi_form.content_type
    }

    response = requests.post(url, headers=headers, data=multi_form)
    return response.json()['data']['file_token']


def write_data_to_cloud(token,spreadsheetToken,sheet_id,left_pos,right_pos,df):
    url = f'https://open.larksuite.com/open-apis/sheets/v2/spreadsheets/{spreadsheetToken}/values'

    range = f"{sheet_id}!{left_pos}:{right_pos}"
    data = {
        "valueRange": {
            "range": range,
            "values": df.values.tolist()
        }
    }

    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    response = requests.put(url, headers=headers, data=json.dumps(data))
    return response.json()


def send_message(token, spreadsheetToken, chat_id, today):
    url = 'https://open.larksuite.com/open-apis/im/v1/messages'

    params = {
        "receive_id_type": "chat_id"
    }

    doc_url = f'https://nzd66m35vu.sg.larksuite.com/sheets/{spreadsheetToken}'

    data = {
        "receive_id": chat_id,
        "content": json.dumps({"text": f"{today.strftime('%Y-%m-%d')} 数据源监控报告：{doc_url}"}),
        "msg_type": "text"
    }

    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    response = requests.post(url, headers=headers, params=params, data=json.dumps(data))
    return response.json()

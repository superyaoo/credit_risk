import os
import requests
from requests_toolbelt import MultipartEncoder
from typing import Optional, Dict
import requests
import time
import pandas as pd
from typing import Dict
from io import BytesIO
import json

class FeishuBase:
    def __init__(self, app_id: str, app_secret: str):
        self.app_id = app_id
        self.app_secret = app_secret
        self.base_url = "https://open.feishu.cn/open-apis"
        self.token = self._get_tenant_access_token()
        
    def _get_tenant_access_token(self) -> str:
        url = f"{self.base_url}/auth/v3/tenant_access_token/internal"
        payload = {
            "app_id": self.app_id,
            "app_secret": self.app_secret
        }
        response = requests.post(url, json=payload)
        return response.json()["tenant_access_token"]

class FeishuFileUploader(FeishuBase):
    def __init__(self, app_id: str, app_secret: str):
        super().__init__(app_id, app_secret)

    def upload_file(self, 
                   file_path: str, 
                   parent_node: str,
                   file_name: Optional[str] = None,
                   parent_type: str = "explorer") -> Dict:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        file_size = os.path.getsize(file_path)

        if file_name is None:
            file_name = os.path.basename(file_path)

        form = {
            'file_name': file_name,
            'parent_type': parent_type,
            'parent_node': parent_node,
            'size': str(file_size),
            'file': (file_name, open(file_path, 'rb'))
        }
        
        multi_form = MultipartEncoder(form)
        
        headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': multi_form.content_type
        }

        url = f"{self.base_url}/drive/v1/files/upload_all"
        try:
            response = requests.post(url, headers=headers, data=multi_form)
            return response.json()
        except Exception as e:
            raise Exception(f"上传文件失败: {str(e)}")
        finally:
            form['file'][1].close()
            

class FeishuXlsxExporter(FeishuBase):
    def __init__(self, app_id: str, app_secret: str):
        super().__init__(app_id, app_secret)
    
    def export_sheet(self, file_token: str, file_type: str = "xlsx") -> Dict:
        url = f"{self.base_url}/drive/v1/export_tasks"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "file_extension": file_type,
            "token": file_token,
            "type": "sheet"
        }
        
        response = requests.post(url, headers=headers, json=payload)
        return response.json()
    
    def get_export_status(self, ticket: str, file_token: str) -> Dict:
        url = f"{self.base_url}/drive/v1/export_tasks/{ticket}"
        headers = {
            "Authorization": f"Bearer {self.token}"
        }
        params = {
            "token": file_token
        }
        response = requests.get(url, headers=headers, params=params)
        return response.json()
    
    def download_exported_file(self, file_token: str):
        url = f"{self.base_url}/drive/v1/export_tasks/file/{file_token}/download"
        headers = {
            "Authorization": f"Bearer {self.token}"
        }
        
        max_retries = 10
        for attempt in range(max_retries):
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return BytesIO(response.content)
            time.sleep(10)
            print(f"Download failed, retrying... ({attempt + 1}/{max_retries})")
        
        return {"code": response.status_code, "msg": "Download failed after retries"}
        
    def read_xlsx_flow(self, file_token: str):
        export_result = self.export_sheet(file_token, "xlsx")
        if export_result["code"] == 0:
            ticket = export_result["data"]["ticket"]
            result = self.get_export_status(ticket, file_token)
            if result['code'] == 0:
                download_file_token = result['data']['result']['file_token']
                content = self.download_exported_file(file_token=download_file_token)
                if isinstance(content, BytesIO):
                    return pd.read_excel(content)
                return content
            else:
                return result
            

class FeishuTableBuilder(FeishuBase):
    def __init__(self, app_id: str, app_secret: str):
        super().__init__(app_id, app_secret)
        
    def build_table_from_template(self, file_token: str, folder_token: str, new_name: str):
        url = f"{self.base_url}/drive/v1/files/{file_token}/copy"
        
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        data = {
            "type": "sheet",
            "folder_token": folder_token,
        }
        
        if new_name:
            data["name"] = new_name

        response = requests.post(
            url=url,
            headers=headers,
            data=json.dumps(data)
        )
        print(response.json())
        if response.status_code != 200:
            raise Exception(f"创建表格失败: {response.json()}")
        return response.json()
    
    def write_df_to_table(self, 
                          file_token: str, 
                          sheet_token: str,
                          left_position: str,
                          right_position: str,
                          df: pd.DataFrame, 
                          ):
        
        url = f"https://open.feishu.cn/open-apis/sheets/v2/spreadsheets/{file_token}/values"
        
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        range_str = f"{sheet_token}!{left_position}:{right_position}"
        data = {
            "valueRange": {
                "range": range_str,
                "values": df.values.tolist()
            }
        }
        
        response = requests.put(
            url=url,
            headers=headers,
            data=json.dumps(data)
        )
        print(response.json())
        if response.status_code != 200:
            raise Exception(f"写入表格失败: {response.json()}")
        return response.json()


if __name__ == "__main__":
    APP_ID = "你的应用 ID"
    APP_SECRET = "你的应用密钥"
    
    uploader = FeishuFileUploader(APP_ID, APP_SECRET)
    
    result = uploader.upload_file(
        file_path="/path/to/your/file.xlsx",
        parent_node="fold_token",
        file_name="test.xlsx"
    )
    
    print("上传结果：", result)
    
    exporter = FeishuXlsxExporter(APP_ID, APP_SECRET)
    
    df = exporter.read_xlsx_flow(
        file_token="fold_token",
        )
    print(df)
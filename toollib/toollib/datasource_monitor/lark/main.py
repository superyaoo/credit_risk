import os
from datetime import *
from app_sms_stat import *
from lark_api import *

sys_list = ['af', 'ath', 'athl']
acq_channel_dict = {'af': ['AMOY', 'LHXD'], 'ath': ['MOEK', 'VIJG'], 'athl': ['KOTX']}
sheet_dict = {'af': '0yhmDn', 'ath': '1xPlAZ', 'athl': '2coucC'}


if __name__ == '__main__':
    today = datetime.today()
    days_ago_63 = today - timedelta(days=70)
    start_date = days_ago_63.strftime('%Y-%m-%d')

    app_id = 'cli_a8cc1e042a7f9028'
    app_secret = 'ZsabOxh6PynXB0RQRie9Wf7evxOCtT6H'
    chat_id = 'oc_a1a7edf0b95a6d4158be6c6f76c6c260'
    spreadsheetToken = 'IYaGsvLpmh8icotfGIYlVqfpgud'

    token = get_tenant_access_token(app_id, app_secret)

    for sys in sys_list:
        df = get_basic_data_stat(start_date,sys,acq_channel_dict[sys])
        result = write_data_to_cloud(token, spreadsheetToken, sheet_dict[sys], 'A2', f'G{len(df)+1}', df)
        print(sys,result)

    result = send_message(token, spreadsheetToken, chat_id,today)
    print(result)

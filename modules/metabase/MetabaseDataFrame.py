import requests
import numpy as np
import os
import pandas as pd
from typing import Dict, List
import warnings
from io import StringIO
# from modelo.normalize import DataFrameProcessor

warnings.filterwarnings('ignore')  # Ignore SSL certificate warnings

class MetabaseDataFrame:
    def __init__(self, session_id: str, base_url: str = 'https://adbikpi.arcosdorados.net/api/'):
        self.session_id = session_id
        self.base_url = base_url
        self.headers = {
            'Accept': '*/*',
            'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36 Edg/113.0.1774.35',
        }
        self.cookies = {'metabase.SESSION': session_id}

    def get_dataframe(self, data_get_data: Dict) -> pd.DataFrame:
        response = requests.post(f'{self.base_url}dataset/csv', cookies=self.cookies, headers=self.headers, data=data_get_data, verify=False)
        data_string = response.text
        data = StringIO(data_string)
        return pd.read_csv(data, sep=",")

    def process_and_save(self, df: pd.DataFrame, parquet_path: str, columns_to_datetime: List[str] = None):
        # Convert columns to datetime
        if columns_to_datetime:
            for col in columns_to_datetime:
                df[col] = pd.to_datetime(df[col])
                
        # Convert columns with dtype float16 to float32 to be able to save in parquet
        for col in df.columns:
            if df[col].dtype == 'float16':
                df[col] = df[col].astype('float32')
                
        # Save the dataframe in a parquet file
        df.to_parquet(parquet_path, index=False)
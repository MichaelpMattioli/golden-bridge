import contextlib
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from unidecode import unidecode

class NormalizeDataframe:

    @staticmethod
    def normalize_strings(df, columns,is_lower_upper):
        def process_string(string):
            if string is None:
                return np.nan
            if 'lower' in is_lower_upper:
                string = string.lower()
            elif 'upper' in is_lower_upper:
                string = string.upper()
            string = unidecode(string)
            string = re.sub(r'[^\w\s+]', '', string)
            string = string.strip()
            return string

        for col in tqdm(columns):
            if isinstance(df[col], pd.Series) and df[col].dtype == 'category':
                df[col] = df[col].apply(process_string)
        return df
    
    @staticmethod
    def process_dtypes(df, columns_not_to_process=[]):

        def get_dtype_INT(value):
            if np.iinfo(np.int8).min <= value <= np.iinfo(np.int8).max:
                return 'int8'
            elif np.iinfo(np.int16).min <= value <= np.iinfo(np.int16).max:
                return 'int16'
            elif np.iinfo(np.int32).min <= value <= np.iinfo(np.int32).max:
                return 'int32'
            elif np.iinfo(np.int64).min <= value <= np.iinfo(np.int64).max:
                return 'int64'
            else:
                return 'object'

        def get_dtype_FLOAT(value):
            if np.finfo(np.float16).min <= value <= np.finfo(np.float16).max:
                return 'float16'
            elif np.finfo(np.float32).min <= value <= np.finfo(np.float32).max:
                return 'float32'
            elif np.finfo(np.float64).min <= value <= np.finfo(np.float64).max:
                return 'float64'
            elif np.finfo(np.float128).min <= value <= np.finfo(np.float128).max:
                return 'float128'
            else:
                return 'object'
            
        colunas_alvos = [coluna for coluna in df.columns if coluna not in columns_not_to_process]

        for col in colunas_alvos:
            with contextlib.suppress(Exception):
                df[col] = pd.to_numeric(df[col]).astype('Int64')
                continue
            with contextlib.suppress(Exception):
                df[col] = pd.to_numeric(df[col]).astype('float64')

        # for col in colunas_alvos:
        #     if 'int' in df[col].dtype.name.lower():
        #         try:
        #             min_value = df[col].min()
        #             max_value = df[col].max()
        #             dtype = get_dtype_INT(max(abs(min_value), abs(max_value)))
        #             df[col] = df[col].astype(dtype)
        #         except ValueError:
        #             df[col] = df[col].astype('float64')
        #     elif 'float' in df[col].dtype.name.lower():
        #         min_value = df[col].min()
        #         max_value = df[col].max()
        #         dtype = get_dtype_FLOAT(max(abs(min_value), abs(max_value)))
        #         df[col] = df[col].astype(dtype)
        #     elif 'object' in df[col].dtype.name.lower():
        #         df[col] = df[col].astype('category')

        # Process each column
        for col in colunas_alvos:
            # Normalize columns with int values
            if 'int' in df[col].dtype.name.lower():
                try:
                    # Try to convert the column to int64
                    min_value = df[col].min()
                    max_value = df[col].max()
                    dtype = get_dtype_INT(max(abs(min_value), abs(max_value)))
                    df[col] = df[col].astype(dtype)
                except ValueError:
                    # If it is not possible to convert to int, try to convert to float possible that have NaN values
                    min_value = df[col].min()
                    max_value = df[col].max()
                    dtype = get_dtype_FLOAT(max(abs(min_value), abs(max_value)))
                    df[col] = df[col].astype(dtype)

            # Normalize columns with float values
            elif 'float' in df[col].dtype.name.lower():
                min_value = df[col].min()
                max_value = df[col].max()
                dtype = get_dtype_FLOAT(max(abs(min_value), abs(max_value)))
                df[col] = df[col].astype(dtype)

            # Normalize columns with object values
            elif 'object' in df[col].dtype.name.lower():
                df[col] = df[col].astype('category')


        return df

import requests
import pandas as pd
from datetime import datetime
import pytz

class StoresExtractorHabibs:
    def __init__(self):
        self.headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36',
        }
        self.df_lojas = self.extract_stores_data()

    def extract_stores_data(self):
        response = requests.get('https://api.habibs.com.br/v1/stores-latlon', headers=self.headers)

        dic_info_loja = {
            'store_name':'',
            'store_latitude':'',
            'store_longitude':'',
            'store_endereco':'',
            'store_telefone':'',
            'store_CEP':'',
            'store_endereco_numero':'',
            'store_codLoja':'',
            'store_tipo_atendimento':''
        }

        df_lojas = pd.DataFrame()

        for loja in eval(response.text):
            contador = 0
            for info in loja.split("|"):
                dic_info_loja[list(dic_info_loja.keys())[contador]] = info
                contador +=1

            df_lojas = pd.concat([
                df_lojas,
                pd.DataFrame([dic_info_loja])
            ])

        df_lojas = df_lojas.reset_index(drop=True)

        # Obter a data e hora atual em UTC
        data_hora_utc = datetime.now(tz=pytz.utc)
        df_lojas['data_extracao_loja_utc'] = data_hora_utc
        df_lojas.rename(columns={'store_lat':'store_latitude', 'store_long': 'store_longitude'}, inplace=True)

        return df_lojas

import requests
import json
import pandas as pd
from datetime import datetime
import pytz

# Importa a classe APIUtils
from modules.competidor.get_info_stores.utils import APIUtils

class StoresExtractorMcDonalds:
    def __init__(self, url='https://api-mcd-ecommerce-br.appmcdonalds.com/restaurants/'):
        self.url = url
        self.df_lojas = self.process_data()

    def get_response(self):
        params = {
            'area': 'MOP',
        }

        response = requests.get(self.url, params=params)
        if response.status_code < 400:
            return json.loads(response.text)
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")

    def process_data(self):

        json_response = self.get_response()

        dic_info_lojas = json_response

        list_of_dic_info_lojas = json_response

        list_palavras_chaves_lojas = ['id', 'name', 'code', 'city', 'neighborhood', 'address', 'id', 'coordinates','longitude','latitude']

        dict_keyword_and_keys_semantic_mapped = APIUtils.get_keyword_and_keys_semantic_similarity(list_palavras_chaves_lojas, dic_info_lojas)

        data_hora_utc = datetime.now(tz=pytz.utc)

        contador = 0
        df_lojas = pd.DataFrame()
        for dic_info_lojas in list_of_dic_info_lojas:
            tuple_info_lojas = APIUtils.extrair_info_api_por_chaves_similares('store_concurrent_', dic_info_lojas, dict_keyword_and_keys_semantic_mapped,None, data_hora_utc)
            df_lojas = pd.concat([df_lojas, pd.DataFrame(tuple_info_lojas[0], index=[contador])])
            contador += 1

        return df_lojas

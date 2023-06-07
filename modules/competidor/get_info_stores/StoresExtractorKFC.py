import requests
import pandas as pd
from datetime import datetime
import pytz

# Importa a classe APIUtils
from modules.competidor.get_info_stores.utils import APIUtils

class StoresExtractorKFC:
    def __init__(self):
        # Realiza uma requisição POST para obter um JSON com as informações das lojas do Burger King
        url_stores = 'https://fb.tictuk.com/webFlowAddress'
        headers = {
            'authority': 'fb.tictuk.com',
            'accept': 'application/json',
            'accept-language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
            'origin': 'https://kfcbrasil.com.br',
            'referer': 'https://kfcbrasil.com.br/',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36',
        }

        response_params = requests.get('https://kfcbrasil.com.br/params.json', headers={'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36',})

        params = {
            'chainId': response_params.json().get('c',None),
            'lang': response_params.json().get('l',None),
            'type': 'getBranchesList',
            'noFilter': 'true',
            'cust': response_params.json().get('cust',None),
        }

        response = requests.get(url_stores, params=params, headers=headers)
    
        # Converte a resposta para um dicionário
        dic_info_lojas = response.json()

        # Define uma lista de palavras-chave a serem utilizadas para procurar similaridades semânticas no JSON
        list_palavras_chaves_lojas = ['msg','id','title', 'address', 'pt_PT', 'latLng','lat','lng','longitude', 'countryCode', 'formatted', 'city']

        # Utiliza a classe APIUtils para obter um dicionário com as chaves combinadas e suas similaridades correspondentes
        dict_keyword_and_keys_semantic_mapped = APIUtils.get_keyword_and_keys_semantic_similarity(list_palavras_chaves_lojas, dic_info_lojas)

        # Obtém a data e hora atual em UTC
        data_hora_utc = datetime.now(tz=pytz.utc)

        # Chama a função extrair_info_api_por_chaves_similares usando a instância criada
        tuple_info_lojas = APIUtils.extrair_info_api_por_chaves_similares('store_concurrent_', dic_info_lojas, dict_keyword_and_keys_semantic_mapped, None, data_hora_utc)

        # Transforma a lista de dicionários em um DataFrame
        self.df_lojas = pd.DataFrame(tuple_info_lojas[1]).reset_index(drop=True)

        # Obtenha a lista de nomes das colunas
        colunas = self.df_lojas.columns.tolist()
        # Percorra a lista de nomes das colunas e atualize os nomes
        novas_colunas = []
        for coluna in colunas:
            if 'latLng' in coluna:
                if 'lat' in coluna:
                    nova_coluna = coluna.replace('latLng', 'coordinates').replace('lat', 'latitude')
                if 'lng' in coluna:
                    nova_coluna = coluna.replace('latLng', 'coordinates').replace('lng', 'longitude')
            else:
                nova_coluna = coluna
            novas_colunas.append(nova_coluna)
        
        self.df_lojas.columns = novas_colunas

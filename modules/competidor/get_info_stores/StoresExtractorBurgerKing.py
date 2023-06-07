# Importa as bibliotecas necessárias
import requests
import json
import pandas as pd
from datetime import datetime
import pytz
from spellchecker import SpellChecker
from nltk.corpus import wordnet
import nltk
import re

# Importa a classe APIUtils
from modules.competidor.get_info_stores.utils import APIUtils

# Define a classe StoresExtractorBurgerKing
class StoresExtractorBurgerKing:
    def __init__(self):
        # Realiza uma requisição POST para obter um JSON com as informações das lojas do Burger King
        url_page_bk = 'https://www.burgerking.com.br'
        headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36',
        }
        json_data = {
            'localization': {
                'lat': -23.5489000,
                'long': -46.6388000,
            },
            'address': '',
            'nonParticipatingStores': '',
        }

        response = requests.post(f'{url_page_bk}/api/nearest', headers=headers, json=json_data)
        
        # Converte a resposta para um dicionário
        dic_info_lojas = json.loads(response.text)
        
        # Define uma lista de palavras-chave a serem utilizadas para procurar similaridades semânticas no JSON
        list_palavras_chaves_lojas = ['title', 'address', 'locality', 'latitude', 'longitude', 'open', 'id', 'entries']
        
        # Utiliza a classe APIUtils para obter um dicionário com as chaves combinadas e suas similaridades correspondentes
        dict_keyword_and_keys_semantic_mapped = APIUtils.get_keyword_and_keys_semantic_similarity(list_palavras_chaves_lojas, dic_info_lojas)
        
        # Obtém a data e hora atual em UTC
        data_hora_utc = datetime.now(tz=pytz.utc)

        # Chama a função extrair_info_api_por_chaves_similares usando a instância criada
        tuple_info_lojas = APIUtils.extrair_info_api_por_chaves_similares('store_concurrent_', dic_info_lojas, dict_keyword_and_keys_semantic_mapped, None, data_hora_utc)

        # Transforma a lista de dicionários em um DataFrame
        self.df_lojas = pd.DataFrame(tuple_info_lojas[1]).reset_index(drop=True)

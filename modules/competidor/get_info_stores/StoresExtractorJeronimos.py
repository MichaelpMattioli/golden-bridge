import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class StoresExtractorJeronimos:
    def __init__(self, url_base="https://jeronimoburger.com.br"):
        self.url_base = url_base
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
        }
        self.df_info_lojas_base = pd.DataFrame()
        self.df_lojas = self.get_store_coordinates()
        self.soup_main_page = None

    def get_main_page(self):
        response_main_page = requests.get(f'{self.url_base}/restaurantes', headers=self.headers)
        self.soup_main_page = BeautifulSoup(response_main_page.text, 'html.parser')

    def get_store_links(self):
        dic_linha_info_rest_jeronimo = {}
        contador = 0

        links = self.soup_main_page.find_all("a", class_="item1-restaurante-page transition")
        for link in links:
            dic_linha_info_rest_jeronimo['nome_restaurante'] = link.text
            dic_linha_info_rest_jeronimo['link_pagina_restaurante'] = self.url_base + "/" + link.get("href")

            self.df_info_lojas_base = pd.concat([
                self.df_info_lojas_base,
                pd.DataFrame(dic_linha_info_rest_jeronimo, index=[contador])
            ])
            contador += 1

    def get_store_coordinates(self):
        self.get_main_page()
        self.get_store_links()

        # URLs fornecidos
        urls = list(self.df_info_lojas_base['link_pagina_restaurante'].values)

        # Função para realizar a requisição para uma única URL
        def make_request(url):
            headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
                }

            try:
                response = requests.get(url, headers=headers)
                # Processar a resposta, se necessário
                return response.text
            except Exception as e:
                # Lidar com erros de requisição
                return str(e)

        # Criar um executor de threads
        executor = ThreadPoolExecutor(max_workers=5)  # Defina o número máximo de threads

        # Fazer as requisições em paralelo
        futures_page_urls = [executor.submit(make_request, url) for url in urls]

        # Processar as respostas conforme elas são concluídas
        list_of_links_google_where_store = []
        for future_page in tqdm(as_completed(futures_page_urls)):

            result = future_page.result()
            soup_rest = BeautifulSoup(result, 'html.parser')
            list_of_links_google_where_store.append(soup_rest.find("a", class_="como-chegar transition").get("href").split('><span')[0])

        # Fazer as requisições em paralelo
        futures_google_coords = [executor.submit(make_request, url) for url in list_of_links_google_where_store]

        # Processar as respostas conforme elas são concluídas
        list_soupHtml_with_coords_google = []
        for future_page in tqdm(as_completed(futures_google_coords)):

            result = future_page.result()
            soup_rest = BeautifulSoup(result, 'html.parser')
            # print(soup_rest)
            list_soupHtml_with_coords_google.append(soup_rest.find("title").string)

        # print(list_soupHtml_with_coords_google)

        lat_data = []
        long_data = []

        for link in list_soupHtml_with_coords_google:
            match = re.search(r"!3d(.+)!", link)
            if match is None:
                match = re.search(r"!2d(.+)", link)
                if match is None:
                    lat_data.append(None)
                else:
                    lat_data.append(match.group(1).split('?')[0])
            else:
                lat = match.group(1)
                lat_data.append(lat)
                
            match = re.search(r"!4d(.+)", link)
            if match is None:
                match = re.search(r"!1d(.+)!", link)
                if match is None:
                    long_data.append(None)
                else:
                    long_data.append(match.group(1))
            else:
                long = match.group(1)[:len(lat)]
                long_data.append(long)

        self.df_info_lojas_base['latitude'] = lat_data
        self.df_info_lojas_base['longitude'] = long_data

        self.df_lojas = self.df_info_lojas_base.copy()

        return self.df_lojas
    
if __name__ == "__main__":
    stores_extractor = StoresExtractorJeronimos()
    df_lojas = stores_extractor.df_lojas
    # print(df_lojas.shape)
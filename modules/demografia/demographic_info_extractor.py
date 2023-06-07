import os
import time
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm

class DemographicInfoExtractor:
    """
    Class for extracting and saving demographic data
    """

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
        }
        self.information_directory = ".\\data\\demografia\\"
        self.municipal_data_file = os.path.join(self.information_directory, 'tabela_cod_regioes_IBGE.parquet')

    def request_ibge_data(self, info_cod, ibge_search_code , unique_ids, column_id):
        """
        Sends request to the IBGE API and transforms the response into a dataframe.
        """
        # Altere a linha dentro do método request_ibge_data
        concatenated_ids = '|'.join(map(str, unique_ids))
        response = requests.get(
            f'https://servicodados.ibge.gov.br/api/v1/pesquisas/indicadores/{ibge_search_code}/resultados/{concatenated_ids}',
            headers=self.headers,
        )

        if response.status_code >= 400:
            print(f"Erro para pegar as informações, status code: {response.status_code}")
            print(response.text)
            return None

        data = response.json()[0]
        return self.transform_to_dataframe(info_cod, data, column_id)

    def transform_to_dataframe(self, info_cod, data, column_id):
        """
        Transforms the data into a dataframe.
        """
        df = pd.DataFrame()
        for district in data['res']:
            row = {column_id: district['localidade']}
            for year in district['res'].keys():
                row[f'{info_cod}_{year}'] = district['res'][year]
            df = pd.concat([df, pd.DataFrame(row, index=[0])])
        return df.reset_index(drop=True)

    def get_info_codIBGE(self, info_cod, df_municipal_info, ibge_search_code, column_id):
        """
        Gets the information about a certain IBGE code.
        """
        num_slices = round(len(df_municipal_info)/500)
        parts_df_municipal_info = np.array_split(df_municipal_info, num_slices)

        df_estimated_pop = pd.DataFrame()
        for part_df_municipal_info in parts_df_municipal_info:
            unique_ids = part_df_municipal_info[column_id].unique()
            df = self.request_ibge_data(info_cod, ibge_search_code, unique_ids, column_id)
            df_estimated_pop = pd.concat([df_estimated_pop, df])

        return df_estimated_pop.reset_index(drop=True)

    def get_districts(self):
        """
        Gets the districts of Brazil.
        """
        response = requests.get('https://servicodados.ibge.gov.br/api/v1/localidades/distritos')
        return response.json()

    def save_districts_to_parquet(self, df_districts):
        """
        Saves the district information to a parquet file.
        """
        df_districts.to_parquet(self.municipal_data_file)

    def get_district_information(self):
        """
        Retrieves or generates the district information data.
        """
        if os.path.exists(self.municipal_data_file):
            return pd.read_parquet(self.municipal_data_file)

        district_data = self.get_districts()
        df_districts = pd.DataFrame()
        counter = 0
        for district in district_data:
            try:
                filho1 = district['municipio']
            except Exception:
                continue

            dic_linha_df = {
                        'id_municipio': district['municipio']['id'],
                        'nome_municipio': district['municipio']['nome'],
                        'id_microrregiao': district['municipio']['microrregiao']['id'],
                        'nome_microrregiao': district['municipio']['microrregiao']['nome'],
                        'id_UF': district['municipio']['microrregiao']['mesorregiao']['UF']['id'],
                        'sigla_UF': district['municipio']['microrregiao']['mesorregiao']['UF']['sigla'],
                        'nome_UF': district['municipio']['microrregiao']['mesorregiao']['UF']['nome'],
                        'id_regiao': district['municipio']['microrregiao']['mesorregiao']['UF']['regiao']['id'],
                        'sigla_regiao': district['municipio']['microrregiao']['mesorregiao']['UF']['regiao']['sigla'],
                        'nome_regiao': district['municipio']['microrregiao']['mesorregiao']['UF']['regiao']['nome'],
                    }

            df_districts = pd.concat([
                df_districts,
                pd.DataFrame(dic_linha_df, index=[counter])
            ])
            counter += 1

        self.save_districts_to_parquet(df_districts)
        return df_districts

    def retrieve_or_generate_ibge_data(self, info_cod, ibge_search_code, file_name):
        """
        Retrieves or generates the IBGE data.
        """
        file_path = os.path.join(self.information_directory, 'municipio', f'{file_name}.parquet')
        if os.path.exists(file_path):
            print(f'Dados de {file_name.replace("_"," ")} já existe na base de dados.')
            return

        df_districts = self.get_district_information()
        df_ibge_data = self.get_info_codIBGE(info_cod, df_districts, ibge_search_code, 'id_municipio')
        df_ibge_data.to_parquet(file_path, index=False)

    def extract_and_save_demographic_data(self):
        """
        Extracts and saves the demographic data.
        """
        ibge_codes = {
            'populacao_estimada':29171,
            'densidade_demografica' : 29168,
            'area_territorial(km²)' : 29167,
            'area_urbanizada(km²)' : 95335,
            'PIB_per_capita' : 47001,
            'IDH': 30255,
            'salario_medio_mensal_trab_formais': 29765
        }

        for info_cod, ibge_search_code in ibge_codes.items():
            file_name = info_cod.split("(")[0] + '_municipios'
            self.retrieve_or_generate_ibge_data(info_cod, ibge_search_code, file_name)

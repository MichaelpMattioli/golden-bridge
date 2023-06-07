import os
import pandas as pd
import numpy as np
import contextlib
import unicodedata
import warnings


class DemographicsProcessor:
    def __init__(self):
        warnings.filterwarnings("ignore")

        self.demographic_folder_path = r'.\data\demografia\municipio'
        self.regions_parquet_path = r'.\data\demografia\tabela_cod_regioes_IBGE.parquet'
        self.stores_path = r'.\data\dim_store\df_dim_store.parquet'
        self.path_df_sk_dim_store_unique_in_fact_plu = r".\data\df_sk_dim_store_unique_in_fact_plu.parquet"
        self.regions = self._preprocess_regions_data()
        self.ibge_by_municipality = self._preprocess_demographic_data()
        self.df_stores = self._preprocess_stores_data()
        self.df_all_data_demographic = self.merge_store_municipality_data()

    def _preprocess_regions_data(self):
        """
        Realiza o pré-processamento dos dados de regiões a partir de um arquivo parquet.
        """
        regions = pd.read_parquet(self.regions_parquet_path)
        regions = regions.sort_values(by=['id_municipio'])
        regions['id_municipio'] = regions['id_municipio'].apply(lambda x: int(str(x)[:-1]))
        regions = regions.reset_index(drop=True)
        
        return regions

    def _preprocess_demographic_data(self):
        """
        Realiza o pré-processamento dos dados demográficos a partir de vários arquivos parquet em um determinado diretório.
        """
        all_files = os.listdir(self.demographic_folder_path)
        parquet_files = [filename for filename in all_files if filename.endswith('.parquet')]
        self.regions['id_municipio'] = self.regions['id_municipio'].astype(str)

        for filename in parquet_files:
            file_path = os.path.join(self.demographic_folder_path, filename)
            current_table = pd.read_parquet(file_path)
            self.regions = self.regions.merge(current_table, on='id_municipio')

        drop_variables = self._get_drop_variables()

        for drop_columns in drop_variables:
            with contextlib.suppress(Exception):
                self.regions = self.regions.drop(columns=drop_columns)

        self.regions['nome_municipio'] = [self._remove_accents(word.lower()) for word in self.regions['nome_municipio']]
        return self.regions

    def _get_drop_variables(self):
        """
        Returns a list of columns to be dropped based on certain patterns.
        """
        drop_area_territorial = self.regions.filter(regex='area_territorial\(km²\)_((?!2017)\d{4})').columns
        drop_area_urbanizada = self.regions.filter(regex='area_urbanizada').columns
        drop_idh = self.regions.filter(regex='IDH\_((?!2010)\d{4})').columns
        drop_pib_per_capita = self.regions.filter(regex='PIB_per_capita\_((?!2010)\d{4})').columns
        drop_populacao_estimada = self.regions.filter(regex='populacao_estimada').columns
        drop_salario_medio_mensal_trab_formais = self.regions.filter(regex='salario_medio_mensal_trab_formais_((?!2010)\d{4})').columns

        return [
            drop_area_territorial,
            drop_area_urbanizada,
            drop_idh,
            drop_pib_per_capita,
            drop_populacao_estimada,
            drop_salario_medio_mensal_trab_formais,
        ]

    def _remove_accents(self, txt):
        """
        Remove os acentos de um texto.
        """
        return ''.join(c for c in unicodedata.normalize('NFD', txt) if unicodedata.category(c) != 'Mn')

    def _preprocess_stores_data(self):
        """
        Realiza o pré-processamento dos dados das lojas.
        """
        # Carrega os dados do arquivo parquet
        key = pd.read_parquet(self.path_df_sk_dim_store_unique_in_fact_plu)

        
        stores = pd.read_parquet(self.stores_path)
        # Realiza o merge com as informações das lojas
        stores = stores.merge(key, on="s_sk_dim_store")
        
        
        # stores['s_active'] = stores['s_active'].astype(str)
        # stores = stores[stores['s_active'] == 'True']
        # stores = stores.drop_duplicates('br_address_dcr')
        stores = stores.reset_index(drop=True)
        stores = stores.iloc[1:]
        stores['s_cidade_brasil'] = stores['s_cidade_brasil'].astype(str)
        stores = stores[stores['s_br_store_id'].notna()]
        stores['s_cidade_brasil'] = [self._remove_accents(word.lower()) for word in stores['s_cidade_brasil']]
        return stores

    # def merge_store_municipality_data(self): # TODO
    #     """
    #     Realiza a fusão dos dados de municípios do IBGE com os dados das lojas.
    #     """
    #     store_municipality_dict = self.df_stores[['br_store_abv', 's_cidade_brasil', ]]
    #     store_municipality_dict.columns = ['br_store_abv', 'nome_municipio']
    #     store_municipality_dict = store_municipality_dict.query('nome_municipio!="nan"')
    #     store_municipality_dict = store_municipality_dict.drop_duplicates(subset=['br_store_abv']).reset_index(drop=True)
    #     cities = self.df_stores['s_cidade_brasil'].unique()
    #     export = self.ibge_by_municipality[self.ibge_by_municipality['nome_municipio'].isin(cities)]
    #     export = export.reset_index(drop=True)
    #     export = export.merge(store_municipality_dict).drop_duplicates(subset='br_store_abv').reset_index(drop=True)

    #     return export
    
    def merge_store_municipality_data(self):
        """
        Realiza a fusão dos dados de municípios do IBGE com os dados das lojas.
        """
        store_municipality_dict = self.df_stores[['s_sk_dim_store', 'br_store_abv', 's_cidade_brasil', 's_estado_brasil', 's_regiao_brasil', 's_regiao_estrategica_brasil', 's_regional_brasil', 's_status_brasil']]
        # store_municipality_dict.columns = ['br_store_abv', 'nome_municipio']
        store_municipality_dict = store_municipality_dict.rename(columns={'s_cidade_brasil':'nome_municipio'})
        store_municipality_dict = store_municipality_dict.query('nome_municipio!="nan"')
        store_municipality_dict = store_municipality_dict.drop_duplicates(subset=['br_store_abv']).reset_index(drop=True)
        cities = store_municipality_dict['nome_municipio'].unique()
        export = self.ibge_by_municipality[self.ibge_by_municipality['nome_municipio'].isin(cities)]
        #export = export.reset_index(drop=True)
        export = export.merge(store_municipality_dict)#.drop_duplicates(subset='br_store_abv').reset_index(drop=True)
        return export
    
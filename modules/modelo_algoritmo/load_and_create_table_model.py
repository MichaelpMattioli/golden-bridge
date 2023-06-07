import pandas as pd
import numpy as np
import contextlib
import warnings
import os
import re
import requests

from modules.demografia.demographic_info_extractor import DemographicInfoExtractor
from modules.demografia.demographics_processor import DemographicsProcessor
from modules.competidor.get_competitor_distances import CompetitorProximity
from modules.clima.update_info_clima import UpdateInfoClima
from modules.clima.store_climate_info import StoreClimateInfo
from modules.metabase.MetabaseSession import MetabaseSession
# from modeles.update_data import UpdateDataMetabase

from typing import Dict
from io import StringIO

# Suprimindo avisos
warnings.filterwarnings("ignore")

class DataProcessor:
    def __init__(self, filters, periodicidade, token_metabase):
        print(50*'-')
        print('Inciando DataProcessor')
        
        self.token_metabase = token_metabase
        self.filters = filters
        self.periodicidade = periodicidade
        
        self.path_gc = r".\data\dim_gc+calendario\dim_gc_calendario.parquet"
        self.path_df_sk_dim_store_unique_in_fact_plu = r".\data\df_sk_dim_store_unique_in_fact_plu.parquet"
        self.path_df_dim_store = r".\data\dim_store\df_dim_store.parquet"
        self.path_df_fact_plu_agrupado_day_s_sk_dim_store = r".\data\fact_plu_agrupado_day por s_sk_dim_store"
        self.path_dim_product = r".\data\dim_product\dim_product.parquet"
        self.path_dim_segment = r".\data\dim_segment\dim_segment.parquet"
        self.path_dim_date = r".\data\dim_date\dim_date.parquet"

        self.dim_store = self.tratamento_dim_store()
        self.target = ["sum|sum_plu_net_sale_calculado", "margem_plu", "gc_count"]

        print('\tInciando CompetitorProximity')
        print(50*'-')
        self.competitor_instance = CompetitorProximity()
        
        # Chama o método process() e recebe o dicionário retornado
        process_dict = self.process()
        print('Finalizando DataProcessor')
        print(50*'-')
        
        # Atribui os valores do dicionário aos atributos da classe
        self.produto = process_dict['produto']
        self.demo = process_dict['demo']
        self.clima = process_dict['clima'] 
        self.compt = process_dict['compt'] 
        self.camp = process_dict['camp'] 
        self.df_model = process_dict['df_model']
        self.grouped_df = process_dict['grouped_df'] 
    
    
    def tratamento_dim_store(self):
        """
        Realiza o tratamento da tabela de dimensão `dim_store`.

        Parâmetros:
            - self: Referência à instância da classe atual.

        Retorna:
            pandas.DataFrame: A tabela `dim_store` após o tratamento.

        Exemplo de uso:
            >>> objeto.tratamento_dim_store()

        Observações:
            - Certifique-se de que os caminhos para os arquivos parquet estejam corretamente definidos antes de chamar esta função.
            - Esta função realiza as seguintes etapas de tratamento:
                1. Carrega a tabela de dimensão `dim_store` a partir de um arquivo parquet.
                2. Seleciona as colunas relevantes da tabela.
                3. Carrega os dados de outra tabela parquet.
                4. Realiza o merge com as informações das lojas.
                5. Identifica as colunas que contêm valores nulos.
                6. Filtra a tabela `dim_store` para incluir apenas as linhas cujo valor de `s_sk_dim_store` está presente na outra tabela.
                7. Define as categorias da coluna 's_regiao_estrategica_brasil'.
                8. Preenche os valores ausentes com 'A DEFINIR'.
                9. Remove as linhas duplicadas com base na coluna 'br_store_abv'.
                10. Remove a coluna 'count'.

            - Certifique-se de que os arquivos parquet estejam corretamente carregados antes de chamar esta função.
        """
        dim_store = pd.read_parquet(self.path_df_dim_store)
        dim_store = dim_store[['s_sk_dim_store','br_store_abv', 's_cidade_brasil', 's_estado_brasil', 's_regiao_brasil', 's_regiao_estrategica_brasil', 's_regional_brasil', 's_status_brasil', 's_type_store_brazil']]

        # Carrega os dados do arquivo parquet
        df_sk_dim_store_unique_in_fact_plu = pd.read_parquet(self.path_df_sk_dim_store_unique_in_fact_plu)

        # Realiza o merge com as informações das lojas
        dim_store = dim_store.merge(df_sk_dim_store_unique_in_fact_plu, on="s_sk_dim_store")

        colunas_nulas = dim_store.columns[dim_store.isnull().any()].tolist()

        dim_store_by_list_available = dim_store.loc[dim_store['s_sk_dim_store'].isin(df_sk_dim_store_unique_in_fact_plu['s_sk_dim_store'])]

        # Definir as categorias da coluna 's_regiao_estrategica_brasil'
        for col in colunas_nulas:
            dim_store_by_list_available.loc[:, col] = dim_store_by_list_available.loc[:, col].astype('category')
            dim_store_by_list_available.loc[:, col] = dim_store_by_list_available.loc[:, col].cat.add_categories('A DEFINIR')

            # Preencher os valores ausentes com 'A DEFINIR'
            dim_store_by_list_available.loc[:, col] = dim_store_by_list_available.loc[:, col].fillna('A DEFINIR')

        return dim_store_by_list_available.drop_duplicates('br_store_abv').drop(columns=['count'])

    
    
    def filter_and_groupby(self, df, agg_dict, date_col=None):
        """
        Filtra e agrupa um DataFrame com base nos filtros definidos, calcula as métricas de agregação especificadas
        e retorna o DataFrame resultante.

        Args:
            df (DataFrame): DataFrame de origem para filtragem e agrupamento.
            agg_dict (dict): Dicionário com as métricas de agregação a serem calculadas no formato {coluna: função}.
            date_col (str, optional): Nome da coluna de data a ser considerada no agrupamento. Defaults para None.

        Returns:
            DataFrame: DataFrame resultante após filtragem, agrupamento e cálculo das métricas de agregação.
        """
        # Inicia com o DataFrame completo
        filtered_df = df.copy()

        # Cria uma nova coluna "filtros_usados" com os valores de filtro
        filter_values_strings = []

        # Filtra o DataFrame com base nos filtros
        for k, v in self.filters.items():
            if k == 'd_date':
                continue
            if isinstance(v, list):
                filtered_df = filtered_df[filtered_df[k].isin(v)]
                filter_values_strings.append('_'.join(v))
            else:
                filtered_df = filtered_df[filtered_df[k] == v]
                filter_values_strings.append(str(v))

        filter_values_string = '_'.join(filter_values_strings)

        filtered_df = filtered_df.assign(filtros_usados=filter_values_string)

        # Define as colunas de agrupamento
        groupby_cols = ["filtros_usados"]
        if date_col and date_col in df.columns:
            groupby_cols.append(date_col)

        # Agrupa o DataFrame filtrado e calcula as métricas de agregação
        grouped_df = filtered_df.groupby(groupby_cols).agg(agg_dict).reset_index().dropna()

        # Define "filtros_usados" e "date_col" (se existir) como índices
        grouped_df.set_index(groupby_cols, inplace=True)

        # Remove apenas as colunas usadas para o filtro que estão presentes no DataFrame
        cols_to_remove = [col for col in self.filters.keys() if col in grouped_df.columns]
        grouped_df = grouped_df.drop(columns=cols_to_remove)

        # Retorna o DataFrame resultante
        return grouped_df

    
    # def get_dim_store_with_fact_plu_avaliable(self):

    #     df_sk_dim_store_uni_fact_plu = pd.read_parquet(self.path_df_sk_dim_store_unique_in_fact_plu)

    #     path_data_fact_plu_by_sk_dim = r'.\data\fact_plu_agrupado_day por s_sk_dim_store' 
    #     file_pattern = r'df_fact_plu_agrupado_day_s_sk_dim_store_(\d+)\.parquet'
    #     file_list = os.listdir(path_data_fact_plu_by_sk_dim)

    #     s_sk_dim_store_list_avaliable = []

    #     for file_name in file_list:
    #         if match := re.match(file_pattern, file_name):
    #             s_sk_dim_store = match[1]
    #             s_sk_dim_store_list_avaliable.append(int(s_sk_dim_store))

    #     dim_store = pd.read_parquet(r'.\data\dim_store\df_dim_store.parquet')

    #     return dim_store.loc[dim_store['s_sk_dim_store'].isin(s_sk_dim_store_list_avaliable)].filter(regex='abv|s_sk_dim_store')
        
    def load_dataframes(self):
        """
        Carrega os dataframes com base no ano e lista de lojas fornecidos.

        Args:
            year (int): Ano máximo dos dados a serem carregados.
            store_list (list): Lista de abreviações de lojas.

        Returns:
            pandas.DataFrame: DataFrame concatenado contendo os dados carregados.
        """
        

        def corrigir_caracteres_manualmente(texto):
            """
            Corrige manualmente caracteres incorretos em um texto.

            Args:
                texto (str): Texto contendo caracteres incorretos.

            Returns:
                str: Texto corrigido.
            """
            # Substitui os caracteres incorretos
            texto = texto.replace("Ã£o", "ão")
            texto = texto.replace("Ã©", "é")

            return texto
        
        def calculate_columns(df):
            """
            Calcula colunas adicionais com base nas colunas existentes de um DataFrame.

            Args:
                df (DataFrame): DataFrame contendo as colunas necessárias para o cálculo.

            Returns:
                DataFrame: DataFrame com as colunas adicionais calculadas.
            """
            # Calcula as colunas adicionais
            df["mean|media_plu_unit_price_calculado"] = (df["sum|sum_plu_unit_price"] / df["sum|sum_plu_unit_sold"])
            df["sum_plu_gross_sale_calculado"] = (df["mean|media_plu_unit_sold"] * df["sum|sum_plu_unit_price"])
            df["sum|sum_plu_net_sale_calculado"] = (df["sum_plu_gross_sale_calculado"] - df["sum|sum_plu_val_tax"])
            df["margem_plu"] = (df["sum|sum_plu_net_sale_calculado"] - df["sum|sum_plu_cost_total"]) / df["sum|sum_plu_net_sale_calculado"]
            df["imposto_medio_por_item_vendido"] = (df["sum|sum_plu_val_tax"] / df["sum|sum_plu_unit_sold"])
            df["sum|sum_plu_cost_total_calculado"] = df["sum|sum_plu_unit_sold"] * df["mean|cost_plu_unit"]
            df["media_plu_unit_net_sale"] = (df["mean|media_plu_unit_price_calculado"] - df["imposto_medio_por_item_vendido"])

            return df
        

        # dim_store = self.get_dim_store_with_fact_plu_avaliable()
        dim_store = self.dim_store
        # Lista de colunas que você quer acessar
        colunas = [
            "d_sk_dim_date",
            "p_sk_dim_product",
            "s_sk_dim_store",
            "s_sk_dim_segment",
            "mean|cost_plu_unit",
            "mean|media_plu_unit_price",
            "mean|media_plu_unit_sold",
            "sum|sum_plu_cost_total",
            "sum|sum_plu_net_sale",
            "sum|sum_plu_unit_price",
            "sum|sum_plu_unit_sold",
            "sum|sum_plu_val_tax",
        ]
        
        # df = pd.read_parquet(r'.\data\df_product.parquet')
##############
        def get_dataframe(data_get_data: Dict, metabase_session_id: str) -> pd.DataFrame:
            headers_get_data = {
                'Accept': '*/*',
                'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36 Edg/113.0.1774.35',
            }

            cookies_get_data = {
                'metabase.SESSION': metabase_session_id,
            }

            response_get_data = requests.post('https://adbikpi.arcosdorados.net/api/dataset/csv', cookies=cookies_get_data, headers=headers_get_data, data=data_get_data, verify=False)

            data_string = response_get_data.text


            data_string = response_get_data.text

            # Use StringIO para ler a string como um arquivo
            data = StringIO(data_string)

            # Use a função read_csv para ler a string como um DataFrame
            return pd.read_csv(data, sep=",")


        def get_product_info_table_by_filters(filters, metabase_session_id):
            string_where_query = 'WHERE+'
            contador = 0
            df_clima = pd.read_parquet('data\clima\BRASIL.parquet')
            data_inicial = df_clima.index.min().strftime('%Y-%m-%d')
            data_final = df_clima.index.max().strftime('%Y-%m-%d')
            # data_inicial = filters['d_date'][0]
            # data_final = filters['d_date'][1]
            
            for column_dim_store_key, value in filters.items():
                contador += 1 
                if column_dim_store_key != 'd_date':
                    string_condition = ''.join(
                        f"""%5C%22Lojas%5C%22.%5C%22{column_dim_store_key}%5C%22+%3D+%27{item}%27+OR+"""
                        for item in value
                    )
                else:
                    counter_pos_d_date = 0
                    for item in value:
                        if counter_pos_d_date == 0:
                            string_condition = ''.join(
                                f"""%5C%22Calendario%5C%22.%5C%22d_date%5C%22+%3E%3D+%27{item}%27+AND+"""
                            )
                            
                        else:
                            string_condition = ''.join(
                                f"""%5C%22Calendario%5C%22.%5C%22d_date%5C%22+%3C%3D+%27{item}%27+AND+"""
                            )
                        counter_pos_d_date += 1
                        string_condition = f'%28{string_condition[:-4]}%29'
                        string_where_query += f'{string_condition}+AND+'
                    continue

                string_condition = f'%28{string_condition[:-4]}%29'
                string_where_query += f'{string_condition}+AND+'
            
            string_where_query = '' if contador == 0 else f'{string_where_query[:-5]}'
            first_part_query = 'query=%7B%22type%22%3A%22native%22%2C%22native%22%3A%7B%22query%22%3A%22'
            select_part_query = '''SELECT+
%5C%22PLUS%5C%22.%5C%22p_familia_brasil%5C%22+AS+%5C%22PLUS__p_familia_brasil%5C%22%2C+
%5C%22dim_segment%5C%22.%5C%22segment%5C%22+AS+%5C%22dim_segment__segment%5C%22%2C+
%5C%22Calendario%5C%22.%5C%22d_date%5C%22+AS+%5C%22Calendario__d_date%5C%22%2C+
avg%28%5C%22public%5C%22.%5C%22fact_plu_agrupado_day%5C%22.%5C%22cost_plu_unit%5C%22%29+AS+%5C%22MEAN|cost_plu_unit%5C%22%2C+
sum%28%5C%22public%5C%22.%5C%22fact_plu_agrupado_day%5C%22.%5C%22sum_plu_unit_price%5C%22%29+AS+%5C%22SUM|sum_plu_unit_price%5C%22%2C+
sum%28%5C%22public%5C%22.%5C%22fact_plu_agrupado_day%5C%22.%5C%22sum_plu_unit_sold%5C%22%29+AS+%5C%22SUM|sum_plu_unit_sold%5C%22%2C+
sum%28%5C%22public%5C%22.%5C%22fact_plu_agrupado_day%5C%22.%5C%22sum_plu_val_tax%5C%22%29+AS+%5C%22SUM|sum_plu_val_tax%5C%22%2C+
sum%28%5C%22public%5C%22.%5C%22fact_plu_agrupado_day%5C%22.%5C%22sum_plu_cost_total%5C%22%29+AS+%5C%22SUM|sum_plu_cost_total%5C%22%2C+
avg%28%5C%22public%5C%22.%5C%22fact_plu_agrupado_day%5C%22.%5C%22media_plu_unit_price%5C%22%29+AS+%5C%22MEAN|media_plu_unit_price%5C%22%2C+
avg%28%5C%22public%5C%22.%5C%22fact_plu_agrupado_day%5C%22.%5C%22media_plu_unit_sold%5C%22%29+AS+%5C%22MEAN|media_plu_unit_sold%5C%22%2C+
sum%28%5C%22public%5C%22.%5C%22fact_plu_agrupado_day%5C%22.%5C%22sum_plu_net_sale%5C%22%29+AS+%5C%22SUM|sum_plu_net_sale%5C%22%5Cn'''.replace('\n', '')
            from_part_query = 'FROM+%5C%22public%5C%22.%5C%22fact_plu_agrupado_day%5C%22%5Cn'
            join_part_query = 'LEFT+JOIN+%5C%22public%5C%22.%5C%22dim_product%5C%22+%5C%22PLUS%5C%22+ON+%5C%22public%5C%22.%5C%22fact_plu_agrupado_day%5C%22.%5C%22p_sk_dim_product%5C%22+%3D+%5C%22PLUS%5C%22.%5C%22p_sk_dim_product%5C%22+LEFT+JOIN+%5C%22public%5C%22.%5C%22dim_date%5C%22+%5C%22Calendario%5C%22+ON+%5C%22public%5C%22.%5C%22fact_plu_agrupado_day%5C%22.%5C%22d_sk_dim_date%5C%22+%3D+%5C%22Calendario%5C%22.%5C%22d_sk_dim_date%5C%22+LEFT+JOIN+%5C%22public%5C%22.%5C%22dim_segment%5C%22+%5C%22dim_segment%5C%22+ON+%5C%22public%5C%22.%5C%22fact_plu_agrupado_day%5C%22.%5C%22s_sk_dim_segment%5C%22+%3D+%5C%22dim_segment%5C%22.%5C%22s_sk_dim_segment%5C%22+LEFT+JOIN+%5C%22public%5C%22.%5C%22dim_store%5C%22+%5C%22Lojas%5C%22+ON+%5C%22public%5C%22.%5C%22fact_plu_agrupado_day%5C%22.%5C%22s_sk_dim_store%5C%22+%3D+%5C%22Lojas%5C%22.%5C%22s_sk_dim_store%5C%22%5Cn'
            end_part_query = '%22%2C%22template-tags%22%3A%7B%7D%7D%2C%22database%22%3A2%2C%22middleware%22%3A%7B%22js-int-to-string%3F%22%3Atrue%2C%22add-default-userland-constraints%3F%22%3Atrue%7D%7D'
            group_by_order_by_query = '%5CnGROUP+BY+%5C%22PLUS%5C%22.%5C%22p_familia_brasil%5C%22%2C+%5C%22dim_segment%5C%22.%5C%22segment%5C%22%2C+%5C%22Calendario%5C%22.%5C%22d_date%5C%22%5CnORDER+BY+%5C%22PLUS%5C%22.%5C%22p_familia_brasil%5C%22+ASC%2C+%5C%22dim_segment%5C%22.%5C%22segment%5C%22+ASC%2C+%5C%22Calendario%5C%22.%5C%22d_date%5C%22+ASC'
            data = first_part_query + select_part_query + from_part_query + join_part_query +string_where_query + group_by_order_by_query + end_part_query
            
            return get_dataframe(data, metabase_session_id)


        

        df = get_product_info_table_by_filters(self.filters, self.token_metabase)
###########################
        # df = self.UpdateDataMetabase(self.filters, self.periodicidade) # todo metabase
        
        # Converte a coluna 'calendario__d_date' para o tipo de data
        df["calendario__d_date"] = pd.to_datetime(df["calendario__d_date"])

        # Cria uma nova coluna "filtros_usados" com os valores de filtro
        filter_values_strings = []

        # Filtra o DataFrame com base nos filtros
        for k, v in self.filters.items():
            if 'd_date' == k:
                continue
            if isinstance(v, list):
                filter_values_strings.append('_'.join(v))
            else:
                filter_values_strings.append(str(v))

        filter_values_string = '_'.join(filter_values_strings)

        df["filtros_usados"] = filter_values_string
        
        # Ordena o DataFrame por data
        df = df.sort_values(by="calendario__d_date", ascending=True).reset_index(drop=True)
        
        # Corrige caracteres manualmente na coluna 'dim_segment__segment'
        df["dim_segment__segment"] = df["dim_segment__segment"].apply(corrigir_caracteres_manualmente)
        
        # Retorna os DataFrames concatenados
        return calculate_columns(df)

    def get_product_dictionary(self, df):
        """
        Obtém o dicionário de produtos contendo as colunas 'plus__p_familia_brasil' e 'plus__p_long_description',
        removendo duplicatas e redefinindo o índice.

        Args:
            df (DataFrame): DataFrame contendo as colunas necessárias para o dicionário de produtos.

        Returns:
            DataFrame: Dicionário de produtos com as colunas 'plus__p_familia_brasil' e 'plus__p_long_description'.
        """
        return (
            df[[
                "plus__p_familia_brasil", 
                "plus__p_long_description"
                ]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

    def group_by_date(self, df):
        """
        Realiza o agrupamento de um DataFrame por data ('calendario__d_date') e calcula a soma das colunas numéricas para cada data.

        Args:
            df (DataFrame): DataFrame contendo as colunas necessárias para o agrupamento.

        Returns:
            DataFrame: DataFrame original com as colunas relevantes e DataFrame agrupado por data.
        """

        # Colunas relevantes para agrupamento
        columns_booking = [
            "filtros_usados",
            "mean|cost_plu_unit",
            "calendario__d_date",
            "mean|media_plu_unit_sold",
            "sum|sum_plu_unit_price",
            "sum|sum_plu_unit_sold",
            "sum|sum_plu_val_tax",
            "dim_segment__segment",
            "plus__p_familia_brasil",
            "sum|sum_plu_net_sale_calculado",
            "margem_plu",
            "imposto_medio_por_item_vendido",
            "sum|sum_plu_cost_total_calculado",
            "media_plu_unit_net_sale",
            "mean|media_plu_unit_price_calculado",
        ]

        # Filtra as colunas relevantes
        df = df[columns_booking]

        # Agrupa o DataFrame por 'calendario__d_date' e realiza a soma das colunas numéricas
        df_grouped = df.groupby("calendario__d_date").sum(numeric_only=True)

        # Calcula as colunas adicionais 'net_dia' e 'margem_dia'
        df_grouped["net_dia"] = df_grouped["sum|sum_plu_net_sale_calculado"]
        df_grouped["margem_dia"] = (
            df_grouped["sum|sum_plu_net_sale_calculado"]
            - df_grouped["sum|sum_plu_cost_total_calculado"]
        ) / df_grouped["sum|sum_plu_net_sale_calculado"]

        return df, df_grouped

    def load_gc_data(self):
        """
        Carrega os dados do Google Cloud a partir de um arquivo parquet,
        realiza algumas transformações e retorna um DataFrame com as informações relevantes.

        Args:
            file_path (str): Caminho do arquivo parquet contendo os dados do Google Cloud.
            store_list (list): Lista de abreviações das lojas desejadas.

        Returns:
            DataFrame: DataFrame com as informações do Google Cloud filtradas e transformadas.
        """



        # Carrega os dados do arquivo parquet
        # df_sk_dim_store_in_fact_plu = pd.read_parquet(self.path_df_sk_dim_store_unique_in_fact_plu)
        # dim_store = pd.read_parquet(self.path_df_dim_store)

        # string_loc = ''
        # for key in self.filters.keys():
        #     if 'd_date' == key:
        #         continue
        #     string_loc += f"(dim_store_in_fact_plu['{key}'].isin(self.filters['{key}'])) & "

        # dim_store_in_fact_plu = dim_store.loc[dim_store['s_sk_dim_store'].isin(df_sk_dim_store_in_fact_plu['s_sk_dim_store'])]

        # dim_store = dim_store_in_fact_plu.loc[eval(string_loc[:-2])]
        dim_store = self.dim_store
        dim_store = dim_store[["s_sk_dim_store", "br_store_abv"]]

        # Lista de colunas que você quer acessar
        colunas = ["d_sk_dim_date", "s_sk_dim_store", "gc_count"]

        # Carrega os dados do arquivo parquet
        gc = pd.read_parquet(self.path_gc)

        # Verifica se as colunas existem no DataFrame
        colunas_existentes = [col for col in colunas if col in gc.columns]

        # Acessa somente as colunas que existem
        gc = gc[colunas_existentes]

        # Realiza o merge com as informações das lojas
        gc = gc.merge(dim_store)

        # Converte a coluna 'd_sk_dim_date' para o tipo de string
        gc["d_sk_dim_date"] = gc["d_sk_dim_date"].astype(str)

        # Carrega as informações de data
        dim_date = pd.read_parquet(self.path_dim_date)
        dim_date = dim_date[["d_date", "d_sk_dim_date"]]
        dim_date["d_sk_dim_date"] = dim_date["d_sk_dim_date"].astype(str)

        # Realiza o merge com as informações de data
        gc = gc.merge(dim_date, on="d_sk_dim_date")
        gc = gc.rename(columns={"d_date": "calendario__d_date"})
        gc = gc.drop(columns="d_sk_dim_date")
        gc = gc.drop(columns="s_sk_dim_store")

        # Converte a coluna 'calendario__d_date' para o formato de data
        gc["calendario__d_date"] = pd.to_datetime(gc["calendario__d_date"])

        # Agrupa os dados por data e calcula a soma
        gc = gc.groupby(by="calendario__d_date").sum(numeric_only=True)

        # Remove as colunas desnecessárias
        with contextlib.suppress(Exception):
            gc = gc.drop(columns=["lojas__br_store_abv"])

        # Retorna o DataFrame resultante
        return gc

    def preprocess_product_data(self, df):
        """
        Realiza o pré-processamento dos dados relacionados aos produtos.

        Args:
            df (DataFrame): DataFrame contendo os dados dos produtos.
            target (str): Nome da coluna alvo.

        Returns:
            DataFrame: DataFrame com os dados dos produtos pré-processados.
        """

        # Remove a última coluna (gc) do alvo
        target = self.target[:-1]

        # Remove as colunas de destino do DataFrame
        df_without_target = df.drop(target, axis=1)
        
        # Renomeia as colunas adicionando o prefixo 'PRODUTO_'
        df_without_target.columns = [f"PRODUTO_{col}" for col in df_without_target.columns]

        # Renomeia as colunas 'PRODUTO_calendario__d_date' e 'PRODUTO_lojas__br_store_abv'
        df_without_target = df_without_target.rename(
            columns={
                "PRODUTO_calendario__d_date": "calendario__d_date",
                "PRODUTO_filtros_usados": "filtros_usados",
            }
        )

        # Define as colunas 'lojas__br_store_abv' e 'calendario__d_date' como índice
        df_without_target.set_index(["filtros_usados", "calendario__d_date"], inplace=True)
        
        # Misturando 3 categorias (unitários, média e soma), faz sentido levar os 3? lembrando que os targets estão agregados por dia
        df_without_target = df_without_target.drop(
            columns=[
                "PRODUTO_mean|media_plu_unit_sold",
                "PRODUTO_imposto_medio_por_item_vendido",
                "PRODUTO_media_plu_unit_net_sale",
                "PRODUTO_mean|media_plu_unit_price_calculado",
            ]
        )

        return df_without_target

    def load_demographic_data(self):
        """
        Carrega os dados demográficos a partir de um arquivo parquet e realiza algumas transformações nos dados.

        Args:
            file_path (str): Caminho para o arquivo parquet contendo os dados demográficos.

        Returns:
            DataFrame: DataFrame contendo os dados demográficos tratados.
        """

        # Criar a instância da classe
        extractor = DemographicInfoExtractor()
        
        # Extrair e salvar os dados demográficos
        extractor.extract_and_save_demographic_data()

        # Instancia a classe de processamento dos dados demográficos
        demographics = DemographicsProcessor()
        demo = demographics.df_all_data_demographic

        # Colunas a serem descartadas
        drop_columns = [
            "id_municipio",
            "nome_municipio",
            "id_microrregiao",
            "nome_microrregiao",
            "id_UF",
            "sigla_UF",
            "nome_UF",
            "id_regiao",
            "sigla_regiao",
            "nome_regiao",
        ]

        # Remove as colunas desnecessárias
        demo = demo.drop(columns=drop_columns)

        # Converter as colunas para float
        for column in demo.columns:
            with contextlib.suppress(ValueError, TypeError):
                demo[column] = demo[column].astype(float)
        numeric_columns = demo.select_dtypes(include='number').columns
        
        agg_dict = {numeric_column: 'mean' for numeric_column in numeric_columns}
        demo = self.filter_and_groupby(demo, agg_dict)
        
        demo = demo.drop(columns=['s_sk_dim_store'])
        demo.columns = [f"DEMO_{col}" for col in demo.columns]
        return demo

    def load_climate_data(self):
        """
        Carrega os dados climáticos a partir de um arquivo parquet e realiza algumas transformações nos dados.

        Args:
            file_path (str): Caminho para o arquivo parquet contendo os dados climáticos.

        Returns:
            DataFrame: DataFrame contendo os dados climáticos tratados.
        """
        print('\tIniciando UpdateInfoClima')
        _ = UpdateInfoClima()
        
        print('\tIniciando StoreClimateInfo')
        print(50*'-')
        self.store_climate_info = StoreClimateInfo(self.competitor_instance.df_lojas_mcd)
        
        # Reinicia o índice do DataFrame
        df_clima = self.store_climate_info.df_climate_data.reset_index()
        
        df_clima = df_clima.filter(regex="lojas|calendario|Temp Max|Radiacao|Umidade Relativa|Precipitacao")
        
        # Carrega os dados do arquivo parquet
        # key = pd.read_parquet(self.path_df_sk_dim_store_unique_in_fact_plu)
        # dim_store = pd.read_parquet(self.path_df_dim_store)

        # # Realiza o merge com as informações das lojas
        # dim_store = dim_store.merge(key, on="s_sk_dim_store")
        dim_store = self.dim_store[['br_store_abv', 's_cidade_brasil', 's_estado_brasil', 's_regiao_brasil', 's_regiao_estrategica_brasil', 's_regional_brasil', 's_status_brasil', ]]
        
        df_clima = df_clima.rename(columns = {'lojas__br_store_abv':'br_store_abv'})
        df_clima = df_clima.merge(dim_store, on='br_store_abv')

        # Calcular a média das colunas numéricas por 's_regiao_estrategica_brasil'
        numeric_columns = df_clima.select_dtypes(include='number').columns
        
        # display(df_clima)
        agg_dict = {numeric_column: 'mean' for numeric_column in numeric_columns}
        df_clima = self.filter_and_groupby(df_clima, agg_dict, 'calendario__d_date')
        df_clima.columns = [f"CLIMA_{col}" for col in df_clima.columns]
        # display(df_clima)
        return df_clima

    def load_competidores_data(self):
        """
        Carrega os dados dos competidores a partir de um arquivo parquet,
        realiza algumas transformações e retorna um DataFrame com as informações dos competidores.

        Args:
            file_path (str): Caminho do arquivo parquet contendo os dados dos competidores.
            lojas (list): Lista de abreviações das lojas desejadas.

        Returns:
            DataFrame: DataFrame com as informações dos competidores filtradas e transformadas.
        """

        # Carrega os dados do arquivo parquet
        df_c = self.competitor_instance.df_distance_mcd_concurrent

        # Renomeia as colunas adicionando um prefixo 'COMPT_'
        df_c.columns = [f"COMPT_{col}" for col in df_c.columns]

        # Renomeia a coluna 'COMPT_sigla_mcd' para 'lojas__br_store_abv'
        df_c = df_c.rename(columns={'COMPT_sigla_mcd': 'lojas__br_store_abv'})

        # Remove a coluna 'COMPT_data_extracao_UTC'
        df_c = df_c.drop(columns='COMPT_data_extracao_UTC')

        # Carrega os dados do arquivo parquet
        # key = pd.read_parquet(self.path_df_sk_dim_store_unique_in_fact_plu)
        # dim_store = pd.read_parquet(self.path_df_dim_store)

        # # Realiza o merge com as informações das lojas
        # dim_store = dim_store.merge(key, on="s_sk_dim_store")
        dim_store = self.dim_store[['br_store_abv', 's_cidade_brasil', 's_estado_brasil', 's_regiao_brasil', 's_regiao_estrategica_brasil', 's_regional_brasil', 's_status_brasil', ]]
        
        df_c = df_c.rename(columns = {'lojas__br_store_abv':'br_store_abv'})
        df_c = df_c.merge(dim_store, on='br_store_abv')

        # Calcular a média das colunas numéricas por 's_regiao_estrategica_brasil'
        numeric_columns = df_c.select_dtypes(include='number').columns
        
        agg_dict = {numeric_column: 'mean' for numeric_column in numeric_columns}
        df_c = self.filter_and_groupby(df_c, agg_dict)

        return df_c

    def resample_by_segment(self, df):
        # Obter os valores únicos na coluna 'PRODUTO_product_segment'
        unique_segments = df['PRODUTO_product_segment'].unique()
        
        # Criar uma lista para armazenar cada DataFrame resample
        df_list = []

        # Para cada segmento único, crie um DataFrame separado, aplique resample e adicione à lista
        for segment in unique_segments:
            df_segment = df[df['PRODUTO_product_segment'] == segment]

            # Remova a coluna 'PRODUTO_product_segment' para agora
            df_segment = df_segment.drop(columns=['PRODUTO_product_segment'])

            # Realize o resample
            agg_dict = {col: 'sum' if col.startswith('PRODUTO_') else 'mean' for col in df_segment.columns}
            # df_resample = df_segment.resample(self.periodicidade).agg(agg_dict)
            df_resample = df_segment.resample(self.periodicidade).agg(agg_dict)

            # Adicione a coluna 'PRODUTO_product_segment' de volta
            df_resample['PRODUTO_product_segment'] = segment

            # Adicione o DataFrame resample à lista
            df_list.append(df_resample)
        return pd.concat(df_list)

    def combine_dataframes(self, df_product, df_climate, df_demo, df_compt, grouped_df):
        """
        Combina os DataFrames df_product, df_climate, df_demo e df_compt para criar um novo DataFrame consolidado.

        Args:
            df_product (DataFrame): DataFrame dos produtos.
            df_climate (DataFrame): DataFrame do clima.
            df_demo (DataFrame): DataFrame demográfico.
            df_compt (DataFrame): DataFrame dos competidores.
            grouped_df (DataFrame): DataFrame agrupado por data.

        Returns:
            DataFrame: DataFrame consolidado.
        """

        # Combina os DataFrames df_product, df_climate, df_demo e df_compt
        df_camp = pd.read_parquet('data\campanhas\campanhas_modeled_v2.parquet').reset_index()
        # tratamento v2
        # df_campanha = df_campanha.rename(columns = {'index':'calendario__d_date','CAMPAIGNS_ACTIVATED':'CAMP_Qtd_Campanhas_Ativas'})
        # tratamento v1
        # Renomeia as colunas adicionando um prefixo 'CAMP_'
        df_camp.columns = [f"CAMP_{col}" for col in df_camp.columns]
        # Renomeia a coluna 'COMPT_sigla_mcd' para 'lojas__br_store_abv'
        df_camp = df_camp.rename(columns={'CAMP_index': 'calendario__d_date'})
        df_camp['filtros_usados'] = df_demo.index[0]
        df_camp = df_camp.set_index('filtros_usados')

        df_combined = df_product.join(df_climate).reset_index()#.set_index("filtros_usados")
        df_combined = df_combined.merge(df_camp, on='calendario__d_date').reset_index(drop=True).set_index("filtros_usados")
        df_combined = df_combined.join(df_compt).reset_index().set_index("filtros_usados")
        df_combined = df_combined.join(df_demo).reset_index()
        
        # print(df_product)
        # # Removendo os dias com linhas zeras
        # df_zeros = df.loc[(df.iloc[:, 1:] == 0).all(axis=1)]
        # indices_zeros = df_zeros.index.tolist()
        # df.drop(indices_zeros, inplace=True)
        
        self.data_min = df_climate.index.get_level_values('calendario__d_date').min()
        self.data_max = df_climate.index.get_level_values('calendario__d_date').max()
        
        # df_combined = df_combined[
        #     (df_combined['calendario__d_date'] >= self.data_min) & 
        #     (df_combined['calendario__d_date'] <= self.data_max)
        # ]

        # Cria a coluna 'PRODUTO_product_segment' a partir das colunas 'PRODUTO_plus__p_familia_brasil', 'PRODUTO_dim_segment__segment' e 'filtros_usados'
        df_combined["PRODUTO_product_segment"] = (
            df_combined["PRODUTO_plus__p_familia_brasil"].astype(str)
            + "_"
            + df_combined["PRODUTO_dim_segment__segment"].astype(str)
            + "_"
            + df_combined["filtros_usados"].astype(str)
        )
        
        # Remove a coluna 'filtros_usados'
        df_combined = df_combined.drop(columns="filtros_usados")
        
        # Define 'calendario__d_date' como índice
        df_combined = df_combined.set_index(["calendario__d_date"])
        
        df_combined = df_combined.drop(columns = ['PRODUTO_dim_segment__segment', 'PRODUTO_plus__p_familia_brasil'])
        
        
        df_combined = self.resample_by_segment(df_combined)
        
        grouped_df = grouped_df.loc[
            (grouped_df.index >= self.data_min) &
            (grouped_df.index <= self.data_max)
        ]

        
        # Obtém as colunas 'margem_dia', 'net_dia' e 'gc_count' do DataFrame grouped_df
        margem_dia = grouped_df["margem_dia"].resample(self.periodicidade).mean()
        net_dia = grouped_df["net_dia"].resample(self.periodicidade).sum()
        gc_count = grouped_df["gc_count"].resample(self.periodicidade).sum()
        
        # Cria um DataFrame df_model pivotando os dados e agregando por soma
        df_model = df_combined.pivot_table(index="calendario__d_date", columns="PRODUTO_product_segment", aggfunc="sum")
        df_model.reset_index(inplace=True)
        df_model.columns = [
            f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df_model.columns
        ]

        # Concatena as colunas 'margem_dia', 'net_dia' e 'gc_count' ao DataFrame df_model
        df_model = pd.concat(
            [
                df_model,
                margem_dia.reset_index(drop=True),
                net_dia.reset_index(drop=True),
                gc_count.reset_index(drop=True),
            ],
            axis=1,
        )
        df_model = df_model.fillna(0)
        df_model = df_model.set_index("calendario__d_date")
        df_model.rename(
            columns={"net_dia": "sum|sum_plu_net_sale_calculado", "margem_dia": "margem_plu"},
            inplace=True,
        )
        return df_model, df_camp

    def process(self):
        
        print('Iniciando load_dataframes')
        print(50*'-')
        df = self.load_dataframes().reset_index(drop=True) 
        # dict_cod_prod = self.get_product_dictionary(df) # usa somente se tiver a coluna de produto
        
        print('Iniciando group_by_date')
        print(50*'-')
        df, grouped_df = self.group_by_date(df)
        
        print('Iniciando load_gc_data')
        print(50*'-')
        gc = self.load_gc_data() 
        
        print('Iniciando preprocess_product_data')
        print(50*'-')
        produto = self.preprocess_product_data(df) 
        
        print('Iniciando load_demographic_data')
        print(50*'-')
        demo = self.load_demographic_data()
        
        print(50*'-')
        print('Iniciando load_climate_data')
        clima = self.load_climate_data() 
        
        print('Iniciando load_competidores_data')
        print(50*'-')
        compt = self.load_competidores_data() 
        
        grouped_df = grouped_df.join(gc)
        with contextlib.suppress(Exception):
            grouped_df = grouped_df.drop(columns=["filtros_usados", "dim_segment__segment", "plus__p_familia_brasil"])
        
        print('Iniciando combine_dataframes')
        print(50*'-')
        df_model, camp = self.combine_dataframes(produto, clima, demo, compt, grouped_df)
        
        # display(df_model)
        # display(demo) 
        # display(compt)
        # display(clima)
        # display(gc)
        # display(produto)
        # display(grouped_df)
        
        return {
            'produto': produto,
            'demo': demo,
            'clima': clima,
            'compt': compt,
            'camp': camp,
            'df_model': df_model,
            'grouped_df':grouped_df
        }
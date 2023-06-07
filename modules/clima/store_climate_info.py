import os
import pandas as pd
import math
import datetime
import warnings
import glob
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import haversine_distances
from sklearn.neighbors import BallTree
import numpy as np

warnings.filterwarnings("ignore")

class StoreClimateInfo:
    def __init__(self, df_mcd_stores=None):
        self.path_of_climate_data = r'.\data\clima\dados meteriologicos'

        self.df_mcd_stores = df_mcd_stores

        self.dim_store_in_fact_plu = self.merge_dim_store_data(r'data\df_sk_dim_store_unique_in_fact_plu.parquet', r'data\dim_store\df_dim_store.parquet')

        self.paths_climate_input_output = self.create_paths_input_output_of_the_years()

        self.df_climate_data = self.calculate_and_generate_weather_averages_by_stores()

        
    
    def merge_dim_store_data(self,key_path, dim_store_path):
            """
            Realiza a mesclagem dos dados do arquivo de chave com os dados da dimensão de lojas.

            Args:
                key_path (str): Caminho do arquivo de chave.
                dim_store_path (str): Caminho do arquivo da dimensão de lojas.

            Returns:
                pd.DataFrame: DataFrame resultante da mesclagem dos dados.
            """
            key = pd.read_parquet(key_path)
            dim_store = pd.read_parquet(dim_store_path)
            dim_store = dim_store.merge(key, on='s_sk_dim_store')
            dim_store = dim_store[['br_store_abv', 's_latitude_brasil', 's_longitude_brasil']]
            dim_store.columns = ['Loja', 'LATITUDE', 'LONGITUDE']

            # atualiza as coordenadas das lojas que não foram encontradas no arquivo de coordenadas com as coordenadas extraidas do site do McDonald's
            if self.df_mcd_stores is not None:
                df_stores_lat_long_founded = self.df_mcd_stores.loc[self.df_mcd_stores[self.df_mcd_stores.filter(regex='code').columns.to_list()[0]].isin(dim_store[dim_store.isnull().any(axis=1)]['Loja'].to_list())].filter(regex='code|long|lat').sort_values(self.df_mcd_stores.filter(regex='code').columns.to_list())
                df_stores_lat_long_founded.rename(columns={self.df_mcd_stores.filter(regex='code').columns.to_list()[0]: 'Loja', self.df_mcd_stores.filter(regex='longitude').columns.to_list()[0]: 'LONGITUDE', self.df_mcd_stores.filter(regex='latitude').columns.to_list()[0]: 'LATITUDE'}, inplace=True)
                dim_store.set_index('Loja', inplace=True)
                df_stores_lat_long_founded.set_index('Loja', inplace=True)
                dim_store.update(df_stores_lat_long_founded)
                dim_store.reset_index(inplace=True)
            
            return dim_store.query('LATITUDE != 0.000000').reset_index(drop=True)
        

    def create_paths_input_output_of_the_years(self):
        paths_climate_input_output = []

        caminhos_parquet = glob.glob(self.path_of_climate_data + '\\*.parquet')

        for caminho in caminhos_parquet:
            ano = caminho.split('_')[-1].split('.')[0]
            path_to_save_data_climate_shaped = fr'.\data\clima\export\climate_data_of_all_stores_{ano}.parquet'
            tuple_input_output_of_the_year = (caminho, path_to_save_data_climate_shaped)
            paths_climate_input_output.append(tuple_input_output_of_the_year)
        return paths_climate_input_output
    
    def media_ponderada_k_estacoes_mais_proximas(self, k_estacoes_mais_proximas, dados_diarios, coluna):
        """
        Calcula a média ponderada para uma coluna específica considerando as estações mais próximas.

        Args:
            k_estacoes_mais_proximas (list): Lista contendo as k estações mais próximas, com suas distâncias.
            dados_diarios (pandas.DataFrame): Dataframe contendo os dados diários das estações meteorológicas.
            coluna (str): Nome da coluna para a qual se deseja calcular a média ponderada.

        Returns:
            float: Média ponderada da coluna especificada.

        """

        # Calcula os pesos inversamente proporcionais à distância para cada estação
        pesos = [1 / dist for _, dist in k_estacoes_mais_proximas]
        soma_pesos = sum(pesos)

        # Normaliza os pesos para que a soma seja igual a 1
        pesos_normalizados = [peso / soma_pesos for peso in pesos]

        # Calcula a média ponderada para a coluna especificada, considerando as estações mais próximas e seus pesos
        return sum(
            dados_diarios.loc[dados_diarios['ESTACAO'] == estacao, coluna].mean()
            * peso
            for (estacao, _), peso in zip(k_estacoes_mais_proximas, pesos_normalizados)
        )

    

    def obter_medias_meteorologicas(self, lojas, dados_diarios, limite_distancia, k):
        """
        Calcula as médias meteorológicas para cada loja com base nos dados diários e nas estações mais próximas.

        Args:
            lojas (pandas.DataFrame): DataFrame contendo as informações das lojas.
            dados_diarios (pandas.DataFrame): DataFrame contendo os dados diários das estações meteorológicas.
            limite_distancia (float): Limite de distância para considerar as estações mais próximas.
            k (int): Número de estações mais próximas a considerar.

        Returns:
            pandas.DataFrame: DataFrame com as médias meteorológicas adicionadas às lojas.

        """

        colunas_meteorologicas = [
            'Precipitacao(mm)',
            'Radiacao Global(Kj/m2)',
            'Temp Max(°C)',
            'Umidade Relativa(%)'
        ]

        # Para cada coluna meteorológica, calcula a média para cada loja
        for coluna_meteorologica in colunas_meteorologicas:
            # Calcula a média ponderada das estações mais próximas ou a média da estação mais próxima
            def calcula_media(loja):
                k_estacoes_mais_proximas = loja[f'{k}EstacoesMaisProximas']
                estacao_mais_proxima, dist_estacao_mais_proxima = k_estacoes_mais_proximas[0]

                if dist_estacao_mais_proxima > limite_distancia:
                    # Calcula a média ponderada das estações mais próximas
                    return self.media_ponderada_k_estacoes_mais_proximas(k_estacoes_mais_proximas, dados_diarios, coluna_meteorologica)
                else:
                    # Calcula a média da estação mais próxima
                    return dados_diarios.loc[dados_diarios['ESTACAO'] == estacao_mais_proxima, coluna_meteorologica].mean()

            # Aplica a função calcula_media a todas as lojas
            lojas[coluna_meteorologica] = lojas.apply(calcula_media, axis=1)

        return lojas

    
    
    def calculate_haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calcula a distância haversine entre dois pontos geográficos definidos pelas coordenadas lat/lon.

        Args:
            lat1 (float): Latitude do ponto 1 em graus decimais.
            lon1 (float): Longitude do ponto 1 em graus decimais.
            lat2 (float): Latitude do ponto 2 em graus decimais.
            lon2 (float): Longitude do ponto 2 em graus decimais.

        Returns:
            float: Distância haversine entre os dois pontos em quilômetros.
        """

        # Convertendo as coordenadas para radianos
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # Calculando as diferenças das latitudes e longitudes
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # Aplicando a fórmula haversine
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))

        # Raio médio da Terra em quilômetros
        R = 6371

        # Calculando a distância final
        return R * c


    def generate_weather_averages(self, lojas, k, input_path, output_path):
        """
        Gera médias meteorológicas para cada loja com base nos dados meteorológicos fornecidos.
        
        Se o arquivo de resultados já existir em output_path, a função retorna None.

        Args:
            lojas (pd.DataFrame): DataFrame contendo informações das lojas.
            k (int): Número de estações meteorológicas mais próximas a considerar.
            input_path (str): Caminho do diretório que contém os arquivos CSV com dados meteorológicos.
            output_path (str): Caminho para salvar o arquivo de resultados.

        Returns:
            None se o arquivo de resultados já existir.
            Caso contrário, um DataFrame com as médias meteorológicas para cada loja e um DataFrame com informações das estações meteorológicas.
        """
            
        # Se o arquivo de resultados já existe, retornar None
        if os.path.exists(output_path):
            ano_caminho = int( output_path.split('_')[-1].split('.')[0])
            ano_atual = datetime.datetime.now().year
            
            # Verifica se o ano do caminho é o ano atual
            if ano_caminho != ano_atual:
                print(f'O arquivo de resultados para {self.path_of_climate_data} já existe. Ignorando a geração de médias meteorológicas para este ano.')
                return None, None
        
        df_inmet_of_one_year = pd.read_parquet(input_path).filter(regex='Data|Precipitacao|Radiacao|Temp Max|Umidade Relativa|REGIAO|UF|ESTACAO|CODIGO|LATITUDE|LONGITUDE')  # Concatena os arquivos CSV em um único DataFrame
        df_inmet_of_one_year['Data'] = pd.to_datetime(df_inmet_of_one_year['Data'])  # Converte a coluna 'Data' para formato de data
        
        # Agrupa os dados diários por data, região, UF, estação, código WMO, latitude e longitude
        df_inmet_of_one_year = df_inmet_of_one_year.groupby([pd.Grouper(key='Data', freq='D'), 'REGIAO', 'UF', 'ESTACAO', 'CODIGO (WMO)', 'LATITUDE', 'LONGITUDE']).agg({
            'Precipitacao(mm)': 'sum',
            'Radiacao Global(Kj/m2)': 'sum',
            'Temp Max(°C)': 'max',
            'Umidade Relativa(%)': 'mean',
        }).reset_index()

        estacoes_meteorologicas = df_inmet_of_one_year[['ESTACAO', 'LATITUDE', 'LONGITUDE']].drop_duplicates()  # DataFrame com informações das estações meteorológicas
        distancias_medias = []

        # Calcula as distâncias médias entre as lojas e as estações meteorológicas
        for idx_loja, loja in lojas.iterrows():
            lat_loja, lon_loja = loja['LATITUDE'], loja['LONGITUDE']
            distancias_estacoes = []
            for idx_estacao, estacao in estacoes_meteorologicas.iterrows():
                lat_estacao, lon_estacao = estacao['LATITUDE'], estacao['LONGITUDE']
                dist = self.calculate_haversine_distance(lat_loja, lon_loja, lat_estacao, lon_estacao)
                distancias_estacoes.append(dist)
            dist_estacao_mais_proxima = min(distancias_estacoes)
            distancias_medias.append(dist_estacao_mais_proxima)

        distancia_media = sum(distancias_medias) / len(distancias_medias)  # Distância média entre as lojas e as estações meteorológicas
        # print(f'Distância média: {distancia_media} km')

        lojas[f'{k}EstacoesMaisProximas'] = None

        # Calcula as k estações meteorológicas mais próximas para cada loja
        for idx_loja, loja in lojas.iterrows():
            lat_loja, lon_loja = loja['LATITUDE'], loja['LONGITUDE']
            distancias_estacoes = []
            for idx_estacao, estacao in estacoes_meteorologicas.iterrows():
                lat_estacao, lon_estacao = estacao['LATITUDE'], estacao['LONGITUDE']
                dist = self.calculate_haversine_distance(lat_loja, lon_loja, lat_estacao, lon_estacao)
                distancias_estacoes.append((estacao['ESTACAO'], dist))
            k_estacoes_mais_proximas = sorted(distancias_estacoes, key=lambda x: x[1])[:k]
            lojas.at[idx_loja, f'{k}EstacoesMaisProximas'] = k_estacoes_mais_proximas

        resultados = []

        # Para cada data, calcula as médias meteorológicas para as lojas com base nos dados diários correspondentes
        for data in df_inmet_of_one_year['Data'].unique():
            lojas_temp = lojas.copy()
            lojas_com_medias_meteorologicas = self.obter_medias_meteorologicas(lojas_temp, df_inmet_of_one_year[df_inmet_of_one_year['Data'] == data], limite_distancia=distancia_media, k=k)
            lojas_com_medias_meteorologicas['Data'] = data
            resultados.append(lojas_com_medias_meteorologicas)

        # Concatena os resultados em um único DataFrame final
        return pd.concat(resultados, ignore_index=True), estacoes_meteorologicas


    def save_resultados_df(self, resultados_df, k, output_path):
        """
        Salva o DataFrame de resultados em um arquivo Parquet.

        Args:
            resultados_df (pd.DataFrame): DataFrame contendo os resultados.
            k (int): Número de estações mais próximas consideradas.

        Returns:
            None
        """

        # Obtém a quantidade de lojas no DataFrame de resultados
        qtd_lojas = len(resultados_df.Loja.unique())

        # Define o nome da loja ou 'BRASIL' se houver mais de uma loja
        loja = resultados_df.Loja.unique()[0] if qtd_lojas == 1 else 'BRASIL'

        # Define as colunas a serem descartadas no DataFrame de resultados
        columns_to_drop = ['LATITUDE', 'LONGITUDE', f'{k}EstacoesMaisProximas']
        resultados_df = resultados_df.drop(columns=columns_to_drop)

        # Define a coluna 'Data' como índice e renomeia para 'calendario__d_date'
        resultados_df = resultados_df.set_index('Data')
        resultados_df.index.name = 'calendario__d_date'

        # Renomeia a coluna 'Loja' para 'lojas__br_store_abv'
        resultados_df = resultados_df.rename(columns={'Loja': 'lojas__br_store_abv'})

        # Salva o DataFrame como arquivo Parquet com o nome da loja
        # resultados_df.to_parquet(rf"C:\Users\ValimKaique(BipGroup\BUSINESS INTEGRATION PARTNERS SPA\Arcos Dorados - General\03. Conteúdos do projeto\Frente 3 - Alavancas de vendas e margens\App-Golden-Bridge\data\clima\{loja}.parquet")
        resultados_df.to_parquet(output_path)
        return resultados_df
            
            
    def calculate_and_generate_weather_averages_by_stores(self):

        list_of_dfs_climate = []
        for input_path, output_path in self.paths_climate_input_output:
            if os.path.exists(output_path):
                # Lendo o arquivo de saída
                df_climate_by_store = pd.read_parquet(output_path)
                if len(df_climate_by_store.index.drop_duplicates()) < 365:

                    df_climate_by_store_base = pd.read_parquet(input_path)
                    try:
                        df_climate_by_store_base['Data'] = pd.to_datetime(df_climate_by_store_base['Data'], format='%Y-%m-%d')
                    except:
                        df_climate_by_store_base['Data'] = pd.to_datetime(df_climate_by_store_base['Data'], format='%Y/%m/%d')
                    if df_climate_by_store.index.min() == df_climate_by_store_base['Data'].min() and df_climate_by_store.index.max() == df_climate_by_store_base['Data'].max():
                        list_of_dfs_climate.append(df_climate_by_store)
                    else:
                        numero_estacoes_para_triangulacao = 3
                        resultados, _ = self.generate_weather_averages(self.dim_store_in_fact_plu, numero_estacoes_para_triangulacao, input_path, output_path)
                        if resultados is not None:
                            # Salva os resultados em um arquivo
                            df_climate_not_completed = self.save_resultados_df(resultados, numero_estacoes_para_triangulacao, output_path)
                            df_climate_by_store = df_climate_not_completed.copy()
                            list_of_dfs_climate.append(df_climate_by_store)
                else:
                    list_of_dfs_climate.append(df_climate_by_store)
        

        # Lendo e concatenando os arquivos
        return pd.concat(list_of_dfs_climate)

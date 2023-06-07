from datetime import datetime

import pandas as pd
import numpy as np
from math import atan2
import os

from modules.competidor.get_info_stores.StoresExtractorBurgerKing import StoresExtractorBurgerKing
from modules.competidor.get_info_stores.StoresExtractorHabibs import StoresExtractorHabibs
from modules.competidor.get_info_stores.StoresExtractorKFC import StoresExtractorKFC
from modules.competidor.get_info_stores.StoresExtractorMcDonalds import StoresExtractorMcDonalds
from modules.competidor.get_info_stores.StoresExtractorSubway import StoresExtractorSubway


class CompetitorProximity:
    def __init__(self):
        try:
            self.df_lojas_mcd = StoresExtractorMcDonalds().df_lojas
        except Exception as e:
            print('Warning: Não foi possível extrair as lojas do McDonalds\n{e}')
            self.df_lojas_mcd = None
        if self.df_lojas_mcd is not None:
            self.df_distance_mcd_concurrent = self.get_competitor_distances()
        else:
            self.df_distance_mcd_concurrent = None

    @staticmethod
    def _haversine_distance_km(lat1, long1, lat2, long2):
        raio_terra_km = 6371
        dif_lat = np.radians(lat2 - lat1)
        dif_long = np.radians(long2 - long1)
        a = np.sin(dif_lat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dif_long / 2) ** 2
        c = 2 * atan2(np.sqrt(a), np.sqrt(1 - a))
        return raio_terra_km * c

    def _create_df_distance(self, df_lojas_mc, df_lojas_concorrente, raio_distancia_km):

        def get_coordinates_and_id(data, is_concorrente):
                coluna_latitude = [coluna for coluna in data._fields if 'lat' in coluna]
                coluna_longitude = [coluna for coluna in data._fields if 'long' in coluna]

                if is_concorrente:
                    coluna_id = [coluna for coluna in data._fields if 'id' in coluna or 'store_name' in coluna or '']
                else:
                    coluna_id = [coluna for coluna in data._fields if 'code' in coluna ]

                latitude = getattr(data, coluna_latitude[0])
                longitude = getattr(data, coluna_longitude[0])
                id_data = getattr(data, coluna_id[0])

                return latitude, longitude, id_data

        lista_teste = []

        for linha_concorrente in df_lojas_concorrente.itertuples():
            latitude_concorrente, longitude_concorrente, id_concorrente = get_coordinates_and_id(linha_concorrente, True)
            # print(f'Concorrente -> tipo: {type(latitude_concorrente)}, valor: {latitude_concorrente} - {type(longitude_concorrente)}, valor: {longitude_concorrente}')
            for linha_mc in df_lojas_mc.itertuples():
                latitude_mc, longitude_mc, id_mc = get_coordinates_and_id(linha_mc, False)
                # print(f'McD -> tipo: {type(latitude_mc)}, valor: {latitude_mc} - {type(longitude_mc)}, valor: {longitude_mc}')
                distancia_lojas = self._haversine_distance_km(float(latitude_mc), float(longitude_mc), float(latitude_concorrente), float(longitude_concorrente))

                distancia = distancia_lojas if distancia_lojas <= raio_distancia_km else np.nan

                lista_teste.append({'sigla_mcd': id_mc, 
                                    'id_loja_concorrente': id_concorrente, 
                                    f'distancia_lojas_ate_{raio_distancia_km}km': distancia, 
                                    'distancia_lojas_km': distancia_lojas})
                    
        return pd.DataFrame(lista_teste)

    def _get_info_distance_concurrent(self, df_lojas_mc, df_lojas_concorrente, nome_concorrente, raio_distancia_km):
        df_lojas_mc = df_lojas_mc.filter(regex='code|longitude|latitude')
        df_distancia_mc_vs_concorrente = self._create_df_distance(df_lojas_mc, df_lojas_concorrente, raio_distancia_km)

        # Definir uma função que encontra a loja mais próxima em cada grupo
        def encontrar_loja_mais_proxima(df):
            # Encontrar o índice da linha com a menor distância
            idx = df['distancia_lojas_km'].idxmin()
            # Acessar o valor da coluna "id_loja_concorrente" correspondente à loja mais próxima
            distancia_loja_mais_proxima = df.loc[idx, 'distancia_lojas_km']
            # Criar uma nova coluna no dataframe com o valor da loja mais próxima
            df['distancia_loja_mais_proxima'] = distancia_loja_mais_proxima
            return df

        df_lojas_mc_distancia = df_distancia_mc_vs_concorrente.groupby('sigla_mcd').apply(encontrar_loja_mais_proxima)

        df_agrupado = df_lojas_mc_distancia.groupby('sigla_mcd').agg({
            f'distancia_lojas_ate_{raio_distancia_km}km': lambda x: x.count() if x.count() > 0 else 0,
        }).reset_index()


        return df_agrupado.rename(columns={f'distancia_lojas_ate_{raio_distancia_km}km': f'quantidade_lojas_ate_{raio_distancia_km}km_{nome_concorrente}'})

    def get_competitor_distances(self):

        path_competidores = r'data\competidores\distance_mcd_concurrent.parquet'

        if os.path.exists(path_competidores):
            return pd.read_parquet(path_competidores)

        df_lojas_bk = StoresExtractorBurgerKing()
        df_lojas_habibs = StoresExtractorHabibs()
        df_lojas_KFC = StoresExtractorKFC()
        df_lojas_subway = StoresExtractorSubway()
        df_lojas_mcdonalds = StoresExtractorMcDonalds()

        df_info_lojas_distancia_burgerKing = self._get_info_distance_concurrent(df_lojas_mcdonalds, df_lojas_bk.df_lojas, 'burgerking', 2)
        df_info_lojas_distancia_habibs = self._get_info_distance_concurrent(df_lojas_mcdonalds.df_lojas, df_lojas_habibs.df_lojas, 'habibs', 2)
        df_info_lojas_distancia_kfc = self._get_info_distance_concurrent(df_lojas_mcdonalds.df_lojas, df_lojas_KFC.df_lojas, 'kfc', 2)
        df_info_lojas_distancia_subway = self._get_info_distance_concurrent(df_lojas_mcdonalds.df_lojas, df_lojas_subway.df_lojas, 'subway', 2)

        df_info_lojas_distancia_concorrentes = df_info_lojas_distancia_burgerKing.merge(df_info_lojas_distancia_habibs, on='sigla_mcd', how='left')
        df_info_lojas_distancia_concorrentes = df_info_lojas_distancia_concorrentes.merge(df_info_lojas_distancia_kfc, on='sigla_mcd', how='left')
        df_info_lojas_distancia_concorrentes = df_info_lojas_distancia_concorrentes.merge(df_info_lojas_distancia_subway, on='sigla_mcd', how='left')
        df_info_lojas_distancia_concorrentes['data_extracao_UTC'] = datetime.utcnow()

        df_info_concorrentes_all_data = pd.read_parquet(r'data\competidores\distance_mcd_concurrent.parquet')
        # use o concat
        df_info_concorrentes_all_data = pd.concat([df_info_concorrentes_all_data, df_info_lojas_distancia_concorrentes])
        # print("PASSEI AQUI")
        df_info_concorrentes_all_data.to_parquet(r'data\competidores\distance_mcd_concurrent.parquet', index=False)
        return df_info_concorrentes_all_data
    

import contextlib
import requests
import zipfile
import os
import glob
import shutil
import pandas as pd
from datetime import datetime
from tqdm import tqdm

class UpdateInfoClima:
    """
    Classe responsável por atualizar as informações climáticas.

    Atributos:
        - url_base (str): URL base para download dos arquivos de dados climáticos.
        - extensao_arquivo (str): Extensão dos arquivos de dados climáticos.
        - ano_inicial (int): Ano inicial para busca dos dados climáticos.
        - headers (dict): Cabeçalhos HTTP para realizar a requisição.
        - ano_atual (int): Ano atual.
        - pasta_tratados (str): Caminho da pasta para armazenar os dados climáticos tratados.
        - pasta_dados_inmet (str): Caminho da pasta para armazenar os dados brutos do INMET.
        - lista_de_dataframes (list): Lista para armazenar os dataframes de dados climáticos.

    Métodos:
        - executar(): Método principal para iniciar a atualização das informações climáticas.
    """

    def __init__(self):
        """
        Inicializa uma nova instância da classe UpdateInfoClima.

        Parâmetros:
            Nenhum parâmetro é necessário para a inicialização.

        Retorna:
            Nenhum valor de retorno.
        """
        self.url_base = 'https://portal.inmet.gov.br/uploads/dadoshistoricos/'
        self.extensao_arquivo = '.zip'
        self.ano_inicial = 2020
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36 Edg/113.0.1774.50'
        }
        self.ano_atual = datetime.now().year
        self.pasta_tratados = r'data\clima\dados meteriologicos'
        self.pasta_dados_inmet = r'data\clima\dados_inmet'
        self.lista_de_dataframes = []
        self.executar()

    def baixar_arquivo(self, url, destino):
        """
        Baixa um arquivo da URL fornecida e o salva no destino especificado.

        Parâmetros:
            - self: Referência à instância da classe atual.
            - url (str): A URL do arquivo a ser baixado.
            - destino (str): O caminho completo onde o arquivo deve ser salvo.

        Retorna:
            Nenhum valor de retorno. A função faz o download do arquivo e o salva no destino especificado.

        Exemplo de uso:
            >>> objeto.baixar_arquivo('https://exemplo.com/arquivo.zip', '/caminho/do/arquivo.zip')

        Observações:
            - Certifique-se de ter permissão de acesso à URL fornecida.
            - Certifique-se de que o destino seja um caminho válido e tenha permissão de escrita.
            - A função usa o cabeçalho (headers) especificado na instância da classe atual ao fazer a solicitação.
        """
        response = requests.get(url, headers=self.headers)
        with open(destino, 'wb') as file:
            file.write(response.content)

    def descompactar_arquivo(self, destino):
        """
        Descompacta um arquivo ZIP para a pasta de dados do INMET.

        Parâmetros:
            - self: Referência à instância da classe atual.
            - destino (str): O caminho completo do arquivo ZIP a ser descompactado.

        Retorna:
            Nenhum valor de retorno. A função apenas extrai os arquivos do arquivo ZIP para a pasta de dados do INMET.

        Exemplo de uso:
            >>> objeto.descompactar_arquivo('/caminho/do/arquivo.zip')

        Observações:
            - Certifique-se de que a pasta de dados do INMET esteja configurada corretamente antes de chamar esta função.
            - O arquivo ZIP especificado deve existir e ser acessível para que a função possa realizar a extração dos arquivos.
        """
        with zipfile.ZipFile(destino, 'r') as zip_ref:
            zip_ref.extractall(self.pasta_dados_inmet)

    def tratar_arquivo(self, ano):
        """
        Trata os arquivos de dados climáticos para um determinado ano.

        Parâmetros:
            - self: Referência à instância da classe atual.
            - ano (int): O ano para o qual os arquivos de dados climáticos serão tratados.

        Retorna:
            Nenhum valor de retorno. A função processa os arquivos de dados climáticos para o ano especificado e os salva em um arquivo parquet.

        Exemplo de uso:
            >>> objeto.tratar_arquivo(2021)

        Observações:
            - Certifique-se de que a pasta de dados do INMET esteja configurada corretamente antes de chamar esta função.
            - Os arquivos de dados climáticos para o ano especificado devem estar presentes na pasta de dados do INMET.
            - Os dados tratados são salvos em um arquivo parquet com o nome "INMET_dados_metereologico_<ano>.parquet" na pasta de tratados.
        """
        pasta = self.pasta_dados_inmet
        arquivos = os.listdir(pasta)

        print(f'Atualizando dados de clima de {ano}')
        for arquivo in tqdm(arquivos):
            caminho_arquivo = fr"{pasta}\{arquivo}"
            df_info_metereologico, df_info_estacao = self.ler_arquivo(caminho_arquivo)
            df_info_metereologico = self.tratar_dataframe(df_info_metereologico, df_info_estacao)
            self.lista_de_dataframes.append(df_info_metereologico)

        pd.concat(self.lista_de_dataframes).to_parquet(f"{self.pasta_tratados}\INMET_dados_metereologico_{ano}.parquet", index=False)

    def ler_arquivo(self, caminho_arquivo):
        """
        Lê um arquivo CSV contendo informações meteorológicas e retorna dois dataframes.

        Parâmetros:
            - self: Referência à instância da classe atual.
            - caminho_arquivo (str): O caminho completo do arquivo CSV a ser lido.

        Retorna:
            Uma tupla contendo dois dataframes pandas: (df_info_metereologico, df_info_estacao).
            - df_info_metereologico: Um dataframe contendo as informações meteorológicas.
            - df_info_estacao: Um dataframe contendo informações sobre a estação meteorológica.

        Exemplo de uso:
            >>> df1, df2 = objeto.ler_arquivo('/caminho/do/arquivo.csv')

        Observações:
            - Certifique-se de que o arquivo CSV especificado exista e seja acessível para que a função possa realizar a leitura.
            - A codificação do arquivo CSV é definida como 'ISO-8859-1' para garantir a leitura correta dos caracteres.
            - O separador de colunas no arquivo CSV é definido como ';'.
            - O dataframe df_info_metereologico pula as primeiras 8 linhas do arquivo CSV.
            - O dataframe df_info_estacao lê apenas as primeiras 8 linhas do arquivo CSV, utilizando as colunas 'info' e 'value'.
        """
        df_info_metereologico = pd.read_csv(caminho_arquivo, encoding='ISO-8859-1', sep=';', skiprows=8)
        df_info_estacao = pd.read_csv(caminho_arquivo, sep=';', names=['info', 'value'], nrows=8, encoding='ISO-8859-1')
        return df_info_metereologico, df_info_estacao

    def tratar_dataframe(self, df_info_metereologico, df_info_estacao):
        """
        Realiza o tratamento de um DataFrame contendo informações meteorológicas.

        Parâmetros:
            - self: Referência à instância da classe atual.
            - df_info_metereologico (pandas.DataFrame): O DataFrame contendo as informações meteorológicas a serem tratadas.
            - df_info_estacao (pandas.DataFrame): O DataFrame contendo as informações da estação meteorológica.

        Retorna:
            pandas.DataFrame: O DataFrame `df_info_metereologico` após o tratamento.

        Exemplo de uso:
            >>> objeto.tratar_dataframe(df_info_metereologico, df_info_estacao)

        Observações:
            - Certifique-se de que os DataFrames `df_info_metereologico` e `df_info_estacao` estejam corretamente carregados antes de chamar esta função.
            - Esta função realiza as seguintes etapas de tratamento:
                1. Renomeia as colunas do DataFrame `df_info_metereologico` de acordo com um padrão específico.
                2. Remove a coluna 'nada' do DataFrame `df_info_metereologico`.
                3. Cria um dicionário com as informações da estação a partir do DataFrame `df_info_estacao`.
                4. Cria um novo DataFrame a partir do dicionário com as informações da estação.
                5. Preenche as colunas do DataFrame `df_info_metereologico` com as informações da estação.
                6. Converte as colunas do tipo 'object' para 'float', substituindo ',' por '.' se necessário.
        """
        # Renomear as colunas do DataFrame df_info_metereologico
        df_info_metereologico.columns = [
            'Data', 'Hora UTC', 'Precipitacao(mm)', 'Pressao Atm Estacao(mB)',
            'Pressao Atm Max(mB)', 'Pressao Atm Min(mB)', 'Radiacao Global(Kj/m2)',
            'Temp Ar Bulbo Seco(°C)', 'Temp Ponto de Orvalho(°C)', 'Temp Max(°C)',
            'Temp Min(°C)', 'Temp Orvalho Max(°C)', 'Temp Orvalho Min(°C)',
            'Umidade Rel Max(%)', 'Umidade Rel Min(%)', 'Umidade Relativa(%)',
            'Dir Vento(graus)', 'Rajada Max(m/s)', 'Vel Vento(m/s)', 'nada'
        ]

        # Remover a coluna 'nada' do DataFrame df_info_metereologico
        df_info_metereologico.drop('nada', axis=1, inplace=True)

        # Criar um dicionário com as informações da estação a partir do DataFrame df_info_estacao
        data = {
            row['info'].replace(":", ""): row['value']
            for index, row in df_info_estacao.iterrows()
        }

        # Criar um DataFrame a partir do dicionário com as informações da estação
        df_info_estacao_metereologica = pd.DataFrame.from_dict([data])

        # Preencher as colunas do DataFrame df_info_metereologico com as informações da estação
        for coluna in df_info_estacao_metereologica.columns:
            df_info_metereologico[coluna] = df_info_estacao_metereologica[coluna][0]

        # Converter colunas do tipo object para float, substituindo ',' por '.' se necessário
        for coluna in df_info_metereologico:
            if df_info_metereologico[coluna].dtype == object:
                with contextlib.suppress(Exception):
                    df_info_metereologico[coluna] = df_info_metereologico[coluna].str.replace(',', '.').astype(float)

        return df_info_metereologico

    def robust_rmtree(self, max_retries=5, retry_delay=1):
        """
        Remove uma pasta e seu conteúdo de forma robusta, com possibilidade de repetição e atraso.

        Parâmetros:
            - self: Referência à instância da classe atual.
            - max_retries (int): O número máximo de tentativas de remoção da pasta. O valor padrão é 5.
            - retry_delay (int): O tempo de espera em segundos entre as tentativas de remoção da pasta. O valor padrão é 1.

        Retorna:
            Nenhum valor de retorno. A função remove a pasta e seu conteúdo se a remoção for bem-sucedida.

        Exemplo de uso:
            >>> objeto.robust_rmtree(max_retries=3, retry_delay=2)

        Observações:
            - Certifique-se de que a pasta a ser removida (`self.pasta_dados_inmet`) exista e tenha permissão de remoção antes de chamar esta função.
            - Esta função realiza as seguintes etapas:
                1. Tenta remover a pasta e seu conteúdo usando a função `shutil.rmtree()`.
                2. Se ocorrer uma exceção do tipo `OSError`, espera o tempo especificado por `retry_delay` e faz uma nova tentativa.
                3. Repete as etapas 1 e 2 até que a pasta seja removida com sucesso ou o número máximo de tentativas (`max_retries`) seja atingido.
            - Se a pasta não puder ser removida após o número máximo de tentativas, a função lançará uma exceção não tratada.
        """
        for _ in range(max_retries):
            try:
                shutil.rmtree(self.pasta_dados_inmet)
                return
            except OSError:
                time.sleep(retry_delay)
        raise

    
    def check_datas_entre_arquivos(self, ano):
        """
        Verifica as datas entre os arquivos de dados e o arquivo Parquet tratado para um determinado ano.

        Parâmetros:
            - self: Referência à instância da classe atual.
            - ano (str ou int): O ano para o qual as datas serão verificadas.

        Retorna:
            Nenhum valor de retorno. A função imprime uma mensagem se não houver atualizações para o ano especificado.

        Exemplo de uso:
            >>> objeto.check_datas_entre_arquivos(2022)

        Observações:
            - Certifique-se de que a pasta de dados do INMET e a pasta de arquivos tratados estejam configuradas corretamente antes de chamar esta função.
            - Esta função realiza as seguintes etapas:
                1. Verifica se existem arquivos CSV descompactados correspondentes ao ano especificado.
                2. Se existirem arquivos CSV, lê o último arquivo CSV e obtém a última data presente.
                3. Se existir um arquivo Parquet previamente tratado, lê o arquivo e obtém a última data presente.
                4. Compara a última data do arquivo Parquet com a última data do arquivo CSV.
                5. Se as datas coincidirem, imprime uma mensagem informando que não há atualizações para o ano especificado.
                6. Caso contrário, chama a função `tratar_arquivo` para realizar o tratamento dos dados para o ano especificado.
        """
        if arquivos_csv := glob.glob(f'{self.pasta_dados_inmet}\*-{ano}.CSV*'):
            ultimo_csv = max(arquivos_csv, key=os.path.getctime)
            df_ultimo_csv = pd.read_csv(ultimo_csv, encoding='ISO-8859-1', sep=';', skiprows=8)
            ultima_data_csv = pd.to_datetime(df_ultimo_csv['Data']).max()

            with contextlib.suppress(Exception):
                # Verificar se, no arquivo .parquet já tratado, a última data coincide com a data dos arquivos descompactados
                df_parquet = pd.read_parquet(f'{self.pasta_tratados}\INMET_dados_metereologico_{ano}.parquet')
                ultima_data_parquet = pd.to_datetime(df_parquet['Data']).max()

                if ultima_data_parquet == ultima_data_csv:
                    print(f'Sem atualizações para {ano}.')
                    return

        self.tratar_arquivo(ano)
        return

    def executar(self):
        """
        Executa o fluxo de trabalho para baixar, descompactar e tratar os arquivos meteorológicos.

        Parâmetros:
            - self: Referência à instância da classe atual.

        Retorna:
            Nenhum valor de retorno.

        Exemplo de uso:
            >>> objeto.executar()

        Observações:
            - Certifique-se de que as configurações necessárias estejam corretamente definidas antes de chamar esta função.
            - Esta função executa as seguintes etapas:
                1. Itera sobre os anos no intervalo entre `ano_inicial` e `ano_atual + 1`.
                2. Gera o nome do arquivo e o caminho de destino.
                3. Verifica se o arquivo já foi tratado para o ano atual ou se ele existe no destino. Se sim, passa para o próximo ano.
                4. Baixa o arquivo a partir da URL especificada.
                5. Descompacta o arquivo baixado.
                6. Verifica as datas entre os arquivos.
                7. Remove o arquivo compactado.
                8. Executa a função `robust_rmtree()`.

            - Durante a execução, são impressas mensagens informando quando um arquivo já foi tratado.
        """
        for ano in range(self.ano_inicial, self.ano_atual + 1):
            nome_arquivo = f'{ano}{self.extensao_arquivo}'
            destino = os.path.join('data', 'clima', nome_arquivo)
            url = f'{self.url_base}/{nome_arquivo}'

            if (os.path.exists(f'{self.pasta_tratados}\INMET_dados_metereologico_{ano}.parquet')) and (ano != self.ano_atual):
                print(f'O arquivo de {ano} já foi tratado.')
                continue

            self.baixar_arquivo(url, destino)
            self.descompactar_arquivo(destino)
            self.check_datas_entre_arquivos(ano)
            
            os.remove(destino)
            self.robust_rmtree()
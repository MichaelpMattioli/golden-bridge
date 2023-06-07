import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.utils import resample

import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.offline as pyo
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from tqdm import tqdm

# Suprimindo avisos
warnings.filterwarnings("ignore")

class RidgeModelMultiTarget:

    def __init__(self, df_model, target, start_month, end_month, residual_max, r2_score_min, n_bootstrap, counter=None, progress_bar=None):
        self.random_state = None
        self.test_size = 0.1
        self.cv = 5
        self.n_bootstrap = n_bootstrap
        self.df_model = df_model
        self.target = target
        self.start_month = start_month
        self.end_month = end_month
        self.residual_max = residual_max
        self.r2_score_min = r2_score_min

        self.counter = counter
        self.progress_bar = progress_bar
        self.total_iterations = self.cv * self.n_bootstrap
        
        # Chama o método train_model_and_get_errors() e recebe o dicionário retornado
        train_dict = self.train_model_and_get_errors()
        
        # Atribui os valores do dicionário aos atributos da classe
        self.error = train_dict['error']
        self.metrics = train_dict['metrics']
        self.all_data_pred_sum_plu_net_sale_calculado = train_dict['all_data_pred_sum_plu_net_sale_calculado']
        self.all_data_pred_margem_plu = train_dict['all_data_pred_margem_plu']
        self.all_data_pred_gc = train_dict['all_data_pred_gc']
        self.X_weighted = train_dict['X_weighted']
        self.feature_names = train_dict['feature_names']
        self.df_importancias = train_dict['df_importancias']
        self.bootstrap_ci = train_dict['bootstrap_ci']
        self.graphic_net_sale = train_dict['graphic_net_sale']
        self.graphic_margem = train_dict['graphic_margem']
        self.graphic_gc_count = train_dict['graphic_gc_count']
        
        # self.graphic_erros = train_dict['graphic_erros']




    def multiply_features_by_importance_and_sum(self, X, importances_list, feature_names, scaler, intercept):
        """
        Multiplica as features pelo valor de importância correspondente e soma os resultados.

        Args:
            X (array-like or DataFrame): Dados de entrada.
            importances_list (list of arrays): Lista de valores de importância para cada conjunto de features.
            feature_names (list): Lista de nomes das features.
            scaler (object): Objeto de normalização/scaler usado para normalizar os dados de entrada.
            intercept (float): Intercepto do modelo.

        Returns:
            list of arrays, float: Lista de arrays contendo os dados de entrada multiplicados pelas importâncias e o intercepto do modelo.
        """
        # Certifique-se de que X seja um DataFrame do pandas
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=feature_names)

        # Normalize os dados de entrada usando o mesmo scaler
        X_scaled = scaler.transform(X)

        # Multiplica cada conjunto de importâncias pelo X normalizado
        X_weighted_list = [X_scaled * importances for importances in importances_list]

        return X_weighted_list, intercept

    def ridge_model_multitarget(self, data, feature_names, target_names, shuffle, metrics=None):
        """
        Modelo Ridge com suporte a múltiplos alvos.

        Args:
            data (DataFrame): Conjunto de dados contendo as features e os alvos.
            feature_names (list): Lista de nomes das features.
            target_names (list): Lista de nomes dos alvos.
            shuffle (bool): Indica se os dados devem ser embaralhados antes da divisão em conjuntos de treinamento e teste.
            metrics (list, optional): Lista de métricas a serem calculadas. Se não for especificado, as métricas padrão serão usadas.
            random_state (int, optional): Semente para o gerador de números aleatórios.

        Returns:
            DataFrame, dict, array-like, array-like, list, array-like, array-like, array-like: Resultados do modelo Ridge.

        """

        # Se a métrica não for especificada, utilize as métricas padrão
        if metrics is None:
            metrics = [r2_score, mean_squared_error, mean_absolute_error]
        print(data)
        # Separando as características (features) e os alvos (targets) do conjunto de dados
        X = data[feature_names]
        y = data[target_names]

        # Dividindo os dados em conjuntos de treinamento e teste
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, shuffle=shuffle
        )

        # Normalizando os dados
        scaler = MinMaxScaler()
        X_train_val_scaled = scaler.fit_transform(X_train_val)
        X_test_scaled = scaler.transform(X_test)

        # Definindo o modelo de regressão
        model = MultiOutputRegressor(Ridge(alpha=1.000, fit_intercept=True))

        # Inicializando as pontuações da validação cruzada
        cv_scores = {metric.__name__: [] for metric in metrics}
        kfold = KFold(n_splits=self.cv, shuffle=shuffle, random_state=self.random_state)

        # Inicializando a importância das características
        importances = [np.zeros(len(feature_names)) for _ in range(len(target_names))]

        # Inicializando os resultados da reamostragem
        bootstrap_results = []

        # Para cada conjunto de treinamento e validação, fazendo a reamostragem e treinando o modelo

        pbar_kfold = tqdm(total=self.cv)  # Cria uma instância do tqdm com um total definido
        pbar_kfold.set_description('k-fold - ')  # Define a descrição para a instância do tqdm



        for train_index, val_index in kfold.split(X_train_val_scaled):
            X_train, X_val = X_train_val_scaled[train_index], X_train_val_scaled[val_index]
            y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]

            pbar_bootstrap = tqdm(total=self.n_bootstrap)  # Cria uma instância do tqdm para o loop de bootstrap
            pbar_bootstrap.set_description('Bootstrap - ')  # Define a descrição para a instância do tqdm

            # Executando a reamostragem várias vezes e registrando a pontuação
            for _ in range(self.n_bootstrap):
                X_resampled, y_resampled = resample(X_train, y_train)
                model.fit(X_resampled, y_resampled)
                y_pred_resampled = model.predict(X_val)
                score = r2_score(y_val, y_pred_resampled, multioutput="raw_values")
                bootstrap_results.append(score)

                # Calculando a importância das características
                for i, estimator in enumerate(model.estimators_):
                    importances[i] += estimator.coef_

                # Fazendo a previsão e calculando a pontuação
                y_pred_val = model.predict(X_val)
                for metric in metrics:
                    score = metric(y_val, y_pred_val, multioutput="raw_values")
                    cv_scores[metric.__name__].append(score)
                if self.progress_bar is not None:
                    self.counter[0] += 1
                    self.progress_bar.progress(self.counter[0] / self.total_iterations, text='Treinando modelo...')

                pbar_bootstrap.update(1)  # Atualiza o progresso do loop de bootstrap

            pbar_kfold.update(1)

        # Calculando o intervalo de confiança da reamostragem
        bootstrap_ci = np.percentile(bootstrap_results, [2.5, 97.5])

        # Calculando a importância média das características
        for i in range(len(importances)):
            importances[i] /= self.cv * self.n_bootstrap

        # Normalizando a importância das características
        normalized_importances = []
        for imp in importances:
            max_abs_imp = np.max(np.abs(imp))
            normalized_imp = imp / max_abs_imp
            normalized_importances.append(normalized_imp)

        # Classificando as variáveis como "alavanca" ou "ofensor"
        variable_classifications = [
            ["alavanca" if imp > 0 else "ofensor" for imp in importances_target]
            for importances_target in importances
        ]

        # Criando dataframes com a classificação e a importância das variáveis para cada alvo
        dfs = []
        for i, target_name in enumerate(target_names):
            df = pd.DataFrame(
                {
                    "Variável": X.columns,
                    f"Classificação_{target_name}": [
                        "alavanca" if imp > 0 else "ofensor" for imp in importances[i]
                    ],
                    f"Importância_{target_name}": importances[i],
                }
            )
            dfs.append(df)

        # Concatenando os dataframes
        df = pd.concat(dfs, axis=1)

        # Treinando o modelo com todos os dados de treinamento e validação
        model.fit(X_train_val_scaled, y_train_val)

        # Fazendo previsões com os dados de teste
        y_pred_test = model.predict(X_test_scaled)

        # Calculando as pontuações de teste
        test_scores = {
            f"Test {metric.__name__}": metric(
                y_test, y_pred_test, multioutput="raw_values"
            )
            for metric in metrics
        }

        # Criando um dicionário com todas as métricas
        metrics = {
            "CV Scores": cv_scores,
            "Mean CV Scores": {
                metric: [np.mean([score[i] for score in scores]) for i in range(len(target_names))] for metric, scores in cv_scores.items()
            },
            "Std CV Scores": {
                metric: [np.std([score[i] for score in scores]) for i in range(len(target_names))] for metric, scores in cv_scores.items()
            },
            "Test Scores": test_scores,
        }


        # Calculando o intercepto para cada estimador
        intercept = [estimator.intercept_ for estimator in model.estimators_]
        X_weighted_list, intercept = self.multiply_features_by_importance_and_sum(
            X, importances, feature_names, scaler, intercept
        )

        # Fazendo previsões com todos os dados
        y_pred_full = model.predict(scaler.transform(X))
        print(metrics)
        import time
        time.sleep(15)
        # Retornando os resultados
        return (
            df,
            metrics,
            y_test,
            y_pred_test,
            X_weighted_list,
            intercept,
            y_pred_full,
            bootstrap_ci,
        )

    def train_model_and_get_errors(self):
        """
        Treina o modelo e obtém os erros entre os valores reais e previstos, bem como outras métricas e previsões.

        Args:
            df_model (DataFrame): DataFrame contendo os dados para treinamento do modelo.
            target (str): Nome da coluna alvo.
            start_month (int): Mês de início para filtragem dos dados.
            end_month (int): Mês de fim para filtragem dos dados.
            residual_max (float): Valor máximo permitido para o resíduo percentual.
            r2_score_min (float): Valor mínimo permitido para o coeficiente de determinação (R2).

        Returns:
            Tuple: Uma tupla contendo o DataFrame de erros, as métricas, as previsões para a coluna 'sum|sum_plu_net_sale_calculado',
            as previsões para a coluna 'margem_plu', as previsões para a coluna 'gc_count', os pesos das features,
            os nomes das features, e o intervalo de confiança bootstrap.
        """

        # Filtra os dados do df_model pelos meses de início e fim
        df_model = self.df_model[self.df_model.index.month >= self.start_month]
        df_model = df_model[df_model.index.month <= self.end_month]

        # Renomeia as colunas 'net_dia' e 'margem_dia' para 'sum|sum_plu_net_sale_calculado' e 'margem_plu', respectivamente
        df_model.rename( columns={"net_dia": "sum|sum_plu_net_sale_calculado", "margem_dia": "margem_plu"}, inplace=True)

        counter = 0
        while counter < 3:
            # Select relevant features and target variable
            feature_names = df_model.drop(columns=self.target).columns.to_list()
            target_names = self.target

            # Run ridge_model function to train a linear regression model with the selected features
            (
                df_importances,
                metrics,
                y_test,
                y_pred_test,
                X_weighted,
                intercept,
                y_pred_full,
                bootstrap_ci,
            ) = self.ridge_model_multitarget(df_model, feature_names, target_names, shuffle=True)

            # Create dataframe to calculate and store errors between real and predicted values
            error = pd.concat([y_test.reset_index(drop=True), pd.DataFrame(y_pred_test, columns=["y_pred_net_sale", "y_pred_margem","y_pred_gc"])], axis=1)

            # Calculate and add to 'error' dataframe the columns of residuals
            error["residuo_margem"] = round(error["y_pred_margem"] - error["margem_plu"], 2)
            error["residuo_margem%"] = round(error["residuo_margem"] / error["y_pred_margem"] * 100, 2)
            error["residuo_net_sale"] = round(error["y_pred_net_sale"] - error["sum|sum_plu_net_sale_calculado"], 2)
            error["residuo_net_sale%"] = round(error["residuo_net_sale"] / error["y_pred_net_sale"] * 100, 2)
            error["residuo_gc"] = round(error["y_pred_gc"] - error["gc_count"], 2)
            error["residuo_gc%"] = round(error["residuo_gc"] / error["y_pred_gc"] * 100, 2)

            print(f"Tentativa nº: {counter}")

            if max(abs(error["residuo_net_sale%"])) < self.residual_max: break
            counter += 1
            self.total_iterations += self.total_iterations 

            if max(abs(error["residuo_net_sale%"])) > self.residual_max: print("Treinamento ruim, escolha um período maior")
            else:
                print(50 * "-")
                # Print the minimum and maximum residuals and percentage residuals
                print("residuo min:\n", error[["residuo_margem", "residuo_net_sale", "residuo_gc"]].min())
                print("residuo% min:\n", error[["residuo_margem%", "residuo_net_sale%", "residuo_gc%"]].min())
                print(50 * "-")
                print("residuo max:\n", error[["residuo_margem", "residuo_net_sale", "residuo_gc"]].max())
                print("residuo% max:\n", error[["residuo_margem%", "residuo_net_sale%", "residuo_gc%"]].max())

                print(50 * "-")
                print("Mean CV Scores:")
                for target_name, scores in metrics["Mean CV Scores"].items():
                    print(f"  {target_name}:", scores)
                print("Std CV Scores:")
                for target_name, scores in metrics["Std CV Scores"].items():
                    print(f"  {target_name}:", scores)
                print("Test Scores:")
                for target_name, scores in metrics["Test Scores"].items():
                    print(f"  {target_name}:", scores)


            # Plota um gráfico comparando os valores reais e previstos
            # plot_real_vs_predicted(df_model.index, y_test.values, y_pred_test) # TODO

                        # Plota um gráfico comparando os valores reais e previstos para a coluna 'sum|sum_plu_net_sale_calculado'
            def plot_real_vs_predicted_total(
                real, predito, title='title = "Título"',
                xaxis_title='xaxis_title = "Legenda X"',
                yaxis_title='yaxis_title = "Legenda Y"',
                real_line_color="#33CFA5",
                predito_line_color="#F06A6A",
                real_marker_symbol="circle",
                predito_marker_symbol="x",
                gridcolor="gray",
                leg_x=0.1,
                leg_y=0.1,
                trace_name_1="Nome marcador circulo",
                trace_name_2="Nome marcador x",
            ):
                """
                Plota os dados reais e previstos em um gráfico.

                Args:
                    real (array-like): Valores reais.
                    predito (array-like): Valores previstos.
                    title (str, optional): Título do gráfico. Default é 'title = "Título"'.
                    xaxis_title (str, optional): Título do eixo x. Default é 'xaxis_title = "Legenda X"'.
                    yaxis_title (str, optional): Título do eixo y. Default é 'yaxis_title = "Legenda Y"'.
                    real_line_color (str, optional): Cor da linha para os dados reais. Default é '#33CFA5'.
                    predito_line_color (str, optional): Cor da linha para os dados previstos. Default é '#F06A6A'.
                    real_marker_symbol (str, optional): Símbolo do marcador para os dados reais. Default é 'circle'.
                    predito_marker_symbol (str, optional): Símbolo do marcador para os dados previstos. Default é 'x'.
                    gridcolor (str, optional): Cor das linhas de grade do gráfico. Default é 'gray'.
                    leg_x (float, optional): Posição horizontal da legenda. Default é 0.1.
                    leg_y (float, optional): Posição vertical da legenda. Default é 0.1.
                    trace_name_1 (str, optional): Nome da série para os dados reais. Default é 'Nome marcador circulo'.
                    trace_name_2 (str, optional): Nome da série para os dados previstos. Default é 'Nome marcador x'.
                """

                # Crie um objeto de figura
                fig = go.Figure()

                # Adicione a série real ao gráfico
                fig.add_trace(
                    go.Scatter(
                        x=real.index,
                        y=real,
                        mode="markers+lines",
                        name=trace_name_1,
                        line=dict(color=real_line_color, dash="solid"),
                        marker=dict(symbol=real_marker_symbol),
                    )
                )

                # Adicione a série predita ao gráfico
                fig.add_trace(
                    go.Scatter(
                        x=predito.index,
                        y=predito,
                        mode="markers+lines",
                        name=trace_name_2,
                        line=dict(color=predito_line_color, dash="dash"),
                        marker=dict(symbol=predito_marker_symbol),
                    )
                )

                # Personalize o layout do gráfico
                fig.update_layout(
                    title=title,
                    xaxis_title=xaxis_title,
                    yaxis_title=yaxis_title,
                    plot_bgcolor="white",
                    xaxis=dict(gridcolor=gridcolor, gridwidth=1, tickmode="auto"),
                    yaxis=dict(gridcolor=gridcolor, gridwidth=1, tickmode="auto"),
                    legend=dict(
                        x=leg_x,
                        y=leg_y,
                        bgcolor="rgba(255, 255, 255, 0.5)",
                        bordercolor="black",
                        borderwidth=0.1,
                    ),
                )

                # Exibir o gráfico
                return fig

            # Cria um DataFrame com as previsões para a coluna 'sum|sum_plu_net_sale_calculado' usando todos os dados
            all_data_pred_sum_plu_net_sale_calculado = pd.DataFrame(y_pred_full, columns=target_names)["sum|sum_plu_net_sale_calculado"]
            all_data_pred_sum_plu_net_sale_calculado.index = df_model.index

            graphic_net_sale = plot_real_vs_predicted_total( # TODO
                df_model[target_names[0]],
                all_data_pred_sum_plu_net_sale_calculado,
                "Todos os dados - Net Sale Real vs Net Sale Prevista",
                xaxis_title="Data",
                yaxis_title="Net Sale",
                leg_x=0.05,
                leg_y=0.85,
                trace_name_1="Net Sale Real",
                trace_name_2="Net Sale Prevista",
            )

            # Cria um DataFrame com as previsões para a coluna 'margem_plu' usando todos os dados
            all_data_pred_margem_plu = pd.DataFrame(y_pred_full, columns=target_names)["margem_plu"]
            all_data_pred_margem_plu.index = df_model.index

            # Plota um gráfico comparando os valores reais e previstos para a coluna 'margem_plu'
            graphic_margem = plot_real_vs_predicted_total( # TODO
                all_data_pred_margem_plu,
                df_model[target_names[1]],
                "Todos os dados - Margem Plu Real vs Margem Plu Prevista",
                xaxis_title="Data",
                yaxis_title="Margem Plu",
                leg_x=0.05,
                leg_y=0.1,
                trace_name_1="Margem Plu Real",
                trace_name_2="Margem Plu Prevista",
                real_line_color="#102A6A",
                predito_line_color="#F02A6A",
            )

            # Cria um DataFrame com as previsões para a coluna 'gc_count' usando todos os dados
            all_data_pred_gc = pd.DataFrame(y_pred_full, columns=target_names)["gc_count"]
            all_data_pred_gc.index = df_model.index

            # Plota um gráfico comparando os valores reais e previstos para a coluna 'gc_count'
            graphic_gc_count = plot_real_vs_predicted_total( # TODO
                all_data_pred_gc,
                df_model[target_names[2]],
                "Todos os dados - GC Real vs GC Prevista",
                xaxis_title="Data",
                yaxis_title="GC",
                leg_x=0.05,
                leg_y=0.1,
                trace_name_1="GC Real",
                trace_name_2="GC Previsto",
                real_line_color="#4169E1",
                predito_line_color="#FFD700",
            )
        
        if self.progress_bar is not None:
            self.progress_bar.progress(1.0, text='Finalizado!')
        
        return {
        'error': error,
        'metrics': metrics,
        'all_data_pred_sum_plu_net_sale_calculado': all_data_pred_sum_plu_net_sale_calculado,
        'all_data_pred_margem_plu': all_data_pred_margem_plu,
        'all_data_pred_gc': all_data_pred_gc,
        'X_weighted': X_weighted,
        'feature_names': feature_names,
        'df_importancias': df_importances,
        'bootstrap_ci': bootstrap_ci,
        'graphic_net_sale': graphic_net_sale,
        'graphic_margem': graphic_margem,
        'graphic_gc_count': graphic_gc_count
    }

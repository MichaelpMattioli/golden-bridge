import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.offline as pyo
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np
import warnings

# Suprimindo avisos
warnings.filterwarnings("ignore")

class ModelAnalysis:
    def __init__(self, X_weighted, feature_names, all_data_pred_sum_plu_net_sale_calculado, df_model, produto, clima, compt, demo, flag,lojas, data_1, data_2):
        self.flag = flag
        self.lojas = [lojas]
        self.produto = produto
        self.clima = clima
        self.compt = compt
        self.demo = demo
        self.df_net_sales = df_model["sum|sum_plu_net_sale_calculado"]
        self.df_margem = df_model["margem_plu"]
        self.df_gc = df_model["gc_count"]
        self.data_1 = data_1
        self.data_2 = data_2
        self.X_weighted = X_weighted
        self.feature_names = feature_names
        
        self.all_data_pred_sum_plu_net_sale_calculado = all_data_pred_sum_plu_net_sale_calculado
        self.pilares = [
            "PREÇOS", "CAMPANHAS", "NOVOS PRODUTOS", "NOVAS LOJAS", 
            "AÇÕES DE CRM", "AÇÕES DE DELIVERY", "AUMENTO NO CMV", 
            "CUSTOS", "TAXAS", "COMPETIDORES", "TEMPERATURA E CLIMA", 
            "DEMOGRAFIA", "QTD_VENDIDAS"
            ]
        self.names = [
            "Qtd. Vendida", "Demográficos", "Clima", "Competidores",
            "Taxas", "Custos", "Preço",
        ]
        self.run()
        
        # GRÁFICOS
        # Plots Golden Bridge
        # Net Sale
        self.plot_golden_bridge_net_sale = self.plot_cascata_diferenca('net_sales', self.df_net_sales, self.df_coef_net_sales_clima, self.df_coef_net_sales_demo, self.df_coef_net_sales_qtd_vendidas, self.df_coef_net_sales_precos, self.df_coef_net_sales_taxas, self.df_coef_net_sales_custos, self.df_coef_net_sales_competidores, self.df_coef_net_sales_campanhas, self.df_coef_net_sales_novos_produtos, self.df_coef_net_sales_crm, self.df_coef_net_sales_novas_lojas, self.df_coef_net_sales_delivery, self.df_coef_net_sales_cmv,)
        
        # Margem
        self.plot_golden_bridge_margem = self.plot_cascata_diferenca('margem', self.df_margem, self.df_coef_margem_clima, self.df_coef_margem_demo, self.df_coef_margem_qtd_vendidas, self.df_coef_margem_precos, self.df_coef_margem_taxas, self.df_coef_margem_custos, self.df_coef_margem_competidores, self.df_coef_margem_campanhas, self.df_coef_margem_novos_produtos, self.df_coef_margem_crm, self.df_coef_margem_novas_lojas, self.df_coef_margem_delivery, self.df_coef_margem_cmv,)
        
        # GC
        self.plot_golden_bridge_gc = self.plot_cascata_diferenca('gc', self.df_gc, self.df_coef_gc_clima, self.df_coef_gc_demo, self.df_coef_gc_qtd_vendidas, self.df_coef_gc_precos, self.df_coef_gc_taxas, self.df_coef_gc_custos, self.df_coef_gc_competidores, self.df_coef_gc_campanhas, self.df_coef_gc_novos_produtos, self.df_coef_gc_crm, self.df_coef_gc_novas_lojas, self.df_coef_gc_delivery, self.df_coef_gc_cmv,)
        
        # Plots do período
        # Net Sale
        self.plot_net_sale_geral = self.plot_peso_var_stack("Influência dos Pilares Diariamente<br>Net Sale", *[self.df_dist_perc_net_sale.filter(regex="PRODUTO").filter(regex="sold"), self.df_dist_perc_net_sale.filter(regex="DEMO"), self.df_dist_perc_net_sale.filter(regex="CLIMA"), self.df_dist_perc_net_sale.filter(regex="COMPT"), self.df_dist_perc_net_sale.filter(regex="PRODUTO").filter(regex="tax|imposto"), self.df_dist_perc_net_sale.filter(regex="PRODUTO").filter(regex="cost"), self.df_dist_perc_net_sale.filter(regex="PRODUTO").filter(regex="price")])
        
        # Margem
        self.plot_margem_geral = self.plot_peso_var_stack("Influência dos Pilares Diariamente<br>Margem", *[self.df_dist_perc_margem.filter(regex="PRODUTO").filter(regex="sold"), self.df_dist_perc_margem.filter(regex="DEMO"), self.df_dist_perc_margem.filter(regex="CLIMA"), self.df_dist_perc_margem.filter(regex="COMPT"), self.df_dist_perc_margem.filter(regex="PRODUTO").filter(regex="tax|imposto"), self.df_dist_perc_margem.filter(regex="PRODUTO").filter(regex="cost"), self.df_dist_perc_margem.filter(regex="PRODUTO").filter(regex="price")])
        
        # GC
        self.plot_gc_geral = self.plot_peso_var_stack("Influência dos Pilares Diariamente<br>GC", *[self.df_dist_perc_gc.filter(regex="PRODUTO").filter(regex="sold"), self.df_dist_perc_gc.filter(regex="DEMO"), self.df_dist_perc_gc.filter(regex="CLIMA"), self.df_dist_perc_gc.filter(regex="COMPT"), self.df_dist_perc_gc.filter(regex="PRODUTO").filter(regex="tax|imposto"), self.df_dist_perc_gc.filter(regex="PRODUTO").filter(regex="cost"), self.df_dist_perc_gc.filter(regex="PRODUTO").filter(regex="price")])
        
        # Plots dos pilares pelo período
        # Net Sale
        self.plot_net_sale_qtd_vendida = self.plot_peso_var(self.df_dist_perc_net_sale, "PRODUTO", "Qtd. Vendida", regex_subpilar="sold")
        self.plot_net_sale_preco = self.plot_peso_var(self.df_dist_perc_net_sale, "PRODUTO", "Preços", regex_subpilar="price")
        self.plot_net_sale_custo = self.plot_peso_var(self.df_dist_perc_net_sale, "PRODUTO", "Custos", regex_subpilar="cost")
        self.plot_net_sale_taxas = self.plot_peso_var(self.df_dist_perc_net_sale, "PRODUTO", "Taxas", regex_subpilar="tax|imposto")
        self.plot_net_sale_clima = self.plot_peso_var(self.df_dist_perc_net_sale, "CLIMA", "Clima")
        self.plot_net_sale_compt = self.plot_peso_var(self.df_dist_perc_net_sale, "COMPT", "Competidores")
        self.plot_net_sale_demo = self.plot_peso_var(self.df_dist_perc_net_sale, "DEMO", "Demografia")
        
        # Margem
        self.plot_margem_qtd_vendida = self.plot_peso_var(self.df_dist_perc_margem, "PRODUTO", "Qtd. Vendida", regex_subpilar="sold")
        self.plot_margem_preco = self.plot_peso_var(self.df_dist_perc_margem, "PRODUTO", "Preços", regex_subpilar="price")
        self.plot_margem_custo = self.plot_peso_var(self.df_dist_perc_margem, "PRODUTO", "Custos", regex_subpilar="cost")
        self.plot_margem_taxas = self.plot_peso_var(self.df_dist_perc_margem, "PRODUTO", "Taxas", regex_subpilar="tax|imposto")
        self.plot_margem_clima = self.plot_peso_var(self.df_dist_perc_margem, "CLIMA", "Clima")
        self.plot_margem_compt = self.plot_peso_var(self.df_dist_perc_margem, "COMPT", "Competidores")
        self.plot_margem_demo = self.plot_peso_var(self.df_dist_perc_margem, "DEMO", "Demografia")
        
        # GC
        self.plot_gc_qtd_vendida = self.plot_peso_var(self.df_dist_perc_gc, "PRODUTO", "Qtd. Vendida", regex_subpilar="sold")
        self.plot_gc_preco = self.plot_peso_var(self.df_dist_perc_gc, "PRODUTO", "Preços", regex_subpilar="price")
        self.plot_gc_custo = self.plot_peso_var(self.df_dist_perc_gc, "PRODUTO", "Custos", regex_subpilar="cost")
        self.plot_gc_taxas = self.plot_peso_var(self.df_dist_perc_gc, "PRODUTO", "Taxas", regex_subpilar="tax|imposto")
        self.plot_gc_clima = self.plot_peso_var(self.df_dist_perc_gc, "CLIMA", "Clima")
        self.plot_gc_compt = self.plot_peso_var(self.df_dist_perc_gc, "COMPT", "Competidores")
        self.plot_gc_demo = self.plot_peso_var(self.df_dist_perc_gc, "DEMO", "Demografia")
        
        # TABELAS
        self.report_qtd_vendidas = self.generate_report("PRODUTO_sum|sum_plu_unit_sold")
        self.report_preco = self.generate_report("PRODUTO_sum|sum_plu_unit_price")
        self.report_custo = self.generate_report("PRODUTO_sum|sum_plu_cost_total_calculado")
        self.report_taxas = self.generate_report("PRODUTO_sum|sum_plu_val_tax")
        
        self.report_clima = self.f_report_clima()
        self.report_compt = self.compt
        self.report_demo = self.demo
        self.report_demo = self.report_demo.set_index('filtros_usados')


    def distribuicao_percentual_por_feature(self, X_weighted):
        """
        Calcula a distribuição percentual por feature dos pesos normalizados.

        Args:
            X_weighted (array-like): Lista de arrays com os pesos normalizados das features.
            index (array-like): Índice dos dados.
            feature_names (list): Lista de nomes das features.

        Returns:
            DataFrame: Distribuição percentual por feature dos pesos normalizados.
        """
        # Cria um DataFrame de pesos com as colunas como os nomes das features e o índice fornecido
        pesos = pd.DataFrame(
            X_weighted, 
            columns=self.feature_names
        ).set_index(self.all_data_pred_sum_plu_net_sale_calculado.index)

        # Calcula a soma dos quadrados dos pesos para cada linha
        soma_dos_quadrados = pesos.mul(pesos).sum(axis=1)

        # Retorna a distribuição percentual por feature dividindo cada valor pelo somatório dos quadrados e multiplicando por 100
        return pesos.mul(pesos).div(soma_dos_quadrados, axis=0) * 100

    def get_dataframes(self):
        self.df_dist_perc_net_sale = self.distribuicao_percentual_por_feature(self.X_weighted[0])
        # ... net_sale bruto
        
        self.df_dist_perc_margem = self.distribuicao_percentual_por_feature(self.X_weighted[1])
        
        self.df_dist_perc_gc = self.distribuicao_percentual_por_feature(self.X_weighted[2])
        # ... gc bruto

    def get_coeficients(self):
        # if self.flag == 0:
        self.df_coef_net_sales_clima = self.df_dist_perc_net_sale.filter(regex="CLIMA").sum(axis=1)
        self.df_coef_net_sales_demo = self.df_dist_perc_net_sale.filter(regex="DEMO").sum(axis=1)
        self.df_coef_net_sales_qtd_vendidas = self.df_dist_perc_net_sale.filter(regex="PRODUTO").filter(regex="sold").sum(axis=1)
        self.df_coef_net_sales_precos = self.df_dist_perc_net_sale.filter(regex="PRODUTO").filter(regex="price").sum(axis=1)
        self.df_coef_net_sales_taxas = self.df_dist_perc_net_sale.filter(regex="PRODUTO").filter(regex="tax|imposto").sum(axis=1)
        self.df_coef_net_sales_custos = self.df_dist_perc_net_sale.filter(regex="PRODUTO").filter(regex="cost").sum(axis=1)
        self.df_coef_net_sales_competidores = self.df_dist_perc_net_sale.filter(regex="COMPT").sum(axis=1)
        self.df_coef_net_sales_campanhas = self.df_dist_perc_net_sale.filter(regex="CAMP").sum(axis=1)
        self.df_coef_net_sales_novos_produtos = self.df_dist_perc_net_sale.filter(regex="NOVOS").sum(axis=1)
        self.df_coef_net_sales_crm = self.df_dist_perc_net_sale.filter(regex="CRM").sum(axis=1)
        self.df_coef_net_sales_novas_lojas = self.df_dist_perc_net_sale.filter(regex="NOVAS").sum(axis=1)
        self.df_coef_net_sales_delivery = self.df_dist_perc_net_sale.filter(regex="DELIVERY").sum(axis=1)
        self.df_coef_net_sales_cmv = self.df_dist_perc_net_sale.filter(regex="CMV").sum(axis=1)
        
        self.df_coef_margem_clima = self.df_dist_perc_margem.filter(regex="CLIMA").sum(axis=1)
        self.df_coef_margem_demo = self.df_dist_perc_margem.filter(regex="DEMO").sum(axis=1)
        self.df_coef_margem_qtd_vendidas = self.df_dist_perc_margem.filter(regex="PRODUTO").filter(regex="sold").sum(axis=1)
        self.df_coef_margem_precos = self.df_dist_perc_margem.filter(regex="PRODUTO").filter(regex="price").sum(axis=1)
        self.df_coef_margem_taxas = self.df_dist_perc_margem.filter(regex="PRODUTO").filter(regex="tax|imposto").sum(axis=1)
        self.df_coef_margem_custos = self.df_dist_perc_margem.filter(regex="PRODUTO").filter(regex="cost").sum(axis=1)
        self.df_coef_margem_competidores = self.df_dist_perc_margem.filter(regex="COMPT").sum(axis=1)
        self.df_coef_margem_campanhas = self.df_dist_perc_margem.filter(regex="CAMP").sum(axis=1)
        self.df_coef_margem_novos_produtos = self.df_dist_perc_margem.filter(regex="NOVOS").sum(axis=1)
        self.df_coef_margem_crm = self.df_dist_perc_margem.filter(regex="CRM").sum(axis=1)
        self.df_coef_margem_novas_lojas = self.df_dist_perc_margem.filter(regex="NOVAS").sum(axis=1)
        self.df_coef_margem_delivery = self.df_dist_perc_margem.filter(regex="DELIVERY").sum(axis=1)
        self.df_coef_margem_cmv = self.df_dist_perc_margem.filter(regex="CMV").sum(axis=1)
        
        self.df_coef_gc_clima = self.df_dist_perc_gc.filter(regex="CLIMA").sum(axis=1)
        self.df_coef_gc_demo = self.df_dist_perc_gc.filter(regex="DEMO").sum(axis=1)
        self.df_coef_gc_qtd_vendidas = self.df_dist_perc_gc.filter(regex="PRODUTO").filter(regex="sold").sum(axis=1)
        self.df_coef_gc_precos = self.df_dist_perc_gc.filter(regex="PRODUTO").filter(regex="price").sum(axis=1)
        self.df_coef_gc_taxas = self.df_dist_perc_gc.filter(regex="PRODUTO").filter(regex="tax|imposto").sum(axis=1)
        self.df_coef_gc_custos = self.df_dist_perc_gc.filter(regex="PRODUTO").filter(regex="cost").sum(axis=1)
        self.df_coef_gc_competidores = self.df_dist_perc_gc.filter(regex="COMPT").sum(axis=1)
        self.df_coef_gc_campanhas = self.df_dist_perc_gc.filter(regex="CAMP").sum(axis=1)
        self.df_coef_gc_novos_produtos = self.df_dist_perc_gc.filter(regex="NOVOS").sum(axis=1)
        self.df_coef_gc_crm = self.df_dist_perc_gc.filter(regex="CRM").sum(axis=1)
        self.df_coef_gc_novas_lojas = self.df_dist_perc_gc.filter(regex="NOVAS").sum(axis=1)
        self.df_coef_gc_delivery = self.df_dist_perc_gc.filter(regex="DELIVERY").sum(axis=1)
        self.df_coef_gc_cmv = self.df_dist_perc_gc.filter(regex="CMV").sum(axis=1)
            
        # elif self.flag == 1:
        #     self.df_coef_net_sales_clima = net_sales_bruto.filter(regex="CLIMA").sum(axis=1)
        #     self.df_coef_net_sales_demo = net_sales_bruto.filter(regex="DEMO").sum(axis=1)
        #     self.df_coef_net_sales_qtd_vendidas = (net_sales_bruto.filter(regex="PRODUTO").filter(regex="sold").sum(axis=1))
        #     self.df_coef_net_sales_precos = (net_sales_bruto.filter(regex="PRODUTO").filter(regex="price").sum(axis=1))
        #     self.df_coef_net_sales_taxas = (net_sales_bruto.filter(regex="PRODUTO").filter(regex="tax|imposto").sum(axis=1))
        #     self.df_coef_net_sales_custos = (net_sales_bruto.filter(regex="PRODUTO").filter(regex="cost").sum(axis=1))
        #     self.df_coef_net_sales_competidores = net_sales_bruto.filter(regex="COMPT").sum(axis=1)
        #     self.df_coef_net_sales_campanhas = net_sales_bruto.filter(regex="CAMP").sum(axis=1)
        #     self.df_coef_net_sales_novos_produtos = net_sales_bruto.filter(regex="NOVOS").sum(axis=1)
        #     self.df_coef_net_sales_crm = net_sales_bruto.filter(regex="CRM").sum(axis=1)
        #     self.df_coef_net_sales_novas_lojas = net_sales_bruto.filter(regex="NOVAS").sum(axis=1)
        #     self.df_coef_net_sales_delivery = net_sales_bruto.filter(regex="DELIVERY").sum(axis=1)
        #     self.df_coef_net_sales_cmv = net_sales_bruto.filter(regex="CMV").sum(axis=1)
            
        #     self.df_coef_margem_clima = df_dist_perc_margem.filter(regex="CLIMA").sum(axis=1)
        #     self.df_coef_margem_demo = df_dist_perc_margem.filter(regex="DEMO").sum(axis=1)
        #     self.df_coef_margem_qtd_vendidas = (df_dist_perc_margem.filter(regex="PRODUTO").filter(regex="sold").sum(axis=1))
        #     self.df_coef_margem_precos = (df_dist_perc_margem.filter(regex="PRODUTO").filter(regex="price").sum(axis=1))
        #     self.df_coef_margem_taxas = (df_dist_perc_margem.filter(regex="PRODUTO").filter(regex="tax|imposto").sum(axis=1))
        #     self.df_coef_margem_custos = (df_dist_perc_margem.filter(regex="PRODUTO").filter(regex="cost").sum(axis=1))
        #     self.df_coef_margem_competidores = df_dist_perc_margem.filter(regex="COMPT").sum(axis=1)
        #     self.df_coef_margem_campanhas = df_dist_perc_margem.filter(regex="CAMP").sum(axis=1)
        #     self.df_coef_margem_novos_produtos = df_dist_perc_margem.filter(regex="NOVOS").sum(axis=1)
        #     self.df_coef_margem_crm = df_dist_perc_margem.filter(regex="CRM").sum(axis=1)
        #     self.df_coef_margem_novas_lojas = df_dist_perc_margem.filter(regex="NOVAS").sum(axis=1)
        #     self.df_coef_margem_delivery = df_dist_perc_margem.filter(regex="DELIVERY").sum(axis=1)
        #     self.df_coef_margem_cmv = df_dist_perc_margem.filter(regex="CMV").sum(axis=1)

        self.dados_pilares_1 = [
            self.df_net_sales.loc[self.data_1],
            self.df_coef_net_sales_precos.loc[self.data_1],
            self.df_coef_net_sales_campanhas.loc[self.data_1],
            self.df_coef_net_sales_novos_produtos.loc[self.data_1],
            self.df_coef_net_sales_novas_lojas.loc[self.data_1],
            self.df_coef_net_sales_crm.loc[self.data_1],
            self.df_coef_net_sales_delivery.loc[self.data_1],
            self.df_coef_net_sales_cmv.loc[self.data_1],
            self.df_coef_net_sales_custos.loc[self.data_1],
            self.df_coef_net_sales_taxas.loc[self.data_1],
            self.df_coef_net_sales_competidores.loc[self.data_1],
            self.df_coef_net_sales_clima.loc[self.data_1],
            self.df_coef_net_sales_demo.loc[self.data_1],
            self.df_coef_net_sales_qtd_vendidas.loc[self.data_1],
        ]
        

        self.dados_pilares_2 = [
            self.df_net_sales.loc[self.data_2],
            self.df_coef_net_sales_precos.loc[self.data_2],
            self.df_coef_net_sales_campanhas.loc[self.data_2],
            self.df_coef_net_sales_novos_produtos.loc[self.data_2],
            self.df_coef_net_sales_novas_lojas.loc[self.data_2],
            self.df_coef_net_sales_crm.loc[self.data_2],
            self.df_coef_net_sales_delivery.loc[self.data_2],
            self.df_coef_net_sales_cmv.loc[self.data_2],
            self.df_coef_net_sales_custos.loc[self.data_2],
            self.df_coef_net_sales_taxas.loc[self.data_2],
            self.df_coef_net_sales_competidores.loc[self.data_2],
            self.df_coef_net_sales_clima.loc[self.data_2],
            self.df_coef_net_sales_demo.loc[self.data_2],
            self.df_coef_net_sales_qtd_vendidas.loc[self.data_2],
        ]

    def plot_cascata_diferenca(self, target, df_target, df_coef_target_clima, df_coef_target_demo, df_coef_target_qtd_vendidas, df_coef_target_precos, df_coef_target_taxas, df_coef_target_custos, df_coef_target_competidores, df_coef_target_campanhas, df_coef_target_novos_produtos, df_coef_target_crm, df_coef_target_novas_lojas, df_coef_target_delivery, df_coef_target_cmv):
        def diferenca_bruta(val2, val1):
            """
            Calcula a diferença bruta entre dois valores.

            Args:
                val2 (float): Valor 2.
                val1 (float): Valor 1.

            Returns:
                float: Diferença bruta entre os dois valores.
            """
            return val2 - val1
        
        def diferenca_percentual(val2, val1):
            """
            Calcula a diferença percentual entre dois valores.

            Args:
                val2 (float): Valor 2.
                val1 (float): Valor 1.

            Returns:
                float: Diferença percentual entre os dois valores.
            """
            a = abs(val2) - abs(val1)
            b = (abs(val2) + abs(val1)) / 2
            c = a / b * 100

            return np.nan_to_num(c)
        
        def proporcao_pilares(val_ini, val_fin, pilar, sum_pilar):
            """
            Calcula a proporção de um pilar em relação à soma de todos os pilares.

            Args:
                val_ini (float): Valor inicial do pilar.
                val_fin (float): Valor final do pilar.
                pilar (float): Valor do pilar específico.
                sum_pilar (float): Soma de todos os pilares.

            Returns:
                float: Proporção do pilar em relação à soma de todos os pilares.
            """
            return -((val_ini - val_fin) * pilar) / sum_pilar
        
        # Crie os valores para o eixo y
        if self.flag == 1:
            # Calcule a diferença entre os dados para as duas datas
            dados_diferenca = [
                diferenca_bruta(df_coef_target_precos.loc[self.data_2], df_coef_target_precos.loc[self.data_1]),
                diferenca_bruta(df_coef_target_campanhas.loc[self.data_2],df_coef_target_campanhas.loc[self.data_1],),
                diferenca_bruta(df_coef_target_novos_produtos.loc[self.data_2],df_coef_target_novos_produtos.loc[self.data_1],),
                diferenca_bruta(df_coef_target_novas_lojas.loc[self.data_2],df_coef_target_novas_lojas.loc[self.data_1],),
                diferenca_bruta(df_coef_target_crm.loc[self.data_2], df_coef_target_crm.loc[self.data_1]),
                diferenca_bruta(df_coef_target_delivery.loc[self.data_2], df_coef_target_delivery.loc[self.data_1]),
                diferenca_bruta(df_coef_target_cmv.loc[self.data_2], df_coef_target_cmv.loc[self.data_1]),
                diferenca_bruta(df_coef_target_custos.loc[self.data_2], df_coef_target_custos.loc[self.data_1]),
                diferenca_bruta(df_coef_target_taxas.loc[self.data_2], df_coef_target_taxas.loc[self.data_1]),
                diferenca_bruta(df_coef_target_competidores.loc[self.data_2],df_coef_target_competidores.loc[self.data_1],),
                diferenca_bruta(df_coef_target_clima.loc[self.data_2], df_coef_target_clima.loc[self.data_1]),
                diferenca_bruta(df_coef_target_demo.loc[self.data_2], df_coef_target_demo.loc[self.data_1]),
                diferenca_bruta(df_coef_target_qtd_vendidas.loc[self.data_2],df_coef_target_qtd_vendidas.loc[self.data_1],),
            ]
            # pilares_, dados_diferenca_ = sort_self.pilares(self.pilares, dados_diferenca)
            pilares_, dados_diferenca_ = self.pilares, dados_diferenca
            valores_y = [df_target.loc[self.data_1]] + dados_diferenca_ + [df_target.loc[self.data_2]]

        else:
            # Calcule a diferença entre os dados para as duas datas
            dados_diferenca = [
                diferenca_percentual(df_coef_target_precos.loc[self.data_2], df_coef_target_precos.loc[self.data_1]),
                diferenca_percentual(df_coef_target_campanhas.loc[self.data_2],df_coef_target_campanhas.loc[self.data_1],),
                diferenca_percentual(df_coef_target_novos_produtos.loc[self.data_2],df_coef_target_novos_produtos.loc[self.data_1],),
                diferenca_percentual(df_coef_target_novas_lojas.loc[self.data_2],df_coef_target_novas_lojas.loc[self.data_1],),
                diferenca_percentual(df_coef_target_crm.loc[self.data_2], df_coef_target_crm.loc[self.data_1]),
                diferenca_percentual(df_coef_target_delivery.loc[self.data_2], df_coef_target_delivery.loc[self.data_1]),
                diferenca_percentual(df_coef_target_cmv.loc[self.data_2], df_coef_target_cmv.loc[self.data_1]),
                diferenca_percentual(df_coef_target_custos.loc[self.data_2], df_coef_target_custos.loc[self.data_1]),
                diferenca_percentual(df_coef_target_taxas.loc[self.data_2], df_coef_target_taxas.loc[self.data_1]),
                diferenca_percentual(df_coef_target_competidores.loc[self.data_2],df_coef_target_competidores.loc[self.data_1],),
                diferenca_percentual(df_coef_target_clima.loc[self.data_2], df_coef_target_clima.loc[self.data_1]),
                diferenca_percentual(df_coef_target_demo.loc[self.data_2], df_coef_target_demo.loc[self.data_1]),
                diferenca_percentual(df_coef_target_qtd_vendidas.loc[self.data_2],df_coef_target_qtd_vendidas.loc[self.data_1],),
            ]
            # pilares_, dados_diferenca_ = sort_self.pilares(self.pilares, dados_diferenca)
            pilares_, dados_diferenca_ = self.pilares, dados_diferenca
            val_ini = [100]
            val_fin = [df_target.loc[self.data_2] * 100 / df_target.loc[self.data_1]]
            sum_pilar = sum(dados_diferenca_)
            dados_diferenca_prop = [
                proporcao_pilares(val_ini[0], val_fin[0], pilar, sum_pilar)
                for pilar in dados_diferenca_
            ]

            valores_y = val_ini + dados_diferenca_prop + val_fin
        
        # pilares_, dados_diferenca_ = sort_self.pilares(self.pilares, dados_diferenca)
        pilares_, dados_diferenca_ = self.pilares, dados_diferenca

        if target == "net_sales":
            target_txt = "NET SALES "
            valores_txt = f"Bruto: {round(df_target.loc[self.data_1].sum()/1000,2)}k e {round(df_target.loc[self.data_2].sum()/1000,2)}k"
        elif target == "margem":
            target_txt = "MARGEM "
            valores_txt = f" {round(df_target.loc[self.data_1]*100,2)}% e {round(df_target.loc[self.data_2]*100,2)}%"
        elif target == "gc":
            target_txt = "GC "
            valores_txt = f"Bruto: {round(df_target.loc[self.data_1].sum()/1000,2)}k e {round(df_target.loc[self.data_2].sum()/1000,2)}k"

        valor_inicial = valores_txt.split(" e ")[0].replace("Bruto: ", "")
        valor_final = valores_txt.split(" e ")[1]

        # Crie os valores para o eixo x
        valores_x = [f"<b>{target_txt}INICIAL<br>{valor_inicial}</b>"] + pilares_ + [f"<b>{target_txt}FINAL<br>{valor_final}</b>"]

        list_valores_acumulados = [0]
        valor_acumulado = 0
        for item in valores_y[:-1]:
            valor_acumulado += int(item)
            list_valores_acumulados.append(valor_acumulado)

        # Definir os valores mínimos e máximos dos eixos X e Y
        min_y = min(list_valores_acumulados)
        max_y = max(list_valores_acumulados)

        # Crie a figura do gráfico cascata
        fig = go.Figure()
        fig.add_trace(
            go.Waterfall(
                name="Pilares",
                orientation="v",
                measure=["absolute"] + ["relative"] * len(pilares_) + ["total"],
                x=valores_x,
                textposition="outside",
                text=valores_y,
                y=valores_y,
                texttemplate="%{text:.2f}",
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            )
        )

        loja_txt = (
            f"Loja: {self.lojas[0]}"
            if len(self.lojas) == 1
            else "Lojas: " + ", ".join(map(str, self.lojas))
        )
        title_graph = f"Golden Bridge {target_txt}<br>{self.data_1} e {self.data_2}<br>{loja_txt}<br>Proporção percentual"
        fig.update_layout(
            title=dict(
                text="<span style='text-align:center;font-size:24px;font-weight:bold;'>" + title_graph + "</span>",
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top'
            ),
            showlegend=False,
            yaxis=dict(
                title="Valor",
                showticklabels=False,
                range=[min_y, max_y + 20],
            ),
            xaxis=dict(title="Pilar"),
        )

        # Exiba o gráfico
        return fig

    def plot_peso_var_stack(self, titulo, *dfs):
        """
        Cria um gráfico de área empilhada para visualizar o peso percentual por pilar ao longo do tempo.

        Args:
            titulo (str): Título do gráfico.
            dfs (DataFrame): Um ou mais DataFrames contendo os dados dos pilares.

        Returns:
            None
        """
        fig = go.Figure()

        for name, df in zip(self.names, dfs):
            # Filtra os dados relevantes com base no intervalo de datas
            # df_raw = df.loc[(df.index >= self.data_1) & (df.index <= self.data_2)].sum(axis=1)
            df_raw = df.sum(axis=1)

            # Adiciona o gráfico dos dados brutos
            fig.add_trace(
                go.Scatter(
                    x=df_raw.index,
                    y=df_raw.values,
                    mode="lines",
                    name=name,
                    line=dict(width=0.5),
                    stackgroup="one",
                    fill="tonexty",
                )
            )

        # Configura o layout do gráfico
        fig.update_layout(
            title=dict(
                    text="<span style='text-align:center;font-size:24px;font-weight:bold;'>" + titulo + "</span>",
                    x=0.5,
                    y=0.95,
                    xanchor='center',
                    yanchor='top'
            ),
            height=600,
            width=1200,
            title_text=titulo,
            title_x=0.5,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
            plot_bgcolor="rgba(0, 0, 0, 0.01)",
        )

        # Configura os títulos dos eixos
        fig.update_yaxes(title_text="Peso Percentual por Pilar")
        fig.update_xaxes(title_text="Data")

        return fig

    def plot_peso_var(self, df_dist_perc_target, regex_pilar, titulo, regex_subpilar=None):
        """
        Cria um gráfico para visualizar o valor diário percentual e a variação acumulada percentual de um pilar ou subpilar ao longo do tempo.

        Args:
            regex_pilar (str): Expressão regular para filtrar o pilar desejado.
            titulo (str): Título do gráfico.
            regex_subpilar (str, optional): Expressão regular para filtrar o subpilar desejado. Default é None.

        Returns:
            None
        """
        # Filtra os dados relevantes com base nos regex fornecidos
        # if regex_subpilar is not None:
        #     df = (df_dist_perc_target.filter(regex=regex_pilar).filter(regex=regex_subpilar).loc[(df_dist_perc_target.index >= self.data_1) &(df_dist_perc_target.index <= self.data_2)].sum(axis=1).diff().cumsum())
        #     df_raw = (df_dist_perc_target.filter(regex=regex_pilar).filter(regex=regex_subpilar).loc[(df_dist_perc_target.index >= self.data_1) & (df_dist_perc_target.index <= self.data_2)].sum(axis=1))
        # else:
        #     df = (df_dist_perc_target.filter(regex=regex_pilar).loc[(df_dist_perc_target.index >= self.data_1) & (df_dist_perc_target.index <= self.data_2)].sum(axis=1).diff().cumsum())
        #     df_raw = (df_dist_perc_target.filter(regex=regex_pilar).loc[(df_dist_perc_target.index >= self.data_1) & (df_dist_perc_target.index <= self.data_2)].sum(axis=1))
        
        if regex_subpilar is not None:
            df = (df_dist_perc_target.filter(regex=regex_pilar).filter(regex=regex_subpilar).sum(axis=1).diff().cumsum())
            df_raw = (df_dist_perc_target.filter(regex=regex_pilar).filter(regex=regex_subpilar).sum(axis=1))
        else:
            df = (df_dist_perc_target.filter(regex=regex_pilar).sum(axis=1).diff().cumsum())
            df_raw = (df_dist_perc_target.filter(regex=regex_pilar).sum(axis=1))

        # Cria a figura com 2 subplots, o segundo subplot terá metade da altura do primeiro
        fig = make_subplots(rows=2, cols=1)

        # Adiciona o gráfico dos dados brutos ao primeiro subplot
        fig.add_trace(
            go.Scatter(
                x=df_raw.index,
                y=df_raw.values,
                mode="lines",
                name="Valor Diário Percentual",
                line=dict(color="royalblue"),
                fill="tozeroy",
            ),
            row=1,
            col=1,
        )

        # Adiciona o gráfico da variação percentual ao segundo subplot
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df.values,
                mode="lines",
                name="Var. Acum. Percentual",
                line=dict(color="firebrick"),
                fill="tozeroy",
            ),
            row=2,
            col=1,
        )

        # Configura o layout do gráfico
        fig.update_layout(
            height=600,
            width=1200,
            title_text=titulo,
            title_x=0.5,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
            plot_bgcolor="rgba(0, 0, 0, 0.01)",
        )

        # Configura os títulos dos eixos para cada subplot
        fig.update_yaxes(title_text="Valor Diário Percentual", row=1, col=1)
        fig.update_yaxes(title_text="Var. Acum. Percentual", row=2, col=1)
        fig.update_xaxes(title_text="Data", row=2, col=1)
        fig.update_xaxes(showticklabels=False, row=1, col=1)

        # Define a altura dos subplots
        fig["layout"]["yaxis"]["domain"] = [0.25, 1.0]
        fig["layout"]["yaxis2"]["domain"] = [0.0, 0.2]

        return fig

    def generate_report(self, column):
        """
        Gera um relatório comparando os dados entre duas datas específicas para uma coluna fornecida.

        Args:
            data (DataFrame): DataFrame de dados.
            column (str): Nome da coluna a ser comparada.

        Returns:
            DataFrame: Relatório com a comparação dos dados.
        """
        # Agrupa os dados por data e família de produtos e calcula a soma
        produto_familia_data = (
            self.produto[[column, "PRODUTO_plus__p_familia_brasil"]]
            .reset_index()
            # .drop(columns="lojas__br_store_abv")
            .drop(columns="filtros_usados")
            .set_index("calendario__d_date")
        )
        produto_familia_data = produto_familia_data.groupby(
            ["calendario__d_date", "PRODUTO_plus__p_familia_brasil"]
        ).sum()

        # Obtém os dados para as datas específicas
        df1 = produto_familia_data.loc[self.data_1].reset_index()
        df2 = produto_familia_data.loc[self.data_2].reset_index()
        print(df1)
        # Faz o merge dos dados para as datas especificadas
        df_merge = df1.merge(df2, on="PRODUTO_plus__p_familia_brasil", how="outer")

        # Calcula a diferença entre as duas datas
        difference = (
            produto_familia_data.loc[self.data_2] - produto_familia_data.loc[self.data_1]
        ).reset_index()

        # Faz o merge dos dados de diferença com o merge anterior
        df_merge_diff = df_merge.merge(
            difference, on="PRODUTO_plus__p_familia_brasil", how="outer"
        )

        # Renomeia as colunas
        df_merge_diff = df_merge_diff.rename(
            columns={
                f"{column}_x": f"{column}_{self.data_1}",
                f"{column}_y": f"{column}_{self.data_2}",
                f"{column}": "Difference",
            }
        )

        # Calcula a diferença percentual
        df_merge_diff["Difference_%"] = (
            df_merge_diff["Difference"] / df_merge_diff[f"{column}_{self.data_1}"] * 100
        )

        # Ordena os dados pela diferença percentual
        df_merge_diff = df_merge_diff.sort_values("Difference_%")

        cols_to_fill = df_merge_diff.filter(regex="Difference|sum_plu").columns
        df_merge_diff[cols_to_fill] = df_merge_diff[cols_to_fill].fillna(0)

        return df_merge_diff

    def f_report_clima(self):
        """
        Gera um relatório comparativo entre dois dataframes de clima.

        Args:
            clima (pd.DataFrame): Dataframe de clima.
            lojas (list): Lista de lojas a serem incluídas no relatório.

        Returns:
            pd.DataFrame: Relatório comparativo do clima para as lojas especificadas.

        """
        # Filtra os dataframes para as datas especificadas
        df1 = self.clima.reset_index().query("calendario__d_date == @self.data_1")
        df2 = self.clima.reset_index().query("calendario__d_date == @self.data_2")
        
        # Calcula a diferença entre os dois dataframes
        df_report_clima = (
            df2.reset_index(drop=True)
            .iloc[:, 2:]
            .sub(df1.reset_index(drop=True).iloc[:, 2:])
        )
        
        df_report_clima["filtros_usados"] = df1["filtros_usados"].reset_index(drop=True)[0]
        
        df_report_clima = df_report_clima.set_index("filtros_usados")
        
        return df_report_clima

    def run(self):
        self.get_dataframes()
        self.get_coeficients()
import streamlit as st
from streamlit_option_menu import option_menu
from datetime import date
import pandas as pd
import os
import re
import plotly.graph_objects as go
from modules.modelo_algoritmo.load_and_create_table_model import DataProcessor
from modules.modelo_algoritmo.training_model import RidgeModelMultiTarget
from modules.visualize_and_report.create_graphs import ModelAnalysis
from modules.metabase.MetabaseSession import MetabaseSession
from datetime import datetime

st.set_page_config (layout="wide")


@st.cache_resource(show_spinner=False,experimental_allow_widgets=True)
def main_page():
    st.markdown("""
    <style>
    .reportview-container {
        background: #f0efeb
    }
    .big-font {
        font-size:40px !important;
        color: #FFC300;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        -webkit-text-stroke: 1px rgba(0, 0, 0, 0.1);
    }
    .sub-header {
        font-size:20px !important;
        border
        color: #004d40;
    }
    .widget-label {
        font-weight: bold;
        font-size:15px;
        color: #333;
    }
    .stButton>button {
        color: white;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        transition-duration: 0.4s;
        cursor: pointer;
        background-color: #FFC300;
        border: none;
        border-radius: 5px;
        box-shadow: 0px 2px 4px 0px rgba(0, 0, 0, 0.2), 0px 3px 10px 0px rgba(0, 0, 0, 0.19);
    }
    .stButton>button:hover {
        background-color: #FFA500;
        color: black;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <h1 class='big-font' style='text-align: center;'>Model Training by Store</h1>
        """,
        unsafe_allow_html=True
    )

    sk_dim_store_in_fact_plu = pd.read_parquet(r'.\data\df_sk_dim_store_unique_in_fact_plu.parquet')
    dim_store = pd.read_parquet(r'.\data\dim_store\df_dim_store.parquet')
    dim_store_in_fact_plu = dim_store.loc[dim_store['s_sk_dim_store'].isin(sk_dim_store_in_fact_plu['s_sk_dim_store'])]

    # streamlit select datetime to d1 and d2
    col1, col2, col3 = st.columns([1,2,1])
    # col1_1 col1_2, col1_3 = st.columns([1,1,1])
    with col2:
        
        selected_period = st.selectbox('Select the period:', [
            # 'Monthly', 'Weekly', 
            'Daily'])

        selected_abv = st.multiselect('Select the stores:', sorted(dim_store_in_fact_plu['s_br_store_abv'].astype(str).unique()))

        selected_cidade_brasil = st.multiselect('Select the cities:', sorted(dim_store_in_fact_plu['s_cidade_brasil'].astype(str).unique()))

        selected_estado_brasil = st.multiselect('Select the states:', sorted(dim_store_in_fact_plu['s_estado_brasil'].astype(str).unique()))

        selected_regiao_brasil = st.multiselect('Select the regions:', sorted(dim_store_in_fact_plu['s_regiao_brasil'].astype(str).unique()))

        selected_regional_brasil = st.multiselect('Select the regionals:', sorted(dim_store_in_fact_plu['s_regional_brasil'].astype(str).unique()))

        selected_regiao_estrategica_brasil = st.multiselect('Select the strategic regions:', sorted(dim_store_in_fact_plu['s_regiao_estrategica_brasil'].astype(str).unique()))

        selected_status_brasil = st.multiselect('Select the status:', sorted(dim_store_in_fact_plu['s_status_brasil'].astype(str).unique()))

        selected_type_store_brazil = st.multiselect('Select the type of store:', sorted(dim_store_in_fact_plu['s_type_store_brazil'].astype(str).unique()))

        filters = { 
            'br_store_abv': selected_abv,
            's_cidade_brasil': selected_cidade_brasil,
            's_estado_brasil': selected_estado_brasil,
            's_regiao_brasil': selected_regiao_brasil,
            's_regional_brasil': selected_regional_brasil,
            's_regiao_estrategica_brasil': selected_regiao_estrategica_brasil,
            's_status_brasil': selected_status_brasil,
            's_type_store_brazil': selected_type_store_brazil,
            'd_date': ['2022-01-01', '2022-12-31'],
        }

        if st.button('Train Model'):
            with st.spinner('Carregando os dados...'):
                if selected_period == 'Monthly':
                    periodicidade = 'M'
                elif selected_period == 'Weekly':
                    periodicidade = 'W'
                elif selected_period == 'Daily':
                    periodicidade = 'D'  
                
                list_of_keys_to_remove = []
                for key, value in filters.items():
                    if value == []:
                        list_of_keys_to_remove.append(key)
                
                for key in list_of_keys_to_remove:
                    del filters[key]
                
                data_processor = DataProcessor(filters, periodicidade, st.session_state['token_metabase']) 
            st.success('Os dados foram carregados com sucesso!')

            progress_bar = st.progress(0, text='Treinando o modelo...')
            counter = [0]
            ridge_model_multi_target = RidgeModelMultiTarget(
                df_model=data_processor.df_model,
                target=data_processor.target,
                start_month=1,
                end_month=12,
                residual_max=300, # definir melhor valor
                r2_score_min=0.75,
                n_bootstrap=5, # definir melhor valor
                counter=counter,
                progress_bar=progress_bar
            )

            st.success('O modelo foi treinado com sucesso!')
            st.session_state['ridge_model_multi_target'] = ridge_model_multi_target
            st.session_state['x_weighted'] = ridge_model_multi_target.X_weighted
            st.session_state['feature_names'] = ridge_model_multi_target.feature_names
            st.session_state['all_data_pred_sum_plu_net_sale_calculado'] = ridge_model_multi_target.all_data_pred_sum_plu_net_sale_calculado
            st.session_state['df_model'] = data_processor.df_model
            st.session_state['produto'] = data_processor.produto
            st.session_state['clima'] = data_processor.clima
            st.session_state['compt'] = data_processor.compt
            st.session_state['camp'] = data_processor.camp
            st.session_state['demo'] = data_processor.demo.reset_index()

            st.session_state['min_date'] = data_processor.df_model.index.min()
            st.session_state['max_date'] = data_processor.df_model.index.max()

            st.session_state['selected_abv'] = selected_abv
            
            st.session_state['filtros_usados'] = st.session_state['produto'].index[0][0]



def page_of_training_stats():
    st.markdown("""
    <style>
    .reportview-container {
        background: #f0efeb
    }
    .big-font {
        font-size:40px !important;
        color: #FFC300;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        -webkit-text-stroke: 1px rgba(0, 0, 0, 0.1);
    }
    .sub-header {
        font-size:20px !important;
        border
        color: #004d40;
    }
    .widget-label {
        font-weight: bold;
        font-size:15px;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.session_state['ridge_model_multi_target'] is None:
        st.error('O modelo não foi treinado!')
        return
    else:
        abv_name = st.session_state['selected_abv']
        title = f"Training Stats - {abv_name}"
        st.markdown(f"<h1  class='big-font' style='text-align: center;'>{title}</h1>", unsafe_allow_html=True)
        with st.expander("Gráficos de acertividade do modelo"):
            st.plotly_chart(st.session_state['ridge_model_multi_target'].graphic_net_sale, use_container_width=True)
            st.plotly_chart(st.session_state['ridge_model_multi_target'].graphic_margem, use_container_width=True)
            st.plotly_chart(st.session_state['ridge_model_multi_target'].graphic_gc_count, use_container_width=True)

        with st.expander("Tabela de Erros"):
            st.dataframe(st.session_state['ridge_model_multi_target'].error)


def graphics_of_day_results():
    st.markdown("""
    <style>
    .reportview-container {
        background: #f0efeb
    }
    .big-font {
        font-size:40px !important;
        color: #FFC300;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        -webkit-text-stroke: 1px rgba(0, 0, 0, 0.1);
    }
    .sub-header {
        font-size:20px !important;
        border
        color: #004d40;
    }
    .widget-label {
        font-weight: bold;
        font-size:15px;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.session_state['ridge_model_multi_target'] is None:
        st.error('O modelo não foi treinado!')
        return
    else:
        abv_name = st.session_state['selected_abv']
        title = f"Golden Bridge - All Days Results - {', '.join(abv_name)}"
        st.markdown(f"<h1 class='big-font' style='text-align: center;'>{title}</h1>", unsafe_allow_html=True)

        # streamlit select datetime to d1 and d2
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            d1 = st.date_input('Select a start date', st.session_state['min_date'], min_value=st.session_state['min_date'], max_value=st.session_state['max_date'])
            d2 = st.date_input('Select an end date', st.session_state['max_date'], min_value=st.session_state['min_date'], max_value=st.session_state['max_date'])

        # Format the date objects as 'YYYY-MM-DD' strings
        d1_str = d1.strftime('%Y-%m-%d')
        d2_str = d2.strftime('%Y-%m-%d')
        
        model_analysis = ModelAnalysis(
            X_weighted = st.session_state['x_weighted'],
            feature_names= st.session_state['feature_names'], 
            all_data_pred_sum_plu_net_sale_calculado = st.session_state['all_data_pred_sum_plu_net_sale_calculado'], 
            df_model = st.session_state['df_model'], 
            produto = st.session_state['produto'],
            clima = st.session_state['clima'],
            compt = st.session_state['compt'],
            camp = st.session_state['camp'],
            demo = st.session_state['demo'],
            flag = 0,
            lojas = st.session_state['filtros_usados'],
            data_1 = d1_str,
            data_2 = d2_str
            )

        st.markdown("---")

        st.plotly_chart(model_analysis.plot_golden_bridge_net_sale, use_container_width=True)

        st.markdown("---")

        st.plotly_chart(model_analysis.plot_golden_bridge_margem, use_container_width=True)

        st.markdown("---")

        st.plotly_chart(model_analysis.plot_golden_bridge_gc, use_container_width=True)

        st.markdown("---")

    with st.expander("Product Report"):

        data_obj = datetime.strptime(d1_str, "%Y-%m-%d")
        d1_str_formatted_BR = data_obj.strftime("%d/%m/%Y")

        data_obj = datetime.strptime(d2_str, "%Y-%m-%d")
        d2_str_formatted_BR = data_obj.strftime("%d/%m/%Y")

        columns = {
            "PRODUTO_plus__p_familia_brasil": "P Familia Brasil",
            "Difference": "Diferença Bruta",
            "Difference_%": "Diferença Percentual",
            f"PRODUTO_sum_plu_unit_sold_{d1_str}": f"Somatória de Qtd Vendidas {d1_str_formatted_BR}",
            f"PRODUTO_sum_plu_unit_sold_{d2_str}": f"Somatória de Qtd Vendidas {d2_str_formatted_BR}",
            f"PRODUTO_sum_plu_unit_price_{d1_str}": f"Somatória de Preço {d1_str_formatted_BR}",
            f"PRODUTO_sum_plu_unit_price_{d2_str}": f"Somatória de Preço {d2_str_formatted_BR}",
            f"PRODUTO_sum_plu_cost_total_calculado_{d1_str}": f"Somatória de Custo {d1_str_formatted_BR}",
            f"PRODUTO_sum_plu_cost_total_calculado_{d2_str}": f"Somatória de Custo {d2_str_formatted_BR}",
            f"PRODUTO_sum_plu_val_tax_{d1_str}": f"Somatória de Taxas {d1_str_formatted_BR}",
            f"PRODUTO_sum_plu_val_tax_{d2_str}": f"Somatória de Taxas {d2_str_formatted_BR}",
            "lojas__br_store_abv": "Abreviação Loja",
            "CLIMA_Precipitacao(mm)": "Precipitação(mm)",
            "CLIMA_Radiacao Global(KJ/m2)": "Radiação Global(KJ/m²)",
            "CLIMA_Temp Max(°C)": "Temperatura Máxima(°C)",
            "CLIMA_Umidade Relativa(%)": "Umidade Relativa(%)",
        }

        title_qtd_vendidas = "Pilar QTD Vendidas - Quantidades Vendidas e Diferenças entre datas"
        st.markdown(f"<h2 style='text-align: center;'>{title_qtd_vendidas}</h2>", unsafe_allow_html=True)
        st.dataframe(model_analysis.report_qtd_vendidas.rename(columns=columns), use_container_width=True)
        st.markdown("---")
        title_preco = "Pilar Preços - Somatória de Preço e Diferenças entre datas"
        st.markdown(f"<h2 style='text-align: center;'>{title_preco}</h2>", unsafe_allow_html=True)
        st.dataframe(model_analysis.report_preco.rename(columns=columns), use_container_width=True)
        st.markdown("---")
        title_custo = "Pilar Custos - Somatória de Custos e Diferenças entre datas"
        st.markdown(f"<h2 style='text-align: center;'>{title_custo}</h2>", unsafe_allow_html=True)
        st.dataframe(model_analysis.report_custo.rename(columns=columns), use_container_width=True)
        st.markdown("---")
        title_taxas = "Pilar Taxas - Somatória de Taxas e Diferenças entre datas"
        st.markdown(f"<h2 style='text-align: center;'>{title_taxas}</h2>", unsafe_allow_html=True)
        st.dataframe(model_analysis.report_taxas.rename(columns=columns), use_container_width=True)
        st.markdown("---")
        title_clima = "Pilar Clima - Clima por loja"
        st.markdown(f"<h2 style='text-align: center;'>{title_clima}</h2>", unsafe_allow_html=True)
        st.dataframe(model_analysis.report_clima, use_container_width=True)
        st.markdown("---")
        title_compt = "Pilar Competidor - Quantidade de Competidores por loja"
        st.markdown(f"<h2 style='text-align: center;'>{title_compt}</h2>", unsafe_allow_html=True)
        st.dataframe(model_analysis.report_compt.rename(columns=columns), use_container_width=True)
        st.markdown("---")
        title_camp = "Pilar Campanha - Quantidade de Campanhas por loja"
        st.markdown(f"<h2 style='text-align: center;'>{title_camp}</h2>", unsafe_allow_html=True)
        st.dataframe(model_analysis.report_camp.rename(columns=columns), use_container_width=True)
        st.markdown("---")
        title_demo = "Pilar Demografia - Informações demográficas por loja"
        st.markdown(f"<h2 style='text-align: center;'>{title_demo}</h2>", unsafe_allow_html=True)
        st.dataframe(model_analysis.report_demo.rename(columns=columns), use_container_width=True)
        st.markdown("---")

    


def graphics_of_all_day_results():
    st.markdown("""
    <style>
    .reportview-container {
        background: #f0efeb
    }
    .big-font {
        font-size:40px !important;
        color: #FFC300;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        -webkit-text-stroke: 1px rgba(0, 0, 0, 0.1);
    }
    .sub-header {
        font-size:20px !important;
        border
        color: #004d40;
    }
    .widget-label {
        font-weight: bold;
        font-size:15px;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.session_state['ridge_model_multi_target'] is None:
        st.error('O modelo não foi treinado!')
        return
    else:
        abv_name = st.session_state['selected_abv']
        title = f"Golden Bridge - All Days Results - {', '.join(abv_name)}"

        st.markdown(f"<h1  class='big-font' style='text-align: center;'>{title}</h1>", unsafe_allow_html=True)


        # streamlit select datetime to d1 and d2
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            d1 = st.date_input('Select a start date', st.session_state['min_date'], min_value=st.session_state['min_date'], max_value=st.session_state['max_date'])
            d2 = st.date_input('Select an end date', st.session_state['max_date'], min_value=st.session_state['min_date'], max_value=st.session_state['max_date'])

        # Format the date objects as 'YYYY-MM-DD' strings
        d1_str = d1.strftime('%Y-%m-%d')
        d2_str = d2.strftime('%Y-%m-%d')
        
        model_analysis = ModelAnalysis(
                                        st.session_state['x_weighted'], 
                                        st.session_state['feature_names'], 
                                        st.session_state['all_data_pred_sum_plu_net_sale_calculado'], 
                                        st.session_state['df_model'],  
                                        st.session_state['produto'],
                                        st.session_state['clima'],
                                        st.session_state['compt'],
                                        st.session_state['camp'],
                                        st.session_state['demo'],
                                        0, 
                                        st.session_state['filtros_usados'],
                                        d1_str,
                                        d2_str
                                        )
        
    with st.expander("Net Sales"):
        st.plotly_chart(model_analysis.plot_net_sale_geral, use_container_width=True)
        st.plotly_chart(model_analysis.plot_net_sale_qtd_vendida, use_container_width=True)
        st.plotly_chart(model_analysis.plot_net_sale_preco, use_container_width=True)
        st.plotly_chart(model_analysis.plot_net_sale_custo, use_container_width=True)
        st.plotly_chart(model_analysis.plot_net_sale_taxas, use_container_width=True)
        st.plotly_chart(model_analysis.plot_net_sale_clima, use_container_width=True)
        st.plotly_chart(model_analysis.plot_net_sale_compt, use_container_width=True)
        st.plotly_chart(model_analysis.plot_net_sale_camp, use_container_width=True)
        st.plotly_chart(model_analysis.plot_net_sale_demo, use_container_width=True)

    with st.expander("Margem"):
        
        st.plotly_chart(model_analysis.plot_margem_geral, use_container_width=True)
        st.plotly_chart(model_analysis.plot_margem_qtd_vendida, use_container_width=True)
        st.plotly_chart(model_analysis.plot_margem_preco, use_container_width=True)
        st.plotly_chart(model_analysis.plot_margem_custo, use_container_width=True)
        st.plotly_chart(model_analysis.plot_margem_taxas, use_container_width=True)
        st.plotly_chart(model_analysis.plot_margem_clima, use_container_width=True)
        st.plotly_chart(model_analysis.plot_margem_compt, use_container_width=True)
        st.plotly_chart(model_analysis.plot_margem_camp, use_container_width=True)
        st.plotly_chart(model_analysis.plot_margem_demo, use_container_width=True)
    
    with st.expander("GC"):
        st.plotly_chart(model_analysis.plot_gc_geral, use_container_width=True)
        st.plotly_chart(model_analysis.plot_gc_qtd_vendida, use_container_width=True)
        st.plotly_chart(model_analysis.plot_gc_preco, use_container_width=True)
        st.plotly_chart(model_analysis.plot_gc_custo, use_container_width=True)
        st.plotly_chart(model_analysis.plot_gc_taxas, use_container_width=True)
        st.plotly_chart(model_analysis.plot_gc_clima, use_container_width=True)
        st.plotly_chart(model_analysis.plot_gc_compt, use_container_width=True)
        st.plotly_chart(model_analysis.plot_gc_camp, use_container_width=True)
        st.plotly_chart(model_analysis.plot_gc_demo, use_container_width=True)


def login():
    st.markdown("""
    <style>
    .reportview-container {
        background: #f0efeb
    }
    .big-font {
        font-size:40px !important;
        color: #FFC300;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        -webkit-text-stroke: 1px rgba(0, 0, 0, 0.1);
    }
    .sub-header {
        font-size:20px !important;
        border
        color: #004d40;
    }
    .widget-label {
        font-weight: bold;
        font-size:15px;
        color: #333;
    }
    .stButton>button {
        color: white;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        transition-duration: 0.4s;
        cursor: pointer;
        background-color: #FFC300;
        border: none;
        border-radius: 5px;
        box-shadow: 0px 2px 4px 0px rgba(0, 0, 0, 0.2), 0px 3px 10px 0px rgba(0, 0, 0, 0.19);
    }
    .stButton>button:hover {
        background-color: #FFA500;
        color: black;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3,col4 = st.columns([1,1,1,5])
        

    with col2:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.image(".\images\Arcos-Dorados_Logo.png", width=200)

    with col4:
        st.markdown("<h1  class='big-font' style='text-align: center;' class='big-font'>Sign In - Metabase</h1>", unsafe_allow_html=True)
        st.write("")
        email = st.text_input("📧 Email")
        password = st.text_input("🔒 Password", type='password',)

        if st.button('🔑 Login'):
            try:
                metabase_session = MetabaseSession(email, password)
                st.session_state['email'] = email
                st.session_state['token_metabase'] = metabase_session.login()
            except Exception as e:
                st.warning('Login Failed')
            st.experimental_rerun()
def main():

    st.markdown("""
<style>
    .raised-box {
        margin: 0 auto;
        width: 80%;
        height: 60px;
        box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.05);
        border-radius: 16px;
        padding: 20px;
        background-color: transparent;
        font-size: 14px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

    if 'token_metabase' not in st.session_state or 'email' not in st.session_state:
        # limpa cache
        st.cache_data.clear()
        login()
        return
        
    
    col1, col2, col3, col4 = st.columns([2,2,2,1])
    
    with col4:
        user_name = st.session_state['email'].split('@')[0].split('.')[0].capitalize()
        st.markdown(
            f"<div class='raised-box'><p style='text-align: center; color: #FFC300; text-shadow: 2px 2px 4px #FFFFF;'>Bem vindo {user_name}</p></div>", 
            unsafe_allow_html=True
        )

        st.write("")
        st.write("")
    
    
    if 'ridge_model_multi_target' not in st.session_state:        
        st.session_state['ridge_model_multi_target'] = None
        st.session_state['x_weighted'] = None
        st.session_state['feature_names'] = None
        st.session_state['all_data_pred_sum_plu_net_sale_calculado'] = None
        st.session_state['df_model'] = None
        st.session_state['produto'] = None
        st.session_state['clima'] = None
        st.session_state['compt'] = None
        st.session_state['camp'] = None
        st.session_state['demo'] = None
        st.session_state['selected_abv'] = None
    

    selection = option_menu(
        menu_title=None,
        options=["Main Page", "Golden Bridge - Day Results", "Golden Bridge - All Period Results", "Training Stats"],
        icons=["house", "book", "envelope"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal"
    )

    if selection == "Main Page":
        #Guarda o ridge_model_multi_target em cache para não precisar treinar o modelo toda vez que mudar de página - ajuste o código para isso
        main_page()
    elif selection == "Golden Bridge - Day Results":
        graphics_of_day_results()
    elif selection == "Golden Bridge - All Period Results":
        graphics_of_all_day_results()
    elif selection == "Training Stats":
        page_of_training_stats()

if __name__ == "__main__":
    main()

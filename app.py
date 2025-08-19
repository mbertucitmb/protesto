# app.py

# ==============================================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import inflection
import joblib
import warnings
import os
import plotly.express as px
from streamlit_option_menu import option_menu

# Pré-processamento e Modelagem
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImblearnPipeline
import xgboost as xgb
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')

# ==============================================================================
# 2. CONFIGURAÇÕES DA PÁGINA E ESTILO
# ==============================================================================
st.set_page_config(
    page_title="Análise de Crédito TMB",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS para customizar a aparência
def load_css():
    st.markdown("""
        <style>
            /* Remove o menu e o footer do Streamlit */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            
            /* Estilo do container principal */
            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                padding-left: 5rem;
                padding-right: 5rem;
            }
            
            /* Ajuste de estilo para os títulos */
            h1, h2, h3 {
                color: #004280; /* Azul corporativo */
            }
        </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 3. CONSTANTES E CONFIGURAÇÕES DO MODELO
# ==============================================================================
PIPELINE_OUTPUT_PATH = 'modelo_protesto_pipeline.joblib'

# CORREÇÃO: Adicionadas as colunas que faltavam ('idade', 'percentual_vencido', 'percentual_pago', 'valor_medio_parcela')
FEATURES = [
    'score', 'idade', 'total_financiado', 'quantidade_parcelas', 'saldo_vencido',
    'quantidade_parcelas_vencidas', 'recebido', 'dias_em_atraso',
    'saldo_vencido_com_juros', 'total_pago_com_juros', 'vencidos_sem_juros_tmb',
    'recebido_sem_juros_tmb', 'score_x_idade', 'score_ao_quadrado', 'score_por_idade',
    'percentual_vencido', 'percentual_pago', 'valor_medio_parcela', 'status_cobranca',
    'faixa_etaria', 'regiao', 'segmento', 'categoria_risco_score', 'modalidade', 'pdd'
]
NUMERIC_FEATURES = [
    'score', 'idade', 'total_financiado', 'quantidade_parcelas', 'saldo_vencido',
    'quantidade_parcelas_vencidas', 'recebido', 'dias_em_atraso',
    'saldo_vencido_com_juros', 'total_pago_com_juros', 'vencidos_sem_juros_tmb',
    'recebido_sem_juros_tmb', 'score_x_idade', 'score_ao_quadrado', 'score_por_idade',
    'percentual_vencido', 'percentual_pago', 'valor_medio_parcela'
]
CATEGORICAL_FEATURES = [
    'status_cobranca', 'faixa_etaria', 'regiao', 'segmento',
    'categoria_risco_score', 'modalidade', 'pdd'
]

# ==============================================================================
# 4. FUNÇÕES AUXILIARES (CACHEADAS PARA PERFORMANCE)
# ==============================================================================
@st.cache_data
def load_data(uploaded_file):
    """Carrega e cacheia os dados do arquivo Excel."""
    return pd.read_excel(uploaded_file)

@st.cache_resource
def load_pipeline(pipeline_path):
    """Carrega e cacheia o pipeline de modelo treinado."""
    if os.path.exists(pipeline_path):
        return joblib.load(pipeline_path)
    return None

def preprocess_data_for_prediction(df):
    """Prepara os dados para predição, aplicando limpeza e engenharia de features."""
    # Renomear colunas
    df_clean = df.copy()
    df_clean.columns = [inflection.underscore(c).strip() for c in df_clean.columns]

    # Engenharia de features
    if 'idade' not in df_clean.columns or 'score' not in df_clean.columns:
        st.error("Erro Crítico: As colunas 'idade' e 'score' são obrigatórias e não foram encontradas no arquivo.")
        st.stop()

    df_clean['faixa_etaria'] = pd.cut(df_clean['idade'], bins=[0, 30, 50, 65, 120], labels=['18-30', '31-50', '51-65', '65+'], right=True)
    df_clean['score_x_idade'] = df_clean['score'] * df_clean['idade']
    df_clean['score_ao_quadrado'] = df_clean['score'] ** 2
    df_clean['score_por_idade'] = df_clean['score'] / (df_clean['idade'] + 1)

    mapa_regioes = {
        'AC': 'Norte', 'AP': 'Norte', 'AM': 'Norte', 'PA': 'Norte', 'RO': 'Norte', 'RR': 'Norte', 'TO': 'Norte',
        'AL': 'Nordeste', 'BA': 'Nordeste', 'CE': 'Nordeste', 'MA': 'Nordeste', 'PB': 'Nordeste', 'PE': 'Nordeste', 'PI': 'Nordeste', 'RN': 'Nordeste', 'SE': 'Nordeste',
        'DF': 'Centro-Oeste', 'GO': 'Centro-Oeste', 'MT': 'Centro-Oeste', 'MS': 'Centro-Oeste',
        'ES': 'Sudeste', 'MG': 'Sudeste', 'RJ': 'Sudeste', 'SP': 'Sudeste',
        'PR': 'Sul', 'RS': 'Sul', 'SC': 'Sul'
    }
    if 'endereco_estado' in df_clean.columns:
        df_clean['regiao'] = df_clean['endereco_estado'].str.upper().str.strip().map(mapa_regioes).fillna('Outra')
    else:
        df_clean['regiao'] = 'Outra'

    # Garante que todas as colunas esperadas pelo modelo existam, preenchendo com 0 se ausentes
    for col in FEATURES:
        if col not in df_clean.columns:
            df_clean[col] = 0
    
    return df_clean[FEATURES]

# ==============================================================================
# 5. PÁGINAS DA APLICAÇÃO
# ==============================================================================

def page_home():
    """Página de boas-vindas e instruções."""
    st.title("Plataforma de Análise de Crédito")
    st.markdown("---")
    st.subheader("Bem-vindo à ferramenta de scoring e previsão de pagamento.")
    st.markdown("""
        Esta plataforma utiliza um modelo de Machine Learning (XGBoost) para prever a probabilidade de um cliente pagar um título protestado. 
        Ela oferece duas funcionalidades principais:

        - **Realizar Previsão:** Envie uma planilha com novos dados de clientes para obter a classificação (Pagará / Não Pagará) e a probabilidade associada.
        - **Treinar Modelo:** Envie um novo conjunto de dados históricos para treinar, avaliar e salvar uma nova versão do modelo.

        **Como usar:**
        1.  Navegue até a página desejada usando o menu à esquerda.
        2.  Siga as instruções para carregar o arquivo de dados em formato `.xlsx`.
        3.  Clique nos botões de ação para gerar previsões ou iniciar o treinamento.
        4.  Explore os resultados e dashboards interativos.

        *Desenvolvido pelo time de Dados.*
    """)

def page_predict():
    """Página para realizar previsões com o modelo."""
    st.header("Realizar Previsão de Pagamento")
    
    pipeline = load_pipeline(PIPELINE_OUTPUT_PATH)
    if not pipeline:
        st.error("Modelo não encontrado! Por favor, treine um modelo na página 'Treinar Modelo' antes de fazer previsões.")
        return

    st.info("Para começar, carregue um arquivo Excel (.xlsx) contendo os dados dos clientes para análise.")
    uploaded_file = st.file_uploader("Carregar arquivo", type=["xlsx"], label_visibility="collapsed")

    if uploaded_file:
        df_original = load_data(uploaded_file)
        
        with st.expander("Visualizar dados carregados"):
            st.dataframe(df_original.head())
        
        if st.button("Gerar Previsões e Dashboard", type="primary"):
            with st.spinner("Processando dados e gerando previsões..."):
                df_to_predict = preprocess_data_for_prediction(df_original)
                
                predictions = pipeline.predict(df_to_predict)
                probabilities = pipeline.predict_proba(df_to_predict)

                # Montagem do resultado
                df_results = df_original.copy()
                df_results['Status_Previsto'] = ['Pagará' if p == 1 else 'Não Pagará' for p in predictions]
                df_results['Probabilidade_de_Pagar'] = (probabilities[:, 1] * 100).round(2)

            st.markdown("---")
            st.subheader("Dashboard de Resultados")

            # Métricas
            pagara_count = (predictions == 1).sum()
            nao_pagara_count = (predictions == 0).sum()
            col1, col2, col3 = st.columns(3)
            col1.metric("Total de Clientes", len(df_results))
            col2.metric("Previsão 'Pagará' ✔️", f"{pagara_count}")
            col3.metric("Previsão 'Não Pagará' ❌", f"{nao_pagara_count}")

            # Gráficos
            col_graf1, col_graf2 = st.columns(2)
            with col_graf1:
                fig_bar = px.bar(
                    x=['Pagará', 'Não Pagará'], 
                    y=[pagara_count, nao_pagara_count],
                    color=['Pagará', 'Não Pagará'],
                    color_discrete_map={'Pagará': '#28A745', 'Não Pagará': '#DC3545'},
                    title="Distribuição das Previsões",
                    labels={'x': 'Status', 'y': 'Contagem'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            with col_graf2:
                fig_hist = px.histogram(
                    df_results, x='Probabilidade_de_Pagar', nbins=20,
                    title="Distribuição das Probabilidades",
                    color_discrete_sequence=['#004280']
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            # Tabela de resultados
            with st.expander("Ver tabela de resultados detalhada"):
                st.dataframe(df_results)
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button("Baixar resultados em CSV", csv, "previsoes.csv", "text/csv")


def page_train():
    """Página para treinar um novo modelo."""
    st.header("Treinar Novo Modelo")
    st.warning("Atenção: O treinamento de um novo modelo pode levar vários minutos e substituirá o modelo atual em produção.")
    
    uploaded_file = st.file_uploader("Carregue o dataset de treino (.xlsx)", type=["xlsx"])
    
    if uploaded_file:
        df_train = load_data(uploaded_file)
        st.write("Amostra dos dados carregados:")
        st.dataframe(df_train.head(5))

        if st.button("Iniciar Treinamento Completo", type="primary"):
            # A lógica de treinamento é complexa e foi mantida como no original
            with st.spinner("Processo de treinamento em andamento..."):
                # Limpeza e preparação dos dados
                st.write("1. Limpando dados...")
                df_train.columns = [inflection.underscore(c).strip() for c in df_train.columns]
                df_train['pagamento'] = df_train['pagamento'].str.strip().replace('Pago no Cartório', 'Pagos TMB')
                df_train = df_train[df_train['pagamento'].isin(['Pagos TMB', 'Não foram pagos'])]
                df_train['pagamento'] = df_train['pagamento'].map({'Não foram pagos': 0, 'Pagos TMB': 1})
                df_train.dropna(subset=['pagamento'], inplace=True)

                st.write("2. Engenharia de features...")
                df_train['faixa_etaria'] = pd.cut(df_train['idade'], bins=[0, 30, 50, 65, 120], labels=['18-30', '31-50', '51-65', '65+'])
                df_train['score_x_idade'] = df_train['score'] * df_train['idade']
                df_train['score_ao_quadrado'] = df_train['score'] ** 2
                df_train['score_por_idade'] = df_train['score'] / (df_train['idade'] + 1)
                mapa_regioes = { 'AC': 'Norte', 'AP': 'Norte', 'AM': 'Norte', 'PA': 'Norte', 'RO': 'Norte', 'RR': 'Norte', 'TO': 'Norte', 'AL': 'Nordeste', 'BA': 'Nordeste', 'CE': 'Nordeste', 'MA': 'Nordeste', 'PB': 'Nordeste', 'PE': 'Nordeste', 'PI': 'Nordeste', 'RN': 'Nordeste', 'SE': 'Nordeste', 'DF': 'Centro-Oeste', 'GO': 'Centro-Oeste', 'MT': 'Centro-Oeste', 'MS': 'Centro-Oeste', 'ES': 'Sudeste', 'MG': 'Sudeste', 'RJ': 'Sudeste', 'SP': 'Sudeste', 'PR': 'Sul', 'RS': 'Sul', 'SC': 'Sul' }
                df_train['regiao'] = df_train['endereco_estado'].str.upper().str.strip().map(mapa_regioes).fillna('Outra')
                
                X = df_train[FEATURES]
                y = df_train['pagamento']

                st.write("3. Divisão de treino e teste...")
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                
                # Definição dos pipelines
                numeric_transformer = SklearnPipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', MinMaxScaler())])
                categorical_transformer = SklearnPipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
                preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, NUMERIC_FEATURES), ('cat', categorical_transformer, CATEGORICAL_FEATURES)], remainder='passthrough')
                
                st.write("4. Otimização de Hiperparâmetros (BayesSearchCV)...")
                pipeline_for_tuning = ImblearnPipeline(steps=[('preprocessor', preprocessor), ('smote', SMOTE(random_state=42)), ('model', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))])
                search_spaces = {'model__n_estimators': Integer(100, 500), 'model__learning_rate': Real(0.01, 0.2, 'log-uniform'), 'model__max_depth': Integer(3, 8), 'model__subsample': Real(0.6, 1.0, 'uniform')}
                bayes_cv = BayesSearchCV(estimator=pipeline_for_tuning, search_spaces=search_spaces, n_iter=20, cv=StratifiedKFold(n_splits=3), scoring='precision', n_jobs=-1, random_state=42)
                bayes_cv.fit(X_train, y_train)

                st.write("5. Treinando modelo final...")
                final_pipeline = bayes_cv.best_estimator_
                final_pipeline.fit(X_train, y_train)

                st.write("6. Avaliando performance...")
                y_pred = final_pipeline.predict(X_test)
                
                # Exibição dos resultados
                st.markdown("---")
                st.subheader("Resultados do Novo Modelo")
                st.text("Melhores Parâmetros Encontrados:")
                st.json(bayes_cv.best_params_)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.text("Relatório de Classificação:")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose())
                with col2:
                    st.text("Matriz de Confusão:")
                    st.dataframe(confusion_matrix(y_test, y_pred))

                # Salvando o pipeline
                st.write("7. Salvando o pipeline...")
                joblib.dump(final_pipeline, PIPELINE_OUTPUT_PATH)
            
            st.success(f"Novo modelo treinado e salvo com sucesso como '{PIPELINE_OUTPUT_PATH}'!")
            st.balloons()


# ==============================================================================
# 6. FUNÇÃO PRINCIPAL E NAVEGAÇÃO
# ==============================================================================
def main():
    load_css()
    
    with st.sidebar:
        st.title("📊 Análise de Crédito")
        selected = option_menu(
            menu_title=None,
            options=["Home", "Realizar Previsão", "Treinar Modelo"],
            icons=["house", "graph-up-arrow", "cpu"],
            menu_icon="cast",
            default_index=0,
        )

    if selected == "Home":
        page_home()
    elif selected == "Realizar Previsão":
        page_predict()
    elif selected == "Treinar Modelo":
        page_train()

if __name__ == "__main__":
    main()

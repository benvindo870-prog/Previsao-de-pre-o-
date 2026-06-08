import streamlit as st
import pandas as pd 
import joblib 

st.set_page_config(page_title="Simulador Imobiliário OLX")

st.title("Simulador de Preços de Imóveis")
st.write("Selecione o tipo de mercado e insira as características para estimar o preço médio com base nos dados do OLX.")

# 1. Escolha do Mercado (Arrendamento vs Venda)
tipo_negocio = st.radio("O que deseja simular?", ["Arrendamento (Renda)", "Compra / Venda (Preço Total)"])

if tipo_negocio == "Arrendamento (Renda)":
    ficheiro_modelo = 'modelo_arrendamento.pkl'
    sinal_moeda = "€ / mês"
else:
    ficheiro_modelo = 'modelo_venda.pkl'
    sinal_moeda = "€"

# Função com cache para a interface carregar instantaneamente
@st.cache_resource 
def carregar_modelo_dinamico(ficheiro):
    return joblib.load(ficheiro)

try:
    # Carrega automaticamente o modelo e a estrutura de colunas correta
    dados = carregar_modelo_dinamico(ficheiro_modelo)
    modelo = dados['modelo']
    colunas_treino = dados['colunas_treino']
   
    # 2. Criação do Formulário de Entrada
    with st.form("formulario_previsao"):
        col1, col2 = st.columns(2)
        with col1:
            area = st.number_input("Área Útil (m²)", min_value=10, max_value=1000, value=75, step=5)
        with col2:
            quartos = st.number_input("Número de Quartos (0 para T0/Estúdio)", min_value=0, max_value=10, value=2, step=1)
        
        # Filtra e extrai as localizações guardadas no ficheiro do modelo
        localizacoes = [col.replace('localizacao_', '') for col in colunas_treino if col.startswith('localizacao_')]
        localizacoes.sort()
        
        if not localizacoes:
            localizacoes = ["Zona Geral"]
            
        localizacao_selecionada = st.selectbox("Selecione la Localização / Freguesia", options=localizacoes)

        botao_prever = st.form_submit_button("Calcular Preço Estimado")
    
    # 3. Processamento e Cálculo da Previsão
    if botao_prever:
        # Monta as características base iguazinhas ao treino do new.py
        dados_input = {
            'area_m2': float(area),
            'quartos': float(quartos),
            'area_por_quarto': float(area / (quartos + 1))  # Proteção contra divisão por zero
        }
        
        # Ativa com o valor 1 apenas a localização que o utilizador escolheu
        dados_input[f'localizacao_{localizacao_selecionada}'] = 1
        
        # Transforma os dados num formato DataFrame de uma linha
        df_predict = pd.DataFrame([dados_input])
        
        # Sincroniza as colunas: mete 0 em todas as outras localizações que existem no modelo
        df_predict_processado = df_predict.reindex(columns=colunas_treino, fill_value=0)
        
        # Faz a previsão matemática
        preco_estimado = modelo.predict(df_predict_processado)[0]
        
        # Impede o modelo de dar valores negativos em simulações extremas
        if preco_estimado < 0:
            preco_estimado = 0.0
            
        # Apresenta o resultado limpo e formatado no ecrã
        st.success(f"O preço estimado para **{tipo_negocio}** é de: **{preco_estimado:,.2f} {sinal_moeda}**")

except FileNotFoundError:
    st.error(f"Erro: O ficheiro '{ficheiro_modelo}' não foi encontrado. Por favor, execute o script `python new.py` primeiro no seu terminal para gerar os dados do modelo.")
except Exception as e:
    st.error(f"Ocorreu um erro ao carregar os dados na interface: {e}")
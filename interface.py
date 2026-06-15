import streamlit as st
import pandas as pd 
import joblib 

st.set_page_config(page_title="Simulador Imobiliário OLX", layout="centered")

st.title("Simulador de Preços de Imóveis")
st.write("Selecione o tipo de mercado e insira as características para estimar o preço médio.")

tipo_negocio = st.radio("O que deseja simular?", ["Arrendamento (Renda)", "Compra / Venda (Preço Total)"])

if tipo_negocio == "Arrendamento (Renda)":
    ficheiro_modelo = 'modelo_arrendamento.pkl'
    sinal_moeda = "€ / mês"
else:
    ficheiro_modelo = 'modelo_venda.pkl'
    sinal_moeda = "€"

@st.cache_resource 
def carregar_modelo_dinamico(ficheiro):
    return joblib.load(ficheiro)

try:
    dados = carregar_modelo_dinamico(ficheiro_modelo)
    modelo = dados['modelo']
    colunas_treino = dados['colunas_treino']
   
    with st.form("formulario_previsao"):
        col1, col2 = st.columns(2)
        with col1:
            area = st.number_input("Área Útil (m²)", min_value=10, max_value=1000, value=75, step=5)
        with col2:
            quartos = st.number_input("Número de Quartos (0 para T0/Estúdio)", min_value=0, max_value=10, value=2, step=1)
        
        localizacoes = [col.replace('localizacao_', '') for col in colunas_treino if col.startswith('localizacao_')]
        localizacoes.sort()
        
        if not localizacoes:
            localizacoes = ["Zona Geral"]
            
        localizacao_selecionada = st.selectbox("Selecione a Localização / Freguesia", options=localizacoes)

        botao_prever = st.form_submit_button("Calcular Preço Estimado")
    
    if botao_prever:
        dados_input = {
            'area_m2': float(area),
            'quartos': float(quartos),
            'area_por_quarto': float(area / (quartos + 1)) 
        }
        

        dados_input[f'localizacao_{localizacao_selecionada}'] = 1
        

        df_predict = pd.DataFrame([dados_input])
        df_predict_processado = df_predict.reindex(columns=colunas_treino, fill_value=0)
        
        preco_estimado = modelo.predict(df_predict_processado)[0]
        
        if preco_estimado < 0:
            preco_estimado = 0.0
            
        st.success(f"O preço estimado para **{tipo_negocio}** em **{localizacao_selecionada}** é de: **{preco_estimado:,.2f} {sinal_moeda}**")

except FileNotFoundError:
    st.error(f"Erro: O ficheiro '{ficheiro_modelo}' não foi encontrado. Por favor, execute o script de treino primeiro para gerar os novos modelos.")
except Exception as e:
    st.error(f"Ocorreu um erro ao carregar os dados na interface: {e}")
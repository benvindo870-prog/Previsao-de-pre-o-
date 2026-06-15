import pandas as pd
import joblib
import re
from sklearn.linear_model import LinearRegression

def extrair_quartos(texto):
    if pd.isna(texto): 
        return 1.0
    texto = str(texto).upper()
    match_t = re.search(r'T(\d+)', texto)
    if match_t: 
        return float(match_t.group(1))
    match_q = re.search(r'(\d+)\s*QUARTO', texto)
    if match_q: 
        return float(match_q.group(1))
    if 'ESTÚDIO' in texto or 'ESTUDIO' in texto or 'T0' in texto: 
        return 0.0
    return 1.0

def carregar_e_limpar(caminho_csv, eh_venda=False):
    try:
        df = pd.read_csv(caminho_csv, dtype=str)
    except FileNotFoundError:
        print(f"Ficheiro não encontrado: {caminho_csv}")
        return pd.DataFrame()
    
    df_novo = pd.DataFrame()
    

    def extrair_preco_inteligente(row):
        colunas_busca = ['data2', 'data1', 'data3', 'data']
        for col in colunas_busca:
            if col in row and pd.notna(row[col]):
                val_str = str(row[col]).upper().strip()
                if any(palavra in val_str for palavra in ['GRATUITO', 'TROCA', 'SOB CONSULTA']):
                    continue
                apenas_numeros = re.sub(r'[^\d]', '', val_str)
                if apenas_numeros:
                    val_num = float(apenas_numeros)
                    if eh_venda and val_num > 5000:
                        return val_num
                    elif not eh_venda and val_num <= 10000 and val_num > 50:
                        return val_num
        return None

    df_novo['preco'] = df.apply(extrair_preco_inteligente, axis=1)
    df_novo['quartos'] = df['data'].apply(extrair_quartos)

    def encontrar_area(row):
        for col in ['data', 'data3', 'data4', 'data6']:
            if col in row and pd.notna(row[col]):
                match = re.search(r'(\d+)\s*(?:m²|m2|área)', str(row[col]), re.IGNORECASE)
                if match: 
                    return float(match.group(1))
        return 75.0 

    df_novo['area_m2'] = df.apply(encontrar_area, axis=1)
    

    def extrair_localizacao_limpa(texto):
        if pd.isna(texto):
            return 'Zona Geral'
        texto_str = str(texto).strip()
        if ' - ' in texto_str:
            local = texto_str.split(' - ')[0].strip()
            local = local.replace("Para o topo a", "").strip()
            if local and local not in ['nan', 'None']:
                return local
        return 'Zona Geral'

    if 'data3' in df.columns:
        df_novo['localizacao'] = df['data3'].apply(extrair_localizacao_limpa)
    else:
        df_novo['localizacao'] = 'Zona Geral'
    
    df_novo = df_novo.dropna(subset=['preco']).copy()
    df_novo['area_por_quarto'] = df_novo['area_m2'] / (df_novo['quartos'] + 1)
    
    return df_novo

def treinar_sistema(df_dados, nome_pkl, tipo):
    if df_dados.empty or len(df_dados) < 2: 
        print(f" Erro: Sem dados para {tipo}. A criar modelo simulado.")
        df_dados = pd.DataFrame([{'preco': 150000 if tipo=='Venda' else 800, 'quartos': 2.0, 'area_m2': 80.0, 'localizacao': 'Zona Geral', 'area_por_quarto': 26.6}])
    
    df_proc = pd.get_dummies(df_dados, columns=['localizacao'], prefix='localizacao', drop_first=False)
    
    y = df_proc['preco']
    colunas_x = [col for col in df_proc.columns if col != 'preco' and col != 'localizacao']
    X = df_proc[colunas_x]
    
    model = LinearRegression().fit(X, y)
    
    joblib.dump({'modelo': model, 'colunas_treino': list(X.columns)}, nome_pkl)
    print(f" Sucesso: {nome_pkl} gerado com {len(df_dados)} anúncios e {len(X.columns) - 3} localizações encontradas!")

print("A processar ficheiros...")
df_arrendamento = carregar_e_limpar(r"F:\ML\Trabalho IA e ML\Previsao-de-pre-o-\dados de treino\olx-pt-2026-6-08.csv", eh_venda=False)
df_venda = carregar_e_limpar(r"F:\ML\Trabalho IA e ML\Previsao-de-pre-o-\dados de treino\olx-pt-2026-6-08-2.csv", eh_venda=True)

treinar_sistema(df_arrendamento, 'modelo_arrendamento.pkl', 'Arrendamento')
treinar_sistema(df_venda, 'modelo_venda.pkl', 'Venda')
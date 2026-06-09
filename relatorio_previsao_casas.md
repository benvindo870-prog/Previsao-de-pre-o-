# Sistema de Previsão de Preços de Imóveis com Machine Learning

**Unidade Curricular:** Projeto Final de Curso — Inteligência Artificial Aplicada  
**Ano Letivo:** 2025/2026 | **Entrega:** Junho de 2026

---

## Resumo

Sistema de previsão de preços de imóveis baseado em Machine Learning, com dados reais recolhidos por web scraping do portal OLX Portugal. Inclui um pipeline de treino (Regressão Linear) e uma interface web em Streamlit para estimativas em tempo real. As principais tecnologias são Python, pandas, scikit-learn, joblib e Streamlit.

---

## 1. Introdução

### 1.1 Contextualização e Problema

O mercado imobiliário português tem registado volatilidade crescente, dificultando a avaliação justa de propriedades. Sem ferramentas acessíveis, compradores e vendedores dependem de avaliadores certificados ou comparações manuais — processos morosos e subjetivos.

**Questão central:** Dado um conjunto de características mensuráveis de um imóvel, é possível estimar automaticamente o seu preço de mercado com base em dados históricos de anúncios reais?

### 1.2 Objetivos

| Tipo | Objetivo |
|---|---|
| Geral | Desenvolver um sistema funcional de previsão de preços acessível via interface web |
| Específico 1 | Recolher e processar dados reais do OLX Portugal |
| Específico 2 | Treinar modelo de regressão linear com features numéricas |
| Específico 3 | Persistir o modelo e disponibilizá-lo via interface Streamlit |
| Específico 4 | Avaliar desempenho com métricas MAE e R² |

---

## 2. Estado da Arte

### 2.1 Conceitos Fundamentais

**Aprendizagem Supervisionada** — O modelo aprende a partir de pares (características do imóvel → preço) e generaliza para novos dados.

**Regressão** — Previsão de um valor contínuo (preço). Métricas: MAE, RMSE, R².

**Regressão Linear** — Modelo que ajusta uma equação linear aos dados. Vantagens: interpretável, simples, adequado para dados de pequena escala e features numéricas.

**Web Scraping** — Extração automatizada de dados de páginas web via extensão Web Scraper (Chrome).

### 2.2 Tecnologias Utilizadas

| Tecnologia | Função |
|---|---|
| Python 3.9+ | Linguagem principal |
| pandas | Leitura, limpeza e transformação do CSV |
| scikit-learn | Regressão Linear e métricas |
| joblib | Serialização do modelo |
| Streamlit | Interface web interativa |
| Web Scraper (ext.) | Recolha de dados do OLX.pt |

### 2.3 Comparação com Trabalhos Semelhantes

| Sistema | Abordagem | Escala |
|---|---|---|
| Zillow "Zestimate" (EUA) | Redes neuronais profundas | Milhões de imóveis |
| Idealista / Conf. Imobiliário (PT) | Índices baseados em transações reais | Nacional |
| Este Projeto | Regressão Linear + OLX scraping | ~240 anúncios, Portugal |

---

## 3. Requisitos

### 3.1 Requisitos Funcionais

| ID | Requisito | Prioridade |
|---|---|---|
| RF01 | Importar dados de imóveis de ficheiro CSV | Alta |
| RF02 | Limpar e processar dados antes do treino | Alta |
| RF03 | Extrair área e quartos de campos de texto por regex | Alta |
| RF04 | Treinar modelo de regressão supervisionada | Alta |
| RF05 | Guardar modelo em disco para reutilização | Alta |
| RF06 | Interface permite inserir área (m²) e tipologia | Alta |
| RF07 | Interface calcula e apresenta preço estimado | Alta |
| RF08 | Validar inputs com limites mínimos/máximos | Média |
| RF09 | Suportar extensão com variável de localização | Baixa |

### 3.2 Requisitos Não Funcionais

| ID | Requisito | Categoria |
|---|---|---|
| RNF01 | Resposta da previsão < 2 segundos | Desempenho |
| RNF02 | Modelo carregado em cache (`@st.cache_resource`) | Desempenho |
| RNF03 | Interface acessível via browser sem instalação extra | Usabilidade |
| RNF04 | Código modular: treino separado da inferência | Manutenibilidade |
| RNF05 | Compatível com Python 3.8+ | Compatibilidade |

### 3.3 Casos de Uso

**UC01 – Treinar o Modelo**
Ator: Desenvolvedor. Executa `new.py` → sistema lê o CSV, processa dados, treina a Regressão Linear e gera `modeloprevisaodeimovel.pkl`.

**UC02 – Estimar Preço de Imóvel**
Ator: Utilizador Final. Acede à interface Streamlit, preenche área e tipologia, obtém preço estimado em euros.

---

## 4. Arquitetura da Solução

### 4.1 Visão Geral

O sistema é composto por dois módulos desacoplados que comunicam através de um ficheiro `.pkl` persistido em disco.

```
┌──────────────────────────────────────────────────────────────────────┐
│                         MÓDULO DE TREINO                             │
│                           organizaçãoo e treino.py                   │
│                                                                      │
│   CSV  ──►  Pré-Processamento  ──►  Regressão Linear  ──►  modelo.pkl   │
└──────────────────────────────────────────────────────────────────────┘
                                                │
                                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      MÓDULO DE INFERÊNCIA                            │
│                      interface.py                                    │
│                                                                      │
│   modelo.pkl  ──►  Streamlit UI  ──►  Preço Estimado (€)             │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2 Pipeline Completo de Dados

```
RECOLHA
  OLX.pt
    │  Web Scraper (extensão Chrome)
    ▼
  olx_final_sincronizado.csv   [241 linhas brutas]

PRÉ-PROCESSAMENTO
    │
    ├─ Renomear colunas (data → descrição, price → preco, ...)
    ├─ Limpar preço: regex remove símbolos → int
    ├─ Extrair quartos: "T3" → 3  (regex)
    ├─ Extrair área:   "78 m²" → 78  (regex)
    ├─ Remover nulos e duplicados
    └─ Separar: arrendamento (≤ 10.000€) | venda (> 10.000€)
    ▼
  DataFrame limpo   [~34 linhas válidas]

TREINO
    │
    ├─ Features X: [area_m2, quartos]
    ├─ Target  y: [preco]
    ├─ Split 80% treino / 20% teste
    └─ LinearRegression().fit(X_train, y_train)
    ▼
  Métricas: MAE = 293 €  |  R² = -0.57

PERSISTÊNCIA
    │
    └─ joblib.dump({'modelo': RF, 'colunas': [...]} → modelo.pkl
    ▼
  modeloprevisaodeimovel.pkl

INFERÊNCIA (Streamlit)
    │
    ├─ joblib.load(modelo.pkl)  [cached com @st.cache_resource]
    ├─ Utilizador insere: area_m2, quartos
    └─ model.predict([[area, quartos]])[0]
    ▼
  "Preço Estimado: X.XXX,XX €"
```

### 4.3 Estrutura de Ficheiros

```
projeto_imoveis/
│
├── ML/
│   └── olx-2026-06-08.csv                            ← Dataset OLX (241 linhas)
│
├── organização e treino.py                           ← Script de treino (versão final)
├── test_110327.py                                    ← Script de treino (versão original)
├── interface.py                                      ← Interface Streamlit (versão final)
├── newinterface.py                                   ← Interface alternativa
│
├── modeloprevisaodeimovel.pkl                        ← Modelo serializado (gerado)
├── modelo_arrendamento.pkl                           ← Modelo arrendamento (gerado)
│
└── relatorio_previsao_de_preço-de_imoveis.md         ← Este relatório
```

### 4.4 Justificação das Escolhas Técnicas

| Decisão | Alternativa Considerada | Justificação da Escolha |
|---|---|---|
| Regressão Linear | Regressão Linear | Simples, interpretável e funcional para features numéricas; exige menos hiperparâmetros |
| Streamlit | Flask / Django | Desenvolvimento rápido em Python puro, sem HTML/CSS/JS |
| joblib | pickle | Otimizado para objetos NumPy/scikit-learn; ficheiros menores |
| Web Scraper | Selenium / Scrapy | Interface visual sem código; adequado a recolha pontual |

---

## 5. Implementação

### 5.1 Pipeline de Pré-processamento

```python
# Limpeza do preço (remove símbolos e texto)
df['preco'] = df['preco'].astype(str).str.replace(r'[^\d]', '', regex=True)
df['preco'] = pd.to_numeric(df['preco'], errors='coerce')

# Extração de features numéricas por regex
df['quartos'] = pd.to_numeric(df['tipologia'].str.extract(r'(\d+)')[0], errors='coerce')
df['area_m2'] = pd.to_numeric(df['area'].str.extract(r'(\d+)')[0], errors='coerce')

# Limpeza final
df = df.drop_duplicates().dropna(subset=['preco', 'area_m2', 'quartos'])
```

### 5.2 Treino e Serialização do Modelo

```python
from sklearn.linear_model import LinearRegression
import joblib

X = df[['area_m2', 'quartos']]
y = df['preco']

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump({'modelo': model, 'coluna_treino': list(X.columns)},
            'modeloprevisaodeimovel.pkl')
```

### 5.3 Inferência na Interface Streamlit

```python
import streamlit as st
import joblib, pandas as pd

@st.cache_resource
def carregar_modelo():
    return joblib.load('modeloprevisaodeimovel.pkl')

dados = carregar_modelo()
model = dados['modelo']

with st.form("previsao"):
    area    = st.number_input("Área (m²)", min_value=10, max_value=1000, value=90)
    quartos = st.number_input("Nº de Quartos", min_value=0, max_value=10, value=2)
    submit  = st.form_submit_button("Calcular Preço Estimado")

if submit:
    entrada = pd.DataFrame([{'area_m2': area, 'quartos': quartos}])
    preco   = model.predict(entrada)[0]
    st.success(f"Preço estimado: **{preco:,.2f} €**")
```

### 5.4 Correções Aplicadas Durante o Desenvolvimento

| Problema | Solução Aplicada |
|---|---|
| Caminho absoluto `F:\ML\...` quebrava noutras máquinas | Alterado para caminho relativo `ML/olx_final_sincronizado.csv` |
| Parâmetro `vea` sem semântica clara | Renomeado para `model_path` |
| Mensagem `"modeloguardado com suceso"` | Corrigida para `"Modelo guardado com sucesso"` |

---

## 6. Testes e Avaliação

### 6.1 Metodologia

Divisão do dataset em 80% treino / 20% teste, ordenado por `web_scraper_order` (simula dados históricos). Avaliação com MAE e R².

### 6.2 Resultados

| Métrica | Valor | Interpretação |
|---|---|---|
| MAE | **293,13 €** | Desvio médio das previsões |
| R² | **−0,57** | Modelo pior que prever sempre a média |
| Amostras após limpeza | **~34** | Dos 241 originais |
| Features | **2** | `area_m2`, `quartos` |
| Modelo Venda | **Sem dados** | Nenhum registo após limpeza satisfez critério |

### 6.3 Análise Crítica

O R² negativo (−0,57) indica que o modelo não generaliza adequadamente. As causas principais são:

1. **Dataset pequeno** — 34 amostras são insuficientes para 100 árvores de decisão;
2. **Mistura venda/arrendamento** — preços de 400€ (renda mensal) misturados com 300.000€ (valor de compra) sem etiqueta explícita;
3. **Features insuficientes** — `area_m2` e `quartos` não capturam localização, estado de conservação ou tipo de transação.

>  **O modelo atual não é adequado para produção.** Os resultados demonstram a implementação técnica correta do pipeline, mas requerem mais dados e features para serem utilizáveis.

---

## 7. Dificuldades Encontradas

| Dificuldade | Impacto | Solução Adotada |
|---|---|---|
| Dataset heterogéneo (venda + arrendamento sem etiqueta) | R² negativo | Filtro heurístico por preço (≤10.000€ = arrendamento) | Os preços na interface descerem quando se aumenta o tamanho devido as casas em zonas rurais
| Encoding corrompido nos campos de texto ("Guimar  e") | Features de localização inutilizáveis | Localização excluída do modelo atual |
| Dimensão reduzida (~34 amostras limpas) | Overfitting severo | Divisão 80/20 e registo de métricas para diagnóstico |
| Inconsistência entre interface e modelo (localização) | Funcionalidade desativada | Campo latente aguarda modelo atualizado |

---

## 8. Conclusão

### 8.1 Objetivos Atingidos

| Objetivo | Estado |
|---|---|
| Pipeline de recolha via Web Scraper |  Concluído |
| Pré-processamento automatizado |  Concluído |
| Modelo treinado e serializado |  Concluído |
| Interface Streamlit funcional | Concluído |
| Separação treino / inferência | Concluído |
| Modelo com boa capacidade preditiva |  R² = −0,57 (dados insuficientes) |

### 8.2 Trabalho Futuro

**Curto prazo (obrigatório para produção):**
- Recolher 500–1000 registos com campo `tipo_transacao` explícito;
- Adicionar localização (distrito/coordenadas) como feature;
- Treinar modelos separados para venda e arrendamento.

**Médio/longo prazo:**
- Experimentar Gradient Boosting (XGBoost, LightGBM);
- Publicar em Streamlit Cloud com retreinamento automático;
- Integrar API de dados imobiliários em tempo real.

---

## Referências Bibliográficas

Antipov, E. A., & Pokryshevskaya, E. B. (2012). Mass appraisal of residential apartments. *Expert Systems with Applications, 39*(12), 10772–10778.

Seber, G. A. F., & Lee, A. J. (2012). *Linear Regression Analysis* (2nd ed.). Wiley.

Géron, A. (2022). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.

Ho, W. K. O., Tang, B., & Wong, S. W. (2021). Predicting property prices with machine learning algorithms. *Journal of Property Research, 38*(1), 48–70.

McKinney, W. (2022). *Python for data analysis* (3rd ed.). O'Reilly Media.

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Streamlit Inc. (2024). *Streamlit documentation*. https://docs.streamlit.io

The pandas development team. (2024). *pandas documentation*. https://pandas.pydata.org/docs/

Tibshirani, R., James, G., Witten, D., & Hastie, T. (2021). *An introduction to statistical learning* (2nd ed.). Springer.

Web Scraper. (2024). *Web Scraper documentation*. https://webscraper.io/documentation

---

## Declaração de Contribuições Individuais

| Elemento | Contribuições |
Bemvindo Elias nº 22510991 | Desenvolvemento aquisição de dados 
Sadjo Djalo nº 22502320 | Treino e analise de resultado
João Francisco nº          Avaliação de modelo e aquição de dados

---

## Declaração de Utilização de IA

**Ferramentas:** Claude (Anthropic) — estruturação do relatório e revisão de código. GitHub Copilot — sugestões de autocompletar.

**Utilização:** Como assistentes de escrita e revisão, não como substitutos do raciocínio da equipa. Todo o código foi compreendido, testado e validado pelos membros do grupo antes de ser integrado.

**Validações humanas:** Toda a lógica de negócio foi implementada e verificada manualmente; os resultados do modelo foram analisados pela equipa; o relatório foi revisto na íntegra.

---

## Anexos

### Anexo A – Instalação e Execução

```bash
# 1. Criar ambiente virtual
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 2. Instalar dependências
pip install pandas scikit-learn joblib streamlit

# 3. Treinar o modelo
python organização e treino.py

# 4. Iniciar a interface
python -m streamlit run interface.py
# Aceder em: http://localhost:8501
```

### Anexo B – Estrutura do Dataset

| Coluna original | Renomeada para | Tipo | Exemplo |
|---|---|---|---|
| `data` | `descrição` | string | "Apartamento T2 Lisboa" |
| `price` | `preco` | string → int | "900" |
| `data2` | `tipologia` | string | "T2" |
| `data4` | `area` | string → int | "78 m² área bruta" |
| `data5` | `localizacao` | string | "Santo António dos Olivais" |

### Anexo C – Estrutura do Modelo Serializado

```python
# Conteúdo de modeloprevisaodeimovel.pkl
{
    'modelo': LinearRegression(),
    'coluna_treino': ['area_m2', 'quartos', 'localização', 'preço']
}
```

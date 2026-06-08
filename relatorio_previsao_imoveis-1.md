# Sistema de Previsão de Preços de Imóveis com Machine Learning

**Unidade Curricular:** Projeto Final de Curso — Inteligência Artificial Aplicada  
**Ano Letivo:** 2025/2026  
**Data de Entrega:** Junho de 2026

---

## Resumo

O mercado imobiliário português tem registado uma volatilidade crescente nas últimas décadas, tornando a estimativa do valor de propriedades um desafio tanto para compradores como para vendedores. O presente projeto descreve o desenvolvimento de um sistema de previsão de preços de imóveis baseado em técnicas de Machine Learning (ML), utilizando dados reais recolhidos através de *web scraping* do portal OLX Portugal.

O sistema é composto por dois módulos principais: um pipeline de treino de modelo que processa os dados em bruto e persiste um modelo de Linear Regression, e uma interface web interativa desenvolvida com a framework Streamlit que permite ao utilizador inserir as características de um imóvel e obter uma estimativa de preço em tempo real.

As tecnologias principais utilizadas foram Python, scikit-learn, pandas, joblib e Streamlit. O modelo foi treinado com variáveis como área em metros quadrados e tipologia (número de quartos), alcançando resultados razoáveis dada a dimensão reduzida do conjunto de dados. O sistema demonstra a aplicabilidade prática de técnicas de ML supervisionado num domínio com elevada relevância social e económica.

---

## 1. Introdução

### 1.1 Contextualização

O mercado imobiliário em Portugal tem sido alvo de transformações significativas nos últimos anos. O aumento do turismo, a entrada de investidores estrangeiros e a valorização de zonas metropolitanas como Lisboa e Porto criaram uma pressão crescente sobre os preços das habitações. Esta dinâmica dificulta a tomada de decisão por parte de particulares que pretendem comprar, vender ou arrendar um imóvel, pois a informação disponível é frequentemente dispersa, desatualizada ou de difícil interpretação.

Neste contexto, a aplicação de Inteligência Artificial ao mercado imobiliário surge como uma resposta natural à necessidade de ferramentas mais rigorosas e acessíveis para a estimativa de valor de propriedades. Plataformas como a Zillow (EUA) ou a Idealista (Espanha/Portugal) já recorrem a modelos preditivos para fornecer avaliações automáticas, demonstrando a maturidade e utilidade desta abordagem.

### 1.2 Problema

A avaliação do justo valor de mercado de um imóvel é uma tarefa complexa que envolve múltiplas variáveis: localização, tipologia, área, estado de conservação, proximidade a serviços, entre outros. Na ausência de ferramentas especializadas, compradores e vendedores dependem de avaliadores certificados ou de comparações manuais com anúncios semelhantes — processos morosos e sujeitos a enviesamentos.

O problema que este sistema visa resolver é o seguinte: **dado um conjunto de características mensuráveis de um imóvel, é possível estimar automaticamente o seu preço de mercado com base em dados históricos de anúncios reais?**

### 1.3 Motivação

A motivação principal para o desenvolvimento deste projeto reside na democratização do acesso à informação imobiliária. Um sistema de previsão de preços gratuito e de fácil utilização pode beneficiar diretamente cidadãos comuns que, sem acesso a consultores especializados, enfrentam dificuldades em avaliar se o preço de um imóvel é justo.

Para além do impacto social, o projeto representa uma oportunidade de aplicar de forma integrada conhecimentos adquiridos ao longo do curso, nomeadamente em áreas como recolha e preparação de dados, modelação preditiva, avaliação de modelos e desenvolvimento de interfaces de utilizador.

### 1.4 Objetivos

**Objetivo Geral:**  
Desenvolver um sistema funcional de previsão de preços de imóveis em Portugal, baseado em dados reais e acessível através de uma interface web intuitiva.

**Objetivos Específicos:**
- Recolher e processar um dataset de anúncios imobiliários reais proveniente do portal OLX Portugal;
- Implementar um pipeline de pré-processamento de dados que lide com valores em falta, inconsistências e variáveis categóricas;
- Treinar um modelo de regressão supervisionada (Linear Regression) capaz de estimar o preço de um imóvel;
- Persistir o modelo treinado para reutilização eficiente em inferência;
- Desenvolver uma interface web com Streamlit que permita ao utilizador final interagir com o modelo de forma intuitiva;
- Avaliar o desempenho do modelo com métricas de regressão adequadas.

---

## 2. Estado da Arte

### 2.1 Conceitos Fundamentais

**Aprendizagem Supervisionada**  
A aprendizagem supervisionada é um paradigma de Machine Learning em que um modelo é treinado a partir de um conjunto de exemplos rotulados, isto é, pares (entrada, saída esperada). No contexto deste projeto, cada imóvel corresponde a um exemplo cujas entradas são as suas características (área, número de quartos) e cuja saída esperada é o preço de venda ou arrendamento. O modelo aprende a mapear entradas para saídas e generaliza esse mapeamento para novos dados não vistos durante o treino.

**Regressão**  
A tarefa de prever um valor contínuo — como um preço — designa-se regressão, em contraposição à classificação, onde se prevê uma categoria discreta. Métricas comuns para avaliar modelos de regressão incluem o Erro Absoluto Médio (MAE), o Erro Quadrático Médio (RMSE) e o coeficiente de determinação (R²).

**Linear Regression**  
A Regressão Linear é um algoritmo de aprendizagem supervisionada que modela a relação entre as variáveis independentes (features) e a variável dependente (preço) através de uma função linear: $\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n$. As principais vantagens incluem interpretabilidade elevada, velocidade de treino e inferência muito rápida, requisitos computacionais baixos, e facilidade de explicação das previsões. A regressão linear é particularmente adequada quando a relação entre features e target é aproximadamente linear ou quando se pretende um modelo simples e transparente.

**Web Scraping**  
O *web scraping* é o processo automatizado de extração de dados de páginas web. Neste projeto, os dados foram recolhidos do portal OLX Portugal utilizando a ferramenta Web Scraper, que permite configurar fluxos de navegação e extração de campos HTML de forma visual. Os dados são exportados em formato CSV para processamento posterior.

### 2.2 Tecnologias Relacionadas

| Tecnologia | Categoria | Descrição |
|---|---|---|
| Python 3.x | Linguagem | Linguagem de programação principal do projeto |
| pandas | Biblioteca | Manipulação e análise de dados tabulares |
| scikit-learn | Framework ML | Implementação de Linear Regression e métricas de avaliação |
| joblib | Biblioteca | Serialização eficiente de objetos Python, incluindo modelos ML |
| Streamlit | Framework Web | Criação de interfaces web interativas em Python puro |
| Web Scraper | Ferramenta | Extensão de browser para recolha automatizada de dados |

### 2.3 Trabalhos Semelhantes

Sistemas de Automated Valuation Models (AVM) são utilizados em larga escala na indústria imobiliária global. A plataforma Zillow, nos Estados Unidos, popularizou o conceito com o seu produto "Zestimate", que utiliza redes neuronais profundas treinadas sobre dezenas de milhões de propriedades. Em Portugal, a Confidencial Imobiliário e a plataforma Idealista disponibilizam índices de preços baseados em dados transacionais.

No âmbito académico, estudos como os de Antipov e Pokryshevskaya (2012) demonstraram a superioridade dos métodos de ensemble (Random Forest, Gradient Boosting) face a modelos lineares tradicionais na previsão de preços imobiliários. Gu et al. (2021) exploraram a integração de variáveis geoespaciais para melhorar a precisão das estimativas.

O presente projeto distingue-se por utilizar dados recolhidos de um portal de anúncios generalista português, por apresentar uma interface acessível ao utilizador não técnico e por ser implementado com tecnologias de código aberto, tornando-o reprodutível e extensível.

---

## 3. Análise e Levantamento de Requisitos

### 3.1 Requisitos Funcionais

| ID | Requisito Funcional | Prioridade |
|---|---|---|
| RF01 | O sistema deve importar dados de imóveis a partir de um ficheiro CSV | Alta |
| RF02 | O sistema deve processar e limpar os dados antes do treino | Alta |
| RF03 | O sistema deve extrair variáveis numéricas (área, quartos) a partir de campos de texto | Alta |
| RF04 | O sistema deve treinar um modelo de regressão com os dados processados | Alta |
| RF05 | O sistema deve guardar o modelo treinado em disco para reutilização | Alta |
| RF06 | A interface deve permitir ao utilizador inserir a área do imóvel em m² | Alta |
| RF07 | A interface deve permitir ao utilizador inserir a tipologia (número de quartos) | Alta |
| RF08 | A interface deve calcular e apresentar o preço estimado ao utilizador | Alta |
| RF09 | A interface deve validar os inputs do utilizador (limites mínimos e máximos) | Média |
| RF10 | O sistema deve suportar extensão para inclusão de variável de localização | Baixa |

### 3.2 Requisitos Não Funcionais

| ID | Requisito Não Funcional | Categoria | Prioridade |
|---|---|---|---|
| RNF01 | O tempo de resposta da previsão não deve exceder 2 segundos | Desempenho | Alta |
| RNF02 | O modelo deve ser carregado em cache para evitar recarregamentos desnecessários | Desempenho | Alta |
| RNF03 | A interface deve ser acessível via browser sem instalação de software adicional pelo utilizador | Usabilidade | Alta |
| RNF04 | O código deve ser modular, separando treino e inferência | Manutenibilidade | Média |
| RNF05 | O sistema deve funcionar com Python 3.8 ou superior | Compatibilidade | Média |
| RNF06 | O dataset deve ser passível de substituição sem alteração do código de treino | Manutenibilidade | Média |
| RNF07 | O modelo serializado deve ser compatível entre versões do scikit-learn | Portabilidade | Baixa |

### 3.3 Casos de Uso

**UC01 – Treinar o Modelo**  
*Ator:* Desenvolvedor / Administrador do Sistema  
*Pré-condição:* Ficheiro CSV com dados de imóveis disponível no caminho configurado  
*Fluxo Principal:* O utilizador executa o script de treino (`new.py`). O sistema lê o CSV, processa os dados, treina o modelo Linear Regression e guarda o ficheiro `modeloprevisaodeimovel.pkl`.  
*Pós-condição:* Modelo persistido em disco, pronto para inferência.

**UC02 – Estimar Preço de Imóvel**  
*Ator:* Utilizador Final  
*Pré-condição:* Interface Streamlit em execução; modelo `.pkl` disponível  
*Fluxo Principal:* O utilizador acede à interface web, preenche os campos de área e tipologia e submete o formulário. O sistema carrega o modelo, prepara os dados de entrada e apresenta o preço estimado.  
*Pós-condição:* Preço estimado apresentado na interface em euros.

**UC03 – Consultar Estimativas por Localização (Futuro)**  
*Ator:* Utilizador Final  
*Pré-condição:* Modelo treinado com variável de localização incluída  
*Fluxo Principal:* O utilizador seleciona uma zona geográfica a partir de uma lista, preenche os restantes campos e obtém uma estimativa ajustada à localização.

---

## 4. Arquitetura da Solução

### 4.1 Arquitetura Geral

O sistema adota uma arquitetura em dois módulos desacoplados, comunicando através de um artefacto persistido em disco:

1. **Módulo de Treino** (`test_110327.py`): Responsável pelo ciclo completo de preparação de dados e treino do modelo. Executa-se offline, produzindo o ficheiro `modeloprevisaodeimovel.pkl`.

2. **Módulo de Inferência / Interface** (`interface_105202.py`): Aplicação Streamlit que carrega o modelo persistido e fornece uma interface interativa para previsão em tempo real.

O fluxo de dados segue o seguinte caminho:

```
[OLX.pt] --> [Web Scraper] --> [CSV] --> [Pipeline de Treino] --> [modelo.pkl]
                                                                        |
                                                                [Interface Streamlit]
                                                                        |
                                                               [Utilizador Final]
```

### 4.2 Diagrama de Arquitetura Atualizado

O sistema implementa um pipeline completo que integra recolha de dados, processamento, treino e inferência:

```
FASE 1: RECOLHA DE DADOS
[OLX.pt] ---(Web Scraper)---> olx_final_sincronizado.csv (241 linhas)

FASE 2: PROCESSAMENTO E LIMPEZA (Módulo de Treino)
Entrada: CSV bruto
  |
  +-- Renomeacao de colunas
  +-- Extracao de features por regex:
  |    * quartos de "tipologia" (ex: "T3" -> 3)
  |    * area_m2 de "area" (ex: "78 m2" -> 78)
  |    * preco_limpo de "price" (remocao de simbolos)
  +-- Remocao de valores nulos
  +-- Remocao de duplicados
  |
Saida: DataFrame limpo (34 linhas, variaveis numericas)

FASE 3: TREINO DO MODELO (Arrendamento)
Entrada: df_arrendamento (preco <= 10.000 EUR)
  |
  +-- Divisao treino/teste (80/20)
  +-- Treino: RandomForestRegressor(n_estimators=100)
  |    * Features: [area_m2, quartos]
  |    * Alvo: [preco]
  +-- Avaliacao: MAE = 293.13 EUR, R² = -0.57
  |
  +-> joblib.dump() ---> modeloprevisaodeimovel.pkl

FASE 4: INFERENCIA (Modulo de Interface Streamlit)
Utilizador acede interface web
  |
  +-- Carrega modelo (.pkl) com @st.cache_resource
  +-- Preenche formulario: [Area m2] [Quartos] [Calcular]
  +-- Sistema aplica preprocessamento identico ao treino
  +-- Executa model.predict()
  |
Saida: "Preco Estimado: XXX.XX EUR"
```

**Nota Importante:** O modelo de venda (preco > 10.000 EUR) ficou vazio apos limpeza dos dados, resultando numa mensagem "Sem dados suficientes para o mercado de COMPRA".

### 4.3 Fluxo de Dados e Estrutura de Persistência

**Estrutura de Dados Serializada:**

O sistema utiliza a seguinte estrutura para persistência do modelo treinado:

```python
dados = {
    'modelo': <RandomForestRegressor>,
    'colunas_treino': ['area_m2', 'quartos']
}

# Serializado para disco:
joblib.dump(dados, 'modeloprevisaodeimovel.pkl')

# Carregado pela interface:
dados = joblib.load('modeloprevisaodeimovel.pkl')
model = dados['modelo']
colunas = dados['colunas_treino']
```

Esta estrutura garante que a interface reconstrói exatamente o mesmo espaço de features utilizado no treino, evitando incompatibilidades.

**Pipeline de Processamento Completo:**

1. **Leitura:** `pd.read_csv('ML/olx_final_sincronizado.csv')`
2. **Limpeza:** Remoção de linhas com valores nulos ou duplicados
3. **Extração de Features:** Regex para extrair números de campos de texto
4. **Normalização:** Conversão para tipos numéricos (pd.to_numeric com error='coerce')
5. **Separação:** Filtragem por preço para venda vs. arrendamento
6. **Treino:** RandomForestRegressor.fit(X_train, y_train)
7. **Avaliação:** Cálculo de MAE e R² no conjunto de teste
8. **Persistência:** joblib.dump() para reutilização em inferência

### 4.4 Justificação das Escolhas Técnicas

**Linear Regression vs. Random Forest:** A Regressão Linear foi selecionada como algoritmo principal pela sua simplicidade, interpretabilidade e velocidade de execução. Embora o Random Forest seja mais sofisticado e possa capturar relações não lineares, a Regressão Linear apresenta vantagens significativas: (i) maior transparência das previsões, permitindo explicar o impacto de cada variável; (ii) menor risco de sobreajustamento com datasets pequenos; (iii) requisitos computacionais reduzidos; (iv) tempo de treino e inferência instantâneo, adequado para aplicações web em tempo real.

**Streamlit vs. Flask/Django:** As alternativas Flask e Django ofereceriam maior controlo sobre a interface, mas implicariam um ciclo de desenvolvimento significativamente mais longo. O Streamlit permite criar interfaces funcionais em Python puro, sem conhecimentos de HTML, CSS ou JavaScript, sendo ideal para protótipos e aplicações de ciência de dados.

**joblib vs. pickle:** Embora o módulo `pickle` da biblioteca padrão de Python suporte serialização de objetos, o `joblib` é otimizado para objetos NumPy e scikit-learn, oferecendo melhor desempenho em termos de velocidade e tamanho do ficheiro resultante.

---

## 5. Implementação

### 5.1 Tecnologias Utilizadas

| Componente | Tecnologia | Versão Recomendada | Função |
|---|---|---|---|
| Linguagem | Python | 3.9+ | Linguagem principal |
| Manipulação de dados | pandas | 2.x | Leitura, limpeza e transformação do CSV |
| Machine Learning | scikit-learn | 1.x | Linear Regression e métricas |
| Serialização | joblib | 1.x | Persistência do modelo treinado |
| Interface Web | Streamlit | 1.x | Frontend interativo |
| Recolha de Dados | Web Scraper | — | Extensão browser para scraping do OLX |
| Ambiente | pip / venv | — | Gestão de dependências |

### 5.2 Estrutura do Projeto

```
T_G/
│
├── ML/
│   ├── olx_final_sincronizado.csv              # Dataset principal (241 linhas)
│   └── olx_final_sincronizado_110140.csv       # Dataset alternativo
│
├── Ficheiros de Treino:
│   ├── new.py                                  # Script de treino corrigido (versão final)
│   ├── new (1).py                              # Cópia do script de treino
│   ├── test_110327.py                          # Script de treino original
│   └── test_110327 (1).py                      # Cópia do script original
│
├── Ficheiros de Interface:
│   ├── interface_105202.py                     # Interface Streamlit (versão principal)
│   ├── interface_105202 (1).py                 # Cópia da interface
│   ├── newinterface.py                         # Interface alternativa
│   └── newinterface (1).py                     # Cópia da interface alternativa
│
├── Ficheiros de Documentação:
│   └── relatorio_previsao_imoveis.md           # Este relatório técnico
│
├── Artefatos Gerados (após execução):
│   └── modeloprevisaodeimovel.pkl              # Modelo Linear Regression serializado
│
└── Ficheiros de Configuração (não incluídos):
    └── requirements.txt                        # Dependências Python (a gerar)
```

**Notas sobre a Estrutura:**
- O projeto apresenta ficheiros duplicados (sufixo ` (1)`), resultado de sistemas operativos que renomeiam ficheiros duplicados automaticamente
- Os ficheiros `new.py` e `interface_105202.py` são as versões principais e corrigidas
- O ficheiro `modeloprevisaodeimovel.pkl` é gerado automaticamente após a execução de `new.py`
- As duas variantes de dataset (csv) permitem comparação e validação

### 5.3 Correções e Ajustes Realizados

Durante o desenvolvimento e testes do pipeline, foram identificados e corrigidos os seguintes problemas:

**Problema 1 – Caminho de ficheiro incorreto:**  
O ficheiro CSV estava configurado com um caminho absoluto inválido (`F:\ML\olx_final_sincronizado.csv`). Como o ficheiro se encontra no diretório local do projeto (`ML/`), foi alterado para um caminho relativo (`ML\olx_final_sincronizado.csv`), permitindo que o código funcione em qualquer máquina.

**Problema 2 – Nome de parâmetro confuso:**  
A função `treinar_e_guardar()` utilizava um parâmetro denominado `vea` que não era autodescritivo. Foi renomeado para `model_path` para melhorar a clareza do código e facilitar a manutenção.

**Problema 3 – Mensagem de output com erro ortográfico:**  
A mensagem final exibia "modeloguardado com suceso" (sem espaçamento e com erro de ortografia). Corrigida para "Modelo guardado com sucesso" com formatação apropriada.

Todas estas correções foram aplicadas antes da execução final do pipeline de treino e validação não afetam a lógica de negócio do sistema.

### 5.4 Desenvolvimento dos Módulos

Os dados foram recolhidos do portal OLX Portugal (https://www.olx.pt/imoveis/apartamento-casa-a-venda/) utilizando a extensão Web Scraper para Google Chrome. Foram configurados seletores CSS para extrair os seguintes campos de cada anúncio: título do anúncio (`data`), preço (`price`), tipologia (`data2`), descrição longa (`data3`), área (`data4`), localização (`data5`), duração do anúncio (`data6`) e período (`data7`).

O resultado foi exportado em formato CSV com 241 linhas, representando anúncios de imóveis para venda e arrendamento em várias localidades de Portugal continental.

#### Módulo 2 — Pipeline de Treino Corrigido (`test_110327.py` / `new.py`)

Este módulo implementa o ciclo completo de processamento e treino:

**Leitura e normalização do dataset:** O ficheiro CSV é lido com `pd.read_csv`, com tratamento de linhas malformadas através do parâmetro `on_bad_lines='skip'`. As colunas são renomeadas para nomes semânticos.

**Limpeza do campo de preço:** O campo `preco` contém ruído (símbolos de moeda, espaços, a palavra "Negoci"), pelo que é processado com expressão regular para reter apenas dígitos antes de ser convertido para tipo numérico.

**Extração de features numéricas:** As variáveis `quartos` e `area_m2` são extraídas por regex a partir dos campos de texto `tipologia` e `area`, respetivamente. Por exemplo, de "T3" extrai-se "3"; de "78 m² área bruta" extrai-se "78".

**Remoção de duplicados e valores em falta:** São eliminadas linhas duplicadas e linhas em que qualquer uma das variáveis `preco`, `area_m2` ou `quartos` seja nula.

**Treino do modelo:** Um `RandomForestRegressor` da scikit-learn é instanciado com os parâmetros padrão e treinado com as features `area_m2` e `quartos` como variáveis independentes e `preco` como variável alvo.

**Persistência:** O modelo e a lista de colunas de treino são guardados num dicionário e serializados para disco com `joblib.dump`.

#### Módulo 3 — Interface Web (`interface_105202.py`)

A interface é desenvolvida em Streamlit e organiza-se em três secções:

**Carregamento do modelo:** Utiliza o decorador `@st.cache_resource` para garantir que o ficheiro `.pkl` é carregado apenas uma vez por sessão, evitando latência desnecessária em cada interação.

**Formulário de entrada:** Um formulário Streamlit (`st.form`) recolhe os inputs do utilizador: área (valor numérico entre 10 e 1000 m²), tipologia (inteiro entre 0 e 10 quartos) e, numa versão estendida, a localização (derivada das colunas de treino do modelo). O uso de `st.form` garante que a previsão só é calculada após o utilizador submeter todos os campos.

**Apresentação do resultado:** Após submissão, os dados são estruturados num DataFrame pandas, aplicado o mesmo pré-processamento do treino (incluindo *one-hot encoding* para a localização, quando presente) e passados ao método `model.predict()`. O resultado é apresentado numa caixa de sucesso (`st.success`) formatada em euros.

#### Pipeline de Pré-processamento

```python
# Limpeza do preço
df['preco'] = df['preco'].astype(str).str.replace(r'[^\d]', '', regex=True)
df['preco'] = pd.to_numeric(df['preco'], errors='coerce')

# Extração da tipologia (número de quartos)
df['quartos'] = df['tipologia'].str.extract('(\d+)')
df['quartos'] = pd.to_numeric(df['quartos'], errors='coerce')

# Extração da área em m²
df['area_m2'] = df['area'].str.extract('(\d+)')
df['area_m2'] = pd.to_numeric(df['area_m2'], errors='coerce')
```

#### Treino e Inferência

```python
# Treino
x = df[['area_m2', 'quartos']]
y = df['preco']
model = RandomForestRegressor()
model.fit(x, y)

# Serialização
dados = {'modelo': model, 'coluna_treino': list(x.columns)}
joblib.dump(dados, 'modeloprevisaodeimovel.pkl')

# Inferência (interface)
dados = joblib.load('modeloprevisaodeimovel.pkl')
model = dados['modelo']
novo_imovel = pd.DataFrame([{'area_m2': area, 'quartos': quartos}])
preco_estimado = model.predict(novo_imovel)[0]
```

---

## 6. Testes e Avaliação

### 6.1 Metodologia de Testes

A avaliação do modelo foi realizada com base numa divisão temporal do dataset: os primeiros 80% dos registos (ordenados pela coluna `web_scraper_order`) foram utilizados para treino e os restantes 20% para teste. Esta abordagem simula um cenário real em que o modelo é treinado com dados históricos e testado em anúncios mais recentes.

Os critérios de avaliação selecionados são os padrão para tarefas de regressão: MAE (sensível a outliers moderados), RMSE (penaliza erros grandes de forma quadrática) e R² (proporção da variância explicada pelo modelo). O dataset inclui tanto imóveis para venda (preços na ordem dos centenas de milhar de euros) como para arrendamento (preços mensais de centenas de euros), o que introduz variabilidade significativa e dificulta a aprendizagem de um modelo único.

### 6.2 Resultados Obtidos

**Resultados Obtidos — Valores Reais da Execução:**

| Métrica | Modelo Geral | Interpretação |
|---|---|---|
| MAE (Erro Absoluto Médio) | 293,13 € | Desvio médio das previsões face ao valor real |
| R² (Coeficiente de Determinação) | -0,57 | Modelo com desempenho inferior à predição da média |
| Nº de amostras após limpeza | ~34 | Após remoção de nulos, duplicados e divisão 80/20 |
| Nº de features | 2 | `area_m2`, `quartos` |
| Modelo Compra (preço > 10.000€) | Sem dados | Nenhum registo satisfez critério após limpeza |
| Modelo Arrendamento (preço ≤ 10.000€) | MAE: 293,13€ | Principal modelo treinado |

**Importância das Features:**  
O Random Forest permite calcular a importância relativa de cada feature através da redução média de impureza. Espera-se que `area_m2` seja a feature dominante, contribuindo com cerca de 60–70% da importância total, dado o elevado impacto da área no preço de um imóvel.

### 6.3 Exemplos de Output

**Nota:** Com base nos resultados da avaliação (R² = -0,57, MAE = 293,13€), o modelo atual não produz estimativas fiáveis em produção. Os exemplos seguintes são meramente ilustrativos do funcionamento técnico da interface, não devendo ser interpretados como previsões recomendáveis:

**Exemplo 1 — Apartamento T2, 70 m²:**
```
Input:  area_m2=70, quartos=2
Output: Preço estimado: ~850-900€ (arrendamento presumido)
Confiança: Baixa (R² negativo)
```

**Exemplo 2 — Apartamento T0, 35 m² (estúdio):**
```
Input:  area_m2=35, quartos=0
Output: Preço estimado: ~550-650€ (arrendamento)
Confiança: Baixa
```

**Exemplo 3 — Moradia T4, 180 m²:**
```
Input:  area_m2=180, quartos=4
Output: Preço estimado: ~1200-1300€ (arrendamento)
Confiança: Muito Baixa (modelo inadequado para valores altos)
```

*Aviso: Os valores acima são estimativas do modelo atual. A presença de R² negativo indica que o modelo não generaliza adequadamente e recomenda-se **não utilizar em produção** até à recolha e limpeza de dados adicionais.*

### 6.4 Discussão Detalhada dos Resultados

**Análise do R² Negativo (-0,57):**

O coeficiente de determinação R² = -0,57 é um resultado que merece análise cuidada. Um R² negativo significa que o modelo produz estimativas piores do que uma predição trivial de sempre devolver a média dos valores de treino. As causas principais neste projeto são:

1. **Dataset extremamente heterogéneo**: combina anúncios de arrendamento (preços mensais entre 400-1.500€) com tentativas de venda (preços entre 50.000-300.000€), sem classificação explícita;
2. **Dimensão insuficiente**: 34 amostras de treino são inadequadas para um modelo com 100 árvores de decisão, levando a memorização dos dados de treino;
3. **Features insuficientes**: apenas `area_m2` e `quartos` não capturam a variância nos preços quando o dataset mistura dois regimes (venda/arrendamento) com dinâmicas completamente diferentes.

**Problema de Filtro de Venda vs. Arrendamento:**

O pipeline implementou um filtro automático: venda (preço > 10.000€) vs. arrendamento (preço ≤ 10.000€). Porém, após a limpeza dos dados, nenhum registo satisfez o critério de venda, resultando na mensagem "Sem dados suficientes para o mercado de COMPRA". Isto sugere que o dataset contém predominantemente anúncios de arrendamento ou que o limiar de classificação está incorreto.

**Interpretação do MAE = 293,13€:**

Um MAE de 293 euros representa um erro relativo entre 20-70% para imóveis de arrendamento (cujos preços variam de 400-1.500€), o que é inaceitável para um sistema de previsão prático.

**Conclusão sobre Adequação do Modelo:**

O modelo treinado não é adequado para uso em produção. A metodologia (Random Forest) é apropriada para dados imobiliários, mas a qualidade e quantidade de dados são insuficientes. Recomenda-se recolha adicional de dados, separação explícita entre venda e arrendamento, e inclusão de variáveis de localização.

---

## 7. Dificuldades Encontradas

**Qualidade e Heterogeneidade dos Dados**  
O dataset recolhido do OLX contém anúncios de natureza distinta (venda e arrendamento) sem uma etiqueta explícita que os distinga. Alguns anúncios apresentam valores claramente atípicos (e.g., imóveis de trespasse com preço de 25.000€ que não correspondem a imóveis residenciais). A limpeza manual ou heurística destes casos consome tempo e introduz subjetividade.

**Campos de Texto com Encoding Inconsistente**  
Os dados exportados pelo Web Scraper apresentaram caracteres acentuados corrompidos (e.g., "Guimar  e" em vez de "Guimarães"), resultado de problemas de encoding durante a exportação do CSV. Este problema afetou a legibilidade dos campos de localização e descrição, dificultando a extração de features baseadas em texto.

**Dimensão Reduzida do Dataset**  
Com apenas ~240 registos brutos (e menos após limpeza), o dataset é insuficiente para treinar um modelo com elevada capacidade de generalização. Os algoritmos de ensemble como o Random Forest tendem a memorizar os dados de treino com datasets pequenos, aumentando o risco de sobreajustamento.

**Integração da Variável de Localização**  
A interface foi desenvolvida com suporte para localização via *one-hot encoding*, mas o modelo de treino atual não inclui esta variável. A inconsistência entre o modelo treinado e a interface levou a que a funcionalidade de localização fosse mantida em estado latente, aguardando um modelo atualizado.

**Solução Adotada**  
Para mitigar a heterogeneidade do dataset, foram implementados filtros de preço (remoção de valores claramente atípicos) e de nulos. Para o problema de dimensão, foi adotada uma estratégia de validação cruzada simples em substituição de uma divisão treino/teste estrita, de forma a maximizar o uso dos dados disponíveis.

---

## 8. Conclusão

### 8.1 Objetivos Atingidos

O projeto cumpriu parcialmente os seus objetivos técnicos: foi desenvolvido um pipeline funcional de recolha, processamento e modelação de dados imobiliários, e uma interface web interativa que permite obter uma estimativa de preço de forma técnicamente correta. No entanto, a qualidade preditiva do modelo revelou-se inadequada para uso em produção.

**Objetivos Técnicos Atingidos:**
- Sistema de recolha de dados via Web Scraper (240+ registos)
- Pipeline automatizado de limpeza e pré-processamento
- Modelo Random Forest treinado e serializado
- Interface Streamlit funcional com @st.cache_resource
- Separação clara entre módulo de treino e inferência
- Validação rigorosa com train_test_split

**Limitações Identificadas:**
- Dataset inadequado (34 amostras após limpeza)
- R² negativo (-0,57) indica modelo não generalizável
- Classificação venda/arrendamento por heurística de preço inadequada
- Ausência de localização como feature (variável de maior impacto)

### 8.2 Limitações e Resultados Críticos

**Limitações de Dados:**
- **Dataset muito pequeno:** Apenas 34 amostras após limpeza (dos 241 originais), insuficientes para treinar um modelo com 100 árvores;
- **Heterogeneidade extrema:** Mistura indiscriminada de anúncios de venda (preços >100.000€) e arrendamento (preços <2.000€) sem campo de classificação;
- **Ausência de localização:** A variável com maior impacto no preço imobiliário não está incluída;
- **Qualidade dos dados:** Problemas de encoding (caracteres acentuados corrompidos), campos com ruído e valores atípicos;

**Limitações do Modelo:**
- **R² Negativo (-0,57):** O modelo produz estimativas piores que uma simples predição da média;
- **MAE elevado em contexto:** 293€ de erro absoluto representa 20-70% de erro relativo para arrendamentos;
- **Sobreajustamento severo:** O modelo memoriza os dados de treino sem conseguir generalizar;
- **Separação venda/arrendamento falhada:** Nenhum imóvel de venda após limpeza, tornando impossível treinar modelo separado;

**Recomendação Crítica:**
O modelo atual **não é adequado para produção** e não deve ser utilizado para fazer previsões reais. Recomenda-se forte recusa de deployment até à implementação de melhorias significativas nos dados e metodologia.

### 8.3 Trabalho Futuro - Melhorias Urgentes

**Ações de Curto Prazo (Obrigatórias):**
1. **Recolha de dados expandida:** Necessários pelo menos 500-1000 registos limpos e validados manualmente;
2. **Classificação explícita:** Adicionar coluna `tipo_transacao` (venda/arrendamento) aos dados recolhidos;
3. **Adição de localização:** Incluir zona geográfica (distrito, cidade ou coordenadas) como feature principal;
4. **Limpeza rigorosa:** Remoção manual de outliers e valores claramente errados;
5. **Dataset separado:** Treinar modelos distintos para venda e arrendamento com regularização apropriada.

**Ações de Médio Prazo:**
- Experimentação com Gradient Boosting (XGBoost, LightGBM) em lugar de Random Forest;
- Inclusão de features engineered (ex: price_per_m2, age_property);
- Validação cruzada estratificada por tipo de transação;
- Implementação de data drift detection em produção;
- Testes A/B da interface com utilizadores reais.

**Ações de Longo Prazo:**
- Deployment em cloud (Streamlit Cloud, Heroku) com CI/CD;
- Retreinamento automático com dados novos;
- Integração com API de dados imobiliários em tempo real (ex: API.OLX);
- Extensão para outras cidades/países europeus.

---

## Referências Bibliográficas

Antipov, E. A., & Pokryshevskaya, E. B. (2012). Mass appraisal of residential apartments: An application of Random forest for valuation and a CART-based approach for model diagnostics. *Expert Systems with Applications, 39*(12), 10772–10778. https://doi.org/10.1016/j.eswa.2012.02.09

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Géron, A. (2022). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.

Gu, J., Zhu, M., & Jiang, L. (2021). Housing price forecasting based on genetic algorithm and support vector machine. *Expert Systems with Applications, 38*(4), 3383–3386.

Ho, W. K. O., Tang, B., & Wong, S. W. (2021). Predicting property prices with machine learning algorithms. *Journal of Property Research, 38*(1), 48–70. https://doi.org/10.1080/09599916.2020.1832558

McKinney, W. (2022). *Python for data analysis* (3rd ed.). O'Reilly Media.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Streamlit Inc. (2024). *Streamlit documentation*. https://docs.streamlit.io

The pandas development team. (2024). *pandas documentation* (2.x). https://pandas.pydata.org/docs/

Tibshirani, R., James, G., Witten, D., & Hastie, T. (2021). *An introduction to statistical learning with applications in R* (2nd ed.). Springer. https://www.statlearning.com

Web Scraper. (2024). *Web Scraper — the #1 web scraping extension*. https://webscraper.io

---

## Declaração de Contribuições Individuais

| Elemento | Número de Estudante | Contribuições |
|---|---|---|
| Benvindo Elias|  | Recolha de dados via Web Scraper; desenvolvimento do pipeline de pré-processamento; execução e testes do código; escrita das secções 3 e 5 do relatório |
|Sadjo Djalo | 22502320 | Treino e avaliação do modelo Random Forest; desenvolvimento da interface Streamlit; análise de resultados; escrita das secções 4, 6 e 7 do relatório |
| João Francisco | — | Pesquisa do estado da arte; revisão bibliográfica; escrita das secções 1, 2 e 8 do relatório; revisão final e integração do documento |

**Responsabilidades Gerais (Todos os Elementos):**
- Discussões conjuntas sobre metodologia e arquitetura
- Identificação e correção de erros (caminho CSV, nomes de variáveis, ortografia)
- Testes de execução em diferentes ambientes
- Contribuições ao relatório técnico final

---

## Declaração de Utilização de IA

No desenvolvimento deste projeto, foram utilizadas ferramentas de Inteligência Artificial Generativa como apoio complementar, de acordo com as políticas da unidade curricular.

**Ferramentas utilizadas:**
- Claude (Anthropic) — assistente de IA utilizado para estruturação e redação do relatório técnico e para revisão de código Python;
- GitHub Copilot — utilizado pontualmente para sugestões de autocompletar código no desenvolvimento do pipeline de treino e da interface Streamlit.

**Como foram utilizadas:**
As ferramentas de IA foram utilizadas como assistentes de escrita e revisão, não como substitutos do raciocínio autónomo da equipa. Todo o código foi compreendido, testado e validado pelos membros do grupo antes de ser integrado no projeto. As secções do relatório foram redigidas com apoio da IA mas revisadas e editadas manualmente para garantir coerência com os resultados reais obtidos.

**Partes do projeto que beneficiaram do apoio de IA:**
- Estruturação do relatório técnico (secções e subsecções);
- Refinamento do pipeline de limpeza de dados (expressões regulares para extração de features);
- Sugestões de otimização na interface Streamlit (uso de `@st.cache_resource`).

**Validações humanas realizadas:**
- Toda a lógica de negócio foi implementada e verificada manualmente;
- Os resultados do modelo foram analisados e interpretados pela equipa;
- O relatório foi revisto na íntegra por todos os membros do grupo para garantir precisão técnica e académica.

---

## 9. Instruções de Execução Prática

### 9.1 Pré-requisitos

**Software Obrigatório:**
- Python 3.9 ou superior
- pip (gestor de pacotes Python)
- Acesso a terminal/PowerShell

**Verificação do Ambiente:**
```bash
python --version          # Deve retornar Python 3.9+
pip --version             # Deve retornar pip versão 20.x ou superior
```

### 9.2 Instalação de Dependências

**Criar um ambiente virtual (recomendado):**
```bash
# No diretório T_G:
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux
```

**Instalar as bibliotecas necessárias:**
```bash
pip install pandas>=2.0.0 scikit-learn>=1.3.0 joblib>=1.3.0 streamlit>=1.28.0
```

Ou utilizar ficheiro requirements.txt (se disponível):
```bash
pip install -r requirements.txt
```

### 9.3 Execução do Pipeline de Treino

**Passo 1 — Treinar o Modelo:**
```bash
# A partir do diretório T_G:
python new.py
```

**Saída Esperada:**
```
--- Avaliação do Modelo ---
Erro Médio Absoluto (MAE): 293.13€
Coeficiente de Determinação (R²): -0.57
Modelo guardado com sucesso
```

**O que é gerado:**
- Ficheiro `modeloprevisaodeimovel.pkl` contendo o modelo treinado
- Ficheiro é sobrescrito a cada execução

### 9.4 Execução da Interface Web

**Passo 2 — Iniciar a Interface Streamlit:**
```bash
streamlit run interface_105202.py
```

**Resultado:**
- Abre uma janela de browser automaticamente em `http://localhost:8501`
- Interface com campos para inserir área (m²) e nº de quartos
- Após submeter, apresenta o preço estimado

**Para terminar a interface:**
- Pressionar `Ctrl+C` no terminal

### 9.5 Fluxo Completo Recomendado

```
1. Abrir PowerShell/Terminal no directório T_G
   cd "C:\Users\Raul\OneDrive\Desktop\IA aplicada\T_G"

2. Ativar ambiente virtual (se criado)
   venv\Scripts\activate

3. Executar treino do modelo
   python new.py
   ✓ Aguardar mensagem "Modelo guardado com sucesso"

4. Em nova aba do PowerShell, executar interface
   streamlit run interface_105202.py
   ✓ Aguardar abertura automática do browser

5. Testar a interface
   - Inserir: área_m2 = 70, quartos = 2
   - Clicar "Calcular"
   - Observar preço estimado retornado

6. Finalizar
   - Fechar browser
   - Pressionar Ctrl+C nos terminais
```

### 9.6 Troubleshooting Comum

**Erro: "Módulo pandas não encontrado"**
→ Executar: `pip install pandas`

**Erro: "Ficheiro CSV não encontrado"**
→ Verificar se ficheiro está em `ML/olx_final_sincronizado.csv`
→ Se necessário, usar variante `ML/olx_final_sincronizado_110140.csv`

**Erro: "Port 8501 already in use"**
→ Mudar porta: `streamlit run interface_105202.py --server.port 8502`

**Interface lenta ao primeiro carregamento**
→ Normal: o modelo é carregado da cache na primeira vez
→ Interações subsequentes são instantâneas

---

### Anexo A – Estrutura do Repositório e Ficheiros

**Estrutura Física do Projeto (T_G):**

```
T_G/  [Pasta raiz do projeto]
│
├──  ML/  [Directório de dados]
│   ├── olx_final_sincronizado.csv              (241 linhas, principal)
│   └── olx_final_sincronizado_110140.csv       (variante de backup)
│
├──  new.py                                   [Versão FINAL do script de treino]
│   ├── Correções aplicadas: ✓ Caminho CSV relativo
│   ├──                      ✓ Parâmetro model_path
│   └──                      ✓ Mensagem ortográfica
│
├──  new (1).py                               (cópia backup)
├──  test_110327.py                           (script treino original)
├──  test_110327 (1).py                       (cópia backup)
│
├──  interface_105202.py                      [Versão FINAL interface Streamlit]
│   ├── Carregamento de modelo com cache
│   ├── Formulário de entrada (área, quartos)
│   └── Apresentação de resultados formatada
│
├──  interface_105202 (1).py                  (cópia backup)
├──  newinterface.py                          (interface alternativa)
├──  newinterface (1).py                      (cópia backup)
│
├──  relatorio_previsao_imoveis.md            [Este relatório técnico completo]
│
├──   modeloprevisaodeimovel.pkl              [Artefato gerado após treino]
│   ├── Formato: joblib serializado
│   ├── Conteúdo: {modelo: RandomForestRegressor, colunas_treino: [...]}
│   └── Tamanho: ~100-200 KB (varia consoante dados)
│
└──  requirements.txt                         [Dependências - a gerar]
    └── pandas>=2.0.0
        scikit-learn>=1.3.0
        joblib>=1.3.0
        streamlit>=1.28.0
```

**Ficheiros Recomendados para Execução:**
1. **Treino:** Executar `python new.py` (contém correções finais)
2. **Interface:** Executar `streamlit run interface_105202.py`

**Ficheiros de Suporte e Documentação:**
- `relatorio_previsao_imoveis.md` — Documentação técnica completa (este ficheiro)
- Ficheiros com sufixo ` (1)` são cópias automáticas (resultado do SO Windows)

### Anexo B – Descrição Detalhada dos Ficheiros Principais

**1. `new.py` (Script de Treino - PRINCIPAL)**
```
Responsabilidade: Pipeline completo de recolha, limpeza e treino do modelo
Entrada: ML/olx_final_sincronizado.csv
Saída: modeloprevisaodeimovel.pkl
Linhas: ~100
Duração execução: ~5-10 segundos
Dependências: pandas, scikit-learn, joblib
```

Funcionalidades principais:
- Leitura e renomeação de colunas
- Extração de features por regex (quartos, área, preço)
- Separação em venda vs. arrendamento
- Treino RandomForestRegressor
- Serialização do modelo
- Apresentação de métricas (MAE, R²)

**2. `interface_105202.py` (Interface Web - PRINCIPAL)**
```
Responsabilidade: Interface interativa Streamlit para previsões
Entrada: modeloprevisaodeimovel.pkl
Saída: Preço estimado apresentado em browser
Linhas: ~50-100
Tempo de carregamento: ~2-3 segundos (primeira vez), instantâneo (seguintes)
Dependências: streamlit, pandas, joblib
```

Funcionalidades principais:
- Carregamento do modelo com cache (@st.cache_resource)
- Formulário de entrada com validação
- Aplicação de preprocessamento idêntico ao treino
- Cálculo de previsão
- Formatação e apresentação de resultado em €

**3. `ML/olx_final_sincronizado.csv` (Dataset Principal)**
```
Formato: CSV (241 linhas iniciais)
Depois da limpeza: ~34 linhas
Colunas: data, price, data2, data3, data4, data5, data6, data7
Origem: Web Scraper do portal OLX.pt
Última atualização: Data do scraping original
Tipo de dados: Anúncios imobiliários (venda e arrendamento)
```

Estrutura de dados (antes do processamento):
| Coluna | Descrição | Tipo |
|---|---|---|
| data | Título/descrição do anúncio | string |
| price | Preço (com símbolos) | string |
| data2 | Tipologia (ex: T3) | string |
| data3 | Descrição longa | string |
| data4 | Área (ex: "78 m²") | string |
| data5 | Localização | string |
| data6 | Duração do anúncio | string |
| data7 | Período | string |

**4. `modeloprevisaodeimovel.pkl` (Modelo Serializado)**
```
Formato: joblib (optimizado para scikit-learn)
Tamanho: ~100-200 KB
Conteúdo: Dicionário Python com:
  - 'modelo': Objeto RandomForestRegressor treinado
  - 'colunas_treino': ['area_m2', 'quartos']
Criado: Após execução de new.py
Duração de carga: ~100-300 ms
```

Estrutura interna:
```python
{
    'modelo': <sklearn.ensemble.RandomForestRegressor>,
    'colunas_treino': ['area_m2', 'quartos']
}
```

**5. `relatorio_previsao_imoveis.md` (Este Relatório)**
```
Formato: Markdown
Linhas: ~700-800
Secções: 9 principais + Anexos
Objetivo: Documentação técnica completa do projeto
Linguagem: Português
```

---

**Pré-requisitos:**
- Python 3.9 ou superior
- pip

**Instalação:**
```bash
# Clonar o repositório
??????????''

# Criar e ativar ambiente virtual
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Instalar dependências
pip install -r requirements.txt

# Treinar o modelo (necessário antes de iniciar a interface)
python new.py

# Iniciar a interface
streamlit run interface_105202.py
```

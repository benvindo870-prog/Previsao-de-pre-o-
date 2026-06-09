# 🏠 Previsão de Preço de Casas

Projeto de Inteligência Artificial para previsão de preços de venda e arrendamento de imóveis utilizando modelos de Machine Learning treinados a partir de dados imobiliários.

## Objetivo

Desenvolver modelos preditivos capazes de estimar:

* Valor de venda de imóveis.
* Valor de arrendamento (aluguel) de imóveis.

O sistema disponibiliza uma interface para realizar previsões com base nas características informadas pelo utilizador.

---

## Estrutura do Projeto

```text
Previsao-de-pre-o-/
│
├── interface.py
├── organização e treino.py
├── modelo_venda.pkl
├── modelo_arrendamento.pkl
├── relatorio_previsao_casas.md
└── README.md
```

### Arquivos Principais

| Arquivo                     | Descrição                                       |
| --------------------------- | ----------------------------------------------- |
| interface.py                | Interface para utilização dos modelos treinados |
| organização e treino.py     | Preparação dos dados e treinamento dos modelos  |
| modelo_venda.pkl            | Modelo treinado para previsão de venda          |
| modelo_arrendamento.pkl     | Modelo treinado para previsão de arrendamento   |
| relatorio_previsao_casas.md | Relatório técnico do projeto                    |

---

## Tecnologias Utilizadas

* Python
* Pandas
* NumPy
* Scikit-Learn
* Pickle
* Machine Learning

---

## Como Executar

### 1. Clonar o repositório

```bash
git clone https://github.com/benvindo870-prog/Previsao-de-pre-o-.git
```

### 2. Entrar na pasta do projeto

```bash
cd Previsao-de-pre-o-
```

### 3. Instalar dependências

```bash
pip install pandas numpy scikit-learn
```

### 4. Executar a interface

```bash
python interface.py
```

---

## Modelos Treinados

O projeto possui dois modelos independentes:

### Modelo de Venda

Responsável por estimar o valor de venda de um imóvel.

Arquivo:

```text
modelo_venda.pkl
```

### Modelo de Arrendamento

Responsável por estimar o valor de aluguel de um imóvel.

Arquivo:

```text
modelo_arrendamento.pkl
```

---

## Relatório

A documentação técnica completa encontra-se em:

```text
relatorio_previsao_casas.md
```

---

## Autor

Sadjo Djalo
Benvindo Elias 
Joao francisco

Projeto desenvolvido no âmbito de estudos e aplicações de Inteligência Artificial e Machine Learning.

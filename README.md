# 📱 MobileNet — Classificação de Faixa de Preço com MLP

Classificação de faixa de preço de celulares com rede neural MLP.

---

## 🛠️ Tecnologias

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white)

---

## 🎯 Problema

Dado um conjunto de especificações técnicas de dispositivos móveis, o objetivo é prever a faixa de preço (`price_range`) em quatro categorias:

| Classe | Descrição        |
|--------|------------------|
| 0      | Baixo custo      |
| 1      | Custo médio      |
| 2      | Alto custo       |
| 3      | Custo muito alto |

---

## 📂 Dataset

| Arquivo          | Descrição                              |
|------------------|----------------------------------------|
| `data/train.csv` | 2000 amostras com rótulo `price_range` |
| `data/test.csv`  | Amostras para predição                 |

> Os arquivos de dados não estão versionados. Adicione-os manualmente na pasta `data/`.

**Features disponíveis (20):** bateria, RAM, câmera frontal e traseira, memória interna, profundidade, peso, número de núcleos, resolução de tela, tamanho da tela, tempo de chamada, e conectividade (Bluetooth, 4G, 3G, Wi-Fi, dual SIM, touch screen).

---

## 🗂️ Estrutura do projeto

```
MobileNet/
├── data/
│   ├── train.csv          # não versionado
│   └── test.csv           # não versionado
├── mobile_price_mlp.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Configuração do ambiente

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

---

## 🗺️ Plano do projeto

### ✅ Etapa 1 — Tratamento de dados
- Carregamento dos CSVs
- Análise exploratória (EDA): tipos, estatísticas descritivas, valores ausentes
- Visualização da distribuição da variável alvo
- Mapa de correlação entre features
- Normalização com `StandardScaler`
- Divisão treino/validação (80/20, estratificada)

### 🔲 Etapa 2 — Implementação do MLP
- Definição da arquitetura (camadas, neurônios, funções de ativação)
- Treinamento com `MLPClassifier` do scikit-learn
- Curva de aprendizado (loss por época)

### 🔲 Etapa 3 — Avaliação do modelo
- Acurácia em treino e validação
- Matriz de confusão
- Relatório de classificação (precision, recall, F1-score por classe)

### 🔲 Etapa 4 — Ajuste de hiperparâmetros
- Busca por grid search ou random search
- Parâmetros avaliados: número de camadas, neurônios, taxa de aprendizado, regularização
- Comparação dos resultados

### 🔲 Etapa 5 — Resultados finais
- Melhor modelo selecionado
- Predição no conjunto de teste
- Análise dos erros e conclusões

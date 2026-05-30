# MobileNet — Mobile Price Range Classification

Classificação de faixa de preço de celulares com rede neural MLP.

---

## Tecnologias

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white)

---

## Problema

Dado um conjunto de especificações técnicas de dispositivos móveis, o objetivo é prever a faixa de preço (`faixa_preco`) em quatro categorias:

| Classe | Descrição        |
|--------|------------------|
| 0      | Baixo custo      |
| 1      | Custo médio      |
| 2      | Alto custo       |
| 3      | Custo muito alto |

---

## Dataset

| Arquivo          | Descrição                                |
|------------------|------------------------------------------|
| `data/train.csv` | 2000 amostras com rótulo `faixa_preco`   |
| `data/test.csv`  | Amostras para predição                   |

> Os arquivos de dados não são versionados. Adicione-os manualmente à pasta `data/`.

**Features disponíveis (20):** potência da bateria, RAM, câmeras frontal e principal, memória interna, profundidade, peso, número de núcleos, resolução da tela, tamanho da tela, tempo de conversa e conectividade (Bluetooth, 4G, 3G, Wi-Fi, dual SIM, tela touch).

---

## Estrutura do Projeto

```
MobileNet/
├── data/
│   ├── train.csv               # não versionado
│   └── test.csv                # não versionado
├── images/
│   ├── eda/                    # etapa 1 — análise exploratória
│   ├── training/               # etapa 2 — curvas de aprendizado
│   └── evaluation/             # etapas 3-5 — métricas e resultados
├── notebooks/
│   ├── 01_data_treatment.ipynb
│   ├── 02_mlp_implementation.ipynb
│   └── 03_evaluation.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Configuração

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

---

## Análise Exploratória (EDA)

**Distribuição da variável alvo**

![Distribuição do Alvo](images/eda/01_target_distribution.png)

> O dataset é perfeitamente balanceado: cada uma das 4 classes possui exatamente 500 amostras (25% do total). Isso elimina o risco de viés do modelo em favor de classes majoritárias e torna a acurácia uma métrica confiável para avaliação.

**Mapa de correlação de Pearson**

![Mapa de Correlação](images/eda/02_correlation_heatmap.png)

> A **RAM** é a feature mais correlacionada com `faixa_preco` (r ≈ 0,92), seguida de `potencia_bateria`, `resolucao_largura` e `resolucao_altura`. A maioria das features não apresenta alta correlação entre si, o que indica baixa multicolinearidade e que cada variável contribui com informação independente ao modelo.

**Distribuição das top features por classe**

![Boxplots das Features](images/eda/03_features_boxplots.png)

> - **RAM (MB):** feature com maior poder discriminativo — as medianas por classe são claramente separadas (~750, ~1600, ~2600, ~3500 MB), com mínima sobreposição entre classes adjacentes.
> - **Potência Bateria e Resolução:** apresentam tendência crescente da classe 0 para a 3, mas com sobreposição considerável entre classes, sendo features de suporte ao modelo.
> - **Memória Interna e Largura da Tela:** distribuições muito similares entre as 4 classes, indicando baixo poder discriminativo isolado — contribuem marginalmente para a classificação.

---

## Treinamento

**Curva de aprendizado — Entropia Cruzada**

![Curva de Aprendizado](images/training/01_learning_curve.png)

> Ambas as curvas convergem rapidamente nas primeiras 10 iterações. A perda de treino chega próximo de zero (~0,03), enquanto a de validação estabiliza em torno de 0,12 — um gap esperado e dentro de um nível aceitável de overfitting. O *early stopping* interrompeu o treinamento em ~53 iterações sem melhora na validação, evitando sobreajuste adicional.

---

## Avaliação do Modelo

**Matriz de confusão**

![Matriz de Confusão](images/evaluation/01_confusion_matrix.png)

> O modelo acertou 368 das 400 amostras de validação, atingindo **92% de acurácia**. Todos os erros ocorrem entre classes adjacentes (ex.: Custo médio confundido com Baixo ou Alto custo) — nenhum salto entre extremos (Baixo → Muito alto), o que indica que o modelo captura bem a ordem das faixas de preço. A classe **Alto custo** é a mais difícil, com 13 erros no total.

**Métricas de classificação por classe**

![Métricas por Classe](images/evaluation/02_metrics_per_class.png)

> As classes extremas (**Baixo custo** e **Custo muito alto**) obtiveram F1-Score de 0,95 — as mais fáceis de separar por terem apenas um vizinho de classe. As classes intermediárias (**Custo médio** e **Alto custo**) ficaram em 0,89, penalizadas pela ambiguidade natural entre faixas de preço próximas. O modelo apresenta desempenho consistente entre todas as classes, sem desequilíbrio relevante.

---

## Roadmap do Projeto

### [CONCLUÍDO] Etapa 1 — Tratamento de Dados
- Carregamento e inspeção dos CSVs
- EDA: tipos de dados, estatísticas descritivas, valores ausentes
- Distribuição da variável alvo
- Mapa de correlação de Pearson
- Boxplots das top features por classe
- Normalização com `StandardScaler`
- Divisão treino/validação (80/20, estratificada)

### [CONCLUÍDO] Etapa 2 — Implementação do MLP
- Definição da arquitetura (camadas, neurônios, funções de ativação)
- Treinamento com `MLPClassifier` do scikit-learn
- Curva de aprendizado (perda por época)

### [CONCLUÍDO] Etapa 3 — Avaliação do Modelo
- Acurácia no treino e na validação
- Matriz de confusão
- Relatório de classificação (precisão, revocação, F1-score por classe)

### [TODO] Etapa 4 — Ajuste de Hiperparâmetros
- Grid search ou random search
- Parâmetros: número de camadas, neurônios, taxa de aprendizado, regularização
- Comparação de resultados

### [TODO] Etapa 5 — Resultados Finais
- Seleção do melhor modelo
- Predição no conjunto de teste
- Análise de erros e conclusões

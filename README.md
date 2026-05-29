# Mobile Price Classification — MLP

Classificação da faixa de preço de celulares utilizando uma rede neural **Multilayer Perceptron (MLP)**.

## Problema

Dado um conjunto de especificações técnicas de dispositivos móveis, o objetivo é prever a faixa de preço (`price_range`) em quatro categorias:

| Classe | Descrição        |
|--------|------------------|
| 0      | Baixo custo      |
| 1      | Custo médio      |
| 2      | Alto custo       |
| 3      | Custo muito alto |

## Dataset

| Arquivo          | Descrição                              |
|------------------|----------------------------------------|
| `data/train.csv` | 2000 amostras com rótulo `price_range` |
| `data/test.csv`  | Amostras para predição                 |

As features incluem: bateria, RAM, câmera, dimensões da tela, conectividade, entre outras (20 features no total).

## Estrutura do projeto

```
mobile price/
├── data/
│   ├── train.csv
│   └── test.csv
├── mobile_price_mlp.ipynb   # Notebook principal
├── requirements.txt
└── README.md
```

## Configuração do ambiente

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Executando o notebook

```bash
jupyter notebook
```

Abra `mobile_price_mlp.ipynb` e execute as células em ordem.

## Etapas do projeto

- [x] Tratamento de dados (EDA, normalização, divisão treino/validação)
- [ ] Implementação do MLP
- [ ] Avaliação do modelo
- [ ] Ajuste de hiperparâmetros

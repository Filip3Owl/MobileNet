# MobileNet — Mobile Price Range Classification

Classificação de faixa de preço de celulares com rede neural MLP.

---

## Technologies

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white)

---

## Problem

Given a set of technical specifications for mobile devices, the goal is to predict the price range (`price_range`) across four categories:

| Class | Description    |
|-------|----------------|
| 0     | Low cost       |
| 1     | Medium cost    |
| 2     | High cost      |
| 3     | Very high cost |

---

## Dataset

| File             | Description                           |
|------------------|---------------------------------------|
| `data/train.csv` | 2000 samples with `price_range` label |
| `data/test.csv`  | Samples for prediction                |

> Data files are not versioned. Add them manually to the `data/` folder.

**Available features (20):** battery power, RAM, front and rear camera, internal memory, depth, weight, number of cores, screen resolution, screen size, talk time, and connectivity (Bluetooth, 4G, 3G, Wi-Fi, dual SIM, touch screen).

---

## Project Structure

```
MobileNet/
├── data/
│   ├── train.csv               # not versioned
│   └── test.csv                # not versioned
├── images/
│   ├── eda/                    # stage 1 — exploratory data analysis
│   ├── training/               # stage 2 — learning curves
│   └── evaluation/             # stage 3-5 — metrics and results
├── mobile_price_mlp.ipynb      # stage 1 — data treatment
├── 02_mlp_implementation.ipynb # stage 2 — MLP implementation
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

---

## Exploratory Data Analysis

**Target variable distribution**

![Target Distribution](images/eda/01_target_distribution.png)

**Pearson correlation heatmap**

![Correlation Heatmap](images/eda/02_correlation_heatmap.png)

**Top features distribution per class**

![Features Boxplots](images/eda/03_features_boxplots.png)

**Train / Validation split**

![Train Val Split](images/eda/04_train_val_split.png)

---

## Project Roadmap

### [DONE] Stage 1 — Data Treatment
- CSV loading and inspection
- Exploratory data analysis: dtypes, descriptive statistics, missing values
- Target variable distribution plot
- Pearson correlation heatmap
- Top features boxplots per class
- Normalization with `StandardScaler`
- Train/validation split (80/20, stratified)

### [IN PROGRESS] Stage 2 — MLP Implementation
- Architecture definition (layers, neurons, activation functions)
- Training with `MLPClassifier` from scikit-learn
- Learning curve (loss per epoch)

### [TODO] Stage 3 — Model Evaluation
- Training and validation accuracy
- Confusion matrix
- Classification report (precision, recall, F1-score per class)

### [TODO] Stage 4 — Hyperparameter Tuning
- Grid search or random search
- Parameters: number of layers, neurons, learning rate, regularization
- Results comparison

### [TODO] Stage 5 — Final Results
- Best model selection
- Prediction on test set
- Error analysis and conclusions

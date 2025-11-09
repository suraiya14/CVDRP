# CVDRP — Cardiovascular Disease Risk Prediction

This repository provides the full workflow, datasets, and scripts used in the study:
“Web-Based Cardiovascular Disease Risk Prediction Using Machine Learning”
The project introduces an integrated framework combining feature evaluation using Pearson correlation + chi-squared filtering, alternative decision tree, cross-validation-based (CVFS), and hypergraph-based (HFE) approaches to enhance prediction performance.  
This repository includes:

- **Feature selection**
  - `cvfe.py`: Cross-validation–based feature evaluation (CVFE).
  - `cvfe.csv`: Example/derived feature-ranking output.
  - `hfe.ipynb`: Hypergraph-based feature evaluation (HFE).
  - `adt.R` – Alternating Decision Tree.
  - `pearsoncorrelation_chi-squared_test.R` – Pearson correlation & χ² screening.
- **Model training (R)**
  - `rf.R` – Random Forest
  - `svm.R` – Support Vector Machine
  - `xgboost.R` – XGBoost
- **Example data**
  - `traindata.csv`, `testdata.csv` (example training/testing splits).

---

## Repository structure

```
CVDRP/
├── adt.R
├── cvfe.csv
├── cvfe.py
├── hfe.ipynb
├── pearsoncorrelation_chi-squared_test.R
├── rf.R
├── svm.R
├── xgboost.R
├── traindata.csv
└── testdata.csv
```

---

## Setup

### Python (for CVFE / HFE utilities)
- Python 3.9+
- Recommended packages:
  - `pandas`, `numpy`, `scikit-learn`, `jupyter`

Create an environment (optional):

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pandas numpy scikit-learn jupyter
```

### R (for modeling & statistics)
Install R (4.2+ recommended) and the following packages:

```r
install.packages(c("data.table","e1071","xgboost","randomForest","ggplot2"))
# If adt.R uses additional packages (e.g., RWeka/partykit), install as needed:
# install.packages(c("RWeka","partykit"))
```

---

## Data

- Place your **training** data in `traindata.csv` and **testing** data in `testdata.csv`.
- Ensure the **target/label column** and feature columns match what the scripts expect.

---

## Feature selection

### 1) Pearson correlation + Chi-squared test

Pearson’s correlation was used to identify highly correlated continuous features (r >= 0.9), and one feature from each correlated pair was removed. Chi-squared tests were applied to categorical variables, retaining only pairs with significant associations (p <= 0.001):

```bash
pearsoncorrelation_chi-squared_test.R

```

### 2) Alternative Decision Tree

Randomly selected values for B=50
```bash
adt.R
```

---

### 3) Cross-Validation Feature Evaluation (CVFE)

Run CVFE to rank features:

```bash
python cvfe.py
```

- Output: check `cvfe.csv` for ranked features or scores.

### 4) Hypergraph Feature Evaluation (HFE)

Launch the notebook and run cells sequentially:

```bash
jupyter notebook hfe.ipynb
```

---

## Modeling (R)

Each model script trains on `traindata.csv`, evaluates on `testdata.csv`, and prints/saves metrics and plots.

### Random Forest
```bash
Rscript rf.R
```

### Support Vector Machine
```bash
Rscript svm.R
```

### XGBoost
```bash
Rscript xgboost.R
```

---

## Workflow

1. Prepare `traindata.csv` / `testdata.csv`
2. Run feature selection (`pearsoncorrelation_chi-squared_test.R` / `adt.R` / `cvfe.py` / `hfe.ipynb`)
3. Train & evaluate (`rf.R`, `svm.R`, `xgboost.R`)
4. Compare results (AUC, accuracy, F1, etc.)

---


## Citation

Explore and run predictions using the interactive web app:
[🔗 Cardiovascular Disease Risk Prediction Web Application](https://shiny.tricities.wsu.edu/cvdr-prediction/)

If you find our web application useful, please consider citing the following works:

Akhter, S. and Miller, J.H., 2025. Evaluating Feature Selection Methods and Feature Contributions for Cardiovascular Disease Risk Prediction. 
medRxiv, pp.2025-07..

---




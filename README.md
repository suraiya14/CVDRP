# CVDRP — Cardiovascular Disease Risk Prediction

This repository provides the full workflow, datasets, and scripts used in the study:
“Web-Based Cardiovascular Disease Risk Prediction Using Machine Learning”
The project introduces an integrated framework combining feature evaluation using Pearson correlation + chi-squared filtering, alternative decision tree, cross-validation-based (CVFS), and hypergraph-based (HFE) approaches to enhance prediction performance.  
This repository includes:

- **Feature selection**
  - `cvfe.py`: Cross-validation–based feature evaluation (CVFE).
  - `hfe.ipynb`: Hypergraph-based feature evaluation (HFE).
  - `cvfe.csv`: Example/derived feature-ranking output.
- **Model training (R)**
  - `rf.R` – Random Forest
  - `svm.R` – Support Vector Machine
  - `xgboost.R` – XGBoost
  - `adt.R` – Alternating Decision Tree
  - `pearsoncorrelation_chi-squared_test.R` – Pearson correlation & χ² screening
- **Example data**
  - `traindata.csv`, `testdata.csv` (example training/testing splits).

> Note: The older README snippet on GitHub references `CVFS.py`, `HFE.ipynb`, and “validate_*” CSVs; the current repository file names are `cvfe.py`, `hfe.ipynb`, and `train/testdata.csv` respectively.

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

### 1) Cross-Validation Feature Evaluation (CVFE)

Run CVFE to rank features:

```bash
python cvfe.py
```

- Output: check `cvfe.csv` for ranked features or scores.

### 2) Hypergraph Feature Evaluation (HFE)

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

### Alternating Decision Tree
```bash
Rscript adt.R
```

### Correlation & χ² screening
```bash
Rscript pearsoncorrelation_chi-squared_test.R
```

---

## Workflow

1. Prepare `traindata.csv` / `testdata.csv`
2. Run feature selection (`cvfe.py` / `hfe.ipynb`)
3. Optional: univariate screening (`pearsoncorrelation_chi-squared_test.R`)
4. Train & evaluate (`rf.R`, `svm.R`, `xgboost.R`, `adt.R`)
5. Compare results (AUC, accuracy, F1, etc.)

---

## License

No license is currently specified in this repository. Add one (MIT, BSD-3-Clause, or Apache-2.0) if public use is intended.

---

## Citation

If you use this code, please cite the repository and related publication(s).

---

## Acknowledgments

Thanks to the maintainers and contributors of the Python and R ecosystems leveraged here.


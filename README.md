# AI Diabetes Risk Predictor
## Dia-Explain

An explainable, calibrated machine learning system for three-class diabetes risk prediction, built on the CDC BRFSS 2015 dataset (253,680 records). Combines a stacking ensemble with SHAP, LIME, model calibration, and a Human-in-the-Loop Streamlit interface.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-red)
![License](https://img.shields.io/badge/License-MIT-green)

> **Status:** Honours research project (Durban University of Technology, 2026). Model training and evaluation complete. Streamlit prototype functional.

---

## What this is

Dia-Explain predicts a patient's diabetes risk class (No Diabetes / Prediabetes / Diabetes) from 21 self-reported health indicators, and explains every prediction at two levels: global feature importance across the population (SHAP) and local contributions for the individual (LIME). It further provides counterfactual suggestions — model-grounded changes that would lower a patient's predicted risk.

The system is deliberately built as a **transparent research artefact**, not a product. Everything from the preprocessing pipeline to the calibration method to the baseline comparison is included so that every reported number can be reproduced.

## Headline results

Evaluated on a held-out test set of 50,736 records:

| Metric | Value |
|---|---|
| Macro F1 | **0.449** (95% CI 0.444 – 0.454) |
| Macro AUC-ROC | **0.757** (95% CI 0.751 – 0.763) |
| Weighted AUC-ROC | 0.816 |
| Expected Calibration Error (ECE) | **0.007** |
| Mean Brier score | 0.075 |
| SHAP fidelity (Pearson *r*) | **0.942** (*p* < 1×10⁻⁹) |

Per-class breakdown:

| Class | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| No Diabetes | 0.93 | 0.78 | 0.85 | 0.814 |
| Prediabetes | 0.03 | 0.10 | 0.04 | 0.671 |
| Diabetes | 0.37 | 0.61 | 0.46 | 0.822 |

Three-fold stratified cross-validation confirms stability: macro F1 = 0.453 ± 0.002, macro AUC = 0.757 ± 0.001.

## What worked and what didn't

The most useful part of this project, from a research perspective, is what it *fails* to do — and why.

**What worked:**
- The calibrated stack is well calibrated (ECE 0.007), meaning predicted probabilities can be trusted as probabilities, not just rankings.
- SHAP explanations are faithful to the underlying model (Pearson *r* = 0.94 under perturbation).
- Diabetes-class detection reaches recall 0.61 with AUC 0.822 — the clinically actionable class works.

**What didn't work:**
- **Prediabetes detection is essentially broken.** Recall 0.10, AUC 0.671. Two independent improvement attempts — heavier SMOTE sampling and a two-stage binary override classifier — both failed to raise macro F1 beyond 0.46.
- **The stacking ensemble does not beat a single Random Forest.** Random Forest alone achieves macro AUC 0.778 versus the stack's 0.757. On macro F1 they are within the bootstrap confidence interval. The stack's added complexity is not earned on this dataset.

These are reported as findings, not hidden.

## Model comparison

All models trained with identical preprocessing (`MinMaxScaler` → `SMOTETomek`):

| Model | Macro F1 | Macro AUC |
|---|---|---|
| DummyClassifier (stratified) | 0.300 | 0.499 |
| Logistic Regression | 0.445 | **0.777** |
| Random Forest | 0.444 | **0.778** |
| XGBoost | 0.416 | 0.776 |
| **Stacking (calibrated, thresholded)** | **0.449** | 0.757 |

The stack produces the highest macro F1 but the *lowest* AUC of any trained model. This is a real result and it drove a substantial rewrite of the paper's discussion section.

## Architecture
CDC BRFSS 2015 (253,680 × 21)
│
▼
Stratified 64/16/20 split (train / val / test)
│
▼
imblearn Pipeline
├── MinMaxScaler
├── SMOTETomek (dynamic ratios derived from training counts)
└── StackingClassifier
├── Random Forest (100 trees, depth 10, class_weight='balanced')
├── XGBoost (200 trees, depth 6, LR 0.08)
└── Logistic Regression (passthrough=True, 3-fold internal CV)
│
▼
CalibratedClassifierCV (isotonic, 3-fold)
│
▼
Threshold tuning on validation set only
│
▼
Explanation layer
├── SHAP: TreeExplainer on RF + native pred_contribs on XGB
├── LIME: LimeTabularExplainer (categorical features declared)
└── Counterfactuals: greedy search over feature modifications
│
▼
Streamlit prototype (Dia-Explain)


## Technical stack

- **Modelling:** `scikit-learn`, `xgboost`, `imbalanced-learn`
- **Explainability:** `shap`, `lime`
- **Calibration:** `sklearn.calibration.CalibratedClassifierCV` (isotonic)
- **Evaluation:** `scipy.stats`, custom bootstrap implementation
- **App:** `streamlit`, `pandas`, `numpy`, `matplotlib`
- **Environment:** conda (`diabetes_ai`, Python 3.10)

## Repository structure
dia-explain/
├── notebooks/
│ └── dia_explain_training.ipynb # Full training + evaluation pipeline
├── src/
│ ├── train.py # Standalone training script
│ └── evaluate.py # Reproduces all reported metrics
├── app/
│ └── app.py # Streamlit HitL prototype
├── data/
│ └── README.md # Dataset source and description
├── figures/
│ ├── confusion_matrix.png
│ ├── roc_curves.png
│ └── reliability_diagram.png
├── paper/
│ └── Dia_Explain_IEEE.pdf # Submitted manuscript
├── requirements.txt
└── README.md


## Reproducing the results

```bash
# 1. Clone
git clone https://github.com/YOUR-USERNAME/dia-explain.git
cd dia-explain

# 2. Environment
conda create -n dia-explain python=3.10
conda activate dia-explain
pip install -r requirements.txt

# 3. Data
# Download diabetes_012_health_indicators_BRFSS2015.csv from Kaggle:
# https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
# Place it in the project root.

# 4. Train (≈30–60 min on CPU)
python src/train.py

# 5. Reproduce metrics
python src/evaluate.py

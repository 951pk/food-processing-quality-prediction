# Industrial Food Processing Quality & Defect Rate Prediction

An end-to-end data science lifecycle and predictive machine learning pipeline designed to monitor, analyze, and forecast batch-level defect rates across industrial food manufacturing operations.

---

## 📌 Project Overview

In commercial food processing, identifying product quality anomalies late in the packaging phase leads to operational waste, excessive energy expenditure, and inventory rejection. 

This repository documents a 5-week data science workflow analyzing **1,200 batch runs** across four key food sectors (**Dairy**, **Cereal/Bakery**, **Meat/Poultry**, and **Beverages**). The objective is to evaluate thermodynamic parameters, unbound moisture dynamics, and process holding times to forecast continuous defect rates (`Defect_Rate_Pct`) prior to packaging.

---

## 🛠️ Tech Stack & Libraries

- **Language:** Python 3.10+
- **Data Manipulation:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning & Modeling:** Scikit-Learn (Linear Regression, Ridge, Random Forest, Gradient Boosting)
- **Environment:** Google Colab, Jupyter Notebook, Git/GitHub

---

## 📂 Multi-Week Architecture & Workflow

### Week 1: Exploratory Data Analysis (EDA)
- **Initial Audit:** Evaluated 1,200 batch records across raw moisture, water activity, process duration, and temperature.
- **Univariate Analysis:** Identified a bimodal distribution in Water Activity ($a_w$), highlighting pathogen proliferation thresholds ($a_w > 0.85$).
- **Multivariate Dynamics:** Quantified primary linear relationships with batch defect rates, finding strong positive correlations with post-processing moisture ($r = 0.65$) and water activity ($r = 0.53$).

### Week 2: Data Preprocessing
- **Missing Value Imputation:** Addressed missing water activity metrics using category-stratified median imputation to preserve matrix-level moisture equilibria.
- **Outlier Mitigation:** Filtered extreme physical sensor anomalies using Tukey’s $1.5 \times \text{IQR}$ bounding rules.
- **Transformation & Scaling:** Executed One-Hot Encoding for categorical variables (`Food_Category`, `Processing_Method`) and standardized continuous metrics via `StandardScaler`.
- **Dimensionality Reduction:** Conducted Principal Component Analysis (PCA), retaining over 86% of continuous system variance within the first 3 components.

### Week 3: Feature Engineering
Synthesized five novel thermodynamic and microbiological interaction metrics:
1. **Thermal Severity Index (TSI):** $(\text{Process\_Temp} \times \text{Processing\_Time}) / 100$
2. **Relative Moisture Removal Efficiency (RMRE):** $(\text{Pre\_Moist} - \text{Post\_Moist}) / \text{Pre\_Moist}$
3. **Microbial Risk Exposure Score (MRES):** $a_w / \ln(1 + \text{Preservative\_PPM})$
4. **Thermal Rate Per Minute:** $\text{Process\_Temp} / (\text{Processing\_Time} + 1)$
5. **Compound Moisture-Retention Index:** $\text{Post\_Moisture} \times a_w$

*Random Forest importance evaluation confirmed that engineered moisture-retention indicators accounted for the top predictive splits.*

### Week 4: Supervised Predictive Modeling
- Implemented an 80/20 train-test partition across 1,200 batch runs.
- Benchmarked 4 regression architectures:
  - Multiple Linear Regression
  - Ridge Regression ($L_2$ Regularization)
  - Random Forest Regressor
  - Gradient Boosting Regressor
- Non-linear tree ensembles outperformed linear benchmarks, establishing Gradient Boosting as the candidate model architecture.

### Week 5: Hyperparameter Optimization & Model Validation
- **GridSearchCV:** Explored 108 candidate hyperparameter combinations across `n_estimators`, `learning_rate`, `max_depth`, and `subsample`.
- **Optimal Hyperparameters:** `{'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.8}`
- **Validation Results:**
  - **10-Fold Cross-Validation $R^2$:** $0.7250 \pm 0.0494$
  - **Hold-Out Test $R^2$:** $0.6983$
  - **Hold-Out Test RMSE:** $0.8869\%$
  - **Hold-Out Test MAE:** $0.7070\%$
  - **Mean Residual Bias:** $+0.0789\%$ (near-zero, confirming zero systematic directional bias)

---

## 📁 Repository Structure

```text
├── Food_Processing_Analytics_Master.ipynb   # Consolidated 5-week executable pipeline
├── food_processing_batches.csv              # Initial batch-level production dataset
├── food_processing_feature_engineered.csv   # Feature-engineered modeling dataset
├── README.md                                # Project summary & technical documentation

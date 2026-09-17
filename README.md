# Customer Churn Prediction

Predicting which customers are likely to stop using a service -- and explaining *why* -- so the analysis can drive retention actions.

## Results

| Model | F1 (Churn) | AUC-ROC | Recall |
| --- | --- | --- | --- |
| **XGBoost (Tuned, holdout)** | **0.6316** | **0.8465** | **0.8021** |
| Random Forest (balanced) | 0.6283 | 0.8444 | 0.7271 |
| Logistic Regression (balanced) | 0.6258 | 0.8445 | 0.7972 |
| LightGBM (balanced) | 0.6209 | 0.8356 | 0.7453 |
| XGBoost (scale_pos_weight, untuned) | 0.6143 | 0.8344 | 0.7266 |
| XGBoost + SMOTE | 0.5919 | 0.8251 | 0.6110 |

**Best model**: Tuned XGBoost (via `RandomizedSearchCV` on a held-out dev split) evaluated on a never-touched holdout set -- AUC-ROC 0.8465 and 80% recall on churners.

## Key Findings (SHAP-validated)

1. **Low tenure** is the #1 churn driver -- new customers haven't built loyalty
2. **Month-to-month contracts** create zero switching cost
3. **No tech support / online security** reduces product stickiness
4. **Fiber optic + high monthly charges** without perceived value
5. **Electronic check** payment correlates with higher churn

## Project Structure

```
Customer Churn/
    README.md
    requirements.txt
    .gitignore
    models/
        churn_model.pkl                                             # Trained, tuned XGBoost model (produced by notebook 2)
    notebooks/
        eda_output_file.ipynb                                       # Exploratory Data Analysis (with outputs)
        feature_engineering_and_modeling_output_file.ipynb          # Modeling + SHAP (with outputs)
```

## Notebooks

### 1. EDA (`eda_output_file.ipynb`)

- Target distribution (73/27 split)
- Feature-wise churn rates
- Correlation analysis
- Key visualizations (churn by contract, tenure, services)

### 2. Modeling + Explainability (`feature_engineering_and_modeling_output_file.ipynb`)

- Feature engineering (`services_count`, `charges_per_tenure_month`, `has_tech_support_and_security`)
- 5 models with stratified 5-fold CV: Logistic Regression, Random Forest, XGBoost, LightGBM, XGBoost + SMOTE
- Class imbalance: class weighting vs SMOTE comparison
- Hyperparameter tuning (`RandomizedSearchCV`) on a dev split, with final metrics reported on a separate, untouched holdout set
- Cost-benefit threshold optimization
- SHAP global importance + individual waterfall plots
- High-risk customer profiling

## Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/nehamalik10/customer-churn-datset) -- 7,043 customers, 21 features.

## Tech Stack

Python, pandas, scikit-learn, XGBoost, LightGBM, imbalanced-learn (SMOTE), SHAP, matplotlib, seaborn

## How to Run

Upload the notebooks to [Kaggle](https://www.kaggle.com) with the dataset linked above, and run cells in order.

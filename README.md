# Customer Churn Prediction

Predicting which customers are likely to stop using a service -- and explaining *why* -- so the analysis can drive retention actions.

## Results

| Model | F1 (Churn) | AUC-ROC | Recall |
| --- | --- | --- | --- |
| **XGBoost (Tuned)** | **0.6316** | **0.8465** | **0.8021** |
| Random Forest | 0.628 | 0.844 | 0.727 |
| Logistic Regression | 0.626 | 0.845 | 0.797 |
| LightGBM | 0.621 | 0.836 | 0.745 |
| XGBoost + SMOTE | 0.592 | 0.825 | 0.611 |

**Best model**: Tuned XGBoost with AUC-ROC 0.847 and 80% recall on churners.

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
        churn_model.pkl                                    # Trained XGBoost model
    notebooks/
        eda output file.ipynb                              # Exploratory Data Analysis (with outputs)
        feature engineering and modeling output file.ipynb  # Modeling + SHAP (with outputs)
```

## Notebooks

### 1. EDA (`eda output file.ipynb`)

- Target distribution (73/27 split)
- Feature-wise churn rates
- Correlation analysis
- Key visualizations (churn by contract, tenure, services)

### 2. Modeling + Explainability (`feature engineering and modeling output file.ipynb`)

- Feature engineering (services_count, charges_per_tenure_month)
- 5 models with stratified 5-fold CV
- Class imbalance: class weighting vs SMOTE comparison
- Cost-benefit threshold optimization
- SHAP global importance + individual waterfall plots
- High-risk customer profiling

## Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/nehamalik10/customer-churn-datset) -- 7,043 customers, 21 features.

## Tech Stack

Python, pandas, scikit-learn, XGBoost, LightGBM, SHAP, matplotlib, seaborn

## How to Run

Upload the notebooks to [Kaggle](https://www.kaggle.com) with the dataset linked above, and run cells in order.

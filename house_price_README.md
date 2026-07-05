# House Price Prediction

Predicts house sale prices from property attributes (area, bedrooms, bathrooms, age,
location, condition, etc.), with feature transformations, missing-value handling, and
comparison of three regression models.

## Dataset

Two ways to run this notebook:

1. **Real data (recommended):** Download the dataset from
   [Kaggle: House Price Prediction](https://www.kaggle.com/datasets/bhanupratapbiswas/house-price-prediction),
   place the CSV next to the notebook, and rename it to `house_prices.csv`.
2. **No download needed:** If `house_prices.csv` isn't found, the notebook automatically
   generates a realistic **synthetic** housing dataset (same feature types/relationships)
   so every cell still runs end-to-end. The version delivered here was run using this
   synthetic fallback — swap in the real CSV and re-run to get results on actual Kaggle data.

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

To pin exact versions used:
```bash
pip freeze > requirements.txt
```

## Running the Notebook

```bash
jupyter notebook house_price_prediction.ipynb
```

Run all cells top to bottom.

## What the Notebook Does

1. **Loads** the dataset (real CSV if present, else synthetic fallback).
2. **EDA**: price distribution (raw vs. log-transformed), price vs. area by location,
   price by condition, and a correlation heatmap.
3. **Feature transformations**:
   - `Age` / `LotSize` — median imputation for missing values.
   - `Price` (target) — log-transformed (`log1p`) to reduce right-skew; predictions are
     converted back with `expm1` before computing metrics, so results are reported in
     original price units.
   - `Location`, `Condition` — one-hot encoded.
4. **Trains & compares**:
   - Linear Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
5. **Evaluates** with RMSE, MAE, and R² (original price scale).
6. **Residual analysis** — residual scatter plots per model, plus a residual distribution
   and predicted-vs-actual plot for the best model.
7. **Saves** the best-performing model (by RMSE):
   - `house_price_model.pkl`
   - `house_price_scaler.pkl`
   - `house_price_feature_columns.pkl`
   - `house_price_numeric_features.pkl`
8. **Inference example** — a ready-to-call `predict_house_price()` function.

## Results (from the synthetic-data run included in this notebook)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 55,393 | 37,986 | 0.86 |
| Random Forest | 59,051 | 39,254 | 0.84 |
| Gradient Boosting | 49,027 | *(see notebook)* | *(see notebook)* |

Gradient Boosting had the lowest RMSE and was saved as the final model. Numbers will
differ once you run this on the real Kaggle dataset — treat these as a demonstration
of the pipeline, not the final reportable metrics.

## Inference Example

```python
import joblib
import pandas as pd
import numpy as np

def predict_house_price(raw_house: dict):
    model = joblib.load("house_price_model.pkl")
    scaler = joblib.load("house_price_scaler.pkl")
    feature_columns = joblib.load("house_price_feature_columns.pkl")
    numeric_features = joblib.load("house_price_numeric_features.pkl")

    input_df = pd.DataFrame([raw_house])
    input_df = pd.get_dummies(input_df, columns=["Location", "Condition"], drop_first=True)
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    input_df[numeric_features] = scaler.transform(input_df[numeric_features])

    log_pred = model.predict(input_df)[0]
    return {"predicted_price": round(float(np.expm1(log_pred)), 2)}

example_house = {
    "Area": 2200, "Bedrooms": 4, "Bathrooms": 3, "Stories": 2,
    "Age": 5, "Garage": 2, "LotSize": 4500,
    "Location": "Suburb", "Condition": "Good",
}
print(predict_house_price(example_house))
# Output: {'predicted_price': 591902.49}
```

## Files in This Project

| File | Description |
|---|---|
| `house_price_prediction.ipynb` | Main notebook: EDA, preprocessing, training, evaluation, residuals, inference |
| `house_price_model.pkl` | Saved best model (Gradient Boosting) |
| `house_price_scaler.pkl` | Saved `StandardScaler` for numeric features |
| `house_price_feature_columns.pkl` | Exact column order/schema used during training |
| `house_price_numeric_features.pkl` | List of numeric columns that were scaled |
| `README.md` | This file |

## Note on the Dataset

Because this notebook was run in an environment without access to Kaggle, the results
shown were generated on a **synthetic** dataset built to mirror realistic housing data
(same columns, realistic value ranges, and price relationships). To get results on the
actual Kaggle dataset for your submission, download `house_prices.csv` from the link
above, place it next to the notebook, and re-run all cells — no code changes needed.

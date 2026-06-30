# Airbnb Price Prediction, Berlin

Predicting Airbnb listing prices in Berlin using tabular, text, and spatial features.

---

## Team Members

Sara Ossman, Qi Guan, Shrabona Mukherjee

---

## Project structure

```
ML_project/
├── data/
│   ├── raw/                          # Original files (not re-downloaded)
│   │   ├── listings.csv.gz
│   │   ├── reviews.csv.gz
│   │   └── calendar.csv.gz
│   └── processed/                    # Outputs from cleaning and feature engineering
│       ├── listings_clean.parquet
│       ├── train_hotenc.parquet
│       ├── test_hotenc.parquet
│       ├── train_targetenc.parquet
│       └── test_targetenc.parquet
│
├── src/                              # Importable Python modules
│   ├── cleaning.py                   # Data cleaning functions
│   ├── features.py                   # Feature engineering (tabular, text, spatial)
│   ├── visualize.py                  # EDA plotting functions
│   └── models.py                     # Model definitions, tuning, evaluation, plots
│
├── scripts/                          # Runnable pipeline scripts
│   ├── run_cleaning.py
│   ├── run_eda.py
│   ├── run_features.py
│   └── run_training.py
│
├── outputs/
│   ├── figures/                      # EDA plots + model evaluation figures
│   └── models/                       # Saved hyper-parameters and evaluation scores
│
├── environment.yml
└── README.md
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate ml_project
```

> Data files are included in `data/raw/`, no download needed.

---

## Pipeline

Run scripts in order from the project root:

### 1. Clean the data

```bash
python scripts/run_cleaning.py
```

Reads `data/raw/listings.csv.gz`. Applies:
- Column drops (redundant/high-missing fields)
- Price parsing, zero-price removal, log-price creation
- Minimum-nights filter (≤ 30 nights, removes long-term rentals)
- Bathroom text parsing → numeric `bathrooms_parsed`
- Bedroom/bed imputation by `room_type` group median
- Boolean t/f string conversion; rate string → float
- Review score imputation (median) + `has_reviews` flag
- Presence flags for text fields (`has_description`, `has_neighborhood_overview`, `has_host_about`)
- Host response fill (unknown → `'unknown'` / `-1`)
- `host_effort_score` composite feature
- `days_since_last_review` from `last_review` date

Saves to `data/processed/listings_clean.parquet`.

---

### 2. Explore the data

```bash
python scripts/run_eda.py
```

Reads `data/processed/listings_clean.parquet` and saves 11 figures to `outputs/figures/`:

| Figure | Description |
|---|---|
| `price_distribution.png` | Raw price and log_price histograms |
| `feature_vs_price_grid.png` | Scatter grid of numerical features vs log_price |
| `binary_vs_price.png` | Median log_price for each binary feature |
| `room_type_vs_price.png` | log_price boxplot by room type |
| `host_response_time_vs_price.png` | log_price boxplot by host response time |
| `property_type_vs_price.png` | log_price boxplot for top 5 property types |
| `price_by_neighbourhood.png` | Price boxplot for top 10 neighbourhoods |
| `price_by_accommodates.png` | Price boxplot by number of guests |
| `spatial_price_map.png` | Map of listings coloured by log_price |
| `correlation_matrix.png` | Heatmap of all numeric features |
| `review_distributions.png` | Distribution of all review score columns |

---

### 3. Build features

```bash
python scripts/run_features.py
```

Reads `data/processed/listings_clean.parquet` and `data/raw/reviews.csv.gz`.

Train/test split (80/20) and price cap (99th percentile of train) are performed first to prevent any leakage. All fitting steps (PCA, target encoding, amenity selection) use training data only.

Feature engineering steps:
1. **Amenity selection**: top 20 amenities by point-biserial correlation with log_price (min 2% frequency in train)
2. **Amenity flags** : one binary column per selected amenity + `amenities_count`
3. **Host tenure** : `host_tenure_days` relative to dataset snapshot date (2025-09-23)
4. **Description length** : `description_word_count`
5. **Description sentiment** : multilingual DistilBERT
6. **Description embeddings** : `paraphrase-multilingual-MiniLM-L12-v2` sentence embeddings, reduced to 50 PCA components (PCA fitted on train only)
7. **Spatial features** : haversine distance to Alexanderplatz and Kreuzberg
8. **Review sentiment** : VADER compound scores (mean, min, std, recency-weighted mean) on up to 10 most recent reviews per listing
9. **Categorical encoding** : produces two variants

Saves:

```
data/processed/train_hotenc.parquet       # one-hot encoded train set
data/processed/test_hotenc.parquet        # one-hot encoded test set
data/processed/train_targetenc.parquet    # target-encoded train set
data/processed/test_targetenc.parquet     # target-encoded test set

outputs/models/pca_model.joblib           # fitted PCA (50 components)
outputs/models/train_embeddings_raw.npy   # full-dim train description embeddings
outputs/models/test_embeddings_raw.npy    # full-dim test description embeddings
outputs/models/selected_amenities.json    # data-driven amenity list
outputs/models/review_sentiments_cache.parquet
```

**Encoding variants:**

- `hotenc`: `neighbourhood_cleansed`, `room_type`, `host_response_time`, `property_type` one-hot encoded (top 10 property types; rest = `Other`). 
- `targetenc`: same columns replaced by smoothed m-estimate target encoding (smoothing=10).

---

### 4. Train models

```bash
python scripts/run_training.py                       # default: hotenc
python scripts/run_training.py --encoding targetenc
```

Loads the pre-split parquet files and runs the full model sequence:

**Baselines**
1. Median Predictor (DummyRegressor — absolute floor)
2. Linear Regression
3. Ridge Regression (default α=1.0)
4. Ridge (tuned)

**Tree models**
5. Random Forest (default)
6. Random Forest (tuned: randomized search, 20 iterations)
7. XGBoost (default)
8. XGBoost (tuned: randomized search with early stopping, 30 iterations)
9. LightGBM (default)
10. LightGBM (tuned: randomized search with early stopping, 30 iterations)
11. Stacking Ensemble (tuned RF + tuned XGB + tuned LGBM → tuned Ridge meta-learner)

All tuned XGBoost and LightGBM candidates use early stopping on the validation set, so each is evaluated at its own optimal tree count rather than a fixed number.

Saves to `outputs/figures/` and `outputs/models/`:

```
outputs/figures/model_comparison_{encoding}.png
outputs/figures/feature_importance_rf.png
outputs/figures/feature_importance_xgb.png
outputs/figures/residuals_{best_model}.png
outputs/models/model_comparison_{encoding}.csv
outputs/models/best_params_{encoding}.json
```

---

## Results

All metrics are computed in EUR space (prices are exponentiated before computing RMSE/MAE/MAPE). R² is computed on log-price

### One-hot encoding (`hotenc`)

| Model | CV R² | Test R² | Test RMSE (€) | Test MAE (€) | Test MAPE (%) |
|---|---|---|---|---|---|
| Median Predictor | −0.003 | −0.000 | 91.77 | 62.02 | 52.53 |
| Linear Regression | 0.698 | 0.697 | 55.00 | 33.85 | 24.62 |
| Ridge (tuned) | 0.696 | 0.698 | 55.07 | 33.84 | 24.64 |
| Random Forest (tuned) | 0.719 | 0.723 | 56.19 | 32.65 | 23.47 |
| XGBoost (tuned) | 0.769 | 0.774 | 49.82 | 29.18 | 20.97 |
| **LightGBM (tuned)** | **0.775** | **0.777** | **49.54** | **28.60** | **20.63** |
| Stacking (RF + XGB + LGBM) | — | 0.772 | 49.03 | 29.10 | 21.05 |

### Target encoding (`targetenc`)

| Model | CV R² | Test R² | Test RMSE (€) | Test MAE (€) | Test MAPE (%) |
|---|---|---|---|---|---|
| Median Predictor | −0.003 | −0.000 | 91.77 | 62.02 | 52.53 |
| Linear Regression | 0.699 | 0.686 | 55.76 | 34.54 | 25.55 |
| Ridge (tuned) | 0.698 | 0.687 | 55.78 | 34.63 | 25.60 |
| Random Forest (tuned) | 0.728 | 0.712 | 56.47 | 32.73 | 23.70 |
| XGBoost (tuned) | 0.779 | 0.769 | 49.87 | 28.95 | 21.15 |
| **LightGBM (tuned)** | **0.775** | **0.769** | **49.56** | **29.01** | **21.14** |
| Stacking (RF + XGB + LGBM) | — | 0.766 | 49.79 | 29.05 | 21.12 |

**Hotenc outperforms targetenc** across all model classes, most noticeably for linear models and Random Forest.

**Best overall model:** LightGBM (tuned) with hotenc : **R² = 0.777, RMSE = €49.54, MAPE = 20.6%**.

The stacking ensemble does not significantly outperform the best individual model.

---

## Loading pre-built features

```python
import pandas as pd

# one-hot variant
train = pd.read_parquet('data/processed/train_hotenc.parquet')
test  = pd.read_parquet('data/processed/test_hotenc.parquet')

X_train = train.drop(columns=['id', 'price', 'log_price'])
y_train = train['log_price']
X_test  = test.drop(columns=['id', 'price', 'log_price'])
y_test  = test['log_price']

# target-encoding variant
train = pd.read_parquet('data/processed/train_targetenc.parquet')
test  = pd.read_parquet('data/processed/test_targetenc.parquet')
```

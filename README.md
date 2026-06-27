# Airbnb Price Prediction, Berlin

Predicting Airbnb listing prices in Berlin using tabular, text, and spatial features.

---

## Team Members
Sara Ossman, Qi Guan, Shrabona Mukherjee  

## Project structure

```
airbnb-price-prediction/
├── data/
│   ├── raw/                    # Original files
│   │   ├── listings.csv.gz
│   │   ├── reviews.csv.gz
│   │   └── calendar.csv.gz
│   └── processed/              # Outputs from cleaning and feature engineering
│
├── notebooks/                  # Exploratory work
│   ├── 01_Data_Processing.ipynb
│   ├── 02_feature_exploration.ipynb        
│   └── 03_models_1.ipynb
│
├── src/                        # Importable Python modules
│   ├── cleaning.py             # Data cleaning functions
│   ├── features.py             # Feature engineering (tabular, text, spatial)
│   ├── visualize.py            # EDA plotting functions
│   ├── models.py               # Model definitions and training
│   └── evaluate.py             # Metrics and evaluation
│
├── scripts/                    # Runnable pipeline scripts
│   ├── run_cleaning.py
│   ├── run_eda.py
│   ├── run_features.py
│   └── run_training.py
│
├── outputs/
│   ├── figures/                # EDA plots
│   └── models/                 # Saved model artifacts and embeddings
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

> Data files are included in `data/raw/` , no download needed.

---

## Pipeline

Run scripts in order:

### 1. Clean the data
```bash
python scripts/run_cleaning.py
```
Reads `data/raw/listings.csv.gz`, applies cleaning (price parsing, imputation, boolean fixes, date features, etc.), and saves to `data/processed/listings_clean.parquet`.

### 2. Explore the data
```bash
python scripts/run_eda.py
```
Reads `data/processed/listings_clean.parquet` and saves 9 figures to `outputs/figures/`:

| Figure | Description |
|---|---|
| `price_distribution.png` | Raw price and log_price histograms |
| `feature_vs_price_grid.png` | Scatter grid of numerical features vs log_price |
| `binary_vs_price.png` | Median log_price for each binary feature |
| `room_type_vs_price.png` | Log_price boxplot by room type |
| `host_response_time_vs_price.png` | Log_price boxplot by host response time |
| `price_by_neighbourhood.png` | Price boxplot for top 10 neighbourhoods |
| `price_by_accommodates.png` | Price boxplot by number of guests |
| `spatial_price_map.png` | Map of listings coloured by log_price |
| `correlation_matrix.png` | Heatmap of all numeric features |
| `review_distributions.png` | Distribution of all review score columns |

### 3. Build features
```bash
python scripts/run_features.py
```
Reads `data/processed/listings_clean.parquet` and `data/raw/reviews.csv.gz`. Applies feature engineering (categorical encoding, amenity flags, host tenure, description length and sentiment, multilingual embeddings with PCA, spatial distances, review sentiment) and saves:
- `data/processed/listings_features.parquet`
- `outputs/models/description_embeddings.npy`
- `outputs/models/description_embeddings_pca50.npy`

### 4. Train models
```bash
python scripts/run_training.py
```
*To be continued*

---

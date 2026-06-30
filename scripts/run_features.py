"""
Run the full feature engineering pipeline.

Usage
-----
    conda activate ml_project
    python scripts/run_features.py

What it does
------------
1. Loads  data/processed/listings_clean.parquet  and  data/raw/reviews.csv.gz
2. Performs train/test split AND price-cap derivation on the training set only
   (no leakage from test into any fitted statistic).
3. Runs the full feature engineering pipeline (amenity selection, sentiment,
   embeddings + PCA, spatial features, review sentiment, encoding).
4. Writes four output files:

   data/processed/train_hotenc.parquet     ← training set, one-hot encoded
   data/processed/test_hotenc.parquet      ← test set,     one-hot encoded
   data/processed/train_targetenc.parquet  ← training set, target encoded
   data/processed/test_targetenc.parquet   ← test set,     target encoded

5. Writes artefacts for reproducibility:

   outputs/models/pca_model.joblib            ← fitted PCA (50 components)
   outputs/models/train_embeddings_raw.npy    ← full-dim train embeddings
   outputs/models/test_embeddings_raw.npy     ← full-dim test  embeddings
   outputs/models/selected_amenities.json     ← amenity list (data-driven)

Loading in model notebooks
---------------------------
    import pandas as pd

    # one-hot variant
    train = pd.read_parquet('data/processed/train_hotenc.parquet')
    test  = pd.read_parquet('data/processed/test_hotenc.parquet')

    X_train = train.drop(columns=['id', 'price', 'log_price'])
    y_train = train['log_price']
    X_test  = test.drop(columns=['id', 'price', 'log_price'])
    y_test  = test['log_price']

    # target-encoding variant — same interface, different file
    train = pd.read_parquet('data/processed/train_targetenc.parquet')
    ...
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# make src/ importable regardless of where the script is run from
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.features import build_features_pipeline, split_and_cap

RAW_DIR       = Path('data/raw')
PROCESSED_DIR = Path('data/processed')
MODELS_DIR    = Path('outputs/models')


def main():
    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("Loading data...")
    listings = pd.read_parquet(PROCESSED_DIR / 'listings_clean.parquet')
    print(f"  Listings loaded: {listings.shape}")

    reviews = pd.read_csv(RAW_DIR / 'reviews.csv.gz', compression='gzip')
    print(f"  Reviews loaded:  {reviews.shape}")

    # ── 2. Train/test split + price cap ──────────────────────────────────────
    # Both operations happen here, before any feature fitting, to prevent
    # leakage of test-set statistics into PCA, target encoding, etc.
    print("\n" + "=" * 60)
    print("Splitting and capping prices (cap fitted on train only)...")
    train, test, cap = split_and_cap(listings)
    print(f"  Price cap: €{cap:.0f}")

    # ── 3. Feature engineering ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Running feature engineering pipeline...")
    results = build_features_pipeline(train, test, reviews)

    train_ohe = results['train_ohe']
    test_ohe  = results['test_ohe']
    train_te  = results['train_te']
    test_te   = results['test_te']

    # ── 4. Save output parquet files ─────────────────────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Saving output files...")

    outputs = {
        PROCESSED_DIR / 'train_hotenc.parquet':    train_ohe,
        PROCESSED_DIR / 'test_hotenc.parquet':     test_ohe,
        PROCESSED_DIR / 'train_targetenc.parquet': train_te,
        PROCESSED_DIR / 'test_targetenc.parquet':  test_te,
    }
    for path, df in outputs.items():
        df.to_parquet(path, index=False)
        print(f"  Saved {path}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")

    # ── 5. Save artefacts ─────────────────────────────────────────────────────
    pca_path = MODELS_DIR / 'pca_model.joblib'
    joblib.dump(results['pca'], pca_path)
    print(f"  Saved {pca_path}")

    train_emb_path = MODELS_DIR / 'train_embeddings_raw.npy'
    test_emb_path  = MODELS_DIR / 'test_embeddings_raw.npy'
    np.save(train_emb_path, results['train_emb_raw'])
    np.save(test_emb_path,  results['test_emb_raw'])
    print(f"  Saved {train_emb_path}")
    print(f"  Saved {test_emb_path}")

    amenity_path = MODELS_DIR / 'selected_amenities.json'
    with open(amenity_path, 'w') as f:
        json.dump(results['selected_amenities'], f, indent=2)
    print(f"  Saved {amenity_path}")

    # ── 6. Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Done. Summary:")
    print(f"  OHE  — train: {train_ohe.shape}, test: {test_ohe.shape}")
    print(f"  TE   — train: {train_te.shape},  test: {test_te.shape}")
    print(f"  PCA variance explained: "
          f"{results['pca'].explained_variance_ratio_.sum():.2%}")
    print(f"  Selected amenities ({len(results['selected_amenities'])}):")
    for a in results['selected_amenities']:
        print(f"    - {a}")


if __name__ == '__main__':
    main()

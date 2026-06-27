"""
Run the full feature engineering pipeline.

Input:  data/processed/listings_clean.parquet
        data/raw/reviews.csv.gz
Output: data/processed/listings_features.parquet
        outputs/models/description_embeddings.npy
        outputs/models/description_embeddings_pca50.npy
"""

import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.features import build_all_features

RAW_DIR = Path('data/raw')
PROCESSED_DIR = Path('data/processed')
MODELS_DIR = Path('outputs/models')


def main():
    print("Loading cleaned listings...")
    listings = pd.read_parquet(PROCESSED_DIR / 'listings_clean.parquet')
    print(f"  Loaded: {listings.shape}")

    print("Loading raw reviews...")
    reviews = pd.read_csv(RAW_DIR / 'reviews.csv.gz', compression='gzip')
    print(f"  Loaded: {reviews.shape}")

    print("Building features...")
    listings_features, embeddings_raw, embeddings_pca = build_all_features(listings, reviews)
    print(f"  After feature engineering: {listings_features.shape}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    out_parquet = PROCESSED_DIR / 'listings_features.parquet'
    listings_features.to_parquet(out_parquet, index=False)
    print(f"Saved → {out_parquet}")

    out_emb = MODELS_DIR / 'description_embeddings.npy'
    np.save(out_emb, embeddings_raw)
    print(f"Saved → {out_emb}")

    out_pca = MODELS_DIR / 'description_embeddings_pca50.npy'
    np.save(out_pca, embeddings_pca)
    print(f"Saved → {out_pca}")


if __name__ == '__main__':
    main()

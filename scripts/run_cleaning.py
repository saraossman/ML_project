"""
Run the full data cleaning pipeline.

Input:  data/raw/listings.csv.gz
Output: data/processed/listings_clean.parquet
"""

import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.cleaning import clean_listings

RAW_DIR = Path('data/raw')
PROCESSED_DIR = Path('data/processed')


def main():
    print("Loading raw listings...")
    listings = pd.read_csv(RAW_DIR / 'listings.csv.gz', compression='gzip', low_memory=False)
    print(f"Loaded listings with shape: {listings.shape}")

    print("Cleaning..")
    listings_clean = clean_listings(listings)
    print(f"Listings after cleaning: {listings_clean.shape}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / 'listings_clean.parquet'
    listings_clean.to_parquet(out, index=False)
    print(f"Saved cleaned data at: {out}")


if __name__ == '__main__':
    main()

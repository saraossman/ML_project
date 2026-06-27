"""
Run all EDA visualisations on the cleaned listings DataFrame.

Input:  data/processed/listings_clean.parquet
Output: outputs/figures/*.png
"""

import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.visualize import (
    plot_price_distribution,
    plot_numeric_vs_price,
    plot_binary_vs_price,
    plot_categorical_vs_price,
    plot_price_by_neighbourhood,
    plot_price_by_accommodates,
    plot_spatial_price_map,
    plot_correlation_matrix,
    plot_review_score_distributions,
)

PROCESSED_DIR = Path('data/processed')


def main():
    print("Loading cleaned listings...")
    df = pd.read_parquet(PROCESSED_DIR / 'listings_clean.parquet')
    print(f"  Loaded: {df.shape}")

    print("\nGenerating plots...")
    plot_price_distribution(df)
    plot_numeric_vs_price(df)
    plot_binary_vs_price(df)
    plot_categorical_vs_price(df)
    plot_price_by_neighbourhood(df)
    plot_price_by_accommodates(df)
    plot_spatial_price_map(df)
    plot_correlation_matrix(df)
    plot_review_score_distributions(df)

    print("\nAll figures saved to outputs/figures/")


if __name__ == '__main__':
    main()

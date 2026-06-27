"""
EDA visualisation functions for Airbnb listings.
All functions save figures to outputs/figures/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

FIGURES_DIR = Path('outputs/figures')


def _save(fig: plt.Figure, filename: str) -> None:
    """Save figure to outputs/figures/ """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out}")


# Price

def plot_price_distribution(df: pd.DataFrame) -> None:
    """Histogram of raw price and log_price """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].hist(df['price'], bins=80, color='steelblue', edgecolor='none')
    axes[0].set_title('Price, raw')
    axes[0].set_xlabel('€ per night')

    axes[1].hist(df['log_price'], bins=80, color='steelblue', edgecolor='none')
    axes[1].set_title('Price, log transformed')
    axes[1].set_xlabel('log(€ per night)')

    plt.tight_layout()
    _save(fig, 'price_distribution.png')


# Numerical features

def plot_numeric_vs_price(df: pd.DataFrame) -> None:
    """Scatter plots of numerical features vs log_price """
    numeric_features = [
        'accommodates', 'bedrooms', 'beds', 'bathrooms_parsed',
        'minimum_nights', 'number_of_reviews', 'review_scores_rating',
        'review_scores_cleanliness', 'review_scores_location',
        'host_listings_count', 'availability_365',
        'reviews_per_month', 'host_effort_score',
        'days_since_last_review',
    ]

    n = len(numeric_features)
    cols = 4
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4))
    axes = axes.flatten()

    for i, feat in enumerate(numeric_features):
        axes[i].scatter(df[feat], df['log_price'], alpha=0.1, s=5, color='steelblue')
        axes[i].set_xlabel(feat, fontsize=10)
        axes[i].set_ylabel('log_price', fontsize=10)

        mask = df[feat].notna()
        z = np.polyfit(df.loc[mask, feat], df.loc[mask, 'log_price'], 1)
        x_range = np.linspace(df[feat].min(), df[feat].max(), 100)
        axes[i].plot(x_range, np.poly1d(z)(x_range), color='coral', linewidth=1.5)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Numerical features vs log(price) — Berlin listings', fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, 'feature_vs_price_grid.png')


# Binary features

def plot_binary_vs_price(df: pd.DataFrame) -> None:
    """Bar chart grid showing median log_price for each binary feature (0 vs 1)."""
    bool_features = [
        'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified',
        'instant_bookable', 'has_reviews', 'has_description',
        'has_neighborhood_overview', 'has_host_about', 'host_location_given',
    ]

    n = len(bool_features)
    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 4))
    axes = axes.flatten()

    for i, feat in enumerate(bool_features):
        medians = df.groupby(feat)['log_price'].median()
        counts = df[feat].value_counts()

        bars = axes[i].bar(
            medians.index.astype(str),
            medians.values,
            color=['steelblue', 'coral'],
            width=0.5,
        )
        for bar, (idx, count) in zip(bars, counts.items()):
            axes[i].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'n={count}',
                ha='center', va='bottom', fontsize=9,
            )

        axes[i].set_title(feat, fontsize=10)
        axes[i].set_ylabel('median log_price', fontsize=9)
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(['No (0)', 'Yes (1)'])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Binary features vs median log(price)', fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, 'binary_vs_price.png')


# Categorical features

def plot_categorical_vs_price(df: pd.DataFrame) -> None:
    """Boxplot of log_price for each category in room_type and host_response_time."""
    cat_features = ['room_type', 'host_response_time']

    for feat in cat_features:
        fig, ax = plt.subplots(figsize=(8, 4))
        order = df.groupby(feat)['log_price'].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x=feat, y='log_price', order=order, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title(f'{feat} vs log(price)')
        plt.tight_layout()
        _save(fig, f'{feat}_vs_price.png')


# Neighbourhood

def plot_price_by_neighbourhood(df: pd.DataFrame, top_n: int = 10) -> None:
    """Boxplot of raw price for the top N neighbourhoods by listing count."""
    top = df['neighbourhood_cleansed'].value_counts().head(top_n).index
    subset = df[df['neighbourhood_cleansed'].isin(top)]
    order = subset.groupby('neighbourhood_cleansed')['price'].median().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=subset, x='neighbourhood_cleansed', y='price', order=order, ax=ax)
    ax.set_ylim(0, 400)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title(f'Price by neighbourhood (top {top_n} by listing count)')
    plt.tight_layout()
    _save(fig, 'price_by_neighbourhood.png')


# Accommodates

def plot_price_by_accommodates(df: pd.DataFrame, max_guests: int = 10) -> None:
    """Boxplot of price by number of guests"""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df[df['accommodates'] <= max_guests], x='accommodates', y='price', ax=ax)
    ax.set_ylim(0, 500)
    ax.set_title('Price vs number of guests')
    plt.tight_layout()
    _save(fig, 'price_by_accommodates.png')


#Spatial map

def plot_spatial_price_map(df: pd.DataFrame) -> None:
    """Scatter map of listings coloured by log_price."""
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        df['longitude'], df['latitude'],
        c=df['log_price'], cmap='YlOrRd',
        alpha=0.4, s=3,
    )
    plt.colorbar(scatter, ax=ax, label='log(price)')
    ax.set_title('Listing prices across Berlin')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.tight_layout()
    _save(fig, 'spatial_price_map.png')


# Correlation matrix

def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """Heatmap of correlations between all numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ['id', 'host_id']]

    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(
        corr,
        annot=True, fmt='.2f',
        cmap='coolwarm', center=0,
        annot_kws={'size': 7},
        ax=ax,
    )
    ax.set_title('Correlation matrix')
    plt.tight_layout()
    _save(fig, 'correlation_matrix.png')


# Review scores

def plot_review_score_distributions(df: pd.DataFrame) -> None:
    """Histogram panel for each review score column."""
    review_cols = [c for c in df.columns if 'review_scores' in c]

    fig, axes = plt.subplots(1, len(review_cols), figsize=(18, 3))
    for i, col in enumerate(review_cols):
        axes[i].hist(df[col], bins=30, color='steelblue', edgecolor='none')
        axes[i].set_title(col.replace('review_scores_', ''), fontsize=9)
        axes[i].set_xlim(0, 5)

    plt.suptitle('Review score distributions', fontsize=12)
    plt.tight_layout()
    _save(fig, 'review_distributions.png')
